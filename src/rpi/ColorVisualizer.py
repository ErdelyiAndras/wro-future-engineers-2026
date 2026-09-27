from __future__ import annotations

import base64
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from processors.BodyFrameColorSampler import BodyFrameColorSampler

_COLOR_RED   = (  0,   0, 220)   # BGR
_COLOR_GREEN = (  0, 200,   0)
_COLOR_NONE  = (  0, 200, 200)   # yellow = sampled but unclassified
_COLOR_STALE = (150, 150, 150)


class ColorVisualizer:
    """
    Debug visualiser for the reactive planner's obstacle-colour detection, served as
    a live JPEG stream over HTTP.

    Shows the live camera frame with the most recent `BodyFrameColorSampler.color_at`
    query overlaid: where the LiDAR obstacle projected into the image, the HSV sample
    patch, the red/green coverage ratios, and the verdict (RED / GREEN / none). This
    is the direct check on the colour half of the obstacle pipeline — whether the
    body-frame LiDAR cluster lands on the pillar in the image and reads the right
    colour.

    The frame streams continuously (subscribe to `camera.on_frame`); the overlay
    appears whenever the planner has recently queried a colour and fades to grey once
    the query is stale (no obstacle in view).

    Open in any browser on the same network:
        http://<rpi-hostname>:8085

    Usage
    -----
        vis = ColorVisualizer(color_sampler)
        vis.start()
        camera.on_frame += vis
        ...
        vis.stop()
    """

    _DEFAULT_PORT: int   = 8085
    _FRESH_S:      float = 1.5      # a detection older than this is drawn as stale

    def __init__(
        self,
        color_sampler: BodyFrameColorSampler,
        *,
        port: int = _DEFAULT_PORT,
        fps:  int = 10,
    ) -> None:
        self._sampler   = color_sampler
        self._port      = port
        self._interval  = 1.0 / fps
        self._last_push = 0.0

        self._frame:      str | None     = None
        self._frame_lock: threading.Lock = threading.Lock()
        self._server:     ThreadingHTTPServer | None = None

    # ------------------------------------------------------------------ #
    #  Frame intake (camera thread) + annotation                          #
    # ------------------------------------------------------------------ #

    def __call__(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_push < self._interval:
            return

        out = self._annotate(frame.copy())
        _, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        with self._frame_lock:
            self._frame = b64
        self._last_push = now

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        det  = self._sampler.last_detection()

        if det is None:
            cv2.putText(frame, "no colour query yet", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLOR_STALE, 2, cv2.LINE_AA)
            return frame

        age    = time.monotonic() - det["t"]
        stale  = age > self._FRESH_S
        colour = {"red": _COLOR_RED, "green": _COLOR_GREEN}.get(det["color"], _COLOR_NONE)
        if stale:
            colour = _COLOR_STALE

        u, v, r = det["u"], det["v"], det["radius"]
        if u is not None and v is not None and 0 <= u < w and 0 <= v < h:
            cv2.rectangle(frame, (max(0, u - r), max(0, v - r)),
                          (min(w, u + r), min(h, v + r)), colour, 2)
            cv2.drawMarker(frame, (u, v), colour, cv2.MARKER_CROSS, 18, 2)

        verdict = (det["color"] or "none").upper()
        status  = det["status"]
        lines = [
            f"colour: {verdict}" + ("  (STALE)" if stale else ""),
            f"R={det['red']:.2f}  G={det['green']:.2f}",
            f"fwd={det['forward']:.0f}  lat={det['lateral']:+.0f} mm",
            f"status: {status}   age: {age:.1f}s",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 26 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

        # If the point projected outside the frame, point toward where it went.
        if status == "off-frame" and u is not None:
            edge_x = 8 if u < 0 else (w - 8)
            cv2.arrowedLine(frame, (w // 2, h - 12), (edge_x, h - 12), colour, 2, tipLength=0.3)

        return frame

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        visualizer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == '/':
                    self._page()
                elif self.path == '/stream':
                    self._sse()
                else:
                    self.send_error(404)

            def _page(self) -> None:
                html = b'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Obstacle Colour</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111; display:flex; flex-direction:column;
         align-items:center; justify-content:center;
         min-height:100vh; overflow:hidden; }
  img { max-width:100vw; max-height:100vh; display:block; image-rendering:pixelated; }
  #status { position:fixed; bottom:8px; right:12px;
            font:11px monospace; color:#444; pointer-events:none; }
</style>
</head>
<body>
<img id="frame" alt="waiting for frame..." />
<div id="status">\xe2\x80\x93</div>
<script>
const img    = document.getElementById('frame');
const status = document.getElementById('status');
let count = 0;
const src = new EventSource('/stream');
src.onmessage = e => {
  img.src = 'data:image/jpeg;base64,' + e.data;
  status.textContent = 'frame ' + (++count);
};
</script>
</body>
</html>'''
                self.send_response(200)
                self.send_header('Content-Type',  'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(html)))
                self.end_headers()
                self.wfile.write(html)

            def _sse(self) -> None:
                self.send_response(200)
                self.send_header('Content-Type',  'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()
                try:
                    while True:
                        with visualizer._frame_lock:
                            frame = visualizer._frame
                        if frame is not None:
                            self.wfile.write(f'data: {frame}\n\n'.encode('utf-8'))
                            self.wfile.flush()
                        time.sleep(1 / 10)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server                = ThreadingHTTPServer(('0.0.0.0', self._port), _Handler)
        self._server.daemon_threads = True
        threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name='ColorVisualizer-HTTP',
        ).start()
        print(f"ColorVisualizer: http://localhost:{self._port}")

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
