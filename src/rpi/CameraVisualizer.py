from __future__ import annotations

import base64
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from control import FieldMap, EgoInformation, ObstacleColor
from processors import CameraIntrinsics, CameraExtrinsics

_COLOR_RED          = (  0,   0, 220)
_COLOR_GREEN        = (  0, 200,   0)
_COLOR_UNCLASSIFIED = (200, 200,   0)


class CameraVisualizer:
    """
    Debug visualiser for camera frames served as a live JPEG stream over HTTP.

    Frames are annotated with projected obstacle positions and their voted
    colors, then pushed to the browser via Server-Sent Events (SSE).

    Open in any browser on the same network:
        http://<rpi-tailscale-hostname>:8081

    Usage
    -----
        vis = CameraVisualizer(field_map, ego_information, intrinsics, extrinsics)
        vis.start()
        camera.on_frame += vis
        ...
        vis.stop()
    """

    _DEFAULT_PORT: int = 8081

    def __init__(
        self,
        field_map:       FieldMap,
        ego_information: EgoInformation,
        intrinsics:      CameraIntrinsics,
        extrinsics:      CameraExtrinsics,
        *,
        sample_radius:   int = 20,
        port:            int = _DEFAULT_PORT,
        fps:             int = 10,
    ) -> None:
        self._field_map:       FieldMap         = field_map
        self._ego_information: EgoInformation   = ego_information
        self._intrinsics:      CameraIntrinsics = intrinsics
        self._extrinsics:      CameraExtrinsics = extrinsics
        self._sample_radius:   int              = sample_radius
        self._port             = port
        self._interval         = 1.0 / fps
        self._last_push        = 0.0

        self._frame:      str | None     = None
        self._frame_lock: threading.Lock = threading.Lock()
        self._server:     ThreadingHTTPServer | None = None

    def __call__(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_push < self._interval:
            return

        ego_pos, yaw = self._ego_information.get_ego_information()
        if ego_pos is not None and yaw is not None:
            obstacles = self._field_map.get_obstacles()
            if obstacles:
                frame = self._annotate(frame, np.asarray(ego_pos, dtype=np.float64), yaw, obstacles)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        with self._frame_lock:
            self._frame = b64
        self._last_push = now

    # ------------------------------------------------------------------ #
    #  Annotation                                                          #
    # ------------------------------------------------------------------ #

    def _annotate(
        self,
        frame:     np.ndarray,
        ego_pos:   np.ndarray,
        yaw:       float,
        obstacles: list,
    ) -> np.ndarray:
        h, w  = frame.shape[:2]
        out   = frame.copy()
        r     = self._sample_radius

        for obs in obstacles:
            P_cam = self._world_to_camera(obs.centroid.reshape(1, 2), ego_pos, yaw)
            if P_cam[0, 2] <= 0:
                continue

            pts, _ = cv2.projectPoints(
                P_cam.astype(np.float32),
                np.zeros(3), np.zeros(3),
                self._intrinsics.K,
                self._intrinsics.dist_coeffs,
            )
            u = int(pts[0, 0, 0])
            v = int(pts[0, 0, 1])

            if not (0 <= u < w and 0 <= v < h):
                continue

            label = obs.color
            if label == ObstacleColor.RED:
                color = _COLOR_RED
            elif label == ObstacleColor.GREEN:
                color = _COLOR_GREEN
            else:
                color = _COLOR_UNCLASSIFIED

            depth_mm = float(P_cam[0, 2])
            cv2.rectangle(out,
                          (max(0, u - r), max(0, v - r)),
                          (min(w, u + r), min(h, v + r)),
                          color, 1)
            cv2.circle(out, (u, v), 6, color, -1)
            cv2.putText(out, f"{depth_mm:.0f}mm", (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return out

    def _world_to_camera(
        self,
        world_points: np.ndarray,
        ego_pos:      np.ndarray,
        yaw:          float,
    ) -> np.ndarray:
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        R_w2e = np.array([
            [ cos_yaw, -sin_yaw],
            [ sin_yaw,  cos_yaw],
        ])

        d_xy     = world_points - ego_pos[[1, 0]]
        P_ego_xy = (R_w2e @ d_xy.T).T

        N     = len(world_points)
        P_ego = np.column_stack([P_ego_xy, np.zeros(N)])

        P_rel = P_ego - self._extrinsics.translation
        P_cam = (self._extrinsics.R @ P_rel.T).T

        return P_cam

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
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
<title>Camera</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111; display:flex; flex-direction:column;
         align-items:center; justify-content:center;
         min-height:100vh; overflow:hidden; }
  img { max-width:100vw; max-height:100vh; display:block; }
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
        thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name='CameraVisualizer-HTTP',
        )
        thread.start()
        print(f"CameraVisualizer: http://localhost:{self._port}")

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
