from __future__ import annotations

import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from utils import degree


class LidarVisualizer:
    """
    Debug visualiser for raw lidar scans served as a live SVG stream over HTTP.

    Each frame shows the latest single scan as a polar scatter plot with the
    robot at the centre.  Points are colour-coded by quality.

    Open in any browser on the same network:
        http://<rpi-tailscale-hostname>:8082

    Usage
    -----
        vis = LidarVisualizer()
        vis.start()
        lidar.on_scan += vis
        ...
        vis.stop()
    """

    _DEFAULT_PORT: int = 8082

    _BG_COLOR:     str = "#0d0d0d"
    _GRID_COLOR:   str = "#2a2a2a"
    _LABEL_COLOR:  str = "#555555"
    _POINT_COLOR:  str = "#00e5ff"
    _ROBOT_COLOR:  str = "#ff6600"

    # SVG canvas size and geometry
    _SIZE:    int   = 700        # px (square)
    _PADDING: int   = 40         # px from edge to the outermost range ring
    _MAX_MM:  float = 4000.0     # mm shown at the edge of the plot
    _RINGS:   list[float] = [1000.0, 2000.0, 3000.0, 4000.0]

    def __init__(self, offset_angle: degree = 0.0, port: int = _DEFAULT_PORT) -> None:
        self._offset_angle: float = math.radians(offset_angle)
        self._port = port

        self._scan:      np.ndarray | None = None
        self._scan_lock: threading.Lock    = threading.Lock()

        self._frame:      str | None      = None
        self._frame_lock: threading.Lock  = threading.Lock()

        self._server: ThreadingHTTPServer | None = None

        self._client_count:    int            = 0
        self._ever_connected:  bool           = False
        self._last_disconnect: float | None   = None
        self._count_lock:      threading.Lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Processor interface — callable by Event                             #
    # ------------------------------------------------------------------ #

    def __call__(self, scan: np.ndarray) -> None:
        with self._scan_lock:
            self._scan = scan

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the HTTP server and render loop in background daemon threads."""
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
                size = visualizer._SIZE
                html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LiDAR Scan</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#000; overflow:hidden; width:100vw; height:100vh;
         display:flex; align-items:center; justify-content:center; }}
  svg {{ cursor:grab; user-select:none; max-width:100vw; max-height:100vh; }}
  svg.dragging {{ cursor:grabbing; }}
  #hint {{ position:fixed; bottom:10px; right:12px;
           color:#444; font:11px monospace; pointer-events:none; }}
</style>
</head>
<body>
<svg id="plot" viewBox="0 0 {size} {size}"
     xmlns="http://www.w3.org/2000/svg"
     width="{size}" height="{size}">
  <g id="content"></g>
</svg>
<div id="hint">scroll: zoom &nbsp;|&nbsp; drag: pan &nbsp;|&nbsp; dbl-click: reset</div>
<script>
const svg = document.getElementById('plot');
const content = document.getElementById('content');
let scale=1, tx=0, ty=0, dragging=false, lx=0, ly=0;

function apply() {{
  svg.style.transform = `translate(${{tx}}px,${{ty}}px) scale(${{scale}})`;
}}

svg.addEventListener('wheel', e => {{
  e.preventDefault();
  const r  = svg.getBoundingClientRect();
  const mx = e.clientX - (r.left + r.width/2);
  const my = e.clientY - (r.top  + r.height/2);
  const d  = e.deltaY < 0 ? 1.15 : 1/1.15;
  const ns = Math.max(1, Math.min(40, scale*d));
  tx += mx*(1 - ns/scale);
  ty += my*(1 - ns/scale);
  scale = ns;
  apply();
}}, {{passive:false}});

svg.addEventListener('mousedown', e => {{
  dragging=true; lx=e.clientX; ly=e.clientY;
  svg.classList.add('dragging'); e.preventDefault();
}});
document.addEventListener('mousemove', e => {{
  if(!dragging) return;
  tx+=e.clientX-lx; ty+=e.clientY-ly; lx=e.clientX; ly=e.clientY; apply();
}});
document.addEventListener('mouseup', () => {{
  dragging=false; svg.classList.remove('dragging');
}});
svg.addEventListener('dblclick', () => {{ scale=1;tx=0;ty=0;apply(); }});

const src = new EventSource('/stream');
src.onmessage = e => {{ content.innerHTML = e.data; }};
</script>
</body>
</html>'''.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(html)))
                self.end_headers()
                self.wfile.write(html)

            def _sse(self) -> None:
                self.send_response(200)
                self.send_header('Content-Type',  'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()
                with visualizer._count_lock:
                    visualizer._client_count   += 1
                    visualizer._ever_connected  = True
                try:
                    while True:
                        with visualizer._frame_lock:
                            frame = visualizer._frame
                        if frame is not None:
                            self.wfile.write(
                                f'data: {frame}\n\n'.encode('utf-8')
                            )
                            self.wfile.flush()
                        time.sleep(1 / 10)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with visualizer._count_lock:
                        visualizer._client_count  -= 1
                        if visualizer._client_count == 0:
                            visualizer._last_disconnect = time.time()

        self._server                = ThreadingHTTPServer(('0.0.0.0', self._port), _Handler)
        self._server.daemon_threads = True
        threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name='LidarVisualizer-HTTP',
        ).start()
        threading.Thread(
            target=self._update_loop,
            daemon=True,
            name='LidarVisualizer-Update',
        ).start()
        print(f"LidarVisualizer: http://localhost:{self._port}")

    def _update_loop(self) -> None:
        while True:
            with self._scan_lock:
                scan = self._scan
            svg_content = self._render(scan)
            with self._frame_lock:
                self._frame = svg_content
            time.sleep(1 / 10)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()

    @property
    def should_stop(self) -> bool:
        with self._count_lock:
            if not self._ever_connected:
                return False
            if self._client_count > 0:
                return False
            if self._last_disconnect is None:
                return False
            return time.time() - self._last_disconnect > 2.0

    # ------------------------------------------------------------------ #
    #  SVG rendering                                                       #
    # ------------------------------------------------------------------ #

    def _render(self, scan: np.ndarray | None) -> str:
        size    = self._SIZE
        pad     = self._PADDING
        cx      = size / 2
        cy      = size / 2
        radius  = cx - pad           # px radius of the outermost ring
        scale   = radius / self._MAX_MM

        parts: list[str] = []

        # Background
        parts.append(f'<rect width="{size}" height="{size}" fill="{self._BG_COLOR}"/>')

        # Range rings and labels
        for ring_mm in self._RINGS:
            r_px = ring_mm * scale
            parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r_px:.1f}" '
                f'fill="none" stroke="{self._GRID_COLOR}" stroke-width="1"/>'
            )
            label = f'{int(ring_mm // 1000)}m'
            parts.append(
                f'<text x="{cx + r_px + 3:.1f}" y="{cy:.1f}" '
                f'fill="{self._LABEL_COLOR}" font-size="11" '
                f'font-family="monospace" dominant-baseline="middle">{label}</text>'
            )

        # Cross-hair axes
        parts.append(
            f'<line x1="{cx}" y1="{pad}" x2="{cx}" y2="{size - pad}" '
            f'stroke="{self._GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{pad}" y1="{cy}" x2="{size - pad}" y2="{cy}" '
            f'stroke="{self._GRID_COLOR}" stroke-width="1"/>'
        )

        # Forward direction label
        parts.append(
            f'<text x="{cx}" y="{pad - 6}" fill="{self._LABEL_COLOR}" '
            f'font-size="11" font-family="monospace" text-anchor="middle">FWD</text>'
        )

        # Scan points
        if scan is not None and len(scan) > 0:
            angles    = scan[:, 0] + self._offset_angle
            distances = scan[:, 1]

            # Ego-frame: x_ego = lateral/East (right in SVG), y_ego = forward/North (up in SVG)
            # Matches LidarProcessor._to_world_coordinates before the world rotation.
            x_ego = distances * np.sin(angles)
            y_ego = distances * np.cos(angles)

            x_px = cx + x_ego * scale
            y_px = cy - y_ego * scale

            in_range = distances <= self._MAX_MM
            x_px = x_px[in_range]
            y_px = y_px[in_range]

            # Build a single <g> with all points as small circles
            point_parts = [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.5" fill="{self._POINT_COLOR}"/>'
                for x, y in zip(x_px, y_px)
            ]
            count = len(x_px)
            parts.append(f'<g opacity="0.85">{"".join(point_parts)}</g>')

            # Point count label
            parts.append(
                f'<text x="8" y="{size - 8}" fill="{self._LABEL_COLOR}" '
                f'font-size="11" font-family="monospace">{count} pts</text>'
            )
        else:
            parts.append(
                f'<text x="{cx}" y="{cy}" fill="{self._LABEL_COLOR}" '
                f'font-size="14" font-family="monospace" '
                f'text-anchor="middle" dominant-baseline="middle">waiting for scan…</text>'
            )

        # Robot dot at centre
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="4" fill="{self._ROBOT_COLOR}"/>'
        )

        return ''.join(parts)
