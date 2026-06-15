from __future__ import annotations

import base64
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from control import CellLabel, FieldMap, EgoInformation, Obstacle, ObstacleColor
from utils import radian

class FieldMapVisualizer:
    """
    Debug visualiser served as a live SVG stream over HTTP.

    Each frame is a vector SVG — crisp at any zoom level, no pixel
    artifacts.  Frames are pushed to the browser via Server-Sent Events
    (SSE) and injected into a persistent <svg> element, so zoom and pan
    state survive updates.

    Open in any browser on the same Tailscale network:
        http://<rpi-tailscale-hostname>:8080

    Three views are available via tab buttons:
        Combined  — semantic labels overlaid on occupancy evidence (default)
        Occupancy — grayscale log-odds heatmap (white=unobserved, black=occupied)
        Semantic  — classified cells only, colour-coded by label

    Controls
    --------
    Scroll       zoom in / out toward cursor
    Drag         pan when zoomed
    Double-click reset zoom

    Usage
    -----
        vis = FieldMapVisualizer(field_map, ego_information)
        vis.start()
        while running:
            vis.update()
            time.sleep(1 / 10)
        vis.stop()
    """

    _DEFAULT_PORT: int = 8080

    # Hex RGB colours for each semantic label.
    # UNKNOWN cells show as a single dark gray if their occupancy > 0.
    _BACKGROUND:       str = "#ffffff"  # unknown / no data
    _OCCUPIED_UNKNOWN: str = "#999999"  # occupied but not yet classified

    _LABEL_COLORS: dict[int, str] = {
        int(CellLabel.WALL):         "#000000",
        int(CellLabel.OBSTACLE):     "#ffd700",  # gold — color not yet known
        int(CellLabel.PARKING_WALL): "#c850c8",  # magenta
    }

    _OBSTACLE_COLOR_RGB: dict[ObstacleColor, tuple[int, int, int]] = {
        ObstacleColor.RED:   (224,  30,  30),
        ObstacleColor.GREEN: ( 30, 200,  30),
    }

    # Palette for cluster view — one colour per cluster ID (wraps around).
    _CLUSTER_PALETTE: list[tuple[int, int, int]] = [
        (230,  25,  75),  # red
        ( 60, 180,  75),  # green
        (255, 225,  25),  # yellow
        ( 67,  99, 216),  # blue
        (245, 130,  49),  # orange
        (145,  30, 180),  # purple
        ( 66, 212, 244),  # cyan
        (240,  50, 230),  # magenta
    ]

    _ORIGIN_COLOR:        str = "#00a5ff"  # orange
    _EGO_COLOR:           str = "#00ffff"  # cyan
    _TARGET_COLOR:        str = "#ff6600"  # orange — pure-pursuit lookahead point
    _NEXT_OBSTACLE_COLOR: str = "#ffffff"  # white ring — next obstacle path planner targets
    _TEXT_COLOR:          str = "#333333"

    _ARROW_LEN:   int = 30    # SVG units = grid cells = 300 mm
    _CROSS_R:     int = 2     # robot circle radius
    _ORIGIN_SIZE: int = 3     # diamond half-size

    def __init__(
        self,
        field_map:       FieldMap,
        ego_information: EgoInformation,
        port:            int = _DEFAULT_PORT,
    ) -> None:
        self._field_map       = field_map
        self._ego_information = ego_information
        self._port            = port

        self._frame:      str | None      = None
        self._frame_lock: threading.Lock  = threading.Lock()
        self._server:     ThreadingHTTPServer | None = None

        self._client_count:    int            = 0
        self._ever_connected:  bool           = False
        self._last_disconnect: float | None   = None
        self._count_lock:      threading.Lock = threading.Lock()

        self._target:      tuple[float, float] | None = None
        self._target_lock: threading.Lock             = threading.Lock()

        self._next_obstacle:      object | None    = None
        self._next_obstacle_lock: threading.Lock   = threading.Lock()

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
                cols = FieldMap.COLS
                rows = FieldMap.ROWS
                html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FieldMap</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#000; overflow:hidden; width:100vw; height:100vh;
         display:flex; align-items:center; justify-content:center; }}
  svg {{ cursor:grab; user-select:none; max-width:100vw; max-height:100vh; }}
  svg.dragging {{ cursor:grabbing; }}
  #hint {{ position:fixed; bottom:10px; right:12px;
           color:#444; font:11px monospace; pointer-events:none; }}
  #tabs {{ position:fixed; top:10px; left:50%; transform:translateX(-50%);
           display:flex; gap:8px; z-index:10; }}
  #tabs button {{ background:#222; color:#aaa; border:1px solid #444;
                  padding:6px 18px; border-radius:4px; cursor:pointer;
                  font:12px monospace; }}
  #tabs button.active {{ background:#444; color:#fff; border-color:#888; }}
</style>
</head>
<body>
<div id="tabs">
  <button id="btn-c" onclick="setView('c')" class="active">Combined</button>
  <button id="btn-o" onclick="setView('o')">Occupancy</button>
  <button id="btn-s" onclick="setView('s')">Semantic</button>
  <button id="btn-k" onclick="setView('k')">Clusters</button>
</div>
<svg id="map" viewBox="0 0 {cols} {rows}"
     xmlns="http://www.w3.org/2000/svg"
     width="{cols}" height="{rows}">
  <defs>
    <marker id="ah" markerWidth="8" markerHeight="6"
            refX="8" refY="3" orient="auto">
      <polygon points="0 0,8 3,0 6" fill="{visualizer._EGO_COLOR}"/>
    </marker>
  </defs>
  <g id="content"></g>
</svg>
<div id="hint">scroll: zoom &nbsp;|&nbsp; drag: pan &nbsp;|&nbsp; dbl-click: reset</div>
<script>
const svg = document.getElementById('map');
const content = document.getElementById('content');
let scale=1, tx=0, ty=0, dragging=false, lx=0, ly=0;
let activeView = 'c';
let frames = {{}};

function apply() {{
  svg.style.transform = `translate(${{tx}}px,${{ty}}px) scale(${{scale}})`;
}}

function setView(v) {{
  activeView = v;
  document.querySelectorAll('#tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + v).classList.add('active');
  if (frames[v]) content.innerHTML = frames[v];
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
src.onmessage = e => {{
  frames = JSON.parse(e.data);
  content.innerHTML = frames[activeView] || '';
}};
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
        self._server.daemon_threads = True  # handler threads die with the main thread
        threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name='FieldMapVisualizer-HTTP',
        ).start()
        threading.Thread(
            target=self._update_loop,
            daemon=True,
            name='FieldMapVisualizer-Update',
        ).start()
        print(f"FieldMapVisualizer: http://localhost:{self._port}")

    def _update_loop(self) -> None:
        while True:
            self.update()
            time.sleep(1 / 10)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()

    @property
    def should_stop(self) -> bool:
        """
        True once all browser clients have disconnected and a 2-second
        grace period has elapsed (long enough to survive a page refresh).
        False until the first client has ever connected.
        """
        with self._count_lock:
            if not self._ever_connected:
                return False
            if self._client_count > 0:
                return False
            if self._last_disconnect is None:
                return False
            return time.time() - self._last_disconnect > 2.0

    # ------------------------------------------------------------------ #
    #  Target / next-obstacle subscription                                 #
    # ------------------------------------------------------------------ #

    def set_next_obstacle(self, obs) -> None:
        with self._next_obstacle_lock:
            self._next_obstacle = obs

    def set_target(self, forward_mm: float, lateral_mm: float, speed: float) -> None:
        pos, yaw = self._ego_information.get_ego_information()
        if pos is None or yaw is None:
            with self._target_lock:
                self._target = None
            return
        dx = forward_mm * math.cos(yaw) - lateral_mm * math.sin(yaw)
        dy = forward_mm * math.sin(yaw) + lateral_mm * math.cos(yaw)
        with self._target_lock:
            self._target = (pos[0] + dx, pos[1] + dy)

    # ------------------------------------------------------------------ #
    #  Frame update                                                        #
    # ------------------------------------------------------------------ #

    def update(self) -> None:
        """Render all views and push to connected browsers as JSON."""
        occupancy, semantic, obstacles = self._grid_snapshot()
        position, yaw                  = self._ego_information.get_ego_information()
        with self._target_lock:
            target = self._target
        with self._next_obstacle_lock:
            next_obstacle = self._next_obstacle

        combined    = self._render_combined_svg(occupancy, semantic, obstacles, position, yaw, target, next_obstacle)
        occ_svg     = self._render_occupancy_svg(occupancy, position, yaw, target, next_obstacle)
        sem_svg     = self._render_semantic_svg(semantic, obstacles, position, yaw, target, next_obstacle)
        cluster_svg = self._render_cluster_svg(obstacles, position, yaw, target, next_obstacle)

        payload = json.dumps({"c": combined, "o": occ_svg, "s": sem_svg, "k": cluster_svg})
        with self._frame_lock:
            self._frame = payload

    # ------------------------------------------------------------------ #
    #  SVG generation — combined view                                      #
    # ------------------------------------------------------------------ #

    def _render_combined_svg(
        self,
        occupancy:     np.ndarray,
        semantic:      np.ndarray,
        obstacles:     list[Obstacle],
        position,
        yaw,
        target,
        next_obstacle,
    ) -> str:
        rows, cols = FieldMap.ROWS, FieldMap.COLS

        img_rgb = np.full((rows, cols, 3), 255, dtype=np.uint8)

        # Unknown cells with occupancy evidence → gray
        unknown_mask = (semantic == int(CellLabel.UNKNOWN)) & (occupancy > 0.0)
        img_rgb[unknown_mask] = self._hex_to_rgb(self._OCCUPIED_UNKNOWN)

        # Semantic labeled cells on top — only where occupancy evidence exists
        for label_int, hex_color in self._LABEL_COLORS.items():
            img_rgb[(semantic == label_int) & (occupancy > 0.0)] = self._hex_to_rgb(hex_color)

        # Obstacle color overlay
        for obs in obstacles:
            if obs.color is not None and obs.cells:
                cell_arr = np.array(list(obs.cells), dtype=int)
                img_rgb[cell_arr[:, 0], cell_arr[:, 1]] = self._OBSTACLE_COLOR_RGB[obs.color]

        img_bgr = cv2.cvtColor(np.flipud(img_rgb), cv2.COLOR_RGB2BGR)
        _, buf  = cv2.imencode('.png', img_bgr)
        b64     = base64.b64encode(buf.tobytes()).decode('ascii')

        parts = [
            f'<rect width="{cols}" height="{rows}" fill="{self._BACKGROUND}"/>',
            f'<image x="0" y="0" width="{cols}" height="{rows}" '
            f'image-rendering="pixelated" href="data:image/png;base64,{b64}"/>',
            self._origin_svg(),
        ]
        if next_obstacle is not None:
            parts.append(self._next_obstacle_svg(next_obstacle))
        if target is not None:
            parts.append(self._target_svg(target))
        if position is not None and yaw is not None:
            parts.append(self._ego_svg(np.array(position, dtype=float), yaw))
        return ''.join(parts)

    # ------------------------------------------------------------------ #
    #  SVG generation — occupancy view                                     #
    # ------------------------------------------------------------------ #

    def _render_occupancy_svg(
        self,
        occupancy:     np.ndarray,
        position,
        yaw,
        target,
        next_obstacle,
    ) -> str:
        """
        Grayscale log-odds heatmap as an embedded PNG with pixelated scaling.

        Colour mapping:
            unobserved (log-odds == 0) → white (255)
            free       (log-odds < 0)  → light gray, approaching 220
            occupied   (log-odds > 0)  → dark gray / black, approaching 0
        """
        rows, cols = FieldMap.ROWS, FieldMap.COLS
        lo_range   = FieldMap.L_MAX - FieldMap.L_MIN

        img          = np.full((rows, cols), 255, dtype=np.uint8)
        observed     = occupancy != 0.0
        vals         = np.clip(occupancy[observed], FieldMap.L_MIN, FieldMap.L_MAX)
        img[observed] = ((1.0 - (vals - FieldMap.L_MIN) / lo_range) * 220).astype(np.uint8)

        _, buf = cv2.imencode('.png', np.flipud(img))
        b64    = base64.b64encode(buf.tobytes()).decode('ascii')

        parts = [
            f'<rect width="{cols}" height="{rows}" fill="{self._BACKGROUND}"/>',
            f'<image x="0" y="0" width="{cols}" height="{rows}" '
            f'image-rendering="pixelated" href="data:image/png;base64,{b64}"/>',
            self._origin_svg(),
        ]
        if next_obstacle is not None:
            parts.append(self._next_obstacle_svg(next_obstacle))
        if target is not None:
            parts.append(self._target_svg(target))
        if position is not None and yaw is not None:
            parts.append(self._ego_svg(np.array(position, dtype=float), yaw))
        return ''.join(parts)

    # ------------------------------------------------------------------ #
    #  SVG generation — semantic view                                      #
    # ------------------------------------------------------------------ #

    def _render_semantic_svg(
        self,
        semantic:      np.ndarray,
        obstacles:     list[Obstacle],
        position,
        yaw,
        target,
        next_obstacle,
    ) -> str:
        """
        Colour-coded semantic label map as an embedded PNG with pixelated scaling.
        Only classified cells are shown; unclassified cells are white.
        """
        rows, cols = FieldMap.ROWS, FieldMap.COLS

        img_rgb = np.full((rows, cols, 3), 255, dtype=np.uint8)
        for label_int, hex_color in self._LABEL_COLORS.items():
            img_rgb[semantic == label_int] = self._hex_to_rgb(hex_color)

        # Obstacle color overlay
        for obs in obstacles:
            if obs.color is not None and obs.cells:
                cell_arr = np.array(list(obs.cells), dtype=int)
                img_rgb[cell_arr[:, 0], cell_arr[:, 1]] = self._OBSTACLE_COLOR_RGB[obs.color]

        img_bgr = cv2.cvtColor(np.flipud(img_rgb), cv2.COLOR_RGB2BGR)
        _, buf  = cv2.imencode('.png', img_bgr)
        b64     = base64.b64encode(buf.tobytes()).decode('ascii')

        parts = [
            f'<rect width="{cols}" height="{rows}" fill="{self._BACKGROUND}"/>',
            f'<image x="0" y="0" width="{cols}" height="{rows}" '
            f'image-rendering="pixelated" href="data:image/png;base64,{b64}"/>',
            self._origin_svg(),
        ]
        if next_obstacle is not None:
            parts.append(self._next_obstacle_svg(next_obstacle))
        if target is not None:
            parts.append(self._target_svg(target))
        if position is not None and yaw is not None:
            parts.append(self._ego_svg(np.array(position, dtype=float), yaw))
        return ''.join(parts)

    # ------------------------------------------------------------------ #
    #  SVG generation — cluster view                                       #
    # ------------------------------------------------------------------ #

    def _render_cluster_svg(
        self,
        obstacles:     list[Obstacle],
        position,
        yaw,
        target,
        next_obstacle,
    ) -> str:
        rows, cols = FieldMap.ROWS, FieldMap.COLS
        palette    = self._CLUSTER_PALETTE

        img_rgb = np.full((rows, cols, 3), 255, dtype=np.uint8)
        for idx, obs in enumerate(obstacles):
            if obs.cells:
                cell_arr = np.array(list(obs.cells), dtype=int)
                img_rgb[cell_arr[:, 0], cell_arr[:, 1]] = palette[idx % len(palette)]

        img_bgr = cv2.cvtColor(np.flipud(img_rgb), cv2.COLOR_RGB2BGR)
        _, buf  = cv2.imencode('.png', img_bgr)
        b64     = base64.b64encode(buf.tobytes()).decode('ascii')

        parts = [
            f'<rect width="{cols}" height="{rows}" fill="{self._BACKGROUND}"/>',
            f'<image x="0" y="0" width="{cols}" height="{rows}" '
            f'image-rendering="pixelated" href="data:image/png;base64,{b64}"/>',
            self._origin_svg(),
        ]
        if next_obstacle is not None:
            parts.append(self._next_obstacle_svg(next_obstacle))
        if target is not None:
            parts.append(self._target_svg(target))
        if position is not None and yaw is not None:
            parts.append(self._ego_svg(np.array(position, dtype=float), yaw))
        return ''.join(parts)

    # ------------------------------------------------------------------ #
    #  SVG helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip('#')
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def _next_obstacle_svg(self, obs) -> str:
        # centroid is (east_mm, north_mm); _grid_idx_from_world_coordinates wants (east, north) row
        centroid = obs.centroid  # np.array([east, north])
        arr = np.array([[centroid[0], centroid[1]]], dtype=float)
        rows_idx, cols_idx = FieldMap._grid_idx_from_world_coordinates(arr)
        col = int(cols_idx[0])
        row = FieldMap.ROWS - 1 - int(rows_idx[0])

        if not FieldMap._valid_mask(np.array([row]), np.array([col]))[0]:
            return ''

        color = obs.color
        if color is not None:
            rgb = self._OBSTACLE_COLOR_RGB[color]
            stroke = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
        else:
            stroke = self._NEXT_OBSTACLE_COLOR

        r = self._CROSS_R + 6
        return (
            f'<circle cx="{col}" cy="{row}" r="{r}" fill="none" '
            f'stroke="{stroke}" stroke-width="2" stroke-dasharray="3 2"/>'
        )

    def _origin_svg(self) -> str:
        rows, cols = FieldMap._grid_idx_from_world_coordinates(np.zeros((1, 2)))
        col = int(cols[0])
        row = FieldMap.ROWS - 1 - int(rows[0])
        s   = self._ORIGIN_SIZE
        pts = f"{col},{row-s} {col+s},{row} {col},{row+s} {col-s},{row}"
        return f'<polygon points="{pts}" fill="{self._ORIGIN_COLOR}"/>'

    def _target_svg(self, pos_mm: tuple[float, float]) -> str:
        arr = np.array([[pos_mm[1], pos_mm[0]]], dtype=float)
        rows_idx, cols_idx = FieldMap._grid_idx_from_world_coordinates(arr)
        col = int(cols_idx[0])
        row = FieldMap.ROWS - 1 - int(rows_idx[0])

        if not FieldMap._valid_mask(np.array([row]), np.array([col]))[0]:
            return ''

        r = self._CROSS_R + 2
        c = self._TARGET_COLOR
        circle = f'<circle cx="{col}" cy="{row}" r="{r}" fill="none" stroke="{c}" stroke-width="1"/>'
        hline  = f'<line x1="{col-r}" y1="{row}" x2="{col+r}" y2="{row}" stroke="{c}" stroke-width="1"/>'
        vline  = f'<line x1="{col}" y1="{row-r}" x2="{col}" y2="{row+r}" stroke="{c}" stroke-width="1"/>'
        return circle + hline + vline

    def _ego_svg(self, pos_mm: np.ndarray, yaw: radian) -> str:
        rows, cols = FieldMap._grid_idx_from_world_coordinates(pos_mm[[1, 0]][np.newaxis])
        col = int(cols[0])
        row = FieldMap.ROWS - 1 - int(rows[0])

        if not FieldMap._valid_mask(np.array([row]), np.array([col]))[0]:
            return ''

        # Heading arrow
        x2 = col + math.sin(yaw) * self._ARROW_LEN
        y2 = row - math.cos(yaw) * self._ARROW_LEN

        arrow = (
            f'<line x1="{col}" y1="{row}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{self._EGO_COLOR}" stroke-width="1" '
            f'marker-end="url(#ah)"/>'
        )
        circle = (
            f'<circle cx="{col}" cy="{row}" r="{self._CROSS_R}" '
            f'fill="{self._EGO_COLOR}"/>'
        )

        yaw_deg = math.degrees(yaw)
        lines   = [
            f'x: {pos_mm[1]:+.0f} mm',
            f'y: {pos_mm[0]:+.0f} mm',
            f'yaw: {yaw_deg:+.1f}°',
        ]
        text = ''.join(
            f'<text x="12" y="{28 + i * 24}" fill="{self._TEXT_COLOR}" '
            f'font-size="22" font-family="monospace">{line}</text>'
            for i, line in enumerate(lines)
        )

        return arrow + circle + text

    # ------------------------------------------------------------------ #
    #  Grid snapshot                                                       #
    # ------------------------------------------------------------------ #

    def _grid_snapshot(self) -> tuple[np.ndarray, np.ndarray, list[Obstacle]]:
        return self._field_map.snapshot()
