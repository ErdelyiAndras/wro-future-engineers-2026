from __future__ import annotations

import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from processors.FollowTheGapPlanner import FollowTheGapPlanner


class FTGVisualizer:
    """
    Debug visualiser for the FollowTheGapPlanner served as a live SVG stream.

    Renders the planner's per-cycle scan-space state as a polar plot with the
    robot at the centre, forward pointing up:

        gray dots      binned raw ranges (full circle)
        purple dots    walls recalled from the occupancy grid (memory fusion),
                       i.e. returns the live scan dropped off black walls
        green/red dots disparity-extended ranges inside the forward window
                       (green = deep enough to count as free, red = blocked)
        green outline  the extended-range profile the gap search runs on
        cyan arrow     travel direction (rear-mask centre) and window edges
        orange ray     committed steering direction, dot at the lookahead point
        dashed ring    the free-space threshold (_CLEAR_MIN)
        pink wedges    wrong side of a coloured pillar, closed by the pass-side mask
        big dots       detected obstacles (red/green = camera colour, grey = colour
                       not yet read) at their bearing and range
        faint wedges   sectors still unknown after fusion (no live or remembered
                       return) — a dropout the map has not yet filled

    Open in any browser on the same network:
        http://<rpi-tailscale-hostname>:8083

    Usage
    -----
        vis = FTGVisualizer()
        vis.start()
        planner.on_debug += vis.set_debug
        ...
        vis.stop()
    """

    _DEFAULT_PORT: int = 8083

    _BG_COLOR:      str = "#0d0d0d"
    _GRID_COLOR:    str = "#2a2a2a"
    _LABEL_COLOR:   str = "#555555"
    _RAW_COLOR:     str = "#4a6a7a"
    _FREE_COLOR:    str = "#30c830"
    _BLOCKED_COLOR: str = "#e01e1e"
    _TRAVEL_COLOR:  str = "#00e5ff"
    _CHOSEN_COLOR:  str = "#ff6600"
    _UNKNOWN_COLOR: str = "#c8a000"
    _MEMORY_COLOR:  str = "#b060ff"
    _PASS_COLOR:    str = "#ff2090"
    _ROBOT_COLOR:   str = "#ffffff"

    # SVG canvas size and geometry
    _SIZE:    int   = 700        # px (square)
    _PADDING: int   = 40         # px from edge to the outermost range ring
    _MAX_MM:  float = 3200.0     # mm shown at the edge of the plot
    _RINGS:   list[float] = [1000.0, 2000.0, 3000.0]

    def __init__(self, port: int = _DEFAULT_PORT) -> None:
        self._port = port

        self._debug:      dict | None    = None
        self._debug_lock: threading.Lock = threading.Lock()

        self._frame:      str | None     = None
        self._frame_lock: threading.Lock = threading.Lock()

        self._server: ThreadingHTTPServer | None = None

        self._client_count:    int            = 0
        self._ever_connected:  bool           = False
        self._last_disconnect: float | None   = None
        self._count_lock:      threading.Lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Subscription — hook to planner.on_debug                            #
    # ------------------------------------------------------------------ #

    def set_debug(self, debug: dict) -> None:
        with self._debug_lock:
            self._debug = debug

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
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
<title>Follow-the-Gap</title>
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
            name='FTGVisualizer-HTTP',
        ).start()
        threading.Thread(
            target=self._update_loop,
            daemon=True,
            name='FTGVisualizer-Update',
        ).start()
        print(f"FTGVisualizer: http://localhost:{self._port}")

    def _update_loop(self) -> None:
        while True:
            with self._debug_lock:
                debug = self._debug
            svg_content = self._render(debug)
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
    #  SVG rendering                                                      #
    # ------------------------------------------------------------------ #

    def _pt(self, angle: float, dist: float) -> tuple[float, float]:
        # Body frame with forward up: x = right = d·sin(a), y_up = d·cos(a).
        scale = (self._SIZE / 2 - self._PADDING) / self._MAX_MM
        d     = min(dist, self._MAX_MM) * scale
        return (self._SIZE / 2 + d * math.sin(angle),
                self._SIZE / 2 - d * math.cos(angle))

    def _render(self, dbg: dict | None) -> str:
        size   = self._SIZE
        pad    = self._PADDING
        cx     = cy = size / 2
        radius = cx - pad
        scale  = radius / self._MAX_MM

        parts: list[str] = [
            f'<rect width="{size}" height="{size}" fill="{self._BG_COLOR}"/>'
        ]

        # Range rings and labels
        for ring_mm in self._RINGS:
            r_px = ring_mm * scale
            parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r_px:.1f}" '
                f'fill="none" stroke="{self._GRID_COLOR}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{cx + r_px + 3:.1f}" y="{cy:.1f}" '
                f'fill="{self._LABEL_COLOR}" font-size="11" '
                f'font-family="monospace" dominant-baseline="middle">'
                f'{int(ring_mm // 1000)}m</text>'
            )

        # Free-space threshold ring
        clear_px = FollowTheGapPlanner._CLEAR_MIN * scale
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{clear_px:.1f}" fill="none" '
            f'stroke="{self._BLOCKED_COLOR}" stroke-width="1" '
            f'stroke-dasharray="4 4" opacity="0.6"/>'
        )

        # Cross-hair axes and forward label
        parts.append(
            f'<line x1="{cx}" y1="{pad}" x2="{cx}" y2="{size - pad}" '
            f'stroke="{self._GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{pad}" y1="{cy}" x2="{size - pad}" y2="{cy}" '
            f'stroke="{self._GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{cx}" y="{pad - 6}" fill="{self._LABEL_COLOR}" '
            f'font-size="11" font-family="monospace" text-anchor="middle">FWD</text>'
        )

        if dbg is None:
            parts.append(
                f'<text x="{cx}" y="{cy}" fill="{self._LABEL_COLOR}" '
                f'font-size="14" font-family="monospace" '
                f'text-anchor="middle" dominant-baseline="middle">'
                f'waiting for planner…</text>'
            )
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{self._ROBOT_COLOR}"/>')
            return ''.join(parts)

        bin_angles = dbg["bin_angles"]
        ranges     = dbg["ranges"]
        rel        = dbg["rel"]
        ext        = dbg["extended"]
        travel     = dbg["travel_body"]
        chosen     = dbg["chosen"]
        lookahead  = dbg["lookahead"]
        speed      = dbg["speed"]
        laps       = dbg["laps"]

        # Unknown sectors (no-return dropout kept open): faint wedges so black-
        # wall absorption is visible at a glance. Older debug dicts lack the key.
        unknown = dbg.get("unknown")
        if unknown is not None and unknown.any():
            half = math.pi / len(bin_angles)   # half bin width
            for a, u in zip(bin_angles, unknown):
                if not u:
                    continue
                x1, y1 = self._pt(a - half, self._MAX_MM)
                x2, y2 = self._pt(a + half, self._MAX_MM)
                parts.append(
                    f'<path d="M {cx} {cy} L {x1:.1f} {y1:.1f} '
                    f'A {radius:.1f} {radius:.1f} 0 0 1 {x2:.1f} {y2:.1f} Z" '
                    f'fill="{self._UNKNOWN_COLOR}" opacity="0.12"/>'
                )

        # Pass-side mask: the wrong side of a coloured pillar, closed so the gap
        # search routes the mandated way. Older/obstacle-free frames carry None.
        pass_masked = dbg.get("pass_masked")
        if pass_masked is not None and pass_masked.any():
            half = math.pi / len(bin_angles)
            for a, m in zip(bin_angles, pass_masked):
                if not m:
                    continue
                x1, y1 = self._pt(a - half, self._MAX_MM)
                x2, y2 = self._pt(a + half, self._MAX_MM)
                parts.append(
                    f'<path d="M {cx} {cy} L {x1:.1f} {y1:.1f} '
                    f'A {radius:.1f} {radius:.1f} 0 0 1 {x2:.1f} {y2:.1f} Z" '
                    f'fill="{self._PASS_COLOR}" opacity="0.20"/>'
                )

        # Raw binned ranges, full circle
        raw = [
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.5" fill="{self._RAW_COLOR}"/>'
            for x, y in (self._pt(a, r) for a, r in zip(bin_angles, ranges))
        ]
        parts.append(f'<g opacity="0.7">{"".join(raw)}</g>')

        # Walls recalled from the occupancy grid (memory fusion). Older debug
        # dicts, or a planner with memory disabled, carry no/None "memory".
        memory = dbg.get("memory")
        if memory is not None:
            mem_dots = [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.6" fill="{self._MEMORY_COLOR}"/>'
                for x, y in (self._pt(a, r) for a, r in zip(bin_angles, memory)
                             if math.isfinite(r))
            ]
            if mem_dots:
                parts.append(f'<g opacity="0.8">{"".join(mem_dots)}</g>')

        # Forward-window edges and travel-direction arrow
        for edge in (travel - FollowTheGapPlanner._FORWARD_HALF,
                     travel + FollowTheGapPlanner._FORWARD_HALF):
            ex, ey = self._pt(edge, self._MAX_MM)
            parts.append(
                f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" '
                f'stroke="{self._TRAVEL_COLOR}" stroke-width="1" '
                f'stroke-dasharray="6 6" opacity="0.4"/>'
            )
        tx, ty = self._pt(travel, 800.0)
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{tx:.1f}" y2="{ty:.1f}" '
            f'stroke="{self._TRAVEL_COLOR}" stroke-width="1.5" opacity="0.8"/>'
        )

        # Disparity-extended profile inside the window: outline + free/blocked dots
        if len(rel) > 0:
            free = ext >= FollowTheGapPlanner._CLEAR_MIN
            pts  = [self._pt(travel + a, r) for a, r in zip(rel, ext)]
            outline = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
            parts.append(
                f'<polyline points="{outline}" fill="none" '
                f'stroke="{self._FREE_COLOR}" stroke-width="1" opacity="0.5"/>'
            )
            dots = [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" '
                f'fill="{self._FREE_COLOR if f else self._BLOCKED_COLOR}"/>'
                for (x, y), f in zip(pts, free)
            ]
            parts.append(f'<g opacity="0.9">{"".join(dots)}</g>')

        # Committed direction and lookahead point
        gx, gy = self._pt(chosen, lookahead)
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{gx:.1f}" y2="{gy:.1f}" '
            f'stroke="{self._CHOSEN_COLOR}" stroke-width="2.5"/>'
        )
        parts.append(
            f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="5" fill="none" '
            f'stroke="{self._CHOSEN_COLOR}" stroke-width="2"/>'
        )

        # Detected obstacles: a dot at each pillar's bearing/range, coloured by
        # what the camera read (red / green / grey = colour not yet known). This
        # is the direct check on the pass-side pipeline: no dot = nothing detected,
        # grey dot = detected but uncoloured, red/green = ready for the mask.
        obstacles = dbg.get("obstacles")
        if obstacles is not None and len(obstacles) > 0:
            fill = {1: self._BLOCKED_COLOR, 2: self._FREE_COLOR}
            for bearing, rng, code in obstacles:
                ox, oy = self._pt(bearing, rng)
                colour = fill.get(int(code), self._UNKNOWN_COLOR)
                parts.append(
                    f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="7" fill="{colour}" '
                    f'stroke="#ffffff" stroke-width="1.5"/>'
                )
                parts.append(
                    f'<text x="{ox + 9:.1f}" y="{oy:.1f}" fill="{self._LABEL_COLOR}" '
                    f'font-size="10" font-family="monospace" dominant-baseline="middle">'
                    f'{int(rng)}mm</text>'
                )

        # Robot dot at centre
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{self._ROBOT_COLOR}"/>')

        # Info block
        lines = [
            f'lap:       {laps}',
            f'speed:     {speed:.0f}',
            f'lookahead: {lookahead:.0f} mm',
            f'travel:    {math.degrees(travel):+.1f}°',
            f'chosen:    {math.degrees(chosen):+.1f}°',
        ]
        if dbg.get("inspecting"):
            lines.append('** INSPECTING **')
        parts.append(''.join(
            f'<text x="8" y="{20 + i * 16}" fill="{self._LABEL_COLOR}" '
            f'font-size="12" font-family="monospace" xml:space="preserve">{line}</text>'
            for i, line in enumerate(lines)
        ))

        return ''.join(parts)
