from __future__ import annotations

import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class SegmentVisualizer:
    """
    Debug visualiser for the ReactiveSegmentPlanner, served as a live SVG stream.

    Renders the planner's per-scan state as a polar plot with the robot at the
    centre and forward pointing up (body frame: x = right = d·sin(a),
    y_up = d·cos(a)). What it shows, and why it is the thing to watch for this
    planner:

        gray dots        the raw LiDAR scan (walls + pillars), full circle
        cyan dots        the returns that fall inside the forward cone this scan
        cyan wedge       the forward cone itself — re-aimed away from the near wall
                         by the current lane (this is the wall/obstacle split window)
        yellow ray       the segment direction the gyro is holding (e_head); the
                         grid heading the straight is driven along
        red dashed arc   FRONT_TRIGGER — when the end wall crosses it, the corner arms
        red tick + label the measured end-wall distance d_front (corner trigger input)
        big dot          the detected obstacle, coloured by the camera read
                         (red / green / grey = colour not resolved)
        purple cross     during LANE_SWITCH, the target point sent to the firmware —
                         forward = beside the pillar, lateral = the mandated lane
        dashed lines     the three lanes (−L / 0 / +L) drawn along the segment
        white dot        the dead-reckoned lateral position (lat_dr) in the corridor
        hollow ring      the target lane offset the switch is driving toward
        orange ray       the commanded aim point (lookahead); drawn behind on reverse
        magenta wedges   during SETTLE, the ±90° side-wall windows, with the left /
                         right distances being equalised to centre the robot
        text block       state, grid angles, distances, lane, corner count, speed

    Open in any browser on the same network:
        http://<rpi-hostname>:8084

    Usage
    -----
        vis = SegmentVisualizer()
        vis.start()
        planner.on_debug += vis.set_debug
        ...
        vis.stop()
    """

    _DEFAULT_PORT: int = 8084

    _BG_COLOR:      str = "#0d0d0d"
    _GRID_COLOR:    str = "#2a2a2a"
    _LABEL_COLOR:   str = "#777777"
    _RAW_COLOR:     str = "#4a6a7a"
    _CONE_COLOR:    str = "#00e5ff"
    _SEG_COLOR:     str = "#ffd21e"
    _TRIGGER_COLOR: str = "#e01e1e"
    _AIM_COLOR:     str = "#ff6600"
    _LANE_COLOR:    str = "#00b070"
    _SETTLE_COLOR:  str = "#ff2090"
    _TARGET_COLOR:  str = "#b060ff"
    _RED_COLOR:     str = "#e01e1e"
    _GREEN_COLOR:   str = "#30c830"
    _GREY_COLOR:    str = "#c8a000"
    _ROBOT_COLOR:   str = "#ffffff"

    _STATE_NAMES = {0: "DRIVE_STRAIGHT", 1: "LANE_SWITCH", 2: "CORNER", 3: "DONE",
                    4: "INIT", 5: "PARK_LEAVE", 6: "PARK_SETTLE", 7: "PARK"}
    _CORNER_STATE = 2   # _S_CORNER
    _CORNER_KINDS = {0: "outer", 1: "center", 2: "inner"}
    _PARK_LEAVE_STATE = 5   # _S_PARK_LEAVE
    _PARK_STEPS = {0: "kturn", 1: "approach", 2: "reverse-out"}

    _SIZE:    int          = 720
    _PADDING: int          = 40
    _MAX_MM:  float        = 2600.0
    _RINGS:   list[float]  = [500.0, 1000.0, 1500.0, 2000.0]

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
<title>Reactive Segment Planner</title>
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
function apply() {{ svg.style.transform = `translate(${{tx}}px,${{ty}}px) scale(${{scale}})`; }}
svg.addEventListener('wheel', e => {{
  e.preventDefault();
  const r  = svg.getBoundingClientRect();
  const mx = e.clientX - (r.left + r.width/2);
  const my = e.clientY - (r.top  + r.height/2);
  const d  = e.deltaY < 0 ? 1.15 : 1/1.15;
  const ns = Math.max(1, Math.min(40, scale*d));
  tx += mx*(1 - ns/scale); ty += my*(1 - ns/scale); scale = ns; apply();
}}, {{passive:false}});
svg.addEventListener('mousedown', e => {{
  dragging=true; lx=e.clientX; ly=e.clientY; svg.classList.add('dragging'); e.preventDefault();
}});
document.addEventListener('mousemove', e => {{
  if(!dragging) return; tx+=e.clientX-lx; ty+=e.clientY-ly; lx=e.clientX; ly=e.clientY; apply();
}});
document.addEventListener('mouseup', () => {{ dragging=false; svg.classList.remove('dragging'); }});
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
                            self.wfile.write(f'data: {frame}\n\n'.encode('utf-8'))
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
        threading.Thread(target=self._server.serve_forever, daemon=True,
                         name='SegmentVisualizer-HTTP').start()
        threading.Thread(target=self._update_loop, daemon=True,
                         name='SegmentVisualizer-Update').start()
        print(f"SegmentVisualizer: http://localhost:{self._port}")

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

    def _scale(self) -> float:
        return (self._SIZE / 2 - self._PADDING) / self._MAX_MM

    def _polar(self, angle: float, dist: float) -> tuple[float, float]:
        # Body frame, forward up: x = right = d·sin(a), y_up = d·cos(a).
        d = min(dist, self._MAX_MM) * self._scale()
        return (self._SIZE / 2 + d * math.sin(angle),
                self._SIZE / 2 - d * math.cos(angle))

    def _xy(self, x_fwd: float, y_lat: float) -> tuple[float, float]:
        # Body cartesian (forward, lateral-right) -> pixels.
        s = self._scale()
        return (self._SIZE / 2 + y_lat * s, self._SIZE / 2 - x_fwd * s)

    @staticmethod
    def _f(dbg: dict, key: str, default: float = float("nan")) -> float:
        v = dbg.get(key)
        return default if v is None else float(v)

    def _render(self, dbg: dict | None) -> str:
        size   = self._SIZE
        pad    = self._PADDING
        cx     = cy = size / 2
        radius = cx - pad
        scale  = self._scale()

        parts: list[str] = [
            f'<rect width="{size}" height="{size}" fill="{self._BG_COLOR}"/>'
        ]

        # Range rings + labels
        for ring_mm in self._RINGS:
            r_px = ring_mm * scale
            parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r_px:.1f}" fill="none" '
                f'stroke="{self._GRID_COLOR}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{cx + r_px + 3:.1f}" y="{cy:.1f}" fill="{self._LABEL_COLOR}" '
                f'font-size="10" font-family="monospace" dominant-baseline="middle">'
                f'{ring_mm/1000:.1f}m</text>'
            )

        # Cross-hair + FWD label
        parts.append(f'<line x1="{cx}" y1="{pad}" x2="{cx}" y2="{size-pad}" '
                     f'stroke="{self._GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<line x1="{pad}" y1="{cy}" x2="{size-pad}" y2="{cy}" '
                     f'stroke="{self._GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<text x="{cx}" y="{pad-6}" fill="{self._LABEL_COLOR}" '
                     f'font-size="11" font-family="monospace" text-anchor="middle">FWD</text>')

        if dbg is None:
            parts.append(
                f'<text x="{cx}" y="{cy}" fill="{self._LABEL_COLOR}" font-size="14" '
                f'font-family="monospace" text-anchor="middle" dominant-baseline="middle">'
                f'waiting for planner…</text>'
            )
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{self._ROBOT_COLOR}"/>')
            return ''.join(parts)

        scan_a = dbg.get("scan_a")
        scan_r = dbg.get("scan_r")
        e_head      = self._f(dbg, "e_head", 0.0)
        cone_center = self._f(dbg, "cone_center")
        cone_half   = self._f(dbg, "cone_half", math.radians(35))
        d_front     = self._f(dbg, "d_front")
        trigger     = self._f(dbg, "front_trigger", 650.0)
        lane_offset = self._f(dbg, "lane_offset", 250.0)
        lat_dr      = self._f(dbg, "lat_dr", 0.0)
        target_off  = self._f(dbg, "target_offset", 0.0)
        aim         = self._f(dbg, "aim")
        lookahead   = self._f(dbg, "lookahead", 400.0)
        speed       = self._f(dbg, "speed", 0.0)
        state       = int(self._f(dbg, "state", 0))
        rear        = self._f(dbg, "rear")
        rear_target = self._f(dbg, "rear_target")

        # FRONT_TRIGGER ring (corner arms when the end wall crosses this)
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{trigger*scale:.1f}" fill="none" '
            f'stroke="{self._TRIGGER_COLOR}" stroke-width="1" stroke-dasharray="4 5" opacity="0.55"/>'
        )

        # Forward cone wedge (lane-compensated), only while it is meaningful
        if math.isfinite(cone_center):
            x1, y1 = self._polar(cone_center - cone_half, self._MAX_MM)
            x2, y2 = self._polar(cone_center + cone_half, self._MAX_MM)
            parts.append(
                f'<path d="M {cx} {cy} L {x1:.1f} {y1:.1f} '
                f'A {radius:.1f} {radius:.1f} 0 0 1 {x2:.1f} {y2:.1f} Z" '
                f'fill="{self._CONE_COLOR}" opacity="0.08"/>'
            )
            for edge in (cone_center - cone_half, cone_center + cone_half):
                ex, ey = self._polar(edge, self._MAX_MM)
                parts.append(
                    f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" '
                    f'stroke="{self._CONE_COLOR}" stroke-width="1" stroke-dasharray="6 6" opacity="0.4"/>'
                )

        # Lane lines (parallel to the segment, at -L / 0 / +L), + position markers
        self._draw_lanes(parts, e_head, lane_offset, lat_dr, target_off)

        # Scan points; highlight the ones inside the forward cone
        if scan_a is not None and scan_r is not None and len(scan_a) > 0:
            if math.isfinite(cone_center):
                rel = ((scan_a - cone_center + math.pi) % (2*math.pi)) - math.pi
                in_cone = abs(rel) < cone_half
            else:
                in_cone = [False] * len(scan_a)
            dots_raw, dots_cone = [], []
            for ang, rng, inc in zip(scan_a, scan_r, in_cone):
                px, py = self._polar(float(ang), float(rng))
                if inc:
                    dots_cone.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.8" fill="{self._CONE_COLOR}"/>')
                else:
                    dots_raw.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.4" fill="{self._RAW_COLOR}"/>')
            parts.append(f'<g opacity="0.65">{"".join(dots_raw)}</g>')
            parts.append(f'<g opacity="0.95">{"".join(dots_cone)}</g>')

        # Segment direction (the gyro-held heading)
        sx, sy = self._polar(e_head, self._MAX_MM * 0.42)
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{sx:.1f}" y2="{sy:.1f}" '
            f'stroke="{self._SEG_COLOR}" stroke-width="1.5" opacity="0.85"/>'
        )

        # End-wall distance marker (d_front) along the cone centre
        if math.isfinite(d_front) and math.isfinite(cone_center):
            wx, wy = self._polar(cone_center, d_front)
            # a short tick perpendicular to the heading to suggest the wall plane
            perp = cone_center + math.pi / 2
            t = 70.0 * scale
            parts.append(
                f'<line x1="{wx - t*math.sin(perp):.1f}" y1="{wy + t*math.cos(perp):.1f}" '
                f'x2="{wx + t*math.sin(perp):.1f}" y2="{wy - t*math.cos(perp):.1f}" '
                f'stroke="{self._TRIGGER_COLOR}" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{wx+8:.1f}" y="{wy:.1f}" fill="{self._TRIGGER_COLOR}" font-size="10" '
                f'font-family="monospace" dominant-baseline="middle">{int(d_front)}mm</text>'
            )

        # Obstacle
        obs_x = self._f(dbg, "obstacle_x")
        obs_y = self._f(dbg, "obstacle_y")
        if math.isfinite(obs_x) and math.isfinite(obs_y):
            code = int(self._f(dbg, "obstacle_color", 0))
            colour = {1: self._RED_COLOR, 2: self._GREEN_COLOR}.get(code, self._GREY_COLOR)
            ox, oy = self._xy(obs_x, obs_y)
            parts.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="8" fill="{colour}" '
                         f'stroke="#ffffff" stroke-width="1.5"/>')

        # Lane-switch target point: forward beside the pillar, lateral = the lane. This
        # is the exact (forward, lateral) sent to the firmware during a switch.
        tgt_x = self._f(dbg, "target_x")
        tgt_y = self._f(dbg, "target_y")
        if math.isfinite(tgt_x) and math.isfinite(tgt_y):
            gx, gy = self._xy(tgt_x, tgt_y)
            parts.append(f'<line x1="{cx}" y1="{cy}" x2="{gx:.1f}" y2="{gy:.1f}" '
                         f'stroke="{self._TARGET_COLOR}" stroke-width="1.5" '
                         f'stroke-dasharray="5 4"/>')
            parts.append(f'<line x1="{gx-6:.1f}" y1="{gy:.1f}" x2="{gx+6:.1f}" y2="{gy:.1f}" '
                         f'stroke="{self._TARGET_COLOR}" stroke-width="2"/>')
            parts.append(f'<line x1="{gx:.1f}" y1="{gy-6:.1f}" x2="{gx:.1f}" y2="{gy+6:.1f}" '
                         f'stroke="{self._TARGET_COLOR}" stroke-width="2"/>')
            parts.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="9" fill="none" '
                         f'stroke="{self._TARGET_COLOR}" stroke-width="1.5"/>')

        # Commanded aim / lookahead point (behind on reverse)
        if math.isfinite(aim):
            look = lookahead if speed >= 0 else -lookahead
            gx, gy = self._polar(aim, abs(look))
            if look < 0:
                gx, gy = 2 * cx - gx, 2 * cy - gy
            parts.append(f'<line x1="{cx}" y1="{cy}" x2="{gx:.1f}" y2="{gy:.1f}" '
                         f'stroke="{self._AIM_COLOR}" stroke-width="2.5"/>')
            parts.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="5" fill="none" '
                         f'stroke="{self._AIM_COLOR}" stroke-width="2"/>')

        # Rear-wall standoff (closed-loop post-turn reverse): the target ring behind the
        # robot and the measured rear-wall tick, along the segment's backward direction.
        rear_dir = e_head + math.pi
        if math.isfinite(rear_target):
            rtx, rty = self._polar(rear_dir, rear_target)
            perp = rear_dir + math.pi / 2
            t = 70.0 * scale
            parts.append(
                f'<line x1="{rtx - t*math.sin(perp):.1f}" y1="{rty + t*math.cos(perp):.1f}" '
                f'x2="{rtx + t*math.sin(perp):.1f}" y2="{rty - t*math.cos(perp):.1f}" '
                f'stroke="{self._SETTLE_COLOR}" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.7"/>'
            )
        if math.isfinite(rear):
            rmx, rmy = self._polar(rear_dir, rear)
            parts.append(f'<circle cx="{rmx:.1f}" cy="{rmy:.1f}" r="4" fill="{self._SETTLE_COLOR}"/>')
            parts.append(
                f'<text x="{rmx+8:.1f}" y="{rmy:.1f}" fill="{self._SETTLE_COLOR}" font-size="10" '
                f'font-family="monospace" dominant-baseline="middle">rear {int(rear)}</text>'
            )

        # Robot
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{self._ROBOT_COLOR}"/>')

        self._draw_info(parts, dbg, state)
        return ''.join(parts)

    def _draw_lanes(
        self, parts: list[str],
        e_head: float, lane_offset: float, lat_dr: float, target_off: float,
    ) -> None:
        if not math.isfinite(e_head) or lane_offset <= 0:
            return
        seg  = (math.cos(e_head), math.sin(e_head))          # body (fwd, lat) of segment dir
        perp = (-math.sin(e_head), math.cos(e_head))         # right of segment
        T    = self._MAX_MM * 0.85
        for o in (-lane_offset, 0.0, lane_offset):
            bx, by = o * perp[0], o * perp[1]
            p1 = self._xy(bx + T * seg[0], by + T * seg[1])
            p2 = self._xy(bx - T * seg[0], by - T * seg[1])
            width = 1.4 if o == 0.0 else 1.0
            parts.append(
                f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                f'stroke="{self._LANE_COLOR}" stroke-width="{width}" '
                f'stroke-dasharray="3 7" opacity="0.5"/>'
            )
        # target lane offset (hollow ring) and dead-reckoned position (filled dot)
        if math.isfinite(target_off):
            tx, ty = self._xy(target_off * perp[0], target_off * perp[1])
            parts.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="7" fill="none" '
                         f'stroke="{self._LANE_COLOR}" stroke-width="2"/>')
        if math.isfinite(lat_dr):
            lx, ly = self._xy(lat_dr * perp[0], lat_dr * perp[1])
            parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4" fill="{self._LANE_COLOR}"/>')

    def _draw_info(self, parts: list[str], dbg: dict, state: int) -> None:
        deg = math.degrees
        seeded = int(self._f(dbg, "theta_seeded", 1))
        gyro   = int(self._f(dbg, "theta_gyro", 0))
        theta_status = "seeding…" if not seeded else ("gyro-seed" if gyro else "ok")
        state_label = self._STATE_NAMES.get(state, state)
        if state == self._CORNER_STATE:
            kind = int(self._f(dbg, "corner_kind", -1))
            step = int(self._f(dbg, "corner_step", 0))
            state_label = f'{state_label} {self._CORNER_KINDS.get(kind, kind)} [{step}]'
        elif state == self._PARK_LEAVE_STATE:
            step = int(self._f(dbg, "corner_step", 0))
            state_label = f'{state_label} {self._PARK_STEPS.get(step, step)}'
        lines = [
            f'state:     {state_label}',
            f'theta:     {theta_status}',
            f'corner:    {int(self._f(dbg, "corner_count", 0))} / 12',
            f'turn_sign: {int(self._f(dbg, "turn_sign", 0)):+d}',
            f'theta:     {deg(self._f(dbg, "theta", 0)):+.1f}°',
            f'yaw:       {deg(self._f(dbg, "yaw", 0)):+.1f}°',
            f'seg_head:  {deg(self._f(dbg, "seg_heading", 0)):+.1f}°',
            f'e_head:    {deg(self._f(dbg, "e_head", 0)):+.1f}°',
            f'd_front:   {self._fmt(self._f(dbg, "d_front"))} mm',
            f'lat_dr:    {self._f(dbg, "lat_dr", 0):+.0f} -> {self._f(dbg, "target_offset", 0):+.0f} mm',
            f'aim:       {deg(self._f(dbg, "aim", 0)):+.1f}°',
            f'speed:     {self._f(dbg, "speed", 0):+.0f}',
        ]
        parts.append(''.join(
            f'<text x="8" y="{18 + i*15}" fill="{self._LABEL_COLOR}" font-size="12" '
            f'font-family="monospace" xml:space="preserve">{line}</text>'
            for i, line in enumerate(lines)
        ))

    @staticmethod
    def _fmt(v: float) -> str:
        return "--" if not math.isfinite(v) else f'{int(v)}'
