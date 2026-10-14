from __future__ import annotations

import sys
import threading
import time
import webbrowser

from recording.Recording import Recording
from visu.LidarVisualizer import LidarVisualizer
from visu.PathPlanningVisualizer import PathPlanningVisualizer

# Must match the mains (main_obstacle.py / main_open.py) so the LiDAR view is oriented the same.
_LIDAR_OFFSET_ANGLE: float = -90.0

_PLANNER = 'planner_debug'
_SCAN    = 'lidar_scan'

_HELP = """commands (type, then Enter):
  <Enter> / p   pause / resume
  n             step forward one planner frame (pauses)
  b             step back one planner frame (pauses)
  g <sec>       go to time <sec> (relative to the start of the run)
  f / s         play faster / slower (x2 / x0.5)
  h             this help
  q             quit"""


class _Controls:
    """Commands typed on stdin, consumed by the playback loop."""

    def __init__(self, speed: float) -> None:
        self.lock    = threading.Lock()
        self.paused  = False
        self.quit    = False
        self.speed   = speed
        self.pending: list[tuple[str, float | None]] = []

    def read_stdin(self) -> None:
        for line in sys.stdin:
            parts = line.strip().split()
            cmd   = parts[0].lower() if parts else 'p'
            arg   = None
            if cmd == 'g':
                try:
                    arg = float(parts[1])
                except (IndexError, ValueError):
                    print("usage: g <seconds>")
                    continue
            with self.lock:
                if cmd == 'q':
                    self.quit = True
                    return
                self.pending.append((cmd, arg))


def _state_label(debug: dict) -> str:
    names = PathPlanningVisualizer._STATE_NAMES
    try:
        state = int(debug['state'])
    except (KeyError, TypeError, ValueError):
        return '?'
    label = names.get(state, str(state))
    try:
        label += f" cc={int(debug['corner_count'])} lane={float(debug['target_offset']):+.0f}"
    except (KeyError, TypeError, ValueError):
        pass
    if 'obs_hits' in debug:
        label += f" obs_hits={int(debug['obs_hits'])}"
    return label


def run_replay(
    path:         str,
    speed:        float = 1.0,
    start:        float = 0.0,
    loop:         bool  = False,
    open_browser: bool  = True,
    planner_port: int   = PathPlanningVisualizer._DEFAULT_PORT,
    lidar_port:   int   = LidarVisualizer._DEFAULT_PORT,
) -> None:
    """Play a recording's planner debug + LiDAR streams into the live visualizers, at the
    recorded timing. Shows what the planner saw and decided during THAT run — it does not
    re-run the current planner code."""
    print(f"loading {path} ...")
    with Recording(path) as recording:
        names  = [n for n in (_PLANNER, _SCAN) if n in recording.streams]
        frames = list(recording.iter_merged(names))
    if not frames or _PLANNER not in names:
        print("no planner_debug stream in this recording — nothing to replay")
        return
    t0       = frames[0][0]
    frames   = [(t - t0, name, payload) for t, name, payload in frames]
    duration = frames[-1][0]
    planner_idx = [i for i, f in enumerate(frames) if f[1] == _PLANNER]
    print(f"{len(planner_idx)} planner frames, {len(frames) - len(planner_idx)} scans, {duration:.1f} s")

    planner_vis = PathPlanningVisualizer(port = planner_port)
    lidar_vis   = LidarVisualizer(offset_angle = _LIDAR_OFFSET_ANGLE, port = lidar_port)
    planner_vis.start()
    lidar_vis.start()
    if open_browser:
        for port in (planner_port, lidar_port):
            webbrowser.open(f"http://localhost:{port}")

    def push(i: int) -> None:
        _, name, payload = frames[i]
        if name == _PLANNER:
            planner_vis.set_debug(payload)
        else:
            lidar_vis(payload[1])          # recorded scan = (pose, points)

    def show(i: int) -> None:
        # Display the state at frame i: the planner frame itself plus the latest scan before it.
        push(i)
        for j in range(i - 1, -1, -1):
            if frames[j][1] == _SCAN:
                push(j)
                break

    def status(i: int, note: str = '') -> None:
        t, _, payload = frames[i]
        print(f"t={t:7.2f}s / {duration:.1f}s  {_state_label(payload)}  {note}", flush = True)

    ctl = _Controls(speed)
    threading.Thread(target = ctl.read_stdin, daemon = True, name = 'replay-stdin').start()
    print(_HELP)

    pos = next((i for i, f in enumerate(frames) if f[0] >= start), len(frames))
    anchor_wall, anchor_t = time.monotonic(), frames[min(pos, len(frames) - 1)][0]
    last_status = 0.0
    ended = False

    def reanchor(i: int) -> tuple[float, float]:
        return time.monotonic(), frames[min(i, len(frames) - 1)][0]

    try:
        while True:
            with ctl.lock:
                if ctl.quit:
                    break
                pending, ctl.pending = ctl.pending, []
            for cmd, arg in pending:
                if cmd == 'p':
                    ctl.paused = not ctl.paused
                    if not ctl.paused and ended:
                        pos, ended = planner_idx[0], False
                    anchor_wall, anchor_t = reanchor(pos)
                    print("paused" if ctl.paused else f"playing at x{ctl.speed:g}")
                elif cmd in ('n', 'b'):
                    ctl.paused = True
                    cur = max((k for k in planner_idx if k < pos), default = planner_idx[0])
                    if cmd == 'n':
                        nxt = next((k for k in planner_idx if k > cur), cur)
                    else:
                        nxt = max((k for k in planner_idx if k < cur), default = cur)
                    show(nxt)
                    pos = nxt + 1
                    status(nxt, '(paused)')
                elif cmd == 'g':
                    target = next((k for k in planner_idx if frames[k][0] >= arg), planner_idx[-1])
                    show(target)
                    pos, ended = target + 1, False
                    anchor_wall, anchor_t = reanchor(pos)
                    status(target, '(paused)' if ctl.paused else '')
                elif cmd in ('f', 's'):
                    ctl.speed = ctl.speed * 2.0 if cmd == 'f' else ctl.speed / 2.0
                    anchor_wall, anchor_t = reanchor(pos)
                    print(f"speed x{ctl.speed:g}")
                elif cmd == 'h':
                    print(_HELP)
                else:
                    print(f"unknown command {cmd!r} (h for help)")

            if ctl.paused or ended:
                time.sleep(0.05)
                continue
            if pos >= len(frames):
                if loop:
                    pos = planner_idx[0]
                    anchor_wall, anchor_t = reanchor(pos)
                    continue
                ended = True
                print("end of recording — g <sec> to jump back, <Enter> to replay from the start, q to quit")
                continue

            t = frames[pos][0]
            wait = anchor_wall + (t - anchor_t) / ctl.speed - time.monotonic()
            if wait > 0:
                time.sleep(min(wait, 0.05))      # short sleeps keep the controls responsive
                continue
            push(pos)
            if frames[pos][1] == _PLANNER and time.monotonic() - last_status >= 1.0:
                status(pos)
                last_status = time.monotonic()
            pos += 1
    except KeyboardInterrupt:
        pass
    finally:
        planner_vis.stop()
        lidar_vis.stop()
