from __future__ import annotations

import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from processors.Processor import Processor
from control.FieldMap import FieldMap, CellLabel, Direction, Obstacle
from control.EgoInformation import EgoInformation
from utils import mm, radian, second, Point, Event

Cell = tuple[int, int]


class TrackPathPlanner(Processor):
    """Path-following planner: build a reference line, then follow it.

    The planner keeps one persistent reference path -- an ordered list of world
    ``(x, y)`` points approximating the corridor *centreline*. Everything reduces
    to "aim at the path a fixed look-ahead ahead":

      * **Explore (lap 1).** The corridor ahead is still unknown, so the robot does
        NOT follow a pre-built line into it. It drives *reactively*: a wide forward
        cone is ray-cast over the observed corridor and the robot heads toward the
        furthest-reaching, most-clear direction (forward-only, and clamped to the
        lap turn-side once known), rounding corners as they come into view. The
        trajectory it drives is recorded as the reference line.
      * **Close.** Once the robot completes a lap (``FieldMap.update_lap_progress``)
        the loop is closed back to the start, resampled to uniform spacing and
        lightly smoothed, then frozen.
      * **Follow (laps 2+).** Pure path following on the frozen loop until
        ``FieldMap.TOTAL_LAPS``, then home to the start and stop.

    This milestone (M1) has no obstacle handling: obstacles are ignored by the
    geometry. Frenet localisation and colour-aware lateral offset are added on top
    of the same follow step in later milestones, so the path-following core stays
    untouched.

    Coordinate convention (shared with the existing planners): the planner works
    in ego-state space ``pos = (x, y)`` with forward ``= (cos yaw, sin yaw)`` and
    ``+yaw`` rotating ``x -> y``. ``_world_to_coarse`` maps that space to the
    coarse grid consistently with how ``LidarProcessor`` populates ``FieldMap``.
    """

    # --- Coarse grid / clearance ---
    _K:                   int    = 5
    _OCCUPANCY_THRESHOLD: float  = 0.0
    _WALL_INFLATION:      mm     = 50.0    # min wall clearance treated as drivable

    # --- Geometry sampling ---
    _LATERAL_RES:         mm     = 25.0    # segment/chord sampling step (~one coarse cell)

    # --- Lap close (resample + smooth) ---
    _RESAMPLE_SPACING:    mm     = 50.0
    _SMOOTH_PASSES:       int    = 2

    # --- Exploration (lap 1): reactive wide-cone goal in the *observed* corridor ---
    # The map ahead is unknown on lap 1, so we do not follow a pre-built line into
    # it. Instead we ray-cast a wide forward cone over the seen corridor and drive
    # toward the furthest-reaching, most-clear direction (clamped forward, and to the
    # lap turn-side once known), recording the trajectory as the reference line.
    _EXPLORE_HALF_ANGLE: radian = math.radians(80.0)
    _EXPLORE_RAYS:       int    = 21
    _EXPLORE_L_MIN:      mm     = 100.0
    _EXPLORE_L_MAX:      mm     = 700.0
    _EXPLORE_K_ADAPTIVE: float  = 3.0      # reach scales with local clearance
    _EXPLORE_W_DIST:     float  = 2.0      # reward reaching far
    _EXPLORE_W_CLEAR:    float  = 1.0      # reward staying clear of walls
    _EXPLORE_CENTER_BAND: radian = math.radians(25.0)  # within this either turn ok; beyond, allowed side only
    _ROUTE_MIN_SPACING:  mm     = 50.0     # breadcrumb decimation along the recorded line

    # --- Follow ---
    _L_LOOKAHEAD:         mm     = 350.0   # look-ahead on straights
    _MIN_LOOKAHEAD:       mm     = 150.0   # shortest adaptive look-ahead (tight corners)
    _MIN_ROUTE_POINTS:    int    = 20      # forward-search window for the cursor
    _SPEED:               float  = 80.0    # building / first lap
    _FOLLOW_SPEED:        float  = 90.0   # frozen loop

    # --- Steering feasibility ---
    # The firmware (Navigator::setTarget) rejects any target that lies inside the
    # minimum turning-radius circles and goes IDLE, so the planner must never
    # command a turn sharper than the car can make. r_min = WHEELBASE / tan(MAX_STEER);
    # keep in sync with src/arduino/Config.h (89.6251 mm / tan(0.4014 rad) = 211 mm).
    _MIN_TURN_RADIUS:     mm     = 211.0
    _TURN_RADIUS_MARGIN:  float  = 1.15    # land safely outside the circle, not on it
    _MIN_FORWARD:         mm     = 50.0    # always keep some headway so the car can pivot

    # --- Home / stop ---
    _STOP_RADIUS:         mm     = 200.0

    # --- Collision guard ---
    _COLLISION_ACTIVE_S:  second = 0.8

    def __init__(
        self,
        field_map:       FieldMap,
        ego_information: EgoInformation,
    ) -> None:
        super().__init__()
        self._field_map       = field_map
        self._ego_information = ego_information
        self.on_target:        Event = Event()
        self.on_next_obstacle: Event = Event()
        self.on_route:         Event = Event()
        self.on_stop:          Event = Event()
        self.on_plan_debug:    Event = Event()

        # Clearance-field cache, keyed on the coarse wall mask.
        self._cached_walls: np.ndarray | None = None
        self._cached_D_mm:  np.ndarray | None = None

        # Reference path (world points) and its lifecycle.
        self._path:        list[Point] = []
        self._closed:      bool        = False
        self._cursor:      int         = 0

        self._stopped: bool = False

        self._last_collision_t: float = -math.inf

    # ------------------------------------------------------------------ #
    #  Top-level loop                                                     #
    # ------------------------------------------------------------------ #

    def _process(self) -> None:
        if self._stopped:
            return

        pos, yaw = self._ego_information.get_ego_information()
        if pos is None or yaw is None:
            return

        laps = self._field_map.update_lap_progress(pos)

        if laps >= FieldMap.TOTAL_LAPS:
            self._home_and_stop(pos, yaw)
            return

        D_mm = self._clearance_field()

        if self._closed:
            # Frozen line: pure path following (the part that already works well).
            target = self._follow(pos, yaw, D_mm)
        elif laps >= 1 and len(self._path) >= 3:
            # First lap complete: freeze the explored trajectory into the loop.
            self._close_loop(pos)
            target = self._follow(pos, yaw, D_mm)
        else:
            # Exploration. The corridor ahead is still unknown, so do NOT follow a
            # pre-built line into it. Drive reactively toward the most open direction
            # in the *observed* corridor and record the trajectory as the reference
            # line for the following laps.
            self._record_pose(pos)
            self.on_route(list(self._path))
            target = self._explore_target(pos, yaw, D_mm)

        self._emit_plan_debug(D_mm, pos)
        if target is not None:
            target = self._clamp_reachable(pos, target, D_mm)
            self._emit_target(pos, yaw, target)

    def notify_collision(self) -> None:
        # Stored by the collision-guard thread; consumed as a "hold" in _follow so a
        # single event cannot latch. Obstacle/recovery behaviour comes in later M's.
        self._last_collision_t = time.monotonic()

    def _collision_active(self) -> bool:
        return (time.monotonic() - self._last_collision_t) < self._COLLISION_ACTIVE_S

    # ------------------------------------------------------------------ #
    #  Reachability / direction helpers                                   #
    # ------------------------------------------------------------------ #

    def _segment_clear(self, a: Point, b: Point, D_mm: np.ndarray) -> bool:
        # True if the straight segment a->b stays inside the observed free corridor
        # (every sample is an unblocked cell). Used so a centreline step can never
        # tunnel through a wall.
        steps = max(1, int(self._dist(a, b) / self._LATERAL_RES))
        for s in range(steps + 1):
            t = s / steps
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if self._lookup(D_mm, p) <= 0.0:
                return False
        return True

    def _clamp_reachable(self, pos: Point, target: Point, D_mm: np.ndarray) -> Point:
        # Safety net: never hand the firmware a target that the straight line from
        # the robot cannot reach without crossing a wall (a follow chord can clip a
        # corner). Walk from the robot toward the target and pull the point back to
        # the last unblocked sample before any wall.
        steps = max(1, int(self._dist(pos, target) / self._LATERAL_RES))
        last_good = pos
        for s in range(1, steps + 1):
            t = s / steps
            p = (pos[0] + t * (target[0] - pos[0]), pos[1] + t * (target[1] - pos[1]))
            if self._lookup(D_mm, p) <= 0.0:
                return last_good
            last_good = p
        return target

    def _allowed_sign(self) -> float | None:
        # +1 = right turns allowed (CW); -1 = left (CCW); None until known.
        direction = self._field_map.direction
        if direction == Direction.CW:
            return +1.0
        if direction == Direction.CCW:
            return -1.0
        return None

    # ------------------------------------------------------------------ #
    #  Lap close: resample + smooth + freeze                              #
    # ------------------------------------------------------------------ #

    def _close_loop(self, pos: Point) -> None:
        path = list(self._path)
        # Connect the tip back to the start so the loop is continuous.
        if self._dist(path[-1], path[0]) > 1e-6:
            path.append(path[0])

        path = self._resample(path, self._RESAMPLE_SPACING)
        path = self._smooth(path, self._SMOOTH_PASSES)

        self._path   = path
        self._closed = True
        self._cursor = self._closest_index(pos)
        self.on_route(list(self._path))

    @staticmethod
    def _resample(path: list[Point], spacing: mm) -> list[Point]:
        if len(path) < 2:
            return list(path)
        out: list[Point] = [path[0]]
        carry = 0.0
        for a, b in zip(path[:-1], path[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg < 1e-6:
                continue
            d = spacing - carry
            while d <= seg:
                t = d / seg
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
                d += spacing
            carry = (carry + seg) % spacing
        return out

    @staticmethod
    def _smooth(path: list[Point], passes: int) -> list[Point]:
        # Closed-loop moving average (weights 1/4, 1/2, 1/4).
        n = len(path)
        if n < 4:
            return list(path)
        pts = list(path)
        for _ in range(passes):
            nxt = []
            for i in range(n):
                a = pts[(i - 1) % n]
                b = pts[i]
                c = pts[(i + 1) % n]
                nxt.append((0.25 * a[0] + 0.5 * b[0] + 0.25 * c[0],
                            0.25 * a[1] + 0.5 * b[1] + 0.25 * c[1]))
            pts = nxt
        return pts

    # ------------------------------------------------------------------ #
    #  Exploration (reactive, lap 1)                                      #
    # ------------------------------------------------------------------ #

    def _record_pose(self, pos: Point) -> None:
        # Lay down the reference line as the robot drives: a decimated breadcrumb
        # trail of where it actually went. Frozen into the loop at lap close.
        if not self._path or self._dist(pos, self._path[-1]) >= self._ROUTE_MIN_SPACING:
            self._path.append(pos)

    def _explore_target(self, pos: Point, yaw: radian, D_mm: np.ndarray) -> Point | None:
        # Reactive goal in the observed corridor: cast a wide forward cone, march
        # each ray through cells that clear the walls by _WALL_INFLATION, and pick
        # the direction that reaches furthest while staying clear. Forward-only (and
        # turn-side clamped once the lap direction is known) so the robot tracks the
        # corridor and rounds corners as they come into view, without ever pointing
        # backwards or into a wall it cannot see past yet.
        cur = self._world_to_coarse(pos)
        if not self._in_bounds(cur, D_mm):
            return None

        coarse_mm = self._K * FieldMap.CELL_SIZE
        d_cur     = float(D_mm[cur])
        l_max_mm  = min(max(self._EXPLORE_K_ADAPTIVE * d_cur, self._EXPLORE_L_MIN),
                        self._EXPLORE_L_MAX)
        l_min_c   = self._EXPLORE_L_MIN / coarse_mm
        l_max_c   = l_max_mm / coarse_mm

        allowed    = self._allowed_sign()
        rows, cols = D_mm.shape
        best_goal: Cell | None = None
        best_score: float      = -math.inf

        for theta in np.linspace(yaw - self._EXPLORE_HALF_ANGLE,
                                 yaw + self._EXPLORE_HALF_ANGLE,
                                 self._EXPLORE_RAYS):
            delta = self._wrap(float(theta) - yaw)
            if allowed is not None and abs(delta) > self._EXPLORE_CENTER_BAND \
               and delta * allowed < 0.0:
                continue   # large turn the wrong way round the loop -> forbidden

            dr, dc    = math.cos(theta), math.sin(theta)   # row<-pos0, col<-pos1
            last_free: Cell | None = None
            d = l_min_c
            while d <= l_max_c:
                r = int(cur[0] + dr * d)
                c = int(cur[1] + dc * d)
                if not (0 <= r < rows and 0 <= c < cols):
                    break
                if D_mm[r, c] < self._WALL_INFLATION:
                    break   # wall / unknown / too tight: corridor ends here
                last_free = (r, c)
                d += 1.0

            if last_free is None:
                continue
            reach = math.hypot(last_free[0] - cur[0], last_free[1] - cur[1])
            score = self._EXPLORE_W_DIST * reach + self._EXPLORE_W_CLEAR * float(D_mm[last_free])
            if score > best_score:
                best_score = score
                best_goal  = last_free

        if best_goal is None:
            return None
        return self._coarse_to_world(best_goal)

    @staticmethod
    def _wrap(angle: radian) -> radian:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # ------------------------------------------------------------------ #
    #  Follow                                                             #
    # ------------------------------------------------------------------ #

    def _follow(self, pos: Point, yaw: radian, D_mm: np.ndarray) -> Point | None:
        # An imminent wall makes the robot hold (no obstacle/recovery yet in M1).
        if self._collision_active():
            self.on_target(0.0, 0.0, 0.0)
            return None

        if len(self._path) < 2:
            # No usable path yet. Do NOT drive blindly forward: at startup that aims
            # straight into still-unobserved space (which may be a wall). Hold until
            # the build has grown a path through the observed corridor -- the LiDAR
            # maps the surroundings without the robot moving, so this is momentary.
            self.on_target(0.0, 0.0, 0.0)
            return None

        self._cursor = self._closest_index(pos)

        # Adaptive look-ahead: take the farthest point on the line within
        # _L_LOOKAHEAD whose straight chord from the robot stays inside the corridor.
        # On a straight that is the full _L_LOOKAHEAD point; in a corner the far
        # point's chord would cut across the inner wall, so the target shortens and
        # rides the curve -- the robot follows the centre line instead of cutting it.
        best = self._point_at_arclength(self._cursor, self._MIN_LOOKAHEAD)
        s    = self._MIN_LOOKAHEAD
        while s <= self._L_LOOKAHEAD:
            p = self._point_at_arclength(self._cursor, s)
            if not self._segment_clear(pos, p, D_mm):
                break
            best = p
            s   += self._LATERAL_RES
        return best

    def _point_at_arclength(self, cursor: int, s: mm) -> Point:
        # The point on the path a given arclength `s` ahead of `cursor`.
        n     = len(self._path)
        acc   = 0.0
        i     = cursor
        steps = n if self._closed else (n - 1 - cursor)
        for _ in range(max(steps, 0)):
            a = self._path[i % n]
            b = self._path[(i + 1) % n]
            seg = self._dist(a, b)
            if seg > 1e-6 and acc + seg >= s:
                t = (s - acc) / seg
                return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            acc += seg
            i   += 1
        return self._path[i % n] if self._closed else self._path[-1]

    def _closest_index(self, pos: Point) -> int:
        n      = len(self._path)
        window = min(n, max(self._MIN_ROUTE_POINTS, n // 4))

        best_i, best_d = self._cursor % n, math.inf
        for k in range(window):
            i = (self._cursor + k) % n if self._closed else min(self._cursor + k, n - 1)
            d = self._dist(pos, self._path[i])
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    # ------------------------------------------------------------------ #
    #  Emit                                                               #
    # ------------------------------------------------------------------ #

    def _emit_target(
        self,
        pos:    Point,
        yaw:    radian,
        target: Point,
        speed:  float | None = None,
    ) -> None:
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]

        forward_mm =  dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral_mm = -dx * math.sin(yaw) + dy * math.cos(yaw)

        # Keep the target physically reachable, otherwise the firmware rejects it
        # and the robot never moves.
        forward_mm, lateral_mm = self._make_reachable(forward_mm, lateral_mm)

        if speed is None:
            speed = self._FOLLOW_SPEED if self._closed else self._SPEED

        self.on_target(forward_mm, lateral_mm, speed)

    def _make_reachable(self, forward: float, lateral: float) -> tuple[float, float]:
        # Clamp the lateral offset so the pursuit arc to the target is never sharper
        # than the car's minimum turning radius. The firmware models the two tightest
        # turns as circles of radius r tangent to the robot at its sides; a target
        # inside either circle is unreachable and gets rejected (-> IDLE). When the
        # look-ahead point demands too sharp a turn we pull it onto the near circle
        # boundary: the car then steers as hard as it can while still driving
        # forward, and converges onto the path over the following cycles.
        # Never let a near-abeam target collapse to (0, 0) (which the firmware reads
        # as "arrived" and stops): keep a little forward so the car can pivot.
        if 0.0 <= forward < self._MIN_FORWARD:
            forward = self._MIN_FORWARD

        r = self._MIN_TURN_RADIUS * self._TURN_RADIUS_MARGIN
        if abs(forward) >= r:
            return forward, lateral                 # beyond the circles' forward reach
        half = math.sqrt(r * r - forward * forward)
        l_lo = r - half                             # |lateral| in (l_lo, l_hi) is infeasible
        l_hi = r + half
        if l_lo < abs(lateral) < l_hi:
            lateral = math.copysign(l_lo, lateral)
        return forward, lateral

    def _home_and_stop(self, pos: Point, yaw: radian) -> None:
        home = self._field_map.start_position
        if home is not None and self._dist(pos, home) > self._STOP_RADIUS:
            self._emit_target(pos, yaw, home)
            return
        self._stopped = True
        self.on_target(0.0, 0.0, 0.0)
        self.on_stop()

    # ------------------------------------------------------------------ #
    #  Clearance field                                                    #
    # ------------------------------------------------------------------ #

    def _clearance_field(self) -> np.ndarray:
        occupancy, semantic, _ = self._field_map.snapshot()
        blocked = self._coarse_blocked(occupancy, semantic)

        if self._cached_walls is not None and np.array_equal(blocked, self._cached_walls):
            return self._cached_D_mm

        D_mm = distance_transform_edt(~blocked).astype(np.float32) \
               * (self._K * FieldMap.CELL_SIZE)
        self._cached_walls = blocked
        self._cached_D_mm  = D_mm
        return D_mm

    def _coarse_blocked(self, occupancy: np.ndarray, semantic: np.ndarray) -> np.ndarray:
        # Coarse mask of cells the centreline must NOT be built through. A cell is
        # blocked when it is a wall, a stray occupied cell, OR *unobserved*. Treating
        # unknown space as blocked is load-bearing: the map is built incrementally,
        # so the line must only grow inside the corridor we have actually seen and
        # stop at the observation frontier -- otherwise the lateral max-clearance
        # search pulls it toward the largest *unmapped* opening and drives the robot
        # into a wall that has not been scanned yet. Obstacles are excluded (treated
        # as free) so the centreline stays the wall corridor; obstacle avoidance is
        # layered on later as a lateral offset, not by carving the corridor.
        #
        # "Observed free" = at least one sub-cell has free evidence (occupancy < 0);
        # "unknown" = no free evidence at all (occupancy stayed 0).
        K    = self._K
        rows = occupancy.shape[0] // K
        cols = occupancy.shape[1] // K

        occ   = occupancy[: rows * K, : cols * K].reshape(rows, K, cols, K)
        sem   = semantic [: rows * K, : cols * K]

        occupied = occ.max(axis=(1, 3)) > self._OCCUPANCY_THRESHOLD   # any occupied evidence
        unknown  = occ.min(axis=(1, 3)) >= 0.0                        # no free evidence -> unseen
        wall     = np.isin(sem, [int(CellLabel.WALL), int(CellLabel.PARKING_WALL)]) \
                     .reshape(rows, K, cols, K).any(axis=(1, 3))
        obstacle = (sem == int(CellLabel.OBSTACLE)) \
                     .reshape(rows, K, cols, K).any(axis=(1, 3))

        return wall | (~obstacle & (occupied | unknown))

    def _lookup(self, D_mm: np.ndarray, point: Point) -> float:
        cell = self._world_to_coarse(point)
        if not self._in_bounds(cell, D_mm):
            return 0.0
        return float(D_mm[cell])

    # ------------------------------------------------------------------ #
    #  Debug                                                              #
    # ------------------------------------------------------------------ #

    def _emit_plan_debug(self, D_mm: np.ndarray, pos: Point) -> None:
        # Keep the visualiser's planning pane alive. The reference line itself is
        # drawn via on_route; here we only ship the coarse clearance grid.
        self.on_plan_debug({
            "K":                self._K,
            "traversable":      D_mm >= self._WALL_INFLATION,
            "D_mm":             D_mm,
            "trail":            np.zeros_like(D_mm),
            "block":            None,
            "current":          self._world_to_coarse(pos),
            "goal":             None,
            "path":             None,
            "thinned":          None,
            "robot_radius":     self._WALL_INFLATION,
            "target_clearance": self._WALL_INFLATION,
        })
        # Stream the live path while it is still being built (closed path is sent
        # once at _close_loop).
        if not self._closed:
            self.on_route(list(self._path))

    # ------------------------------------------------------------------ #
    #  Geometry helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dist(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _world_to_coarse(pos: Point) -> Cell:
        K   = TrackPathPlanner._K
        col = int((pos[1] + FieldMap.ORIGIN[0]) / (FieldMap.CELL_SIZE * K))
        row = int((pos[0] + FieldMap.ORIGIN[1]) / (FieldMap.CELL_SIZE * K))
        return (row, col)

    @staticmethod
    def _coarse_to_world(cell: Cell) -> Point:
        K      = TrackPathPlanner._K
        offset = (K * FieldMap.CELL_SIZE) / 2.0
        north  = cell[0] * K * FieldMap.CELL_SIZE - FieldMap.ORIGIN[1] + offset
        east   = cell[1] * K * FieldMap.CELL_SIZE - FieldMap.ORIGIN[0] + offset
        return (north, east)

    @staticmethod
    def _in_bounds(cell: Cell, grid: np.ndarray) -> bool:
        return 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]
