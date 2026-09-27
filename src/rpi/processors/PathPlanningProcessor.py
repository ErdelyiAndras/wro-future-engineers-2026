from __future__ import annotations

import heapq
import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from processors.Processor import Processor
from control.FieldMap import FieldMap, CellLabel, Obstacle, ObstacleColor
from control.EgoInformation import EgoInformation
from utils import mm, radian, second, Point, Event

Cell = tuple[int, int]

class PathPlanningProcessor(Processor):
    # TEMPORARY feature switches. Core navigation (clearance around occupied cells,
    # breadcrumb trail, backward penalty, route record/follow) is always on.
    #   _OBSTACLE_RULES_ENABLED: wrong-side block and wall-orientation snap.
    #   _INSPECT_ENABLED:        look at the next obstacle to read its colour.
    #   _REVERSE_ENABLED:        reverse-out recovery when blocked / near a wall.
    _OBSTACLE_RULES_ENABLED: bool = True
    _INSPECT_ENABLED:        bool = False
    _REVERSE_ENABLED:        bool = False

    _K:                   int    = 5
    _OCCUPANCY_THRESHOLD: float  = 0.0
    _WALL_INFLATION:      mm     = 200.0   # keep the robot centre this far from walls/unknown
    _OBSTACLE_INFLATION:  mm     = 30.0   # keep the robot centre this far from obstacles
    _HALF_ANGLE:          radian = math.radians(80.0)
    _N_RAYS:              int    = 17
    _L_MIN:               mm     = 100.0
    _L_MAX:               mm     = 500.0
    _W_DIST:              float  = 2.0
    _W_CLEAR:             float  = 1.0
    _K_ADAPTIVE:          float  = 3.0
    _TARGET_CLEARANCE:    mm     = 200.0
    _CLEARANCE_WEIGHT:    float  = 5.0
    _BACKWARD_PENALTY:    float  = 150.0

    # Around-obstacle reward. Obstacles only sit on the track, so the cells beside
    # them mark the direction of travel — the planner should pass them, not flee.
    # A ring around each obstacle (clear of it, but still near) is rewarded along
    # the goal ray, peaking at _OBSTACLE_RING_DIST and fading to 0 by ±_HALF.
    _OBSTACLE_RING_DIST:   mm    = 200.0   # preferred distance from an obstacle (peak)
    _OBSTACLE_RING_HALF:   mm    = 200.0   # reward fades to 0 this far from the peak
    _OBSTACLE_RING_WEIGHT: float = 300.0   # strength of the around-obstacle reward

    # Wrong-side block: the wrong side of a colour-confirmed pillar is sealed as
    # non-traversable over a bounded along-track window, so neither _select_goal
    # nor _astar can route there at all — the side rule is enforced by construction
    # rather than by a cost that could be outweighed. If no correct-side path
    # remains, _plan returns no target and the robot holds/stops (a reverse-out
    # recovery is planned once the Arduino supports reversing). Geometry is built
    # from the latched anchor/normal, so the block is static in the world.
    _BLOCK_REACH:          mm    = 700.0  # lateral extent into the wrong side (past wall)
    _BLOCK_WINDOW:         mm    = 100.0  # along-track half-extent of the block (leg 1)
    _BLOCK_BACK:           mm    = 50.0    # overlap behind centroid, no gap to pillar
    _BLOCK_ARM_THICKNESS:  mm    = 150.0  # thickness of the wall-hugging arm (leg 2 of the L)
    _BLOCK_CLEARANCE:      mm    = 30.0   # how close the robot centre may get to the block
    _WALL_ORIENT_MIN_CELLS: int  = 60     # min coarse wall edge cells before trusting the angle
    _L_LOOKAHEAD:          mm    = 450.0
    _SPEED:                float = 70.0
    _FOLLOW_SPEED:         float = 80.0   # full speed when driving the memorized line

    # Breadcrumb trail: a decaying penalty laid down behind the robot to
    # encode "the way I came from" without a global track model. Stamped
    # behind the robot, wide across-track (to fill the lane so the planner
    # cannot reverse beside its own trail), thin along-track. Consumed in
    # _select_goal (load-bearing: stops backward goals being chosen) and,
    # softly, in _astar.
    _TRAIL_DECAY:         float  = 0.95  # per planner cycle; main tuning knob
    _TRAIL_HALF_WIDTH:    mm     = 300.0   # across-track half-extent (~lane half)
    _TRAIL_HALF_LENGTH:   mm     = 200.0   # along-track behind, bridges inter-frame motion
    _TRAIL_WEIGHT:        float  = 10000.0  # goal-selection: dominate to reject
    _TRAIL_WEIGHT_ASTAR:  float  = 50.0     # A* step cost: soft, never a hard wall

    # Recorded racing line. On lap 1 the planner runs reactively and records the
    # travelled poses; after the first completed lap (counted by FieldMap) it
    # switches to following the recorded line (pure pursuit). After
    # FieldMap.TOTAL_LAPS it homes to the start and stops.
    _ROUTE_MIN_SPACING:   mm     = 50.0    # decimation between recorded points
    _MIN_ROUTE_POINTS:    int    = 20      # forward-search window for the cursor
    _STOP_RADIUS:         mm     = 200.0   # stop once this close to the start

    # Reverse-out recovery (reactive lap only). When the planner finds no forward
    # path OR the collision guard reports an imminent wall, the robot reverses
    # (negative speed, steered reverse pursuit on the Arduino) along its own
    # breadcrumb trail until a forward path exists again. The retraced poses are
    # trimmed from the route so the recorded racing line follows the corrected path
    # rather than the dead-end approach.
    _REVERSE_SPEED:       float  = 60.0     # reverse speed magnitude (sent negated)
    _COLLISION_ACTIVE_S:  second = 0.8      # treat a collision signal as active this long

    # Heading-free travel reference: the pass side is decided against the direction
    # the robot actually travelled over the last few breadcrumbs, not the noisy
    # instantaneous IMU heading. Falls back to the IMU yaw until the route is this long.
    _TRAVEL_TANGENT_POINTS: int  = 5

    # Active colour inspection. The next uncoloured obstacle ahead is looked at
    # directly: the robot aims at it and slows to a standoff so the camera reads its
    # colour head-on, instead of driving on and hoping it falls into frame. Once the
    # colour is confirmed (obstacle drops out of the uncoloured set) or the timeout
    # elapses, normal planning resumes.
    _INSPECT_RANGE:       mm     = 1500.0            # only inspect obstacles within this
    _INSPECT_HALF_ANGLE:  radian = math.radians(60.0)  # forward cone for "the next obstacle"
    _INSPECT_STANDOFF:    mm     = 400.0             # hold this far away while looking
    _INSPECT_SPEED:       float  = 60.0              # slow approach speed while inspecting
    _INSPECT_TIMEOUT:     second = 3.0               # give up looking after this, plan anyway
    _INSPECT_MATCH:       mm     = 100.0             # same-obstacle tolerance for the timer

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

        self._cached_blocked_walls: np.ndarray | None = None  # walls only -> wall distance
        self._cached_D_mm:          np.ndarray | None = None
        self._cached_obstacle:      np.ndarray | None = None  # obstacles -> distance + ring
        self._cached_D_obs:         np.ndarray | None = None
        self._cached_ring:          np.ndarray | None = None

        self._trail:              np.ndarray | None = None

        # Track's rectilinear orientation (rad, mod 90°), deduced once from the
        # observed wall cells. Used to snap the wrong-side L to the walls instead of
        # to the robot's heading. None until enough wall is seen to estimate it.
        self._wall_theta:         float | None      = None

        self._following:    bool         = False
        self._stopped:      bool         = False
        self._route:        list[Point]  = []
        self._route_cursor: int          = 0

        self._reversing:        bool  = False
        self._last_collision_t: float = -math.inf

        # Active colour inspection state: when the current look started and the
        # obstacle (world point) being looked at, so the timeout resets per obstacle.
        self._inspect_start:    float | None = None
        self._inspect_anchor:   Point | None = None

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

        if self._following:
            target = self._follow_route(pos)
            if target is not None:
                self._emit_target(pos, yaw, target)
            return

        if laps >= 1:
            self._finalize_route(pos)
            self._following = True
            target = self._follow_route(pos)
            if target is not None:
                self._emit_target(pos, yaw, target)
            return

        # Reactive lap: plan forward, or reverse-out if blocked / about to hit a wall.
        self._reactive_step(pos, yaw)

    def notify_collision(self) -> None:
        # Called from the collision guard thread when the forward LiDAR cone is too
        # close. Recorded as a timestamp; _collision_active() decays it so a single
        # event does not latch reversing forever.
        self._last_collision_t = time.monotonic()

    def _collision_active(self) -> bool:
        return (time.monotonic() - self._last_collision_t) < self._COLLISION_ACTIVE_S

    def _reactive_step(self, pos: Point, yaw: radian) -> None:
        # Before committing to a forward plan, explicitly look at the next
        # uncoloured obstacle: aim at it and slow down until the camera confirms its
        # colour (or the per-obstacle timeout elapses), so the colour is read from a
        # head-on look rather than an incidental glimpse.
        if self._INSPECT_ENABLED and self._inspect_obstacle(pos, yaw):
            return

        target, next_obstacle = self._plan(pos, yaw)
        self.on_next_obstacle(next_obstacle)

        # Forward is OK if the planner found a path (and, when reverse-out is on,
        # nothing is about to be hit).
        collision = self._REVERSE_ENABLED and self._collision_active()
        if target is not None and not collision:
            self._reversing = False
            self._record_pose(pos)
            self._emit_target(pos, yaw, target)
            return

        if self._REVERSE_ENABLED:
            self._reversing = True
            rev_target = self._reverse_target(pos)
            if rev_target is None:
                self.on_target(0.0, 0.0, 0.0)   # nothing to back toward: hold
                return
            self._emit_target(pos, yaw, rev_target, reverse = True)
            return

        # Reverse-out disabled: no forward path -> hold and let planning retry.
        self.on_target(0.0, 0.0, 0.0)

    def _reverse_target(self, pos: Point) -> Point | None:
        # Reverse toward the most recent breadcrumb (guaranteed drivable: we were
        # just there). Pops poses as we reach them, which both advances the reverse
        # target and trims the dead-end approach from the recorded route, so the
        # racing line ends up following the corrected path.
        while len(self._route) > 1 and \
              self._dist(pos, self._route[-1]) < self._ROUTE_MIN_SPACING:
            self._route.pop()
        return self._route[-1] if self._route else None

    def _inspect_obstacle(self, pos: Point, yaw: radian) -> bool:
        # Look at the next uncoloured obstacle ahead to read its colour. Returns
        # True if it owns this cycle (the robot is aiming/holding to look), False to
        # let normal planning run (nothing to inspect, or the look timed out).
        obstacles    = self._field_map.get_obstacles()
        current_cell = self._world_to_coarse(pos)
        obs          = self._obstacle_to_inspect(obstacles, current_cell, yaw)

        if obs is None:
            self._inspect_start  = None
            self._inspect_anchor = None
            return False

        obs_pt = (float(obs.centroid[1]), float(obs.centroid[0]))   # (north, east)
        now    = time.monotonic()
        if self._inspect_anchor is None or \
           self._dist(obs_pt, self._inspect_anchor) > self._INSPECT_MATCH:
            self._inspect_start  = now      # a new obstacle -> restart the look timer
            self._inspect_anchor = obs_pt

        if now - self._inspect_start >= self._INSPECT_TIMEOUT:
            return False                    # gave up looking; plan around it as unknown

        self.on_next_obstacle(obs)
        self._aim_at_obstacle(pos, yaw, obs_pt)
        return True

    def _obstacle_to_inspect(
        self,
        obstacles:    list[Obstacle],
        current_cell: Cell,
        heading:      radian,
    ) -> Obstacle | None:
        # Nearest colour-unknown obstacle within the forward cone and range.
        hx, hy    = math.cos(heading), math.sin(heading)
        coarse_mm = self._K * FieldMap.CELL_SIZE
        best:      Obstacle | None = None
        best_dist: float           = math.inf

        for obs in obstacles:
            if obs.color is not None:
                continue
            cell = self._world_to_coarse((float(obs.centroid[1]), float(obs.centroid[0])))
            vx   = cell[1] - current_cell[1]
            vy   = cell[0] - current_cell[0]
            dist = math.hypot(vx, vy)
            if dist == 0 or dist * coarse_mm > self._INSPECT_RANGE:
                continue
            angle = math.acos(max(-1.0, min(1.0, (hy * vx + hx * vy) / dist)))
            if angle <= self._INSPECT_HALF_ANGLE and dist < best_dist:
                best_dist = dist
                best      = obs
        return best

    def _aim_at_obstacle(self, pos: Point, yaw: radian, obs_pt: Point) -> None:
        # Point the robot at the obstacle and creep toward a standoff so the camera
        # frames it; hold once at standoff and let the colour vote accumulate.
        dist = self._dist(pos, obs_pt)
        if dist <= self._INSPECT_STANDOFF:
            self.on_target(0.0, 0.0, 0.0)
            return
        t   = (dist - self._INSPECT_STANDOFF) / dist
        aim = (pos[0] + t * (obs_pt[0] - pos[0]),
               pos[1] + t * (obs_pt[1] - pos[1]))
        self._emit_target(pos, yaw, aim, speed = self._INSPECT_SPEED)

    def _home_and_stop(self, pos: Point, yaw: radian) -> None:
        home = self._field_map.start_position
        if home is not None and self._dist(pos, home) > self._STOP_RADIUS:
            self._emit_target(pos, yaw, home)   # drive the last stretch to the start
            return

        self._stopped = True
        self.on_target(0.0, 0.0, 0.0)
        self.on_stop()

    def _emit_target(
        self,
        pos:     Point,
        yaw:     radian,
        target:  Point,
        reverse: bool         = False,
        speed:   float | None = None,
    ) -> None:
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]

        forward_mm =  dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral_mm = -dx * math.sin(yaw) + dy * math.cos(yaw)

        if reverse:
            # Negative speed = reverse pursuit on the Arduino; the target is a
            # breadcrumb behind the robot, so forward_mm comes out negative too.
            out_speed = -self._REVERSE_SPEED
        elif speed is not None:
            out_speed = speed
        else:
            out_speed = self._FOLLOW_SPEED if self._following else self._SPEED
        self.on_target(forward_mm, lateral_mm, out_speed)

    def _record_pose(self, pos: Point) -> None:
        if not self._route or self._dist(pos, self._route[-1]) >= self._ROUTE_MIN_SPACING:
            self._route.append(pos)

    def _finalize_route(self, pos: Point) -> None:
        if self._route:
            self._route.append(self._route[0])   # close the loop back to the start
        self._route_cursor = self._closest_route_index(pos)
        self.on_route(list(self._route))

    def _follow_route(self, pos: Point) -> Point | None:
        if len(self._route) < 2:
            return None

        self._route_cursor = self._closest_route_index(pos)

        n          = len(self._route)
        accumulated = 0.0
        i           = self._route_cursor
        for _ in range(n):
            a = self._route[i % n]
            b = self._route[(i + 1) % n]
            seg_len = self._dist(a, b)
            if seg_len > 1e-6 and accumulated + seg_len >= self._L_LOOKAHEAD:
                t = (self._L_LOOKAHEAD - accumulated) / seg_len
                return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            accumulated += seg_len
            i += 1

        return self._route[self._route_cursor]

    def _closest_route_index(self, pos: Point) -> int:
        # Search a forward window from the current cursor so progress is
        # monotonic around the loop and we never snap backwards onto an earlier
        # crossing of the same point.
        n      = len(self._route)
        window = min(n, max(self._MIN_ROUTE_POINTS, n // 4))

        best_i, best_d = self._route_cursor, math.inf
        for k in range(window):
            i = (self._route_cursor + k) % n
            d = self._dist(pos, self._route[i])
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    @staticmethod
    def _dist(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _travel_yaw(self, fallback_yaw: radian) -> radian:
        # Heading-free travel direction from the breadcrumb tangent (where the robot
        # actually went over the last few recorded poses), which is far steadier
        # than the instantaneous IMU yaw that has been mislabelling the pass side.
        # Falls back to the IMU yaw until enough route exists to form a tangent.
        n = self._TRAVEL_TANGENT_POINTS
        if len(self._route) >= n:
            a = self._route[-n]
            b = self._route[-1]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            if math.hypot(dx, dy) > 1e-6:
                return math.atan2(dy, dx)
        return fallback_yaw

    def _plan(self, pos: Point, yaw: radian) -> tuple[Point | None, Obstacle | None]:
        occupancy, semantic, obstacles = self._field_map.snapshot()

        blocked_walls, obstacle, unknown = self._coarse_masks(occupancy, semantic)
        current_cell = self._world_to_coarse(pos)

        if not self._in_bounds(current_cell, blocked_walls):
            return None, None

        if self._trail is None or self._trail.shape != blocked_walls.shape:
            self._trail = np.zeros(blocked_walls.shape, dtype = np.float32)
        self._update_trail(current_cell, yaw)
        trail = self._trail

        block:    np.ndarray | None = None
        region:   np.ndarray | None = None
        upcoming = self._find_upcoming_obstacle(obstacles, current_cell, yaw)

        if self._OBSTACLE_RULES_ENABLED:
            # Deduce the track's wall orientation to snap the wrong-side L to it,
            # refined as more wall is mapped.
            theta = self._estimate_wall_orientation(semantic)
            if theta is not None:
                self._wall_theta = theta

            # Every already-latched pillar's wrong side, applied every cycle (a wall).
            region = self._latched_blocks(obstacles, blocked_walls.shape, current_cell)

            # The pillar currently being approached: decide its side now and include
            # it this cycle too (the snapshot copies above predate this latch).
            if upcoming is not None and upcoming.color is not None:
                travel_yaw     = self._travel_yaw(yaw)
                normal, anchor = self._latched_pass(upcoming, travel_yaw)
                up_region      = self._wrong_side_region(blocked_walls.shape, anchor, normal, current_cell)
                region         = up_region if region is None else (region | up_region)

        # Inflate walls and obstacles independently from their own distance fields,
        # so each keeps its own clearance. Walls also drive the clearance reward;
        # obstacles drive the around-obstacle ring. Unknown cells (never observed
        # as free space) are also excluded from traversal.
        D_mm        = self._clearance(blocked_walls)
        D_obs, ring = self._obstacle_fields(obstacle)
        traversable = (D_mm >= self._WALL_INFLATION) & ~unknown
        if D_obs is not None:
            traversable = traversable & (D_obs >= self._OBSTACLE_INFLATION)

        # Apply the virtual wrong-side block separately, with its own (smaller)
        # clearance baked into the region extents. Subtracted after wall inflation so
        # it never weakens wall clearance.
        if region is not None:
            block       = traversable & region
            traversable = traversable & ~region

        if not traversable[current_cell]:
            r0, c0 = current_cell
            r_cells = int(math.ceil(self._WALL_INFLATION / (self._K * FieldMap.CELL_SIZE)))
            rows_t, cols_t = traversable.shape
            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    nr, nc = r0 + dr, c0 + dc
                    if (0 <= nr < rows_t and 0 <= nc < cols_t
                            and not blocked_walls[nr, nc]):
                        traversable[nr, nc] = True

        goal_cell = self._select_goal(traversable, D_mm, trail, current_cell, yaw, ring)
        if goal_cell is None:
            self._emit_plan_debug(traversable, D_mm, trail, current_cell, None, None, None, block)
            return None, upcoming

        path = self._astar(traversable, D_mm, trail, current_cell, goal_cell, yaw)
        if path is None:
            self._emit_plan_debug(traversable, D_mm, trail, current_cell, goal_cell, None, None, block)
            return None, upcoming

        thinned = self._thin_path(path, traversable)
        gx, gy  = self._lookahead_point(thinned)
        self._emit_plan_debug(traversable, D_mm, trail, current_cell, goal_cell, path, thinned, block)
        return (gx, gy), upcoming

    def _emit_plan_debug(
        self,
        traversable:  np.ndarray,
        D_mm:         np.ndarray,
        trail:        np.ndarray,
        current_cell: Cell,
        goal_cell:    Cell | None,
        path:         list[Cell] | None,
        thinned:      list[Cell] | None,
        block:        np.ndarray | None = None,
    ) -> None:
        # Snapshot of the coarse planning grid for the visualiser. trail is copied
        # because the live array is mutated in place every cycle; the rest are
        # freshly built (traversable possibly via the wrong-side block) or read-only.
        self.on_plan_debug({
            "K":                self._K,
            "traversable":      traversable,
            "D_mm":             D_mm,
            "trail":            trail.copy(),
            "block":            block,
            "current":          current_cell,
            "goal":             goal_cell,
            "path":             path,
            "thinned":          thinned,
            "robot_radius":     self._WALL_INFLATION,
            "target_clearance": self._TARGET_CLEARANCE,
        })

    def _update_trail(self, current_cell: Cell, heading: radian) -> None:
        trail  = self._trail
        trail *= self._TRAIL_DECAY

        rows, cols = trail.shape
        coarse_mm  = self._K * FieldMap.CELL_SIZE
        W  = self._TRAIL_HALF_WIDTH  / coarse_mm   # cells (across-track)
        L  = self._TRAIL_HALF_LENGTH / coarse_mm   # cells (along-track)
        Wi = int(math.ceil(W))                     # bounding-box half-size

        r0, c0     = current_cell
        r_lo, r_hi = max(0, r0 - Wi), min(rows, r0 + Wi + 1)
        c_lo, c_hi = max(0, c0 - Wi), min(cols, c0 + Wi + 1)
        if r_lo >= r_hi or c_lo >= c_hi:
            return

        DR, DC = np.meshgrid(
            np.arange(r_lo, r_hi) - r0,
            np.arange(c_lo, c_hi) - c0,
            indexing = "ij",
        )

        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        # Forward axis is (cos h, sin h) in (row, col); behind is along < 0.
        # Skip the robot's own cross-section (along in (-1, 0]) so that nearly
        # perpendicular goal rays, which ride that cross-section, stay clean —
        # only genuinely backward travel gets penalised.
        along  =  DR * cos_h + DC * sin_h
        across = -DR * sin_h + DC * cos_h

        stamp = ((along <= -1.0) & (along >= -L) &
                 (np.abs(across) <= W)).astype(np.float32)

        sub = trail[r_lo:r_hi, c_lo:c_hi]
        np.maximum(sub, stamp, out = sub)

    def _coarse_masks(
        self,
        occupancy: np.ndarray,
        semantic:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Three coarse boolean maps:
        #   blocked_walls - walls, parking walls and any unclassified occupied cell.
        #                   Inflated by _WALL_INFLATION; also drives the clearance
        #                   reward so the robot stays centred in the corridor.
        #   obstacle      - obstacle cells only. Inflated by _OBSTACLE_INFLATION (so
        #                   the robot may pass them at a different margin) and drives
        #                   the around-obstacle ring reward.
        #   unknown       - coarse cells where no fine cell has ever received a
        #                   free-space vote (log-odds < 0). Excluded from traversal
        #                   so the robot does not enter unobserved space.
        K    = self._K
        rows = occupancy.shape[0] // K
        cols = occupancy.shape[1] // K

        occ = occupancy[: rows * K, : cols * K]
        sem = semantic [: rows * K, : cols * K]

        coarse_occ = occ.reshape(rows, K, cols, K)
        occ_hit  = (coarse_occ > self._OCCUPANCY_THRESHOLD).any(axis=(1, 3))
        unknown  = ~(coarse_occ < 0.0).any(axis=(1, 3))
        wall     = np.isin(sem, [int(CellLabel.WALL), int(CellLabel.PARKING_WALL)]) \
                     .reshape(rows, K, cols, K).any(axis=(1, 3))
        obstacle = (sem == int(CellLabel.OBSTACLE)) \
                     .reshape(rows, K, cols, K).any(axis=(1, 3))

        blocked_walls = (occ_hit & ~obstacle) | wall
        return blocked_walls, obstacle, unknown

    def _clearance(self, blocked_walls: np.ndarray) -> np.ndarray:
        # Distance (mm) to the nearest wall/unknown cell. Drives both the wall
        # inflation (traversable) and the clearance reward, so the robot stays
        # centred in the corridor. Cached on blocked_walls.
        if self._cached_blocked_walls is not None and \
           np.array_equal(blocked_walls, self._cached_blocked_walls):
            return self._cached_D_mm

        D_mm = distance_transform_edt(~blocked_walls).astype(np.float32) \
               * (self._K * FieldMap.CELL_SIZE)
        self._cached_blocked_walls = blocked_walls
        self._cached_D_mm          = D_mm
        return self._cached_D_mm

    def _obstacle_fields(
        self,
        obstacle: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        # Distance (mm) to the nearest obstacle, plus the around-obstacle ring reward.
        # The distance drives obstacle inflation (traversable); the ring rewards rays
        # that pass near an obstacle (they mark the track direction). Both None when
        # there are no obstacles. Cached on the obstacle mask.
        if not obstacle.any():
            return None, None
        if self._cached_obstacle is not None and \
           np.array_equal(obstacle, self._cached_obstacle):
            return self._cached_D_obs, self._cached_ring

        D_obs = distance_transform_edt(~obstacle).astype(np.float32) \
                * (self._K * FieldMap.CELL_SIZE)
        ring  = self._OBSTACLE_RING_WEIGHT * np.clip(
            1.0 - np.abs(D_obs - self._OBSTACLE_RING_DIST) / self._OBSTACLE_RING_HALF,
            0.0, 1.0,
        ).astype(np.float32)

        self._cached_obstacle = obstacle
        self._cached_D_obs    = D_obs
        self._cached_ring     = ring
        return D_obs, ring

    @staticmethod
    def _select_goal(
        traversable:  np.ndarray,
        D_mm:         np.ndarray,
        trail:        np.ndarray,
        current_cell: Cell,
        heading:      radian,
        ring:         np.ndarray | None = None,
    ) -> Cell | None:
        D_current = float(D_mm[current_cell])
        L_max_mm  = float(np.clip(
            PathPlanningProcessor._K_ADAPTIVE * D_current,
            PathPlanningProcessor._L_MIN,
            PathPlanningProcessor._L_MAX
        ))
        L_min_c   = PathPlanningProcessor._L_MIN / (PathPlanningProcessor._K * FieldMap.CELL_SIZE)
        L_max_c   = L_max_mm / (PathPlanningProcessor._K * FieldMap.CELL_SIZE)

        angles = np.linspace(
            heading - PathPlanningProcessor._HALF_ANGLE,
            heading + PathPlanningProcessor._HALF_ANGLE,
            PathPlanningProcessor._N_RAYS
        )

        rows, cols = traversable.shape
        best_goal:  Cell | None = None
        best_score: float       = -math.inf

        for theta in angles:
            dx, dy = math.cos(theta), math.sin(theta)
            last_free: Cell | None = None
            max_trail: float       = 0.0
            max_ring:  float       = 0.0

            d = L_min_c
            while d <= L_max_c:
                r = int(current_cell[0] + dx * d)
                c = int(current_cell[1] + dy * d)
                if not (0 <= r < rows and 0 <= c < cols):
                    break
                if not traversable[r, c]:
                    break
                last_free = (r, c)
                t = float(trail[r, c])
                if t > max_trail:
                    max_trail = t
                if ring is not None:
                    rg = float(ring[r, c])
                    if rg > max_ring:
                        max_ring = rg
                d += 1.0

            if last_free is None:
                continue

            reach = math.hypot(
                last_free[0] - current_cell[0],
                last_free[1] - current_cell[1],
            )
            # max_ring rewards rays that travel alongside an obstacle (i.e. go
            # around it down the track) rather than fleeing to open space.
            score = PathPlanningProcessor._W_DIST * reach + \
                    PathPlanningProcessor._W_CLEAR * float(D_mm[last_free]) - \
                    PathPlanningProcessor._TRAIL_WEIGHT * max_trail + \
                    max_ring

            if score > best_score:
                best_score = score
                best_goal  = last_free

        return best_goal

    @staticmethod
    def _astar(
        traversable:   np.ndarray,
        D_mm:          np.ndarray,
        trail:         np.ndarray,
        start:         Cell,
        goal:          Cell,
        robot_heading: radian,
    ) -> list[Cell] | None:
        SQRT2 = math.sqrt(2)
        hx    = math.cos(robot_heading)
        hy    = math.sin(robot_heading)

        rows, cols = traversable.shape
        g_score:   dict[Cell, float] = {start: 0.0}
        came_from: dict[Cell, Cell]  = {}

        open_heap: list[tuple[float, float, int, int]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start[0], start[1]))

        neighbours = [
            (-1,  0, 1.0),   ( 1,  0, 1.0),
            ( 0, -1, 1.0),   ( 0,  1, 1.0),
            (-1, -1, SQRT2), (-1,  1, SQRT2),
            ( 1, -1, SQRT2), ( 1,  1, SQRT2),
        ]

        while open_heap:
            _, g, r, c = heapq.heappop(open_heap)
            cell = (r, c)

            if cell == goal:
                return PathPlanningProcessor._reconstruct_path(came_from, goal)

            if g > g_score.get(cell, math.inf):
                continue

            for dr, dc, move_cost in neighbours:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if not traversable[nr, nc]:
                    continue

                D_n      = float(D_mm[nr, nc])
                cl_pen   = max(0.0, PathPlanningProcessor._TARGET_CLEARANCE - D_n) / \
                           PathPlanningProcessor._TARGET_CLEARANCE * PathPlanningProcessor._CLEARANCE_WEIGHT
                back_pen  = PathPlanningProcessor._BACKWARD_PENALTY * \
                            max(0.0, -(dr * hx + dc * hy) / move_cost)
                trail_pen = PathPlanningProcessor._TRAIL_WEIGHT_ASTAR * float(trail[nr, nc])
                new_g     = g + move_cost + cl_pen + back_pen + trail_pen

                neighbour = (nr, nc)
                if new_g < g_score.get(neighbour, math.inf):
                    g_score[neighbour]   = new_g
                    came_from[neighbour] = cell
                    f = new_g + math.hypot(nr - goal[0], nc - goal[1])
                    heapq.heappush(open_heap, (f, new_g, nr, nc))

        return None

    @staticmethod
    def _find_upcoming_obstacle(
        obstacles:    list[Obstacle],
        current_cell: Cell,
        heading:      radian,
    ) -> Obstacle | None:
        hx = math.cos(heading)
        hy = math.sin(heading)

        best_obs:  Obstacle | None = None
        best_dist: float           = math.inf

        for obs in obstacles:
            if obs.color is None:
                continue

            centroid_cell = PathPlanningProcessor._world_to_coarse(
                (float(obs.centroid[1]), float(obs.centroid[0]))
            )
            vx = centroid_cell[1] - current_cell[1]
            vy = centroid_cell[0] - current_cell[0]

            dist = math.hypot(vx, vy)
            if dist == 0:
                continue

            dot   = (hy * vx + hx * vy) / dist
            angle = math.acos(max(-1.0, min(1.0, dot)))

            if angle <= PathPlanningProcessor._HALF_ANGLE and dist < best_dist:
                best_dist = dist
                best_obs  = obs

        return best_obs

    def _latched_pass(
        self,
        obstacle: Obstacle,
        heading:  radian,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Return the frozen (wrong-side normal, world anchor) for this obstacle,
        # deciding and latching them the first time (or re-deciding if the
        # confirmed colour has flipped). After latching, neither the heading nor
        # the live centroid is consulted again, so the block is fully static in
        # the world while the robot swerves past.
        color = obstacle.color
        if (obstacle.pass_normal is not None
                and obstacle.pass_anchor is not None
                and obstacle.pass_color == color):
            return obstacle.pass_normal, obstacle.pass_anchor

        normal = self._wrong_side_normal(heading, color)
        if self._wall_theta is not None:
            normal = self._snap_to_walls(normal, self._wall_theta)   # align L to the walls
        anchor = np.asarray(obstacle.centroid, dtype=np.float32)   # world [x, y]
        self._field_map.latch_pass_side(obstacle, normal, color, anchor)
        return normal, anchor

    @staticmethod
    def _wrong_side_normal(heading: radian, color: ObstacleColor) -> np.ndarray:
        # Unit vector in (row, col) pointing to the side the robot must NOT pass.
        # Matches the original cross-product convention: red wrong side = left of
        # travel (cross > 0), green wrong side = right (cross < 0).
        s, c = math.sin(heading), math.cos(heading)
        if color == ObstacleColor.RED:
            return np.array([ s, -c], dtype=np.float32)
        return np.array([-s,  c], dtype=np.float32)

    @staticmethod
    def _snap_to_walls(vec: np.ndarray, theta: radian) -> np.ndarray:
        # Snap a (row, col) direction to the nearest arm of the track's rectilinear
        # grid {theta + k·90°}, so both legs of the L align with the actual walls
        # rather than the robot's heading at latch time. Rounds to the nearest of
        # the four arms, which keeps the snapped normal within 45° of the original
        # and therefore on the same (wrong) side.
        phi = math.atan2(float(vec[1]), float(vec[0]))
        k   = round((phi - theta) / (math.pi / 2.0))
        a   = theta + k * (math.pi / 2.0)
        return np.array([math.cos(a), math.sin(a)], dtype=np.float32)

    def _estimate_wall_orientation(self, semantic: np.ndarray) -> float | None:
        # Deduce the track's rectilinear orientation (rad, mod 90°) from the wall
        # cells. The wall edges form two perpendicular families; averaging their
        # gradient angles at 4× (structure-tensor trick) makes both families add
        # coherently, recovering the grid angle regardless of field rotation.
        # Runs on a coarse wall mask so it is cheap enough to re-run every cycle
        # while the wall map is still being refined.
        K    = self._K
        rows = semantic.shape[0] // K
        cols = semantic.shape[1] // K
        sem  = semantic[: rows * K, : cols * K]
        wall = (sem == int(CellLabel.WALL)).reshape(rows, K, cols, K).any(axis=(1, 3))

        gr, gc = np.gradient(wall.astype(np.float32))
        mag2   = gr * gr + gc * gc
        edge   = mag2 > 1e-6
        if int(edge.sum()) < self._WALL_ORIENT_MIN_CELLS:
            return None

        phi = np.arctan2(gc[edge], gr[edge])          # edge-normal orientation
        z   = np.sum(mag2[edge] * np.exp(4j * phi))   # weighted 4θ average
        if abs(z) < 1e-9:
            return None
        return float(np.angle(z) / 4.0)

    def _latched_blocks(
        self,
        obstacles:    list[Obstacle],
        shape:        tuple[int, int],
        current_cell: Cell,
    ) -> np.ndarray | None:
        # Union of the wrong-side regions of every pillar whose pass-side has been
        # latched. Rebuilt every cycle from the latched anchor/normal carried on the
        # obstacles, so the block is a persistent wall: it applies whether or not
        # the pillar is currently the one being approached, and does not flicker
        # when the pillar leaves the forward cone or is briefly unseen. (An obstacle
        # only drops out once it goes stale in FieldMap, by which point it is behind
        # the robot and its block no longer matters.)
        region: np.ndarray | None = None
        for obs in obstacles:
            if obs.pass_normal is None or obs.pass_anchor is None:
                continue
            r = self._wrong_side_region(shape, obs.pass_anchor, obs.pass_normal, current_cell)
            region = r if region is None else (region | r)
        return region

    def _wrong_side_region(
        self,
        shape:        tuple[int, int],
        anchor:       np.ndarray,
        normal:       np.ndarray,
        current_cell: Cell,
    ) -> np.ndarray:
        # L-shaped wrong-side block of a colour-confirmed pillar:
        #   Leg 1 (perpendicular to travel): from the pillar out to _BLOCK_REACH on
        #     the wrong side, thin (±_BLOCK_WINDOW) along travel — the original line
        #     that stops the robot crossing to the wrong side at the pillar.
        #   Leg 2 (along the wall): from the far tip of leg 1 back toward the robot,
        #     so the robot cannot drive up the wrong-side lane and round leg 1's tip
        #     where the bounding wall is not (yet) observed. It ends level with the
        #     robot's current along-position ("a wall point close to the robot").
        # Orientation comes from the latched normal (heading-free); only leg 2's
        # length tracks the robot, shrinking monotonically as it approaches.
        rows, cols = shape
        cr, cc     = self._world_to_coarse((float(anchor[1]), float(anchor[0])))

        coarse_mm = self._K * FieldMap.CELL_SIZE
        reach     = self._BLOCK_REACH         / coarse_mm
        window    = self._BLOCK_WINDOW        / coarse_mm
        back      = self._BLOCK_BACK          / coarse_mm
        arm       = self._BLOCK_ARM_THICKNESS / coarse_mm
        clr       = self._BLOCK_CLEARANCE     / coarse_mm   # baked-in margin (no wall inflation)

        nr, nc = float(normal[0]), float(normal[1])
        fr, fc = -nc, nr   # along-travel axis ⟂ normal (heading-free)

        RR, CC = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        dR, dC = RR - cr, CC - cc

        lateral = dR * nr + dC * nc
        along   = dR * fr + dC * fc

        # Leg 1: perpendicular line at the pillar, out to the wrong-side wall. Grown
        # by clr (the block is applied without wall inflation, so its clearance is
        # baked into the extents here).
        leg1 = (lateral >= -back - clr) & (lateral <= reach + clr) & \
               (np.abs(along) <= window + clr)

        # Leg 2: wall-hugging arm at leg 1's tip, spanning from the pillar back to
        # the robot's along-position.
        along_robot = (current_cell[0] - cr) * fr + (current_cell[1] - cc) * fc
        a_lo = min(along_robot, -window) - clr
        a_hi = max(along_robot,  window) + clr
        leg2 = (lateral >= reach - arm - clr) & (lateral <= reach + clr) & \
               (along >= a_lo) & (along <= a_hi)

        return leg1 | leg2

    @staticmethod
    def _thin_path(
        path:        list[Cell],
        traversable: np.ndarray,
    ) -> list[Cell]:
        if len(path) <= 2:
            return path

        anchor         = path[0]
        control_points = [anchor]

        for i in range(1, len(path)):
            if not PathPlanningProcessor._line_of_sight(anchor, path[i], traversable):
                control_points.append(path[i - 1])
                anchor = path[i - 1]

        control_points.append(path[-1])
        return control_points

    @staticmethod
    def _lookahead_point(
        thinned: list[Cell]
    ) -> Point:
        waypoints = [
            PathPlanningProcessor._coarse_to_world(cell)
            for cell in thinned
        ]

        accumulated = 0.0
        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i]
            x1, y1 = waypoints[i + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)

            if accumulated + seg_len >= PathPlanningProcessor._L_LOOKAHEAD:
                t = (PathPlanningProcessor._L_LOOKAHEAD - accumulated) / seg_len
                return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))

            accumulated += seg_len

        return waypoints[-1]

    @staticmethod
    def _world_to_coarse(pos: tuple[float, float]) -> Cell:
        K   = PathPlanningProcessor._K
        col = int((pos[1] + FieldMap.ORIGIN[0]) / (FieldMap.CELL_SIZE * K))
        row = int((pos[0] + FieldMap.ORIGIN[1]) / (FieldMap.CELL_SIZE * K))
        return (row, col)

    @staticmethod
    def _coarse_to_world(cell: Cell) -> tuple[float, float]:
        K      = PathPlanningProcessor._K
        offset = (K * FieldMap.CELL_SIZE) / 2.0
        north  = cell[0] * K * FieldMap.CELL_SIZE - FieldMap.ORIGIN[1] + offset
        east   = cell[1] * K * FieldMap.CELL_SIZE - FieldMap.ORIGIN[0] + offset
        return (north, east)

    @staticmethod
    def _in_bounds(cell: Cell, grid: np.ndarray) -> bool:
        return 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]

    @staticmethod
    def _reconstruct_path(came_from: dict[Cell, Cell], goal: Cell) -> list[Cell]:
        path    = [goal]
        current = goal
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _line_of_sight(a: Cell, b: Cell, traversable: np.ndarray) -> bool:
        r0, c0 = a
        r1, c1 = b
        dr  = abs(r1 - r0)
        dc  = abs(c1 - c0)
        sr  = 1 if r1 > r0 else -1
        sc  = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        rows, cols = traversable.shape

        while True:
            if not (0 <= r < rows and 0 <= c < cols):
                return False
            if not traversable[r, c]:
                return False
            if r == r1 and c == c1:
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r   += sr
            if e2 < dr:
                err += dr
                c   += sc
