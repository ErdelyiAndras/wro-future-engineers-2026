from __future__ import annotations

import heapq
import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from processors.Processor import Processor
from control.FieldMap import FieldMap, CellLabel, Direction, Obstacle, ObstacleColor
from control.EgoInformation import EgoInformation
from utils import mm, radian, second, Point, Event

Cell = tuple[int, int]


class DirectionalPathPlanner(Processor):
    """Direction-enforcing, two-stage path planner.

    The track is a rectangular ring with four corners. Going clockwise means
    every corner is a *right* turn; counter-clockwise means every corner is a
    *left* turn. In this codebase yaw=0 faces north and +yaw rotates
    north->east->south->west (clockwise), so a right turn is an increasing yaw.
    The lap direction therefore fixes the sign of every large turn the robot is
    allowed to make:

        Direction.CW  -> allowed large-turn sign = +1  (turn right)
        Direction.CCW -> allowed large-turn sign = -1  (turn left)

    Goal selection enforces this invariant by construction: a ray that would
    require a large turn in the forbidden direction is never admissible, so the
    robot physically cannot commit a wrong-way (U-turn) corner. If the corridor
    only opens the wrong way -- which happens when the robot has somehow been
    turned around -- the invariant cannot hold and the planner enters a
    turn-back recovery instead of driving the wrong way.

    Planning is two-stage:
      Stage A -- a goal is chosen from the *walls only* (direction-clamped ray
                 cast on the wall corridor). This is the directed centreline.
      Stage B -- obstacles modify the wall plan: obstacle inflation plus, for
                 colour-confirmed pillars, the wrong-side L-block are subtracted
                 from the traversable grid and A* weaves the wall path past the
                 pillars on the correct side (RED on the robot's right, GREEN on
                 its left).

    Before FieldMap.direction has latched (the very start, a straight section)
    the turn-sign clamp is disabled: the robot just follows the corridor forward
    and stays centred. Recovery is disabled until the direction is known.
    """

    # Feature switches.
    #   _OBSTACLE_RULES_ENABLED: colour pass-side wrong-side block + wall snap.
    #   _INSPECT_ENABLED:        look at the next obstacle to read its colour.
    #   _REVERSE_ENABLED:        reverse-out recovery when blocked / near a wall.
    # Geometric obstacle avoidance (inflation) is always on, independent of
    # _OBSTACLE_RULES_ENABLED.
    _OBSTACLE_RULES_ENABLED: bool = True
    _INSPECT_ENABLED:        bool = True
    _REVERSE_ENABLED:        bool = False

    _K:                   int    = 5
    _OCCUPANCY_THRESHOLD: float  = 0.0
    _WALL_INFLATION:      mm     = 50.0   # keep the robot centre this far from walls/unknown
    _OBSTACLE_INFLATION:  mm     = 30.0   # keep the robot centre this far from obstacles

    # Goal-selection ray cast.
    _HALF_ANGLE:          radian = math.radians(80.0)
    _N_RAYS:              int    = 17
    _L_MIN:               mm     = 100.0
    _L_MAX:               mm     = 500.0
    _W_DIST:              float  = 2.0
    _W_CLEAR:             float  = 1.0
    _K_ADAPTIVE:          float  = 3.0

    # Direction invariant. Rays that demand a turn larger than _CENTER_BAND in
    # the *forbidden* direction are inadmissible. Within the band, turns of
    # either sign are allowed so the robot can still centre itself in the lane.
    _CENTER_BAND:         radian = math.radians(25.0)

    # A* step cost.
    _TARGET_CLEARANCE:    mm     = 100.0
    _CLEARANCE_WEIGHT:    float  = 1.0
    _BACKWARD_PENALTY:    float  = 150.0

    # Wrong-side block of a colour-confirmed pillar (reused from the old planner).
    _BLOCK_REACH:          mm    = 700.0  # lateral extent into the wrong side (past wall)
    _BLOCK_WINDOW:         mm    = 100.0  # along-track half-extent of the block (leg 1)
    _BLOCK_BACK:           mm    = 50.0   # overlap behind centroid, no gap to pillar
    _BLOCK_ARM_THICKNESS:  mm    = 150.0  # thickness of the wall-hugging arm (leg 2 of the L)
    _BLOCK_CLEARANCE:      mm    = 30.0   # how close the robot centre may get to the block
    _WALL_ORIENT_MIN_CELLS: int  = 60     # min coarse wall edge cells before trusting the angle

    _L_LOOKAHEAD:          mm    = 450.0
    _SPEED:                float = 80.0
    _FOLLOW_SPEED:         float = 100.0   # full speed when driving the memorized line

    # Turn-back recovery (used once the direction is latched).
    _RECOVER_TURN:        radian = math.radians(45.0)  # steer this far to the correct side
    _RECOVER_SPEED:       float  = 50.0

    # Recorded racing line.
    _ROUTE_MIN_SPACING:   mm     = 50.0
    _MIN_ROUTE_POINTS:    int    = 20
    _STOP_RADIUS:         mm     = 200.0

    # Reverse-out recovery.
    _REVERSE_SPEED:       float  = 60.0
    _COLLISION_ACTIVE_S:  second = 0.8

    # Heading-free travel reference for the pass side.
    _TRAVEL_TANGENT_POINTS: int  = 5

    # Active colour inspection.
    _INSPECT_RANGE:       mm     = 1500.0
    _INSPECT_HALF_ANGLE:  radian = math.radians(60.0)
    _INSPECT_STANDOFF:    mm     = 400.0
    _INSPECT_SPEED:       float  = 60.0
    _INSPECT_TIMEOUT:     second = 3.0
    _INSPECT_MATCH:       mm     = 100.0

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

        self._cached_blocked_walls: np.ndarray | None = None
        self._cached_D_mm:          np.ndarray | None = None
        self._cached_obstacle:      np.ndarray | None = None
        self._cached_D_obs:         np.ndarray | None = None

        # Track's rectilinear orientation (rad, mod 90 deg), to snap the
        # wrong-side L to the walls. None until enough wall is seen.
        self._wall_theta:         float | None      = None

        self._following:    bool         = False
        self._stopped:      bool         = False
        self._route:        list[Point]  = []
        self._route_cursor: int          = 0

        self._reversing:        bool  = False
        self._last_collision_t: float = -math.inf

        self._inspect_start:    float | None = None
        self._inspect_anchor:   Point | None = None

    # ------------------------------------------------------------------ #
    #  Top-level state machine                                            #
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

        # Reactive lap: directed forward plan, or turn-back / reverse recovery.
        self._directed_step(pos, yaw)

    def notify_collision(self) -> None:
        self._last_collision_t = time.monotonic()

    def _collision_active(self) -> bool:
        return (time.monotonic() - self._last_collision_t) < self._COLLISION_ACTIVE_S

    def _allowed_sign(self) -> float | None:
        # +1 = right turns allowed (CW); -1 = left turns allowed (CCW);
        # None until the lap direction is known.
        direction = self._field_map.direction
        if direction == Direction.CW:
            return +1.0
        if direction == Direction.CCW:
            return -1.0
        return None

    def _directed_step(self, pos: Point, yaw: radian) -> None:
        if self._INSPECT_ENABLED and self._inspect_obstacle(pos, yaw):
            return

        target, next_obstacle, status = self._plan(pos, yaw)
        self.on_next_obstacle(next_obstacle)

        collision = self._REVERSE_ENABLED and self._collision_active()
        if target is not None and not collision:
            self._reversing = False
            self._record_pose(pos)
            self._emit_target(pos, yaw, target)
            return

        if status == "recover" or collision:
            if self._REVERSE_ENABLED:
                self._reversing = True
                rev_target = self._reverse_target(pos)
                if rev_target is not None:
                    self._emit_target(pos, yaw, rev_target, reverse = True)
                    return
            # No reverse support (or nothing to back toward): turn back in place
            # toward the allowed side until a forward corridor reappears.
            self._recover(pos, yaw)
            return

        # Boxed in with the direction not yet known: hold and let planning retry.
        self.on_target(0.0, 0.0, 0.0)

    def _recover(self, pos: Point, yaw: radian) -> None:
        # Steer toward the correct (allowed) side at low speed. The invariant has
        # been violated (the corridor only opens the wrong way), which means the
        # robot is facing backwards, so turning the allowed way turns it back onto
        # the track.
        sign = self._allowed_sign()
        if sign is None:
            self.on_target(0.0, 0.0, 0.0)
            return
        aim_yaw = yaw + sign * self._RECOVER_TURN
        aim     = (pos[0] + self._L_MIN * math.cos(aim_yaw),
                   pos[1] + self._L_MIN * math.sin(aim_yaw))
        self._emit_target(pos, yaw, aim, speed = self._RECOVER_SPEED)

    def _reverse_target(self, pos: Point) -> Point | None:
        while len(self._route) > 1 and \
              self._dist(pos, self._route[-1]) < self._ROUTE_MIN_SPACING:
            self._route.pop()
        return self._route[-1] if self._route else None

    # ------------------------------------------------------------------ #
    #  Two-stage plan                                                     #
    # ------------------------------------------------------------------ #

    def _plan(
        self,
        pos: Point,
        yaw: radian,
    ) -> tuple[Point | None, Obstacle | None, str]:
        occupancy, semantic, obstacles = self._field_map.snapshot()

        blocked_walls, obstacle = self._coarse_masks(occupancy, semantic)
        current_cell            = self._world_to_coarse(pos)

        if not self._in_bounds(current_cell, blocked_walls):
            return None, None, "stop"

        allowed_sign = self._allowed_sign()
        upcoming     = self._find_upcoming_obstacle(obstacles, current_cell, yaw)

        # --- Stage A: directed goal from the walls only ---
        D_mm              = self._clearance(blocked_walls)
        traversable_walls = D_mm >= self._WALL_INFLATION

        goal_cell = self._select_directed_goal(
            traversable_walls, D_mm, current_cell, yaw, allowed_sign,
        )
        if goal_cell is None:
            self._emit_plan_debug(traversable_walls, D_mm, current_cell, None, None, None, None)
            # Direction known -> wrong-way / boxed -> turn back. Otherwise hold.
            return None, upcoming, ("recover" if allowed_sign is not None else "stop")

        # --- Stage B: modify the wall plan to avoid obstacles ---
        traversable = traversable_walls.copy()

        D_obs, _ = self._obstacle_fields(obstacle)
        if D_obs is not None:
            traversable &= (D_obs >= self._OBSTACLE_INFLATION)

        block:  np.ndarray | None = None
        region: np.ndarray | None = None
        if self._OBSTACLE_RULES_ENABLED:
            theta = self._estimate_wall_orientation(semantic)
            if theta is not None:
                self._wall_theta = theta

            region = self._latched_blocks(obstacles, blocked_walls.shape, current_cell)

            if upcoming is not None and upcoming.color is not None:
                travel_yaw     = self._travel_yaw(yaw)
                normal, anchor = self._latched_pass(upcoming, travel_yaw)
                up_region      = self._wrong_side_region(blocked_walls.shape, anchor, normal, current_cell)
                region         = up_region if region is None else (region | up_region)

            if region is not None:
                block       = traversable & region
                traversable = traversable & ~region

        # The wall goal may now sit inside obstacle inflation or a block; pull it
        # back along the ray to the nearest traversable cell so A* still has a
        # reachable target rather than spuriously triggering recovery.
        goal_cell = self._pull_goal_in(goal_cell, current_cell, traversable)
        if goal_cell is None:
            self._emit_plan_debug(traversable, D_mm, current_cell, None, None, None, block)
            return None, upcoming, ("recover" if allowed_sign is not None else "stop")

        path = self._astar(traversable, D_mm, current_cell, goal_cell, yaw)
        if path is None:
            self._emit_plan_debug(traversable, D_mm, current_cell, goal_cell, None, None, block)
            return None, upcoming, ("recover" if allowed_sign is not None else "stop")

        thinned = self._thin_path(path, traversable)
        gx, gy  = self._lookahead_point(thinned)
        self._emit_plan_debug(traversable, D_mm, current_cell, goal_cell, path, thinned, block)
        return (gx, gy), upcoming, "ok"

    def _select_directed_goal(
        self,
        traversable:  np.ndarray,
        D_mm:         np.ndarray,
        current_cell: Cell,
        heading:      radian,
        allowed_sign: float | None,
    ) -> Cell | None:
        # Ray cast over a forward cone, but a ray demanding a large turn in the
        # forbidden direction is inadmissible (the direction invariant). Among
        # admissible rays, pick the one that reaches furthest while staying clear
        # of the walls.
        D_current = float(D_mm[current_cell])
        L_max_mm  = float(np.clip(self._K_ADAPTIVE * D_current, self._L_MIN, self._L_MAX))
        coarse_mm = self._K * FieldMap.CELL_SIZE
        L_min_c   = self._L_MIN / coarse_mm
        L_max_c   = L_max_mm / coarse_mm

        angles = np.linspace(
            heading - self._HALF_ANGLE,
            heading + self._HALF_ANGLE,
            self._N_RAYS,
        )

        rows, cols = traversable.shape
        best_goal:  Cell | None = None
        best_score: float       = -math.inf

        for theta in angles:
            delta = self._wrap(theta - heading)
            if allowed_sign is not None and \
               abs(delta) > self._CENTER_BAND and (delta * allowed_sign) < 0.0:
                continue   # large turn the wrong way -> forbidden

            dx, dy = math.cos(theta), math.sin(theta)
            last_free: Cell | None = None

            d = L_min_c
            while d <= L_max_c:
                r = int(current_cell[0] + dx * d)
                c = int(current_cell[1] + dy * d)
                if not (0 <= r < rows and 0 <= c < cols):
                    break
                if not traversable[r, c]:
                    break
                last_free = (r, c)
                d += 1.0

            if last_free is None:
                continue

            reach = math.hypot(
                last_free[0] - current_cell[0],
                last_free[1] - current_cell[1],
            )
            score = self._W_DIST * reach + self._W_CLEAR * float(D_mm[last_free])
            if score > best_score:
                best_score = score
                best_goal  = last_free

        return best_goal

    @staticmethod
    def _pull_goal_in(
        goal:        Cell,
        current:     Cell,
        traversable: np.ndarray,
    ) -> Cell | None:
        # Walk from goal back toward current and return the first traversable cell.
        if traversable[goal]:
            return goal

        r0, c0 = current
        r1, c1 = goal
        n      = max(abs(r1 - r0), abs(c1 - c0))
        if n == 0:
            return goal if traversable[goal] else None

        for k in range(n, -1, -1):
            t = k / n
            r = int(round(r0 + (r1 - r0) * t))
            c = int(round(c0 + (c1 - c0) * t))
            if traversable[r, c]:
                return (r, c)
        return None

    # ------------------------------------------------------------------ #
    #  Debug                                                              #
    # ------------------------------------------------------------------ #

    def _emit_plan_debug(
        self,
        traversable:  np.ndarray,
        D_mm:         np.ndarray,
        current_cell: Cell,
        goal_cell:    Cell | None,
        path:         list[Cell] | None,
        thinned:      list[Cell] | None,
        block:        np.ndarray | None = None,
    ) -> None:
        # Same dict shape the visualiser expects. There is no breadcrumb trail in
        # this planner, so an all-zero array is emitted purely for compatibility.
        self.on_plan_debug({
            "K":                self._K,
            "traversable":      traversable,
            "D_mm":             D_mm,
            "trail":            np.zeros_like(D_mm),
            "block":            block,
            "current":          current_cell,
            "goal":             goal_cell,
            "path":             path,
            "thinned":          thinned,
            "robot_radius":     self._WALL_INFLATION,
            "target_clearance": self._TARGET_CLEARANCE,
        })

    # ------------------------------------------------------------------ #
    #  Coarse grids / clearance / obstacle fields                         #
    # ------------------------------------------------------------------ #

    def _coarse_masks(
        self,
        occupancy: np.ndarray,
        semantic:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        K    = self._K
        rows = occupancy.shape[0] // K
        cols = occupancy.shape[1] // K

        occ = occupancy[: rows * K, : cols * K]
        sem = semantic [: rows * K, : cols * K]

        occ_hit  = occ.reshape(rows, K, cols, K).max(axis=(1, 3)) > self._OCCUPANCY_THRESHOLD
        wall     = np.isin(sem, [int(CellLabel.WALL), int(CellLabel.PARKING_WALL)]) \
                     .reshape(rows, K, cols, K).any(axis=(1, 3))
        obstacle = (sem == int(CellLabel.OBSTACLE)) \
                     .reshape(rows, K, cols, K).any(axis=(1, 3))

        blocked_walls = (occ_hit & ~obstacle) | wall
        return blocked_walls, obstacle

    def _clearance(self, blocked_walls: np.ndarray) -> np.ndarray:
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
    ) -> tuple[np.ndarray | None, None]:
        if not obstacle.any():
            return None, None
        if self._cached_obstacle is not None and \
           np.array_equal(obstacle, self._cached_obstacle):
            return self._cached_D_obs, None

        D_obs = distance_transform_edt(~obstacle).astype(np.float32) \
                * (self._K * FieldMap.CELL_SIZE)
        self._cached_obstacle = obstacle
        self._cached_D_obs    = D_obs
        return D_obs, None

    # ------------------------------------------------------------------ #
    #  A*                                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _astar(
        traversable:   np.ndarray,
        D_mm:          np.ndarray,
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
                return DirectionalPathPlanner._reconstruct_path(came_from, goal)

            if g > g_score.get(cell, math.inf):
                continue

            for dr, dc, move_cost in neighbours:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if not traversable[nr, nc]:
                    continue

                D_n      = float(D_mm[nr, nc])
                cl_pen   = max(0.0, DirectionalPathPlanner._TARGET_CLEARANCE - D_n) / \
                           DirectionalPathPlanner._TARGET_CLEARANCE * \
                           DirectionalPathPlanner._CLEARANCE_WEIGHT
                back_pen = DirectionalPathPlanner._BACKWARD_PENALTY * \
                           max(0.0, -(dr * hx + dc * hy) / move_cost)
                new_g    = g + move_cost + cl_pen + back_pen

                neighbour = (nr, nc)
                if new_g < g_score.get(neighbour, math.inf):
                    g_score[neighbour]   = new_g
                    came_from[neighbour] = cell
                    f = new_g + math.hypot(nr - goal[0], nc - goal[1])
                    heapq.heappush(open_heap, (f, new_g, nr, nc))

        return None

    # ------------------------------------------------------------------ #
    #  Obstacle selection + wrong-side block                              #
    # ------------------------------------------------------------------ #

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

            centroid_cell = DirectionalPathPlanner._world_to_coarse(
                (float(obs.centroid[1]), float(obs.centroid[0]))
            )
            vx = centroid_cell[1] - current_cell[1]
            vy = centroid_cell[0] - current_cell[0]

            dist = math.hypot(vx, vy)
            if dist == 0:
                continue

            dot   = (hy * vx + hx * vy) / dist
            angle = math.acos(max(-1.0, min(1.0, dot)))

            if angle <= DirectionalPathPlanner._HALF_ANGLE and dist < best_dist:
                best_dist = dist
                best_obs  = obs

        return best_obs

    def _latched_pass(
        self,
        obstacle: Obstacle,
        heading:  radian,
    ) -> tuple[np.ndarray, np.ndarray]:
        color = obstacle.color
        if (obstacle.pass_normal is not None
                and obstacle.pass_anchor is not None
                and obstacle.pass_color == color):
            return obstacle.pass_normal, obstacle.pass_anchor

        normal = self._wrong_side_normal(heading, color)
        if self._wall_theta is not None:
            normal = self._snap_to_walls(normal, self._wall_theta)
        anchor = np.asarray(obstacle.centroid, dtype=np.float32)
        self._field_map.latch_pass_side(obstacle, normal, color, anchor)
        return normal, anchor

    @staticmethod
    def _wrong_side_normal(heading: radian, color: ObstacleColor) -> np.ndarray:
        # Unit vector in (row, col) pointing to the side the robot must NOT pass.
        # red wrong side = left of travel (cross > 0), green wrong side = right.
        s, c = math.sin(heading), math.cos(heading)
        if color == ObstacleColor.RED:
            return np.array([ s, -c], dtype=np.float32)
        return np.array([-s,  c], dtype=np.float32)

    @staticmethod
    def _snap_to_walls(vec: np.ndarray, theta: radian) -> np.ndarray:
        phi = math.atan2(float(vec[1]), float(vec[0]))
        k   = round((phi - theta) / (math.pi / 2.0))
        a   = theta + k * (math.pi / 2.0)
        return np.array([math.cos(a), math.sin(a)], dtype=np.float32)

    def _estimate_wall_orientation(self, semantic: np.ndarray) -> float | None:
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

        phi = np.arctan2(gc[edge], gr[edge])
        z   = np.sum(mag2[edge] * np.exp(4j * phi))
        if abs(z) < 1e-9:
            return None
        return float(np.angle(z) / 4.0)

    def _latched_blocks(
        self,
        obstacles:    list[Obstacle],
        shape:        tuple[int, int],
        current_cell: Cell,
    ) -> np.ndarray | None:
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
        rows, cols = shape
        cr, cc     = self._world_to_coarse((float(anchor[1]), float(anchor[0])))

        coarse_mm = self._K * FieldMap.CELL_SIZE
        reach     = self._BLOCK_REACH         / coarse_mm
        window    = self._BLOCK_WINDOW        / coarse_mm
        back      = self._BLOCK_BACK          / coarse_mm
        arm       = self._BLOCK_ARM_THICKNESS / coarse_mm
        clr       = self._BLOCK_CLEARANCE     / coarse_mm

        nr, nc = float(normal[0]), float(normal[1])
        fr, fc = -nc, nr   # along-travel axis perpendicular to normal

        RR, CC = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        dR, dC = RR - cr, CC - cc

        lateral = dR * nr + dC * nc
        along   = dR * fr + dC * fc

        leg1 = (lateral >= -back - clr) & (lateral <= reach + clr) & \
               (np.abs(along) <= window + clr)

        along_robot = (current_cell[0] - cr) * fr + (current_cell[1] - cc) * fc
        a_lo = min(along_robot, -window) - clr
        a_hi = max(along_robot,  window) + clr
        leg2 = (lateral >= reach - arm - clr) & (lateral <= reach + clr) & \
               (along >= a_lo) & (along <= a_hi)

        return leg1 | leg2

    # ------------------------------------------------------------------ #
    #  Active colour inspection                                           #
    # ------------------------------------------------------------------ #

    def _inspect_obstacle(self, pos: Point, yaw: radian) -> bool:
        obstacles    = self._field_map.get_obstacles()
        current_cell = self._world_to_coarse(pos)
        obs          = self._obstacle_to_inspect(obstacles, current_cell, yaw)

        if obs is None:
            self._inspect_start  = None
            self._inspect_anchor = None
            return False

        obs_pt = (float(obs.centroid[1]), float(obs.centroid[0]))
        now    = time.monotonic()
        if self._inspect_anchor is None or \
           self._dist(obs_pt, self._inspect_anchor) > self._INSPECT_MATCH:
            self._inspect_start  = now
            self._inspect_anchor = obs_pt

        if now - self._inspect_start >= self._INSPECT_TIMEOUT:
            return False

        self.on_next_obstacle(obs)
        self._aim_at_obstacle(pos, yaw, obs_pt)
        return True

    def _obstacle_to_inspect(
        self,
        obstacles:    list[Obstacle],
        current_cell: Cell,
        heading:      radian,
    ) -> Obstacle | None:
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
        dist = self._dist(pos, obs_pt)
        if dist <= self._INSPECT_STANDOFF:
            self.on_target(0.0, 0.0, 0.0)
            return
        t   = (dist - self._INSPECT_STANDOFF) / dist
        aim = (pos[0] + t * (obs_pt[0] - pos[0]),
               pos[1] + t * (obs_pt[1] - pos[1]))
        self._emit_target(pos, yaw, aim, speed = self._INSPECT_SPEED)

    # ------------------------------------------------------------------ #
    #  Targets / route / lap                                              #
    # ------------------------------------------------------------------ #

    def _home_and_stop(self, pos: Point, yaw: radian) -> None:
        home = self._field_map.start_position
        if home is not None and self._dist(pos, home) > self._STOP_RADIUS:
            self._emit_target(pos, yaw, home)
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
            self._route.append(self._route[0])
        self._route_cursor = self._closest_route_index(pos)
        self.on_route(list(self._route))

    def _follow_route(self, pos: Point) -> Point | None:
        if len(self._route) < 2:
            return None

        self._route_cursor = self._closest_route_index(pos)

        n           = len(self._route)
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

    def _travel_yaw(self, fallback_yaw: radian) -> radian:
        n = self._TRAVEL_TANGENT_POINTS
        if len(self._route) >= n:
            a = self._route[-n]
            b = self._route[-1]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            if math.hypot(dx, dy) > 1e-6:
                return math.atan2(dy, dx)
        return fallback_yaw

    # ------------------------------------------------------------------ #
    #  Path post-processing + geometry helpers                            #
    # ------------------------------------------------------------------ #

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
            if not DirectionalPathPlanner._line_of_sight(anchor, path[i], traversable):
                control_points.append(path[i - 1])
                anchor = path[i - 1]

        control_points.append(path[-1])
        return control_points

    @staticmethod
    def _lookahead_point(thinned: list[Cell]) -> Point:
        waypoints = [
            DirectionalPathPlanner._coarse_to_world(cell)
            for cell in thinned
        ]

        accumulated = 0.0
        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i]
            x1, y1 = waypoints[i + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)

            if accumulated + seg_len >= DirectionalPathPlanner._L_LOOKAHEAD:
                t = (DirectionalPathPlanner._L_LOOKAHEAD - accumulated) / seg_len
                return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))

            accumulated += seg_len

        return waypoints[-1]

    @staticmethod
    def _world_to_coarse(pos: tuple[float, float]) -> Cell:
        K   = DirectionalPathPlanner._K
        col = int((pos[1] + FieldMap.ORIGIN[0]) / (FieldMap.CELL_SIZE * K))
        row = int((pos[0] + FieldMap.ORIGIN[1]) / (FieldMap.CELL_SIZE * K))
        return (row, col)

    @staticmethod
    def _coarse_to_world(cell: Cell) -> tuple[float, float]:
        K      = DirectionalPathPlanner._K
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

    @staticmethod
    def _dist(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi
