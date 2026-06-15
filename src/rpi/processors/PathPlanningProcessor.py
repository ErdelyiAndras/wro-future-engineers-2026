from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from processors.Processor import Processor
from control.FieldMap import FieldMap, CellLabel, Obstacle, ObstacleColor
from control.EgoInformation import EgoInformation
from utils import mm, radian, Point, Event

Cell = tuple[int, int]

class PathPlanningProcessor(Processor):
    _K:                   int    = 10
    _OCCUPANCY_THRESHOLD: float  = 0.0
    _ROBOT_RADIUS:        mm     = 150.0
    _HALF_ANGLE:          radian = math.radians(80.0)
    _N_RAYS:              int    = 17
    _L_MIN:               mm     = 100.0
    _L_MAX:               mm     = 500.0
    _W_DIST:              float  = 2.0
    _W_CLEAR:             float  = 0.5
    _K_ADAPTIVE:          float  = 10.0
    _TARGET_CLEARANCE:    mm     = 100.0
    _CLEARANCE_WEIGHT:    float  = 1.2
    _WRONG_SIDE_PENALTY:  float  = 150.0
    _BACKWARD_PENALTY:    float  = 150.0
    _L_LOOKAHEAD:         mm     = 450.0
    _SPEED:               float  = 150.0

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

        self._cached_blocked:     np.ndarray | None = None
        self._cached_traversable: np.ndarray | None = None
        self._cached_D_mm:        np.ndarray | None = None

    def _process(self) -> None:
        pos, yaw = self._ego_information.get_ego_information()
        if pos is None or yaw is None:
            return

        pose, next_obstacle = self._plan(pos, yaw)
        self.on_next_obstacle(next_obstacle)
        if pose is None:
            return

        dx = pose[0] - pos[0]
        dy = pose[1] - pos[1]

        forward_mm =  dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral_mm = -dx * math.sin(yaw) + dy * math.cos(yaw)

        self.on_target(forward_mm, lateral_mm, PathPlanningProcessor._SPEED)

    def _plan(self, pos: Point, yaw: radian) -> tuple[Point | None, Obstacle | None]:
        occupancy, semantic, obstacles = self._field_map.snapshot()

        traversable, D_mm = self._prepare_grid(occupancy, semantic)
        current_cell      = self._world_to_coarse(pos)

        if not self._in_bounds(current_cell, traversable):
            return None, None

        goal_cell = self._select_goal(traversable, D_mm, current_cell, yaw)
        if goal_cell is None:
            return None, None

        upcoming = self._find_upcoming_obstacle(obstacles, current_cell, yaw)
        path = self._astar(traversable, D_mm, upcoming, current_cell, goal_cell, yaw)
        if path is None:
            return None, upcoming

        thinned = self._thin_path(path, traversable)
        gx, gy  = self._lookahead_point(thinned)
        return (gx, gy), upcoming

    def _prepare_grid(
        self,
        occupancy: np.ndarray,
        semantic:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        K    = self._K
        rows = occupancy.shape[0] // K
        cols = occupancy.shape[1] // K

        occ = occupancy[: rows * K, : cols * K]
        sem = semantic [: rows * K, : cols * K]

        occ_coarse = occ.reshape(rows, K, cols, K).max(axis=(1, 3))

        blocking   = np.isin(sem, [int(CellLabel.WALL),
                                   int(CellLabel.OBSTACLE),
                                   int(CellLabel.PARKING_WALL)])
        sem_coarse = blocking.reshape(rows, K, cols, K).any(axis=(1, 3))

        blocked = (occ_coarse > self._OCCUPANCY_THRESHOLD) | sem_coarse

        if self._cached_blocked is not None and \
           np.array_equal(blocked, self._cached_blocked):
            return self._cached_traversable, self._cached_D_mm

        D_cells: np.ndarray = distance_transform_edt(~blocked).astype(np.float32)
        D_mm: np.ndarray    = D_cells * (K * FieldMap.CELL_SIZE)

        self._cached_blocked = blocked
        self._cached_traversable = D_mm >= self._ROBOT_RADIUS
        self._cached_D_mm = D_mm

        return self._cached_traversable, self._cached_D_mm

    @staticmethod
    def _select_goal(
        traversable:  np.ndarray,
        D_mm:         np.ndarray,
        current_cell: Cell,
        heading:      radian,
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
            score = PathPlanningProcessor._W_DIST * reach + \
                    PathPlanningProcessor._W_CLEAR * float(D_mm[last_free])
            if score > best_score:
                best_score = score
                best_goal  = last_free

        return best_goal

    @staticmethod
    def _astar(
        traversable:   np.ndarray,
        D_mm:          np.ndarray,
        upcoming:      Obstacle | None,
        start:         Cell,
        goal:          Cell,
        robot_heading: radian,
    ) -> list[Cell] | None:
        SQRT2 = math.sqrt(2)
        hx    = math.cos(robot_heading)
        hy    = math.sin(robot_heading)

        dir_pen = PathPlanningProcessor._direction_penalties(D_mm, upcoming, robot_heading)

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
                back_pen = PathPlanningProcessor._BACKWARD_PENALTY * \
                           max(0.0, -(dr * hx + dc * hy) / move_cost)
                new_g    = g + move_cost + cl_pen + float(dir_pen[nr, nc]) + back_pen

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

    @staticmethod
    def _direction_penalties(
        D_mm:          np.ndarray,
        next_obstacle: Obstacle | None,
        robot_heading: radian,
    ) -> np.ndarray:
        penalties  = np.zeros(D_mm.shape, dtype=np.float32)
        hx         = math.cos(robot_heading)
        hy         = math.sin(robot_heading)
        rows, cols = D_mm.shape

        if next_obstacle is None or next_obstacle.color is None:
            return penalties

        centroid_cell = PathPlanningProcessor._world_to_coarse(
            (float(next_obstacle.centroid[1]), float(next_obstacle.centroid[0]))
        )
        cr, cc = centroid_cell

        c_idx = np.arange(cols)
        r_idx = np.arange(rows)
        CC, RR = np.meshgrid(c_idx, r_idx)

        cross = hy * (RR - cr) - hx * (CC - cc)

        if next_obstacle.color == ObstacleColor.RED:
            penalties[cross > 0] += PathPlanningProcessor._WRONG_SIDE_PENALTY
        elif next_obstacle.color == ObstacleColor.GREEN:
            penalties[cross < 0] += PathPlanningProcessor._WRONG_SIDE_PENALTY

        return penalties

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
