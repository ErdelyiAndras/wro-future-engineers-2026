from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from threading import Lock

import numpy as np

from utils import mm, Point

class ObstacleColor(Enum):
    RED   = "red"
    GREEN = "green"

@dataclass
class Obstacle:
    cells:                  set[tuple[int, int]]
    red_votes:              int = 0
    green_votes:            int = 0
    observation_count:      int = 0
    frames_since_last_seen: int = 0

    @property
    def centroid(self) -> np.ndarray:
        cell_arr = np.array(list(self.cells), dtype = np.float64)  # (N, 2): (row, col)
        x = (cell_arr[:, 1] * FieldMap.CELL_SIZE - FieldMap.ORIGIN[0]).mean()
        y = (cell_arr[:, 0] * FieldMap.CELL_SIZE - FieldMap.ORIGIN[1]).mean()
        return np.array([x, y])

    @property
    def color(self) -> ObstacleColor | None:
        t = FieldMap._COLOR_THRESHOLD
        if self.red_votes >= t and self.red_votes > self.green_votes:
            return ObstacleColor.RED
        if self.green_votes >= t and self.green_votes > self.red_votes:
            return ObstacleColor.GREEN
        return None

    def copy(self) -> Obstacle:
        return Obstacle(
            cells                  = set(self.cells),
            red_votes              = self.red_votes,
            green_votes            = self.green_votes,
            observation_count      = self.observation_count,
            frames_since_last_seen = self.frames_since_last_seen,
        )

class Direction(Enum):
    CW  = "clockwise"
    CCW = "counter clockwise"

class CellLabel(IntEnum):
    UNKNOWN      = 0
    WALL         = auto()
    OBSTACLE     = auto()
    PARKING_WALL = auto()

class FieldMap:
    CELL_SIZE: mm    = 5.0
    WIDTH:     mm    = 6000.0
    HEIGHT:    mm    = 6000.0
    ROWS:      int   = int(HEIGHT // CELL_SIZE)
    COLS:      int   = int(WIDTH  // CELL_SIZE)
    ORIGIN:    Point = (3000.0, 3000.0)

    _L_OCC:  float =  0.85
    _L_FREE: float = -0.40
    L_MIN:   float = -2.0
    L_MAX:   float =  3.5

    _MATCH_RADIUS:    mm  = 80.0
    _STALE_FRAMES:    int = 5
    _COLOR_THRESHOLD: int = 10

    def __init__(self) -> None:
        self._lock:      Lock             = Lock()
        self._direction: Direction | None = None
        self._occupancy: np.ndarray       = np.zeros((self.ROWS, self.COLS), dtype = np.float32)
        self._semantic:  np.ndarray       = np.zeros((self.ROWS, self.COLS), dtype = np.uint8)
        self._obstacles: list[Obstacle]   = []

    @property
    def direction(self) -> Direction | None:
        return self._direction

    def initialize_direction(self, direction: Direction) -> None:
        if self._direction is not None:
            raise RuntimeError("Direction already initialized")
        with self._lock:
            self._direction = direction

    def update_occupancy(
        self,
        ego_pos:    np.ndarray,
        hit_points: np.ndarray,
    ) -> None:
        ego_pos    = np.asarray(ego_pos, dtype = np.float64)
        hit_points = np.atleast_2d(hit_points).astype(np.float64)

        rows_hit, cols_hit = self._grid_idx_from_world_coordinates(hit_points)
        valid_hit          = self._valid_mask(rows_hit, cols_hit)

        diffs   = hit_points - ego_pos            # (N, 2)
        lengths = np.linalg.norm(diffs, axis = 1) # (N,)

        valid_rays         = lengths > self.CELL_SIZE
        diffs_v, lengths_v = diffs[valid_rays], lengths[valid_rays]

        free_rows = np.empty(0, dtype = int)
        free_cols = np.empty(0, dtype = int)

        if len(diffs_v) > 0:
            n_steps = int(np.ceil(lengths_v.max() / self.CELL_SIZE))

            if n_steps >= 2:
                t = np.linspace(0.0, 1.0, n_steps + 2)[1:-1]  # (n_steps,)

                sample_lengths = t[np.newaxis, :] * lengths_v[:, np.newaxis]   # (N, n_steps)
                before_hit     = sample_lengths < (lengths_v[:, np.newaxis] - self.CELL_SIZE * 0.5)

                sample_points = ego_pos[np.newaxis, np.newaxis, :] + \
                                t[np.newaxis, :, np.newaxis] * \
                                diffs_v[:, np.newaxis, :] # (N, n_steps, 2)

                flat_points = sample_points[before_hit]  # (M, 2)

                if len(flat_points) > 0:
                    rows_f, cols_f = self._grid_idx_from_world_coordinates(flat_points)
                    in_bounds      = self._valid_mask(rows_f, cols_f)
                    free_rows      = rows_f[in_bounds]
                    free_cols      = cols_f[in_bounds]

        with self._lock:
            np.add.at(self._occupancy, (rows_hit[valid_hit], cols_hit[valid_hit]), self._L_OCC)

            if len(free_rows) > 0:
                # unknown = (self._semantic[free_rows, free_cols] == CellLabel.UNKNOWN) | True
                np.add.at(self._occupancy, (free_rows, free_cols), self._L_FREE)

            np.clip(self._occupancy, self.L_MIN, self.L_MAX, out = self._occupancy)

    def update_obstacles(self, clusters: list[np.ndarray]) -> None:
        with self._lock:
            for obs in self._obstacles:
                obs.frames_since_last_seen += 1

            for points in clusters:
                points           = np.atleast_2d(points).astype(np.float64)
                cluster_centroid = points.mean(axis = 0)

                best_obs  = None
                best_dist = float(self._MATCH_RADIUS)
                for obs in self._obstacles:
                    d = float(np.linalg.norm(obs.centroid - cluster_centroid))
                    if d < best_dist:
                        best_dist = d
                        best_obs  = obs

                rows, cols = self._grid_idx_from_world_coordinates(points)
                valid      = self._valid_mask(rows, cols)
                new_cells  = set(zip(rows[valid].tolist(), cols[valid].tolist()))

                if best_obs is not None:
                    best_obs.cells                  |= new_cells
                    best_obs.observation_count      += 1
                    best_obs.frames_since_last_seen  = 0
                else:
                    self._obstacles.append(Obstacle(
                        cells                  = new_cells,
                        observation_count      = 1,
                        frames_since_last_seen = 0,
                    ))

            self._obstacles = [
                obs for obs in self._obstacles
                if obs.frames_since_last_seen < self._STALE_FRAMES
            ]

            self._semantic[self._semantic == int(CellLabel.OBSTACLE)] = int(CellLabel.UNKNOWN)
            for obs in self._obstacles:
                if obs.cells:
                    cell_arr = np.array(list(obs.cells), dtype = int)
                    self._semantic[cell_arr[:, 0], cell_arr[:, 1]] = int(CellLabel.OBSTACLE)

    def vote_obstacle_color(
        self,
        obstacle: Obstacle,
        color:    ObstacleColor,
    ) -> None:
        with self._lock:
            for obs in self._obstacles:
                if obs.cells & obstacle.cells:
                    if color == ObstacleColor.RED:
                        obs.red_votes  += 1
                        obs.green_votes = max(0, obs.green_votes - 1)
                    else:
                        obs.green_votes += 1
                        obs.red_votes   = max(0, obs.red_votes - 1)
                    break

    def get_obstacles(self) -> list[Obstacle]:
        with self._lock:
            return [obs.copy() for obs in self._obstacles]

    def set_label(
        self,
        coords:              np.ndarray,
        label:               CellLabel,
        occupancy_threshold: float = L_MIN,
    ) -> None:
        coords     = np.atleast_2d(coords)
        rows, cols = self._grid_idx_from_world_coordinates(coords)
        valid      = self._valid_mask(rows, cols)
        r, c       = rows[valid], cols[valid]

        with self._lock:
            if occupancy_threshold > self.L_MIN:
                occupied = self._occupancy[r, c] > occupancy_threshold
                r, c     = r[occupied], c[occupied]

            self._semantic[r, c] = int(label)

    def get_occupied_world(
        self,
        threshold: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            rows, cols = np.where(self._occupancy > threshold)
            labels     = self._semantic[rows, cols].copy()

        coords = self._world_from_grid_idx(rows, cols)
        return coords, labels

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, list[Obstacle]]:
        with self._lock:
            return self._occupancy.copy(), \
                   self._semantic.copy(), \
                   [obs.copy() for obs in self._obstacles]

    def apply_semantic_updates(self, updates: dict[CellLabel, np.ndarray]) -> None:
        with self._lock:
            for label, mask in updates.items():
                self._semantic[mask] = int(label)

    def get_cells(
        self,
        coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords     = np.atleast_2d(coords)
        rows, cols = self._grid_idx_from_world_coordinates(coords)
        valid      = self._valid_mask(rows, cols)

        with self._lock:
            occupancy = self._occupancy[rows[valid], cols[valid]].copy()
            labels    = self._semantic [rows[valid], cols[valid]].copy()

        return occupancy, labels, valid

    @staticmethod
    def _grid_idx_from_world_coordinates(
        coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert coords.ndim == 2 and coords.shape[1] == 2, \
            f"Expected (N, 2) array, got shape {coords.shape}"

        idx  = ((coords + FieldMap.ORIGIN) // FieldMap.CELL_SIZE).astype(int)
        cols = idx[:, 0]
        rows = idx[:, 1]
        return rows, cols

    @staticmethod
    def _world_from_grid_idx(
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        x = cols * FieldMap.CELL_SIZE - FieldMap.ORIGIN[0]
        y = rows * FieldMap.CELL_SIZE - FieldMap.ORIGIN[1]
        return np.column_stack([x, y])

    @staticmethod
    def _valid_mask(
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        return (0 <= rows) & (rows < FieldMap.ROWS) & \
               (0 <= cols) & (cols < FieldMap.COLS)
