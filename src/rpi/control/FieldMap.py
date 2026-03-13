from __future__ import annotations

from enum import Enum, IntEnum, auto
from threading import Lock

import numpy as np

class Direction(Enum):
    CW  = "clockwise"
    CCW = "counter clockwise"

class CellLabel(IntEnum):
    UNKNOWN        = 0
    WALL           = auto()
    OBSTACLE       = auto()
    OBSTACLE_RED   = auto()
    OBSTACLE_GREEN = auto()
    PARKING_WALL   = auto()

class FieldMap:
    CELL_SIZE: int = 5    # mm
    WIDTH:     int = 6000 # mm
    HEIGHT:    int = 6000 # mm
    ROWS:      int = HEIGHT // CELL_SIZE
    COLS:      int = WIDTH  // CELL_SIZE
    ORIGIN:    tuple[int, int] = (3000, 3000) # mm, mm

    _L_OCC:  float =  0.85
    _L_FREE: float = -0.40
    _L_MIN:  float = -2.0
    _L_MAX:  float =  3.5

    def __init__(self) -> None:
        self._lock:      Lock             = Lock()
        self._direction: Direction | None = None
        self._occupancy: np.ndarray       = np.zeros((self.ROWS, self.COLS), dtype=np.float32)
        self._semantic:  np.ndarray       = np.zeros((self.ROWS, self.COLS), dtype=np.uint8)

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
        robot_pos:  np.ndarray,
        hit_points: np.ndarray,
    ) -> None:
        robot_pos  = np.asarray(robot_pos, dtype=float)
        hit_points = np.atleast_2d(hit_points).astype(float)

        rows_hit, cols_hit = self._grid_idx_from_world_coordinates(hit_points)
        valid_hit          = self._valid_mask(rows_hit, cols_hit)

        diffs   = hit_points - robot_pos          # (N, 2)
        lengths = np.linalg.norm(diffs, axis=1)   # (N,)

        valid_rays         = lengths > self.CELL_SIZE
        diffs_v, lengths_v = diffs[valid_rays], lengths[valid_rays]

        free_rows = np.empty(0, dtype=int)
        free_cols = np.empty(0, dtype=int)

        if len(diffs_v) > 0:
            n_steps = int(np.ceil(lengths_v.max() / self.CELL_SIZE))

            if n_steps >= 2:
                t = np.linspace(0.0, 1.0, n_steps + 2)[1:-1]  # (n_steps,)

                sample_lengths = t[np.newaxis, :] * lengths_v[:, np.newaxis]   # (N, n_steps)
                before_hit     = sample_lengths < (lengths_v[:, np.newaxis] - self.CELL_SIZE * 0.5)

                sample_points = (
                    robot_pos[np.newaxis, np.newaxis, :]
                    + t[np.newaxis, :, np.newaxis] * diffs_v[:, np.newaxis, :]
                )  # (N, n_steps, 2)

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

            np.clip(self._occupancy, self._L_MIN, self._L_MAX, out=self._occupancy)

    def set_label(self, coords: np.ndarray, label: CellLabel, occupancy_threshold: float = _L_MIN) -> None:
        coords     = np.atleast_2d(coords)
        rows, cols = self._grid_idx_from_world_coordinates(coords)
        valid      = self._valid_mask(rows, cols)
        r, c       = rows[valid], cols[valid]

        with self._lock:
            if occupancy_threshold > self._L_MIN:
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
        return (
            (0 <= rows) & (rows < FieldMap.ROWS) &
            (0 <= cols) & (cols < FieldMap.COLS)
        )
