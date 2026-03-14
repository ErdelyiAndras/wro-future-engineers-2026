from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from processors.Processor import Processor
from control import FieldMap, EgoInformation, CellLabel
from utils import Point


@dataclass
class LineSegment:
    endpoints: np.ndarray

    @property
    def centroid(self) -> np.ndarray:
        return (self.endpoints[0] + self.endpoints[1]) / 2

    @property
    def direction(self) -> np.ndarray:
        diff = self.endpoints[1] - self.endpoints[0]
        norm = np.linalg.norm(diff)
        return diff / norm if norm > 1e-10 else diff

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.endpoints[1] - self.endpoints[0]))

class LidarProcessor(Processor):
    _SCAN_WRAP_THRESHOLD: float = 5.0 # degrees

    _RANSAC_ITERATIONS: int   = 50
    _INLIER_THRESHOLD:  float = 10.0 # mm

    _MIN_WALL_LENGTH:     float = 400.0  # mm
    _MAX_NUMBER_OF_WALLS: int   = 4

    _MIN_PARKING_WALL_LENGTH:     float = 100.0 # mm
    _MAX_PARKING_WALL_LENGTH:     float = 250.0 # mm
    _MAX_NUMBER_OF_PARKING_WALLS: int   = 2

    _PILLAR_CLUSTER_RADIUS: float = 75.0   # mm
    _MIN_PILLAR_POINTS:     int   = 3

    def __init__(
        self,
        field_map:          FieldMap,
        ego_information:    EgoInformation,
        lidar_offset_angle: float = 0.0,
    ) -> None:
        super().__init__()

        self._field_map:          FieldMap       = field_map
        self._ego_information:    EgoInformation = ego_information
        self._lidar_offset_angle: float          = math.radians(lidar_offset_angle)

        self._current_scan: list[tuple[float, float]] = []
        self._prev_angle:   float | None              = None

        # debug
        self.points:        list[np.ndarray]        = []
        self.line_segments: list[list[LineSegment]] = []

    def _process(self, angle: float, distance: float, quality: int) -> None:
        if (self._prev_angle is not None
                and angle < self._prev_angle - math.radians(self._SCAN_WRAP_THRESHOLD)):
            self._process_scan()
            self._current_scan = []

        self._current_scan.append((angle, distance))
        self._prev_angle = angle

    def _process_scan(self) -> None:
        ego_position, yaw = self._ego_information.get_ego_information()
        if ego_position is None or yaw is None:
            return

        ego_position = np.array(ego_position, dtype = float)
        points       = self._to_world_coordinates(ego_position, yaw)

        self.points.append(points)  # debug

        self._field_map.update_occupancy(ego_position, points)

        # walls, remaining = self._detect_segments(
        #     points,
        #     min_length         = self._MIN_WALL_LENGTH,
        #     max_length         = float('inf'),
        #     number_of_segments = self._MAX_NUMBER_OF_WALLS
        # )

        # parking_walls, remaining = self._detect_segments(
        #     remaining,
        #     min_length         = self._MIN_PARKING_WALL_LENGTH,
        #     max_length         = self._MAX_PARKING_WALL_LENGTH,
        #     number_of_segments = self._MAX_NUMBER_OF_PARKING_WALLS
        # )

        # all_segments = walls + parking_walls
        # self.line_segments.append(all_segments)  # debug

        # for segment in walls:
        #     self._field_map.set_label(
        #         self._sample_segment(segment.endpoints),
        #         CellLabel.WALL,
        #         occupancy_threshold = 0.0
        #     )

        # for segment in parking_walls:
        #     self._field_map.set_label(
        #         self._sample_segment(segment.endpoints),
        #         CellLabel.PARKING_WALL,
        #         occupancy_threshold = 0.0
        #     )

        # for cluster in self._cluster_points(remaining):
        #     self._field_map.set_label(cluster, CellLabel.OBSTACLE)

    def _to_world_coordinates(self, ego_position: Point, yaw: float) -> np.ndarray:
        ego_pos_x, ego_pos_y = ego_position
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)

        scan      = np.array(self._current_scan)
        angles    = scan[:, 0] + self._lidar_offset_angle
        distances = scan[:, 1]

        x_ego = distances * np.sin(angles)
        y_ego = distances * np.cos(angles)

        x_world =  x_ego * cos_yaw + y_ego * sin_yaw + ego_pos_x
        y_world = -x_ego * sin_yaw + y_ego * cos_yaw + ego_pos_y

        return np.column_stack((x_world, y_world))

    def _detect_segments(
        self,
        points:             np.ndarray,
        min_length:         float,
        max_length:         float,
        number_of_segments: int
    ) -> tuple[list[LineSegment], np.ndarray]:

        remaining = points.copy()
        segments  = []

        for _ in range(number_of_segments):
            if len(remaining) < 2:
                break

            inlier_mask = self._ransac_line(remaining)
            if not inlier_mask.any():
                break

            segment = self._fit_line_segment(remaining[inlier_mask])

            if segment.length < min_length:
                break

            if segment.length <= max_length:
                segments.append(segment)
                remaining = remaining[~inlier_mask]

        return segments, remaining

    def _ransac_line(self, points: np.ndarray) -> np.ndarray:
        n         = len(points)
        best_mask = np.zeros(n, dtype = bool)

        for _ in range(self._RANSAC_ITERATIONS):
            i, j = np.random.choice(n, 2, replace = False)
            d    = points[j] - points[i]
            norm = np.linalg.norm(d)
            if norm < 1e-10:
                continue

            d /= norm
            mask = self._point_line_distances(points, points[i], d) < self._INLIER_THRESHOLD

            if mask.sum() > best_mask.sum():
                best_mask = mask

        return best_mask

    def _fit_line_segment(self, points: np.ndarray) -> LineSegment:
        centroid    = points.mean(axis = 0)
        _, _, vh    = np.linalg.svd(points - centroid)
        direction   = vh[0]
        projections = (points - centroid) @ direction
        return LineSegment(np.stack([
            centroid + projections.min() * direction,
            centroid + projections.max() * direction,
        ]))

    @staticmethod
    def _point_line_distances(
        points:    np.ndarray,
        centroid:  np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        normal = np.array([-direction[1], direction[0]])
        return np.abs((points - centroid) @ normal)

    def _cluster_points(self, points: np.ndarray) -> list[np.ndarray]:
        if len(points) < self._MIN_PILLAR_POINTS:
            return []

        clusters = []
        assigned = np.zeros(len(points), dtype = bool)

        for i in range(len(points)):
            if assigned[i]:
                continue
            dists   = np.linalg.norm(points - points[i], axis = 1)
            members = (dists <= self._PILLAR_CLUSTER_RADIUS) & ~assigned
            if members.sum() >= self._MIN_PILLAR_POINTS:
                clusters.append(points[members])
                assigned |= members

        return clusters

    @staticmethod
    def _sample_segment(endpoints: np.ndarray) -> np.ndarray:
        p1, p2 = endpoints[0].astype(float), endpoints[1].astype(float)
        length = float(np.linalg.norm(p2 - p1))
        if length == 0:
            return p1[np.newaxis]

        n = max(2, int(length / FieldMap.CELL_SIZE) + 1)
        t = np.linspace(0.0, 1.0, n)[:, np.newaxis]
        return p1 + t * (p2 - p1)
