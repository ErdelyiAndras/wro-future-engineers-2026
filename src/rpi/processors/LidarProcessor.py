from __future__ import annotations

import math

import numpy as np

from processors.Processor import Processor
from control import FieldMap, EgoInformation
from utils import Point

class LidarProcessor(Processor):
    _SCAN_WRAP_THRESHOLD: float = 5.0 # degrees

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

        self._field_map.update_occupancy(ego_position, points)

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
