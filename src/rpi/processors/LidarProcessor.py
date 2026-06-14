from __future__ import annotations

import math

import numpy as np

from processors.Processor import Processor
from control import FieldMap, EgoInformation
from utils import Point, degree, radian, mm

class LidarProcessor(Processor):
    def __init__(
        self,
        field_map:          FieldMap,
        ego_information:    EgoInformation,
        *,
        offset_angle: degree,
        mount_offset: Point,
    ) -> None:
        super().__init__()

        self._field_map:       FieldMap       = field_map
        self._ego_information: EgoInformation = ego_information
        self._offset_angle:    radian         = math.radians(offset_angle)
        self._mount_offset:    Point          = mount_offset

        self._current_scan: list[tuple[radian, mm]] = []

    def _process(self, scan: list[tuple[radian, mm, int]]) -> None:
        self._current_scan = [(angle, distance) for angle, distance, _ in scan]
        self._process_scan()

    def _process(self, scan: np.ndarray) -> None:
        scan = scan[:, :2]

        ego_position, yaw = self._ego_information.get_ego_information()
        if ego_position is None or yaw is None:
            return

        ego_position = np.array(ego_position, dtype = np.float64)
        points       = self._to_world_coordinates(scan, ego_position, yaw)

        self._field_map.update_occupancy(ego_position[[1, 0]], points)

    def _to_world_coordinates(self, scan: np.ndarray, ego_position: Point, yaw: radian) -> np.ndarray:
        ego_pos_x, ego_pos_y = ego_position
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)

        angles    = scan[:, 0] + self._offset_angle
        distances = scan[:, 1]

        mount_x, mount_y = self._mount_offset
        x_ego = distances * np.sin(angles) + mount_x
        y_ego = distances * np.cos(angles) + mount_y

        x_world =  x_ego * cos_yaw + y_ego * sin_yaw + ego_pos_y
        y_world = -x_ego * sin_yaw + y_ego * cos_yaw + ego_pos_x

        return np.column_stack((x_world, y_world))
