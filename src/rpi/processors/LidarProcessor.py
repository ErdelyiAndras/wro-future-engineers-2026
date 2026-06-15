from __future__ import annotations

import math

import numpy as np

from processors.Processor import Processor
from processors.LidarPoseEstimator import LidarPoseEstimator
from control import FieldMap, EgoInformation
from utils import Point, degree, radian, mm

class LidarProcessor(Processor):
    _MIN_TRANSLATION_CHANGE: mm     = 3.0
    _MIN_ROTATION_CHANGE:    radian = math.radians(0.5)

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

        self._pose_estimator: LidarPoseEstimator = LidarPoseEstimator(
            field_map,
            self._offset_angle,
            self._mount_offset,
        )

    def _process(self, scan: np.ndarray) -> None:
        scan = scan[:, :2]

        prior_pos, prior_yaw = self._ego_information.get_ego_information()
        if prior_pos is None or prior_yaw is None:
            return

        ego_position = np.array(prior_pos, dtype = np.float64)
        yaw          = prior_yaw
        points       = self._to_world_coordinates(scan, ego_position, yaw)

        estimate = self._pose_estimator.estimate(scan, prior_pos, prior_yaw)
        estimate = None

        if estimate is not None:
            estimated_pos, estimated_yaw = estimate
            epx, epy = estimated_pos
            ppx, ppy = prior_pos
            translation = math.hypot(epx - ppx, epy - ppy)
            rotation    = abs((estimated_yaw - prior_yaw + math.pi) % (2 * math.pi) - math.pi)

            if translation >= self._MIN_TRANSLATION_CHANGE or rotation >= self._MIN_ROTATION_CHANGE:
                self._ego_information.update_lidar(estimated_pos[0], estimated_pos[1], estimated_yaw)
                ego_position = np.array(estimated_pos, dtype = np.float64)
                yaw          = estimated_yaw
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
