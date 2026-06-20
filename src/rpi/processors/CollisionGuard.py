from __future__ import annotations

import math

import numpy as np

from processors.Processor import Processor
from utils import mm, degree, Event


class CollisionGuard(Processor):
    _MIN_DISTANCE:    mm     = 150.0
    _CONE_HALF_ANGLE: degree = 30.0

    def __init__(self, lidar_offset_angle: degree) -> None:
        super().__init__()
        self._offset_rad: float = math.radians(lidar_offset_angle)
        self.on_collision: Event = Event()

    def _process(self, scan: np.ndarray) -> None:
        if len(scan) == 0:
            return

        raw_angles = scan[:, 0]
        distances  = scan[:, 1]

        body_angles = raw_angles + self._offset_rad
        valid       = distances > 0.0
        body_angles = body_angles[valid]
        distances   = distances[valid]

        if len(distances) == 0:
            return

        half_rad = math.radians(CollisionGuard._CONE_HALF_ANGLE)
        in_cone  = np.abs(body_angles) <= half_rad

        if not np.any(in_cone):
            return

        if float(distances[in_cone].min()) < CollisionGuard._MIN_DISTANCE:
            self.on_collision()
