from __future__ import annotations

import numpy as np

from processors.Processor import Processor
from control.FieldMap import FieldMap, Direction
from utils import mm


class DirectionDetector(Processor):
    _X_THRESHOLD: mm  = 1500.0
    _MIN_POINTS:  int = 80
    _MARGIN:      int = 40

    def __init__(self, field_map: FieldMap) -> None:
        super().__init__()
        self._field_map: FieldMap = field_map

    def _process(self, scan: np.ndarray) -> None:
        if self._field_map.direction is not None:
            return

        coords, _ = self._field_map.get_occupied_world()
        if len(coords) == 0:
            return

        x     = coords[:, 0]
        right = int(np.count_nonzero(x >  self._X_THRESHOLD))
        left  = int(np.count_nonzero(x < -self._X_THRESHOLD))

        if max(right, left) < self._MIN_POINTS:
            return
        if abs(right - left) < self._MARGIN:
            return

        direction = Direction.CW if right > left else Direction.CCW
        if self._field_map.direction is None:
            self._field_map.initialize_direction(direction)
