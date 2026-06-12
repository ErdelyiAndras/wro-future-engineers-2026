from __future__ import annotations

from processors.Processor import Processor
from control.EgoInformation import EgoInformation
from utils import mm, radian


class ArduinoProcessor(Processor):
    def __init__(self, ego_information: EgoInformation) -> None:
        super().__init__()
        self._ego_information = ego_information

    def _process(self, heading: radian, speed: mm, steering: radian, distance: mm) -> None:
        self._ego_information.update_odometry(distance, steering)
        self._ego_information.update_imu(heading)
