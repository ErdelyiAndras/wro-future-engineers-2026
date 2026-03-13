from utils import Point
from threading import Lock

class EgoInformation:
    def __init__(self) -> None:
        self._lock:     Lock         = Lock()
        self._position: Point | None = None
        self._yaw:      float | None = None

    def get_ego_information(self) -> tuple[Point | None, float | None]:
        with self._lock:
            return self._position, self._yaw

    def set_ego_information(self, position: Point, yaw: float) -> None:
        with self._lock:
            self._position = position
            self._yaw = yaw

    @property
    def position(self) -> Point | None:
        with self._lock:
            return self._position

    @property
    def yaw(self) -> float | None:
        with self._lock:
            return self._yaw
