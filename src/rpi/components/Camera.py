from __future__ import annotations
import cv2
from components.Component import Component
from utils import Event

class Camera(Component):
    def __init__(
        self,
        port:   int = 0,
        width:  int = 640,
        height: int = 480,
        fps:    int = 30,
    ) -> None:
        super().__init__()
        self.on_frame: Event = Event()

        self._port:   int = port
        self._width:  int = width
        self._height: int = height
        self._fps:    int = fps
        self._cap:    cv2.VideoCapture | None = None

    def _setup(self) -> None:
        self._cap = cv2.VideoCapture(self._port)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Camera: could not open device {self._port!r}")

    def _teardown(self) -> None:
        cap       = self._cap
        self._cap = None
        if cap:
            try:
                cap.release()
            except Exception:
                pass

    def _run(self) -> None:
        while self._is_running:
            ret, frame = self._cap.read()
            if not ret:
                break
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            self.on_frame(frame)
