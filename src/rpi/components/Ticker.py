from __future__ import annotations
from components.Component import Component
from utils import Event
from threading import Event as StopEvent

class Ticker(Component):
    def __init__(self, interval: float) -> None:
        super().__init__()
        self.on_tick:     Event     = Event()
        self._interval:   float     = interval
        self._stop_event: StopEvent = StopEvent()

    def _setup(self) -> None:
        pass

    def _teardown(self) -> None:
        pass

    def _on_stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        self._stop_event.clear()
        while self._is_running:
            self._stop_event.wait(timeout=self._interval)
            if self._is_running:
                self.on_tick()
