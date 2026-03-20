from __future__ import annotations
from abc import ABC, abstractmethod
from threading import Thread

class Component(ABC):
    def __init__(self) -> None:
        self._is_running: bool          = False
        self._thread:     Thread | None = None

    def __enter__(self) -> Component:
        try:
            self._setup()
        except BaseException:
            self._teardown()
            raise
        return self

    def __exit__(self, *_) -> None:
        self.stop()
        self._teardown()

    def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._thread     = Thread(target = self._run, name = type(self).__name__, daemon = True)
        self._thread.start()

    def stop(self) -> None:
        if not self._is_running:
            return
        self._on_stop()
        self._is_running = False
        if self._thread:
            self._thread.join(timeout = 3)
            self._thread = None

    def _on_stop(self) -> None:
        pass

    @abstractmethod
    def _setup(self) -> None:
        pass

    @abstractmethod
    def _teardown(self) -> None:
        pass

    @abstractmethod
    def _run(self) -> None:
        pass
