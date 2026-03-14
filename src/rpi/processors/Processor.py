from abc import ABC, abstractmethod

class Processor(ABC):
    def __init__(self) -> None:
        self._paused: bool = False

    def __call__(self, *args, **kwargs) -> None:
        if not self._paused:
            self._process(*args, **kwargs)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @abstractmethod
    def _process(self, *args, **kwargs) -> None:
        pass
