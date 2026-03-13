from __future__ import annotations
from abc import ABC, abstractmethod

class Component(ABC):
    def __enter__(self) -> Component:
        try:
            self._setup()
        except BaseException:
            self._teardown()
            raise
        return self

    def __exit__(self, *_) -> None:
        self._teardown()

    @abstractmethod
    def _setup(self) -> None:
        pass

    @abstractmethod
    def _teardown(self) -> None:
        pass
