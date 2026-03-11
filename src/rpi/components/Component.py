from abc import ABC, abstractmethod

class Component(ABC):
    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, *_):
        self._teardown()

    @abstractmethod
    def _setup(self):
        pass

    @abstractmethod
    def _teardown(self):
        pass
