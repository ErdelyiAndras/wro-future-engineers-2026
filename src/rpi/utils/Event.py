from __future__ import annotations
from threading import Lock
from typing import Any, Callable

Handler = Callable[..., Any]

class Event:
    def __init__(self) -> None:
        self._handlers: list[Handler] = []
        self._lock:     Lock          = Lock()

    def __iadd__(self, func: Handler) -> Event:
        with self._lock:
            if func not in self._handlers:
                self._handlers.append(func)
        return self

    def __isub__(self, func: Handler) -> Event:
        with self._lock:
            self._handlers = [h for h in self._handlers if h is not func]
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            handlers = list(self._handlers)
        for func in handlers:
            func(*args, **kwargs)
