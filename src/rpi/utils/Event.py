from threading import Lock

class Event:
    def __init__(self):
        self._handlers = []
        self._lock = Lock()

    def __iadd__(self, func):
        with self._lock:
            if func not in self._handlers:
                self._handlers.append(func)
        return self

    def __isub__(self, func):
        with self._lock:
            self._handlers = [h for h in self._handlers if h is not func]
        return self

    def __call__(self, *args, **kwargs):
        with self._lock:
            handlers = list(self._handlers)
        for func in handlers:
            func(*args, **kwargs)
