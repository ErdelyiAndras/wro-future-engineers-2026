from __future__ import annotations

import io
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

import h5py
import numpy as np

from utils import Event

_SENTINEL = object()


def _append_row(dataset: h5py.Dataset, value: Any) -> None:
    n = dataset.shape[0]
    dataset.resize(n + 1, axis = 0)
    dataset[n] = value


class _Stream:
    """One recorded signal: owns its HDF5 group and append logic.

    ``create`` runs once on the writer side before SWMR is enabled;
    ``append`` runs only on the writer thread.
    """

    kind: str = ''

    def __init__(self, name: str) -> None:
        self.name:    str = name
        self.count:   int = 0
        self.dropped: int = 0   # bumped on producer threads (int += is GIL-safe)
        self._t: h5py.Dataset | None = None

    def create(self, file: h5py.File) -> h5py.Group:
        group = file.create_group(self.name)
        group.attrs['kind'] = self.kind
        self._t = group.create_dataset(
            't', shape = (0,), maxshape = (None,), dtype = np.float64, chunks = (1024,))
        return group

    def append(self, t: float, payload: Any) -> None:
        raise NotImplementedError

    def _append_t(self, t: float) -> None:
        _append_row(self._t, t)
        self.count += 1


class _MarkerStream(_Stream):
    kind = 'marker'

    def append(self, t: float, payload: Any) -> None:
        self._append_t(t)


class _ArrayStream(_Stream):
    kind = 'array'

    def __init__(self, name: str, fields: tuple[str, ...]) -> None:
        super().__init__(name)
        self._fields = tuple(fields)
        self._data: h5py.Dataset | None = None

    def create(self, file: h5py.File) -> h5py.Group:
        group = super().create(file)
        width = len(self._fields)
        group.attrs['fields'] = ','.join(self._fields)
        self._data = group.create_dataset(
            'data', shape = (0, width), maxshape = (None, width),
            dtype = np.float32, chunks = (1024, width))
        return group

    def append(self, t: float, payload: np.ndarray) -> None:
        _append_row(self._data, payload)
        self._append_t(t)


class _ScanStream(_Stream):
    kind = 'scan'

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._points:  h5py.Dataset | None = None
        self._offsets: h5py.Dataset | None = None
        self._counts:  h5py.Dataset | None = None
        self._pose:    h5py.Dataset | None = None

    def create(self, file: h5py.File) -> h5py.Group:
        group = super().create(file)
        group.attrs['columns']     = 'angle_rad,distance_mm,quality'
        group.attrs['pose_fields'] = 'x,y,yaw'
        self._points = group.create_dataset(
            'points', shape = (0, 3), maxshape = (None, 3),
            dtype = np.float32, chunks = (32768, 3))
        self._offsets = group.create_dataset(
            'offsets', shape = (0,), maxshape = (None,), dtype = np.uint64, chunks = (1024,))
        self._counts = group.create_dataset(
            'counts', shape = (0,), maxshape = (None,), dtype = np.uint32, chunks = (1024,))
        self._pose = group.create_dataset(
            'pose', shape = (0, 3), maxshape = (None, 3),
            dtype = np.float32, chunks = (1024, 3))
        return group

    def append(self, t: float, payload: tuple[np.ndarray, tuple]) -> None:
        scan, pose = payload
        start = self._points.shape[0]
        n     = len(scan)
        self._points.resize(start + n, axis = 0)
        if n:
            self._points[start:start + n] = scan
        _append_row(self._offsets, start)
        _append_row(self._counts, n)
        _append_row(self._pose, pose)
        self._append_t(t)


class _VlenStream(_Stream):
    """Variable-length binary blobs (JPEG frames, npz-encoded debug dicts)."""

    def __init__(self, name: str, kind: str, dataset_name: str,
                 attrs: dict[str, Any] | None = None) -> None:
        super().__init__(name)
        self.kind          = kind
        self._dataset_name = dataset_name
        self._attrs        = dict(attrs or {})
        self._blobs: h5py.Dataset | None = None

    def create(self, file: h5py.File) -> h5py.Group:
        group = super().create(file)
        for key, value in self._attrs.items():
            group.attrs[key] = value
        self._blobs = group.create_dataset(
            self._dataset_name, shape = (0,), maxshape = (None,),
            dtype = h5py.vlen_dtype(np.dtype('uint8')), chunks = (16,))
        return group

    def append(self, t: float, payload: np.ndarray) -> None:
        _append_row(self._blobs, payload)
        self._append_t(t)


class Recorder:
    """Records every frame of subscribed Event signals into one HDF5 file.

    Producer-side handlers only timestamp, cheaply pack the payload and
    enqueue; a single writer thread owns the h5py file (h5py is not
    thread-safe). When the queue is full the frame is dropped and counted —
    a producer thread is never blocked. SWMR mode keeps the file readable
    after a crash or power cut, losing at most ``flush_interval_s`` of data.

    All ``attach_*`` calls must happen before ``__enter__`` (datasets are
    created there, and SWMR forbids creating them later).
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str, notes: str = '', enabled: bool = True,
                 queue_size: int = 256, flush_interval_s: float = 1.0) -> None:
        self._path             = path
        self._notes            = notes
        self._enabled          = enabled
        self._flush_interval_s = flush_interval_s
        self._queue: queue.Queue = queue.Queue(maxsize = queue_size)
        self._streams: list[_Stream]                 = []
        self._taps:    list[tuple[Event, Callable]]  = []
        self._planner_name: str = ''
        self._file:    h5py.File | None        = None
        self._dropped: h5py.Dataset | None     = None
        self._thread:  threading.Thread | None = None
        self._started = False

    @staticmethod
    def create_default(notes: str = '', enabled: bool = True) -> Recorder:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path  = os.path.join('recordings', f'run_{stamp}.h5')
        return Recorder(path, notes = notes, enabled = enabled)

    @property
    def path(self) -> str:
        return self._path

    # ------------------------------------------------------------------ #
    #  Attach primitives (one per payload kind)                          #
    # ------------------------------------------------------------------ #

    def attach_scan(self, event: Event, name: str = 'lidar_scan', ego = None) -> None:
        if not self._enabled:
            return
        stream = self._add(_ScanStream(name))

        def handler(scan) -> None:
            t    = time.monotonic()
            data = np.array(scan, dtype = np.float32, copy = True).reshape(-1, 3)
            self._put(stream, t, (data, self._sample_pose(ego)))

        self._tap(event, handler)

    def attach_jpeg(self, event: Event, name: str = 'camera_frame',
                    quality: int = 85, encode: Callable | None = None) -> None:
        if not self._enabled:
            return
        if encode is None:
            import cv2

            def encode(frame):
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                return buf if ok else None

        stream = self._add(_VlenStream(
            name, kind = 'jpeg', dataset_name = 'jpeg',
            attrs = {'width': 640, 'height': 480, 'channels': 'bgr', 'quality': quality}))

        def handler(frame) -> None:
            t   = time.monotonic()
            buf = encode(frame)
            if buf is not None:
                self._put(stream, t, np.asarray(buf, dtype = np.uint8).reshape(-1))

        self._tap(event, handler)

    def attach_array(self, event: Event, name: str, fields: tuple[str, ...]) -> None:
        if not self._enabled:
            return
        stream = self._add(_ArrayStream(name, fields))

        def handler(*args) -> None:
            t = time.monotonic()
            self._put(stream, t, np.asarray(args, dtype = np.float32))

        self._tap(event, handler)

    def attach_marker(self, event: Event, name: str) -> None:
        if not self._enabled:
            return
        stream = self._add(_MarkerStream(name))

        def handler(*args) -> None:
            self._put(stream, time.monotonic(), None)

        self._tap(event, handler)

    def attach_npz(self, event: Event, name: str) -> None:
        if not self._enabled:
            return
        stream = self._add(_VlenStream(name, kind = 'npz', dataset_name = 'npz'))

        def handler(*args) -> None:
            t = time.monotonic()
            if len(args) == 1 and isinstance(args[0], dict):
                payload = args[0]
            else:
                payload = {f'arg{i}': value for i, value in enumerate(args)}
            bio = io.BytesIO()
            np.savez_compressed(bio, **payload)
            self._put(stream, t, np.frombuffer(bio.getvalue(), dtype = np.uint8))

        self._tap(event, handler)

    def attach_pose(self, trigger_event: Event, ego, name: str = 'ego_pose') -> None:
        if not self._enabled:
            return
        stream = self._add(_ArrayStream(name, ('x', 'y', 'yaw')))

        def handler(*args) -> None:
            t = time.monotonic()
            self._put(stream, t, np.asarray(self._sample_pose(ego), dtype = np.float32))

        self._tap(trigger_event, handler)

    def attach_planner(self, planner, name: str = 'planner') -> None:
        if not self._enabled:
            return
        self._planner_name = type(planner).__name__
        if hasattr(planner, 'on_target'):
            self.attach_array(planner.on_target, f'{name}_target',
                              fields = ('fwd_mm', 'lat_mm', 'speed'))
        if hasattr(planner, 'on_stop'):
            self.attach_marker(planner.on_stop, f'{name}_stop')
        if hasattr(planner, 'on_debug'):
            self.attach_npz(planner.on_debug, f'{name}_debug')
        if hasattr(planner, 'on_plan_debug'):
            self.attach_npz(planner.on_plan_debug, f'{name}_debug')
        if hasattr(planner, 'on_route'):
            self.attach_npz(planner.on_route, f'{name}_route')
        if hasattr(planner, 'on_next_obstacle'):
            self.attach_npz(planner.on_next_obstacle, f'{name}_next_obstacle')
        if hasattr(planner, 'on_track_heading'):
            self.attach_array(planner.on_track_heading, 'track_heading',
                              fields = ('track_axis_rad', 'target_heading_rad'))

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                         #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> Recorder:
        if not self._enabled:
            return self
        assert not self._started
        directory = os.path.dirname(self._path)
        if directory:
            os.makedirs(directory, exist_ok = True)

        self._file = h5py.File(self._path, 'w', libver = 'latest')
        self._file.attrs['schema_version'] = self.SCHEMA_VERSION
        self._file.attrs['t0_wall']        = time.time()
        self._file.attrs['t0_mono']        = time.monotonic()
        self._file.attrs['created_utc']    = datetime.now(timezone.utc).isoformat()
        self._file.attrs['script']         = os.path.basename(sys.argv[0])
        self._file.attrs['planner']        = self._planner_name
        self._file.attrs['git_commit']     = self._git_commit()
        self._file.attrs['notes']          = self._notes

        for stream in self._streams:
            stream.create(self._file)

        meta = self._file.create_group('meta')
        meta.attrs['streams'] = ','.join(s.name for s in self._streams)
        self._dropped = meta.create_dataset(
            'dropped', shape = (max(len(self._streams), 1),), dtype = np.uint64)

        self._file.swmr_mode = True

        self._started = True
        self._thread  = threading.Thread(
            target = self._writer_loop, name = 'Recorder', daemon = True)
        self._thread.start()
        print(f"Recorder: writing {self._path}")
        return self

    def __exit__(self, *exc) -> bool:
        if not self._enabled:
            return False
        for event, handler in self._taps:
            event -= handler
        self._taps.clear()
        self._queue.put(_SENTINEL)
        self._thread.join(timeout = 10.0)
        self._flush()
        self._file.close()
        summary = ', '.join(
            f"{s.name}: {s.count}" + (f" (dropped {s.dropped})" if s.dropped else '')
            for s in self._streams)
        print(f"Recorder: closed {self._path} [{summary}]")
        return False

    # ------------------------------------------------------------------ #
    #  Internals                                                         #
    # ------------------------------------------------------------------ #

    def _add(self, stream: _Stream) -> _Stream:
        assert not self._started, "attach_* must be called before __enter__"
        assert all(s.name != stream.name for s in self._streams), \
            f"duplicate stream {stream.name!r}"
        self._streams.append(stream)
        return stream

    def _tap(self, event: Event, handler: Callable) -> None:
        event += handler
        self._taps.append((event, handler))

    def _put(self, stream: _Stream, t: float, payload: Any) -> None:
        try:
            self._queue.put_nowait((stream, t, payload))
        except queue.Full:
            stream.dropped += 1

    @staticmethod
    def _sample_pose(ego) -> tuple[float, float, float]:
        if ego is None:
            return (np.nan, np.nan, np.nan)
        pos, yaw = ego.get_ego_information()
        if pos is None or yaw is None:
            return (np.nan, np.nan, np.nan)
        return (float(pos[0]), float(pos[1]), float(yaw))

    @staticmethod
    def _git_commit() -> str:
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output = True, text = True, timeout = 2.0)
            return result.stdout.strip() if result.returncode == 0 else ''
        except Exception:
            return ''

    def _writer_loop(self) -> None:
        next_flush = time.monotonic() + self._flush_interval_s
        running    = True
        while running:
            try:
                item = self._queue.get(timeout = 0.2)
            except queue.Empty:
                item = None
            if item is _SENTINEL:
                running = False
            elif item is not None:
                self._write(item)
            now = time.monotonic()
            if now >= next_flush:
                self._flush()
                next_flush = now + self._flush_interval_s
        while True:   # drain anything enqueued before the sentinel raced in
            try:
                self._write(self._queue.get_nowait())
            except queue.Empty:
                break

    def _write(self, item) -> None:
        stream, t, payload = item
        try:
            stream.append(t, payload)
        except Exception as exc:
            print(f"Recorder: write error on {stream.name}: {exc!r}", file = sys.stderr)

    def _flush(self) -> None:
        try:
            self._dropped[:len(self._streams)] = [s.dropped for s in self._streams]
            self._file.flush()
        except Exception as exc:
            print(f"Recorder: flush error: {exc!r}", file = sys.stderr)
