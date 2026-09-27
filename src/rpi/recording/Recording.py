from __future__ import annotations

import heapq
import io
from dataclasses import dataclass
from typing import Any, Iterator

import h5py
import numpy as np


@dataclass
class StreamInfo:
    name:       str
    kind:       str
    count:      int
    t_first:    float | None
    t_last:     float | None
    rate_hz:    float | None
    disk_bytes: int
    dropped:    int
    fields:     tuple[str, ...] | None


class Recording:
    """Read side of a Recorder HDF5 file.

    ``iter_stream`` yields ``(t, payload)`` per frame; ``iter_merged``
    interleaves several streams in monotonic-time order — the entry point
    for replaying a run against a planner.
    """

    def __init__(self, path: str) -> None:
        try:
            self._file = h5py.File(path, 'r')
        except OSError:
            # A crashed writer leaves the SWMR "open for write" flag set;
            # such files must be opened as an SWMR reader.
            self._file = h5py.File(path, 'r', libver = 'latest', swmr = True)
        self.path   = path
        self.attrs  = dict(self._file.attrs)
        self._dropped: dict[str, int] = {}
        if 'meta' in self._file:
            meta  = self._file['meta']
            names = [n for n in str(meta.attrs.get('streams', '')).split(',') if n]
            drops = meta['dropped'][()] if 'dropped' in meta else []
            self._dropped = {name: int(d) for name, d in zip(names, drops)}

    def __enter__(self) -> Recording:
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self._file.close()

    # ------------------------------------------------------------------ #
    #  Introspection                                                     #
    # ------------------------------------------------------------------ #

    @property
    def streams(self) -> dict[str, StreamInfo]:
        infos: dict[str, StreamInfo] = {}
        for name, group in self._file.items():
            if name == 'meta' or not isinstance(group, h5py.Group) or 't' not in group:
                continue
            t       = group['t']
            count   = t.shape[0]
            t_first = float(t[0])  if count else None
            t_last  = float(t[-1]) if count else None
            span    = (t_last - t_first) if count > 1 else 0.0
            rate    = (count - 1) / span if count > 1 and span > 0 else None
            disk    = sum(ds.id.get_storage_size()
                          for ds in group.values() if isinstance(ds, h5py.Dataset))
            fields  = group.attrs.get('fields')
            infos[name] = StreamInfo(
                name       = name,
                kind       = str(group.attrs.get('kind', '')),
                count      = count,
                t_first    = t_first,
                t_last     = t_last,
                rate_hz    = rate,
                disk_bytes = disk,
                dropped    = self._dropped.get(name, 0),
                fields     = tuple(str(fields).split(',')) if fields is not None else None,
            )
        return infos

    def t_rel(self, t: float) -> float:
        return t - float(self.attrs['t0_mono'])

    def t_wall(self, t: float) -> float:
        return float(self.attrs['t0_wall']) + self.t_rel(t)

    # ------------------------------------------------------------------ #
    #  Iteration                                                         #
    # ------------------------------------------------------------------ #

    def iter_stream(self, name: str, decode: bool = True) -> Iterator[tuple[float, Any]]:
        group = self._file[name]
        kind  = str(group.attrs.get('kind', ''))
        times = group['t'][()]
        # Lengths are truncated to the shortest dataset so files cut short by
        # a crash (last unflushed rows missing on some datasets) still read.
        if kind == 'marker':
            for t in times:
                yield float(t), None
        elif kind == 'array':
            data = group['data'][()]
            for t, row in zip(times, data):
                yield float(t), row
        elif kind == 'scan':
            offsets = group['offsets'][()]
            counts  = group['counts'][()]
            poses   = group['pose'][()]
            points  = group['points']
            for t, off, cnt, pose in zip(times, offsets, counts, poses):
                off, cnt = int(off), int(cnt)
                yield float(t), (pose, points[off:off + cnt])
        elif kind == 'jpeg':
            blobs = group['jpeg']
            n     = min(len(times), blobs.shape[0])
            if decode:
                import cv2
            for i in range(n):
                raw = blobs[i]
                if decode:
                    yield float(times[i]), cv2.imdecode(raw, cv2.IMREAD_COLOR)
                else:
                    yield float(times[i]), raw.tobytes()
        elif kind == 'npz':
            blobs = group['npz']
            n     = min(len(times), blobs.shape[0])
            for i in range(n):
                if decode:
                    with np.load(io.BytesIO(blobs[i].tobytes())) as npz:
                        payload = {key: npz[key] for key in npz.files}
                else:
                    payload = blobs[i].tobytes()
                yield float(times[i]), payload
        else:
            raise ValueError(f"unknown stream kind {kind!r} for {name!r}")

    def iter_merged(self, names: list[str] | None = None,
                    decode: bool = True) -> Iterator[tuple[float, str, Any]]:
        if names is None:
            names = list(self.streams.keys())
        iterators = (
            ((t, name, payload) for t, payload in self.iter_stream(name, decode = decode))
            for name in names)
        yield from heapq.merge(*iterators, key = lambda item: item[0])
