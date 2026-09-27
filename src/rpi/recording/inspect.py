from __future__ import annotations

import math

import numpy as np

from recording.Recording import Recording


def _longest_empty_run(occupied: np.ndarray) -> int:
    # Longest circular run of False bins.
    if occupied.all():
        return 0
    if not occupied.any():
        return len(occupied)
    empty = np.concatenate((~occupied, ~occupied))  # doubled to cross the seam
    best = run = 0
    for e in empty:
        run = run + 1 if e else 0
        best = max(best, run)
    return min(best, len(occupied))


def print_scan_stats(
    path:      str,
    stream:    str = 'lidar_scan',
    bin_count: int = 180,
    worst:     int = 10,
) -> None:
    """Per-scan dropout analysis: bin each scan the way FollowTheGapPlanner does
    and report how much of the circle produced no return. Large empty sectors on
    a closed field are dropout (matte black walls, grazing angles), not open
    space, so this is the first thing to check on a recorded run."""
    bin_width = 2.0 * math.pi / bin_count

    rows = []
    with Recording(path) as recording:
        t0 = None
        for t, (pose, points) in recording.iter_stream(stream):
            if t0 is None:
                t0 = t
            dists = points[:, 1]
            valid = dists > 0.0
            angles = points[valid, 0]
            idx = ((angles + math.pi) / bin_width).astype(int) % bin_count
            occupied = np.zeros(bin_count, dtype = bool)
            occupied[idx] = True
            rows.append((
                t - t0,
                len(points),
                int(valid.sum()),
                bin_count - int(occupied.sum()),
                _longest_empty_run(occupied),
                float(dists[valid].min()) if valid.any() else float('nan'),
                pose,
            ))

    if not rows:
        print("No scans in stream.")
        return

    empty_bins  = np.array([r[3] for r in rows])
    longest_run = np.array([r[4] for r in rows])
    deg = 360.0 / bin_count

    print(f"{len(rows)} scans, {bin_count} bins of {deg:.1f} deg")
    print(f"  empty bins   mean {empty_bins.mean():6.1f}   max {empty_bins.max():4d}"
          f"   ({empty_bins.mean() * deg:.0f} deg / {empty_bins.max() * deg:.0f} deg)")
    print(f"  longest run  mean {longest_run.mean():6.1f}   max {longest_run.max():4d}"
          f"   ({longest_run.mean() * deg:.0f} deg / {longest_run.max() * deg:.0f} deg)")

    order = np.argsort(longest_run)[::-1][:worst]
    print()
    print(f"  {'t_rel':>7} {'points':>7} {'valid':>7} {'empty':>6} {'run':>8} {'min_mm':>8}  pose")
    for i in sorted(order):
        t_rel, n_pts, n_valid, n_empty, run, min_mm, pose = rows[i]
        pose_txt = ('-' if pose is None or np.isnan(pose).any()
                    else f"({pose[0]:.0f}, {pose[1]:.0f}, {math.degrees(pose[2]):.0f} deg)")
        print(f"  {t_rel:7.2f} {n_pts:>7} {n_valid:>7} {n_empty:>6}"
              f" {run:>4} bins {min_mm:8.0f}  {pose_txt}")


def _preview(payload) -> str:
    if payload is None:
        return '-'
    if isinstance(payload, bytes):
        return f"{len(payload)} bytes"
    if isinstance(payload, dict):
        parts = []
        for key, value in payload.items():
            if isinstance(value, np.ndarray) and value.size > 8:
                parts.append(f"{key}: {value.dtype}{value.shape}")
            else:
                parts.append(f"{key}: {value}")
        return '{' + ', '.join(parts) + '}'
    if isinstance(payload, tuple):                       # scan: (pose, points)
        pose, points = payload
        return f"pose {np.array2string(pose, precision = 1)}, points {points.shape}"
    if isinstance(payload, np.ndarray):
        if payload.size > 12:
            return f"{payload.dtype}{payload.shape}"
        return np.array2string(payload, precision = 3)
    return repr(payload)


def print_dump(
    path:   str,
    stream: str,
    start:  int = 0,
    count:  int = 20,
    decode: bool = True,
) -> None:
    """Print decoded events of one stream, `count` events from index `start`."""
    with Recording(path) as recording:
        t0   = None
        for i, (t, payload) in enumerate(recording.iter_stream(stream, decode = decode)):
            if t0 is None:
                t0 = t
            if i < start:
                continue
            if i >= start + count:
                break
            print(f"  [{i:>6}] t={t - t0:9.3f}  {_preview(payload)}")
