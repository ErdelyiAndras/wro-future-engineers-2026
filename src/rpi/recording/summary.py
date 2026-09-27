from __future__ import annotations

from datetime import datetime, timezone

from recording.Recording import Recording


def _size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if value < 1024.0 or unit == 'GiB':
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GiB"


def print_summary(path: str) -> None:
    with Recording(path) as recording:
        print(f"Recording: {path}")
        for key in ('schema_version', 'created_utc', 'script', 'planner',
                    'git_commit', 'notes'):
            value = recording.attrs.get(key, '')
            if value != '':
                print(f"  {key:<15} {value}")
        t0_wall = recording.attrs.get('t0_wall')
        if t0_wall is not None:
            stamp = datetime.fromtimestamp(float(t0_wall), tz = timezone.utc)
            print(f"  {'t0_wall':<15} {stamp.isoformat()}")

        streams = recording.streams
        firsts = [s.t_first for s in streams.values() if s.t_first is not None]
        lasts  = [s.t_last  for s in streams.values() if s.t_last  is not None]
        if firsts and lasts:
            print(f"  {'duration':<15} {max(lasts) - min(firsts):.1f} s")

        header = f"  {'stream':<24} {'kind':<7} {'count':>7} {'rate':>9} {'dropped':>8} {'size':>10}"
        print()
        print(header)
        print('  ' + '-' * (len(header) - 2))
        for name in sorted(streams):
            info = streams[name]
            rate = f"{info.rate_hz:.1f} Hz" if info.rate_hz is not None else '-'
            print(f"  {name:<24} {info.kind:<7} {info.count:>7} {rate:>9} "
                  f"{info.dropped:>8} {_size(info.disk_bytes):>10}")
