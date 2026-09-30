# Recording & replay

The `Recorder` writes **every frame of every subscribed signal** into one HDF5 file per run,
so a run can be replayed and inspected offline. It is how we debug the planner: almost every
fix in [Previous approaches](../../../docs/software/previous-approaches.md) was root-caused
from a `.h5` file. Recording is on by default; pass `--no-record` to disable it. Files land
in `recordings/run_<timestamp>.h5`.

## How it works — [`Recorder.py`](Recorder.py)

The `Recorder` taps [`Event`s](../utils/Event.py) exactly like any other subscriber. The
producer-side handler only timestamps, cheaply packs the payload, and enqueues it; a **single
writer thread** owns the h5py file (h5py is not thread-safe). If the queue is full the frame
is **dropped and counted** — a sensor thread is never blocked by disk I/O. The file is opened
in **SWMR** mode and flushed every second, so it stays readable after a crash or power cut,
losing at most ~1 s of data.

All `attach_*` calls happen in the main **before** the run starts (SWMR forbids creating
datasets later). Each maps a payload kind to a stream:

| `attach_*` | Stream kind | Used for |
|---|---|---|
| `attach_scan` | `scan` | LiDAR scans + the ego pose sampled at that instant |
| `attach_array` | `array` | fixed-width rows — the `STATE` frame, the ego pose, planner targets |
| `attach_marker` | `marker` | timestamp-only events (e.g. `on_stop`) |
| `attach_npz` | `npz` | variable dicts — the planner's per-scan `on_debug` payload |
| `attach_jpeg` | `jpeg` | encoded camera frames |
| `attach_planner` | — | convenience: wires up a planner's `on_target` / `on_stop` / `on_debug` / `on_track_heading` in one call |

Every file also stores metadata: schema version, wall/monotonic `t0`, the launching script,
the planner class, the **git commit**, and any notes — so a recording is self-describing.

## Reading a run — [`Recording.py`](Recording.py)

`Recording` is the read side. `iter_stream(name)` yields `(t, payload)` per frame;
`iter_merged(...)` interleaves several streams in monotonic-time order — the entry point for
replaying a run against a planner. It transparently handles a file left "open for write" by a
crashed run by reopening it as an SWMR reader, and truncates to the shortest dataset so a
half-flushed file still reads.

## The CLI — `python -m recording`

Run from `src/rpi/` (so the package imports resolve):

```bash
python -m recording summary <run.h5>              # per-stream counts, rates, sizes, drops
python -m recording scans   <run.h5> [--worst N]  # per-scan LiDAR dropout stats
python -m recording dump     <run.h5> <stream>    # decode and print a stream's events
```

`python -m recording <run.h5>` with no subcommand defaults to `summary`. See the
[developer guide](../../../docs/software/setup-guide.md) for the full recording/replay
workflow.
