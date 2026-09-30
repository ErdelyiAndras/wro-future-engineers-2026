# Components — hardware drivers

Components are the robot's hardware drivers. Each one wraps a single device, runs its own
background daemon thread, and fires an [`Event`](../utils/Event.py) when fresh data is ready.
They sit at the bottom of the Raspberry Pi software; everything else subscribes to their
events. See the [architecture overview](../../../docs/software/architecture.md) for how they
fit together.

## The `Component` lifecycle

Every component inherits from [`Component`](Component.py) and shares one lifecycle, so they
are all started, stopped, and cleaned up the same way:

| Step | Method | What happens |
|---|---|---|
| open | `__enter__` → `_setup()` | open the serial port / camera; on failure `_teardown()` runs and the error re-raises, so a half-open device is never leaked |
| run | `start()` → `_run()` | launch the background daemon thread |
| stop | `stop()` | signal the thread, join it (3 s timeout) |
| close | `__exit__` → `_teardown()` | release the hardware |

In the mains, components are opened inside an `ExitStack`, so a failure part-way through
setup unwinds everything already opened. A component fires its event **on its own thread**;
subscribed handlers therefore run there too (see the [event model](../../../docs/software/architecture.md)).

## `Lidar` — [`Lidar.py`](Lidar.py)

Drives the RPLIDAR A2M8 over serial (`adafruit_rplidar`). The thread reads individual
measurements and buffers them; on each **new full rotation** it fires `on_scan` with a
NumPy array of shape `(N, 3)` — `(angle_rad, distance_mm, quality)` — for every valid
return in that rotation. Zero-quality and zero-distance returns are dropped before the event
fires. The spinner PWM is set to `660`. A `RPLidarException` (e.g. a USB hiccup) flushes the
serial buffer and retries after a short pause, so transient disconnects recover on their own.

| | |
|---|---|
| Port | `/dev/lidar` (udev symlink) |
| Baud | 115200 |
| Scan mode | normal |
| Event | `on_scan(scan: np.ndarray[N,3])` |

## `Camera` — [`Camera.py`](Camera.py)

Captures frames from the USB camera with OpenCV `VideoCapture`. Each frame is rotated 180°
(the camera is mounted upside down) and handed to `on_frame` as a BGR array. If reads fail
60× in a row the capture loop stops rather than spinning forever. The camera is optional:
`main_obstacle.py --no-camera` skips it entirely.

| | |
|---|---|
| Device | index `0` |
| Resolution | 640 × 480 |
| Target FPS | 30 |
| Event | `on_frame(frame: np.ndarray)` |

## `Arduino` — [`Arduino.py`](Arduino.py)

Manages the framed UART link to the Arduino Nano. The background thread reads bytes, parses
the protocol, and fires `on_state` on each valid `STATE` frame. Two thread-safe send methods
are exposed to the rest of the system:

- `set_target(forward_mm, lateral_mm, speed)` → a `SET_TARGET` frame (body-frame pursuit
  point; negative `speed` = reverse).
- `stop()` → a `STOP` frame (immediate halt).

The wire format — start byte `0xAA`, `MSG_ID`, `LEN`, payload, CRC-8 (poly `0x07`) — is the
mirror of the firmware's [`Protocol`](../../arduino/README.md#protocol--framed-uart); this
file is the Python half. `STATE` carries `(heading_rad, speed_mm_s, steering_rad,
distance_mm)`.

| | |
|---|---|
| Port | `/dev/arduino` (udev symlink) |
| Baud | 115200 |
| Send | `set_target(...)`, `stop()` |
| Event | `on_state(heading, speed, steering, distance)` |

## Adding a component

Subclass `Component`, expose an `Event`, open the device in `_setup`, release it in
`_teardown`, and fire the event from `_run`. Nothing else has to change: subscribers
(planner, recorder, visualizer) attach with `+=`.
