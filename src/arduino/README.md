# Arduino firmware

The Arduino Nano runs the **real-time motion layer**. It owns the BNO055 IMU and the wheel
encoder, receives body-frame targets from the Raspberry Pi over a framed serial link, and
drives the motor and steering to reach them — all in a tight loop (≥ 50 Hz). The Pi decides
*where* to go; this firmware decides *how* to get there. See the
[architecture overview](../../docs/software/architecture.md) for the split.

Build and flash with the repo `Makefile` (`make flash`, FQBN
`arduino:avr:nano:cpu=atmega328`); see the
[developer guide](../../docs/software/setup-guide.md#flashing-the-arduino).

## Main loop — [`arduino.ino`](arduino.ino)

`setup()` initialises the encoder, IMU, motor, steering, and protocol callbacks. `loop()`
runs every iteration:

```
encoder.update();      // refresh speed
imuCtrl.update();      // refresh heading
protocol.update();     // parse any incoming frames -> onSetTarget / onStop
navigator.update();    // pursue the current target
motor.update();        // apply the motor command (with kickstart)
// every 20 ms: protocol.sendState(heading, speed, steering, delta_distance)
```

So the Pi receives a `STATE` frame at **50 Hz** (`STATE_REPORT_INTERVAL_MS = 20`). The
modules are cooperating singletons (`navigator`, `motor`, `encoder`, …) configured entirely
from [`Config.h`](Config.h).

## `Protocol` — framed UART — [`Protocol.cpp`](Protocol.cpp)

The binary framing layer. Frame format:

```
[0xAA] [MSG_ID] [LEN] [PAYLOAD × LEN] [CRC8]
```

CRC-8 (polynomial `0x07`) is computed over `MSG_ID + LEN + PAYLOAD`; on a bad start byte or
CRC the parser drops one byte and resyncs. Messages:

| ID | Dir | Name | Payload |
|---|---|---|---|
| `0x01` | Pi → Nano | `SET_TARGET` | `forward_mm, lateral_mm, speed` (3× f32) |
| `0x02` | Nano → Pi | `STATE` | `heading_rad, speed, steering_rad, distance_mm` (4× f32) |
| `0x03` | Pi → Nano | `STOP` | empty |

`STATE.distance_mm` is the encoder distance **since the last report** (the Pi's EKF predict
step). The Python half of this protocol is [`components/Arduino.py`](../rpi/components/README.md).

## `Navigator` — pursuit + heading PID — [`Navigator.cpp`](Navigator.cpp)

The motion controller. `setTarget(forward, lateral, speed)`:

- **rejects unreachable targets** — a point inside either minimum-turn-radius circle
  (`r = wheelbase / tan(MAX_STEER)`) can't be reached with Ackermann steering, so the state
  goes `IDLE`. (This is why the planner runs every target through `_make_reachable`.)
- resets its local body frame (`x = y = 0`, heading reference = current IMU heading) and
  starts `PURSUING`. A **negative speed means reverse** (the steering sign is flipped).

`doPursuing()` each tick: integrate encoder distance into the local `(x, y)` using the IMU
heading; compute the heading error to the target (switching to a **heading-hold** mode within
100 mm to avoid oscillation); run a **PID** on that error (filtered derivative, clamped
integral) to a clamped steering angle; and set a speed with an **acceleration ramp** over the
first 200 mm and a **deceleration ramp** over the last 500 mm. Arrival (within 10 mm) brakes
and idles.

Key constants (`Config.h`): `HEADING_KP/KI/KD = −2.5 / −0.7 / −0.155`,
`HEADING_HOLD_RADIUS = 100 mm`, `ARRIVAL = 10 mm`, `MIN_SPEED = 80`, `MAX_STEER = 0.4014 rad`.

> **No serial watchdog.** The firmware does not auto-stop if the Pi goes silent; a `STOP`
> frame (or the start button on the Pi) ends a run. Worth adding before competition.

## `Motor` — [`Motor.cpp`](Motor.cpp)

Drives the DC motor via two PWM channels (forward `D5`, reverse `D6`). Implements a
**kickstart**: starting from rest below the deadband, it briefly applies `MIN_SPEED` (80)
and decays toward the requested speed at `KICKSTART_DECAY = 200` units/s, overcoming static
friction without a jerk. `brake()` drives both channels high (active brake); `stop()` coasts.
`MOTOR_DIR = −1` accounts for the wiring polarity.

## `Steering` — [`Steering.cpp`](Steering.cpp)

Maps a steering angle (rad) to a servo pulse width, with **asymmetric** left/right ranges
because the servo's mechanical centre is off the electrical centre: centre `1050 µs`, full
left `1400 µs`, full right `500 µs`, max `±0.4014 rad`. These values live in
[`Config.h`](Config.h), measured from the servo pulse-width → wheel-angle mapping.

## `Encoder` — [`Encoder.cpp`](Encoder.cpp)

Quadrature decoding on interrupt pins `D2` / `D3`; `TICKS_PER_MM = 14.0`. Speed is sampled
every 50 ms; `distance` reported to the Pi is the delta since the last `STATE`.

## `IMU` — [`IMU.cpp`](IMU.cpp)

The BNO055 over I²C in **`OPERATION_MODE_IMUPLUS`** — gyro + accelerometer fusion, **no
magnetometer** (a magnetometer is unreliable near motors and metal). Heading is therefore
**relative**: it is zeroed at boot, so `yaw = 0` points wherever the robot happened to face
at power-on. Recovering "parallel to the wall" from that arbitrary zero is exactly the job of
the principal angle `theta` on the Pi side (see the
[processors chapter](../rpi/processors/README.md)).
