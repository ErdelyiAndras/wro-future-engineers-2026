# Control — the blackboard

`control/` holds the **shared state** of the Raspberry Pi software: the thread-safe
"blackboard" that sensor threads write and the planner reads, plus the small value types the
colour pipeline uses. It holds **no maps or grids** — only the scalars the reactive stack
reasons about. See the [architecture overview](../../../docs/software/architecture.md) for
the blackboard pattern.

Each accessor takes a lock for only as long as it takes to read or write a few numbers, so
threads running at different rates never block each other.

## `EgoInformation` — [`EgoInformation.py`](EgoInformation.py)

The robot's pose estimate `[x, y, heading]`, maintained by a small **Extended Kalman
Filter**. It is updated by [`ArduinoProcessor`](../processors/README.md) on every `STATE`
frame (50 Hz):

- **Predict** — `update_odometry(ds, steering)` propagates the state with Ackermann
  kinematics (straight-line when steering ≈ 0, arc otherwise) and advances the covariance
  `P` through the analytic Jacobian plus a process-noise term scaled by the distance
  travelled.
- **Correct** — `update_imu(heading)` folds the BNO055 heading in as a scalar measurement,
  correcting heading drift.

The planner consumes **`yaw`** (heading) and, for dead-reckoning lateral drift within a
straight, the **`position`** deltas. It does **not** use the absolute `(x, y)` as a world
position — there is no map to place it in. The noise constants (`_PROCESS_NOISE_V`,
`_PROCESS_NOISE_W`, `_MEAS_NOISE_IMU_HDG`) are measured empirically on the assembled robot.

> Historical note: the EKF also has a LiDAR `[x, y, heading]` correction path in earlier
> revisions; the current reactive pipeline does not wire one in. See
> [Previous approaches](../../../docs/software/previous-approaches.md).

## `TrackModel` — [`TrackModel.py`](TrackModel.py)

The track as the robot currently understands it — a lock-protected bag of scalars written by
two processors and read by the planner:

| Field | Meaning | Written by |
|---|---|---|
| `theta` | principal angle: grid orientation, mod 90°, absolute (gyro) frame | `PrincipalAngleDetector` |
| `theta_seeded`, `theta_from_gyro`, `theta_frozen` | seed status / detection freeze | detector + planner |
| `seg_k` | grid multiple of the **current** straight; latched per straight, bumped by `turn_sign` at each corner | planner |
| `turn_sign` | `+1` CW / `−1` CCW, latched at the first corner | planner |
| `corner_count` | corners completed (12 = 3 laps) | planner |
| `current_lane`, `target_offset` | the lane held vs the lane aimed at | planner |
| `target_heading` | corner target heading | planner |

The one derived quantity, `segment_heading(yaw) = wrap(theta + seg_k·90°)`, lives here
because both its inputs do. Latching `seg_k` per straight (rather than re-deriving it from
the live yaw each scan) is what makes the mid-straight 90° "grid flip" structurally
impossible — a story told in [Previous approaches](../../../docs/software/previous-approaches.md).

## Colour types

- [`ColorRange.py`](ColorRange.py) — an HSV colour as the **union of one or more
  `(lower, upper)` bands**, so a hue that straddles the 0/180 seam (red, or the magenta
  parking wall) is two bands instead of being clipped. `mask()` / `ratio()` test a patch
  against it. The three ranges (red, green, parking) are produced by the
  [obstacle-colour calibration](../../../calibration/README.md#obstacle-colours) and pasted
  into `main_obstacle.py`.
- [`ObstacleColor.py`](ObstacleColor.py) — the `RED` / `GREEN` enum the pillar reader
  returns.
