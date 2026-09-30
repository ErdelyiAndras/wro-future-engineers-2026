# Visualizers — live debug views

Three lightweight web visualizers stream the robot's live state to a browser, so we can
watch what the planner sees during a run (on the robot or over the
[private network](../../../docs/software/setup-guide.md#remote-access-with-tailscale)). Each
runs its own threaded HTTP server and, like everything else, is just an
[`Event`](../utils/Event.py) subscriber — the hot path does not know they exist. They are
started with `--visu` on either main.

The package is kept **import-light on purpose**: import the specific module you need so a
consumer that only wants the LiDAR view does not pull in the OpenCV/camera stack.

| Visualizer | Port | Subscribes to | Shows |
|---|---|---|---|
| [`LidarVisualizer`](LidarVisualizer.py) | **8082** | `lidar.on_scan` | the raw scan as a polar scatter, robot at centre, coloured by return quality |
| [`PathPlanningVisualizer`](PathPlanningVisualizer.py) | **8084** | `planner.on_debug` | the planner's full per-scan state (below) |
| [`ColorVisualizer`](ColorVisualizer.py) | **8085** | `camera.on_frame` | the live frame with the last `color_at` query overlaid |

Open `http://<robot-host>:<port>` in any browser on the same network. With `--visu` (and
without `--no-browser`) the mains open them automatically.

## `PathPlanningVisualizer` (the main one)

Renders the planner's per-scan debug payload as a polar plot, robot at centre, forward up.
It is the single most useful view for this planner — it shows, in one picture:

- the raw scan (grey) and the returns inside the **forward cone** (cyan wedge, re-aimed by
  the current lane) — the wall/obstacle split window;
- the **segment direction** the gyro is holding (yellow ray) and the measured **end-wall
  distance** against the `FRONT_TRIGGER` arc (red);
- the detected **obstacle**, coloured by the camera read (red / green / grey = unresolved);
- the three **lanes** and the dead-reckoned **lateral position** in the corridor;
- during `LANE_SWITCH`, the **target point** sent to the firmware (purple cross);
- the commanded **aim point** (orange), and the rear-wall standoff during a corner;
- a text block: state, grid angles, distances, lane, corner count, speed.

## `ColorVisualizer`

The direct check on the colour half of the obstacle pipeline: the live frame with the most
recent `CameraProcessor.color_at` query drawn on it — where the LiDAR obstacle projected into
the image, the HSV sample patch, the red/green coverage ratios, and the verdict. If the
projected point does not land on the pillar, the camera **extrinsics** need re-measuring
(see [calibration](../../../calibration/README.md)). The overlay fades to grey when no
obstacle has been queried recently.

Because the same `on_debug` stream is archived by the [`Recorder`](../recording/README.md),
anything the planner visualizer shows live can also be reconstructed from a `.h5` after the
run.
