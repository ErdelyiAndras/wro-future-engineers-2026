# Previous approaches

The software that runs today — the deterministic, sensor-only
[reactive segment planner](../../src/rpi/processors/README.md) — is our **third** navigation
stack. This chapter records what came before and why we moved on, because the reasoning is as
much a part of the engineering as the final code. Each approach still lives in the git
history; the branch to check out is noted so the code and its commits can be read in full.

| # | Approach | Core idea | Why we dropped it | Branches |
|---|---|---|---|---|
| 1 | **Occupancy-grid SLAM + A\*** | build a world map, localise in it, plan a path each cycle | localisation never robust enough in the time we had | `path-planning`, `lidar-pose-estimation`, `test/path-lidar`, `field-visu` |
| 2 | **Follow-the-Gap (FTG)** | geometry-agnostic reactive steering toward the largest gap | smooth but imprecise; couldn't place the robot in a *lane* for obstacle passing | `path-planning` |
| 3 | **Reactive segment planner** *(current)* | assume the 4-straight/4-corner layout; gyro + LiDAR + per-pillar colour | — (this is what we run) | `segment-planners` → `main` |

## 1. Occupancy-grid SLAM + A\* planner

Our first full stack treated the problem as **map-build + localise + plan**, the classic
autonomous-driving pipeline — preserved in the git history (the `path-planning`,
`lidar-pose-estimation`, and `test/path-lidar` branches). The pieces:

- **`FieldMap`** — a 1200 × 1200 @ 5 mm/cell **occupancy grid** (log-odds) plus a semantic
  grid and an obstacle list, all updated from the LiDAR by a Bresenham ray-march.
- **`EgoInformation` EKF** with a LiDAR `[x, y, heading]` correction step, so the robot could
  place itself in the map (scan-matching lived on `lidar-pose-estimation` /
  `lidar-rework` as `LidarPoseEstimator`, merged into `test/path-lidar`).
- **`SemanticClassifier`** — DBSCAN clustering + RANSAC line fitting to label cells as wall /
  obstacle / parking wall; **`DirectionDetector`**; **`CollisionGuard`** as a raw-scan safety
  layer.
- **`PathPlanningProcessor`** — ray-cast goal selection → clearance-weighted, colour-aware
  **A\*** → line-of-sight path thinning → pure-pursuit lookahead, at 5 Hz. Lap 1 planned
  reactively and recorded a breadcrumb route; laps 2–3 followed the memorised route.

**Why we dropped it.** The whole stack stands on **accurate localisation**, and we could not
get the map + EKF pose reliable enough on the real track within the schedule. Drift in the
pose corrupts the occupancy grid, which corrupts the plan — a failure that is hard to debug
and harder to trust in a one-shot competition run. It was a lot of machinery to keep
calibrated.

## 2. Follow-the-Gap (FTG)

To escape the localisation problem we went **map-free**: a geometry-agnostic reactive
controller that steers toward the largest gap in the LiDAR scan, fused with a black-wall
memory. It needs no world position and no assumption about the track — it just drives down
whatever corridor is open, clockwise or anti-clockwise.

**Why we dropped it.** FTG drives a smooth centre-of-corridor line, but the obstacle
challenge needs the robot to sit in a **specific lane** to pass a pillar on the correct side
(red → right, green → left), and to do repeatable 90° corners. "Steer toward the gap" gives
no clean notion of *which lane am I in* or *where does this corner start*. It was the right
tool for the open challenge and the wrong one for obstacles.

## 3. Reactive segment planner (current)

The pivot (Aug 2026) was to **give up generality and exploit the known WRO layout**: a
rectangular loop of four straights and four 90° corners. That single assumption buys
determinism — the planner always knows it is on a straight or in a corner, which lane it
holds, and how many corners it has done. Localisation shrinks to a **principal angle** (grid
orientation from the walls) plus **lane dead-reckoning**, both of which are cheap and
self-correcting. The live code and its state machine are the
[processors chapter](../../src/rpi/processors/README.md).

Even within this approach a lot changed on hardware — the interesting pivots, all
reconstructed from `.h5` [recordings](../../src/rpi/recording/README.md):

- **Corner maneuvers: closed-loop → open-loop → hybrid.** Corners first used a closed-loop
  LiDAR **settle** onto the corridor centre. That was replaced by fully **open-loop scripted**
  recipes (tuned to land centred) to remove a fragile two-wall centering that spun the robot
  at a corner's open mouth — then partly **restored**: investigation of the recordings showed
  the old corner was precise *because of the settle, not the turn* (the arc always
  overshoots), so a closed-loop **rear-wall align** step went back in. The corner is now one
  scripted state ending in that align step.
- **Arc precision = anticipation, not slowing.** Because the planner re-emits the target every
  scan, the firmware's arrival-decel never fires and the arc coasts past the target. Slowing
  the arc hit the motor deadband and stalled; the fix was to terminate each arc a **lead
  angle** early so the coast lands it on target.
- **Lane switches and the 45° "grid flip".** Early lane switches could yaw the robot past 45°
  off the segment, at which point `segment_heading` snapped to the next grid line and the
  switch oscillated forever. Fixed structurally by **latching the grid line per straight**
  (`seg_k`) so it can no longer snap, after which the switch was redesigned to "drive beside
  the pillar."
- **Parking-slot start: detect → declare.** Auto-detecting a parked start kept confusing the
  corner for the exit, so it became an **explicit operator flag** (`--parking-start`) with the
  exit side **voted** over a few scans.
- **Firmware reverse geometry.** A subtle one: the firmware resets its frame on every
  `setTarget`, so a reverse target with a fixed lateral offset is re-chased each scan and the
  robot rotates unbounded. Straight reverses must therefore be **lateral = 0**; intentional
  rotation-while-reversing is the reverse-**arc** turn, gyro-terminated and distance-capped.

These are the kind of hardware-only failure modes that no amount of simulation surfaces — the
reason the [recording framework](../../src/rpi/recording/README.md) exists.

---

*A hardware pivot worth noting alongside these: the Pi↔Arduino link was originally a 10-wire
level shifter and is now a single USB cable — see
[Processing Units](../hardware/processing-units.md).*
