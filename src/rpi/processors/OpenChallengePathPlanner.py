from __future__ import annotations

import math

import numpy as np

from processors.Processor import Processor
from control.FieldMap import FieldMap, Direction
from control.EgoInformation import EgoInformation
from utils import mm, degree, radian, Event

# Axis detection parameters
_AXIS_MIN_SEGMENTS: int = 10     # min wall segments needed to trust the estimate
_AXIS_MAX_SEG_LEN:  mm  = 150.0  # max gap between consecutive same-wall points
_AXIS_MAX_DIST:     mm  = 4000.0 # ignore lidar points beyond this


class OpenChallengePathPlanner(Processor):
    """Open-challenge path planner driven directly from raw LiDAR scans.

    On the first scan, estimates the track's rectilinear axis from consecutive
    wall segments and snaps it to the nearest 90° heading. Subsequent straights
    actively steer toward that axis heading; corners advance the target heading
    by ±90° (sign from track direction) and the turn ends when the IMU yaw
    reaches the new target within _HEADING_TOLERANCE. After _LAPS * 4
    completed turns the robot stops.
    """

    _WALL_DISTANCE:     mm     = 480.0
    _CONE_HALF_ANGLE:   float  = math.radians(10.0)
    _HEADING_TOLERANCE: radian = math.radians(10.0)  # turn done within this of target
    _FORWARD_LOOKAHEAD: mm     = 500.0
    _STARTUP_DISTANCE:    mm     = 300.0  # drive this far before turning is allowed
    _STARTUP_SPEED:       float  = 70.0
    _STRAIGHT_SPEED:      float  = 70.0
    _TURN_SPEED:          float  = 70.0
    _CENTERING_SPEED:     float  = 70.0
    _CENTERING_STOP_GAP:  mm     = 550.0  # fire stop this many mm before gap=0; tune to land on midpoint
    _LAPS:              int    = FieldMap.TOTAL_LAPS

    def __init__(
        self,
        field_map:          FieldMap,
        ego_information:    EgoInformation,
        lidar_offset_angle: degree,
    ) -> None:
        super().__init__()
        self._field_map:       FieldMap       = field_map
        self._ego_information: EgoInformation = ego_information
        self._offset_rad:      float          = math.radians(lidar_offset_angle)
        self.on_target:        Event          = Event()
        self.on_stop:          Event          = Event()
        self.on_track_heading: Event          = Event()

        self._track_axis:     float | None = None
        self._target_heading: float | None = None
        self._start_pos:      tuple | None = None
        self._turning:        bool         = False
        self._centering: bool = False  # True after final turn: drive to wall midpoint
        self._wall_stopped:   bool         = False
        self._turn_count:     int          = 0
        self._stopped:        bool         = False

    def _process(self, scan: np.ndarray) -> None:
        if self._stopped or len(scan) == 0:
            return

        pos, yaw = self._ego_information.get_ego_information()

        # Register the start position on the first call
        self._field_map.update_lap_progress(pos)
        if self._start_pos is None:
            self._start_pos = pos

        # Startup phase: go straight for _STARTUP_DISTANCE before any turns
        if math.hypot(pos[0] - self._start_pos[0], pos[1] - self._start_pos[1]) < self._STARTUP_DISTANCE:
            self.on_target(self._FORWARD_LOOKAHEAD, 0.0, self._STARTUP_SPEED)
            return

        # Latch the track axis from wall segments on the first reliable estimate
        if self._track_axis is None:
            axis = _estimate_track_axis(scan, self._offset_rad, yaw)
            if axis is not None:
                self._track_axis     = axis
                self._target_heading = axis
                self.on_track_heading(self._track_axis, self._target_heading)

        # Forward wall distance from raw scan
        raw_angles  = scan[:, 0]
        distances   = scan[:, 1]
        valid       = distances > 0.0
        body_angles = (raw_angles + self._offset_rad)[valid]
        distances   = distances[valid]
        if len(distances) == 0:
            return

        in_cone    = np.abs(body_angles) <= self._CONE_HALF_ANGLE
        front_dist = float(distances[in_cone].min()) if np.any(in_cone) else math.inf

        # Centering phase: drive until equidistant between front and rear walls.
        # The committed forward distance per command is capped to gap/2 so the
        # Arduino cannot physically overshoot the midpoint in a single step.
        # Heading correction still uses the full _FORWARD_LOOKAHEAD so that
        # the pure-pursuit steering stays smooth even when the gap is small.
        if self._centering:
            in_rear   = np.abs(np.abs(body_angles) - math.pi) <= self._CONE_HALF_ANGLE
            rear_dist = float(distances[in_rear].min()) if np.any(in_rear) else math.inf
            print(f"[OpenChallangePathPlanner] Rear dist: {rear_dist}, front dist: {front_dist}")
            if front_dist == math.inf:
                return  # lidar dropout — hold current Arduino command, re-evaluate next scan
            gap = front_dist - rear_dist if rear_dist != math.inf else math.inf
            if front_dist <= rear_dist or gap < self._CENTERING_STOP_GAP:
                self._stopped = True
                self.on_target(0.0, 0.0, 0.0)
                self.on_stop()
                return
            fwd = min(gap / 2.0, self._FORWARD_LOOKAHEAD) if rear_dist != math.inf else self._FORWARD_LOOKAHEAD
            if self._target_heading is not None:
                error = _wrap(self._target_heading - yaw)
                # Forward component bounded by gap/2; lateral uses full L for smooth steering
                lat = self._FORWARD_LOOKAHEAD * math.sin(error)
                self.on_target(fwd, lat, self._CENTERING_SPEED)
            else:
                self.on_target(fwd, 0.0, self._CENTERING_SPEED)
            return

        direction = self._field_map.direction

        # --- State transitions ---
        if self._turning:
            if self._target_heading is not None:
                if abs(_wrap(yaw - self._target_heading)) < self._HEADING_TOLERANCE:
                    self._turning = False
                    if self._turn_count >= self._LAPS * 4:
                        self._centering = True
                        return
        else:
            if front_dist < self._WALL_DISTANCE:
                if direction is None or self._target_heading is None:
                    # Direction not yet known — stop and wait
                    self.on_target(0.0, 0.0, 0.0)
                    return
                if not self._wall_stopped:
                    # First detection: stop momentarily before committing to the turn
                    self._wall_stopped = True
                    self.on_target(0.0, 0.0, 0.0)
                    return
                # Already stopped at wall — now start the turn
                self._wall_stopped   = False
                sign                 = +1.0 if direction == Direction.CW else -1.0
                self._target_heading = _wrap(self._target_heading + sign * math.pi / 2)
                self._turning        = True
                self._turn_count    += 1
                self.on_track_heading(self._track_axis, self._target_heading)
            else:
                self._wall_stopped = False

        # --- Emit target ---
        # Place a lookahead point at _FORWARD_LOOKAHEAD in the target heading
        # direction (robot frame). As the heading error decreases the lateral
        # component shrinks and the forward component grows, giving a natural arc.
        if self._target_heading is None:
            self.on_target(self._FORWARD_LOOKAHEAD, 0.0, self._STRAIGHT_SPEED)
            return

        L     = self._FORWARD_LOOKAHEAD
        speed = self._TURN_SPEED if self._turning else self._STRAIGHT_SPEED
        error = _wrap(self._target_heading - yaw)
        self.on_target(L * math.cos(error), L * math.sin(error), speed)


def _estimate_track_axis(
    scan:       np.ndarray,
    offset_rad: float,
    yaw:        float,
) -> float | None:
    """Return the world-frame track axis heading closest to the robot's current yaw.

    Converts consecutive lidar scan points to body-frame Cartesian, filters
    out inter-wall jumps, then uses the structure tensor trick (exp(2i*theta)
    accumulation) to find the dominant wall orientation mod π. Returns the
    candidate from {wall_dir, wall_dir ± 90°, wall_dir + 180°} that is nearest
    to yaw (i.e. the direction the robot is already roughly travelling).
    Returns None if there are insufficient wall segments.
    """
    body_angles = scan[:, 0] + offset_rad
    distances   = scan[:, 1]
    valid       = (distances > 0) & (distances < _AXIS_MAX_DIST)
    body_angles = body_angles[valid]
    distances   = distances[valid]

    if len(body_angles) < 20:
        return None

    # Body-frame Cartesian (x = right, y = forward)
    x = distances * np.sin(body_angles)
    y = distances * np.cos(body_angles)

    dx      = np.diff(x)
    dy      = np.diff(y)
    seg_len = np.hypot(dx, dy)
    on_wall = (seg_len > 1.0) & (seg_len < _AXIS_MAX_SEG_LEN)

    if int(on_wall.sum()) < _AXIS_MIN_SEGMENTS:
        return None

    angles = np.arctan2(dx[on_wall], dy[on_wall])

    # Structure tensor: exp(2i*theta) average -> dominant direction mod π
    z = np.sum(np.exp(2j * angles))
    if abs(z) < 1e-6:
        return None

    body_wall_dir  = float(np.angle(z) / 2.0)   # dominant wall dir in body frame
    world_wall_dir = _wrap(body_wall_dir + yaw)  # same in world frame

    # The track axis is world_wall_dir or one of its 90° neighbours.
    # Pick the one that is closest to the robot's current heading (yaw).
    candidates = [
        world_wall_dir,
        _wrap(world_wall_dir + math.pi / 2),
        _wrap(world_wall_dir - math.pi / 2),
        _wrap(world_wall_dir + math.pi),
    ]
    return min(candidates, key=lambda a: abs(_wrap(a - yaw)))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
