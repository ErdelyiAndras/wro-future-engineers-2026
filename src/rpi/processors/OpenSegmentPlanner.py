from __future__ import annotations

import math
import time

import numpy as np

from processors.Processor import Processor
from control.EgoInformation import EgoInformation
from control.TrackModel import TrackModel
from utils import mm, degree, radian, Event


class OpenSegmentPlanner(Processor):
    """Minimal reactive planner for the WRO FE **open challenge** (no obstacles).

    A deliberately stripped-down sibling of ``ReactiveSegmentPlanner``: the open
    challenge is a bare rectangular loop with no pillars, no parking, and no lane
    choices, so this planner keeps only the two things that matter:

      * **Follow the current segment's principal angle.** A straight is driven with
        no lateral wall term — the firmware's pure-pursuit holds heading toward an aim
        point placed down the segment (``aim = e_head``), so the robot stays parallel
        to the walls using the gyro + the LiDAR-fit principal angle ``theta`` (both
        supplied by ``PrincipalAngleDetector`` via the shared ``TrackModel``).

      * **A forward turn at the wall.** When the end wall is within the corner
        trigger, play the same *outer-lane* corner as the reactive planner — a
        gyro-terminated forward quarter-turn to ``seg_heading + turn_sign*90`` followed
        by a closed-loop rear-wall standoff that cleans up the arc overshoot. No
        reverse-arc, no lane classification: the robot is always effectively "outer".

    The turn direction (``turn_sign``) is latched once, at the first corner, from the
    LiDAR opening — deferred until the opening is genuinely in view so the straight's
    side walls can't force a wrong early pick (identical logic to the reactive planner).

    This is a **standalone duplicate**: the shared geometry / maneuver primitives
    (``_wall_distance``, ``_emit`` / ``_make_reachable``, the forward-turn +
    rear-wall step logic, ``_detect_turn_sign``) are copied here rather than inherited,
    so the planner carries none of the reactive planner's parking / obstacle / colour
    machinery. The corner constants are copied verbatim, so the turn matches the tuning
    already dialed in for the reactive planner's outer maneuver. The common parts can be
    factored into a shared base later.

    Output contract is identical to every other planner (``on_target`` / ``on_stop`` /
    ``on_debug``), and the debug payload matches ``SegmentVisualizer`` / the Recorder.
    """

    # --- Geometry / scan (identical body-point transform to the reactive planner) ---
    _OFFSET:            degree = -90.0     # sensor-frame -> body: a = wrap(scan_angle + offset)
    _R_MAX:             mm     = 3000.0    # range cap / drop beyond

    # --- Forward cone (front-wall distance) ---
    _CONE_HALF:         radian = math.radians(35.0)   # half-angle of the forward cone

    # --- Aim / pursuit ---
    _LOOKAHEAD:         mm     = 400.0     # forward distance of the aim point on a straight
    _MAX_AIM:           radian = math.radians(40.0)   # cap on the commanded aim bearing

    # --- Corner: forward turn (copied verbatim from the reactive planner's OUTER maneuver) ---
    _TURN_ANGLE:        radian = math.radians(38.0)   # steady aim bearing during the FORWARD turn
    _TURN_LOOK:         mm     = 350.0
    _TURN_LEAD:         radian = math.radians(15.0)   # gyro-terminate this far before target (coast lands it)
    _SPEED_TURN:        float  = 80.0

    # --- Corner trigger (per direction; the robot is not exactly symmetric) ---
    # How close to the end wall the forward turn starts. No rear-wall reverse follows, so this can
    # sit closer than the obstacle planner's outer triggers. _gate() = the larger of the two, used
    # before turn_sign is latched.
    _FRONT_TRIGGER_CW:  mm     = 475.0
    _FRONT_TRIGGER_CCW: mm     = 550.0

    # --- First-corner direction latch (copied from the reactive planner) ---
    # Comparing the +-90deg side walls only tells the turn direction once the CORNER OPENING is
    # in view; far from the corner the reading is just the straight's side walls. Defer the latch
    # until one side is clearly open (>= _TURN_OPENING_MIN AND >= _TURN_OPENING_RATIO x the other),
    # with a forced fallback close in so it always commits before the corner triggers.
    _SIDE_HALF:         radian = math.radians(30.0)   # window around +-90deg used to read a side wall
    _TURN_OPENING_MIN:   mm    = 1200.0
    _TURN_OPENING_RATIO: float = 2.0
    _TURN_SIGN_FORCE_DIST: mm  = 450.0

    # --- Steering feasibility (same clamp as the reactive planner / firmware min-turn circle) ---
    _MIN_TURN_RADIUS:   mm     = 211.0
    _TURN_RADIUS_MARGIN: float = 1.15
    _MIN_FORWARD:       mm     = 50.0

    # --- Speeds ---
    _SPEED_CRUISE:      float  = 80.0

    # --- Laps / finish ---
    _TOTAL_CORNERS:     int    = 12        # 3 laps x 4 corners
    _FINISH_STOP_DIST:  mm     = 1500.0    # after the last corner, stop this far from the end wall

    # --- States (numeric values match the reactive planner so SegmentVisualizer names them) ---
    _S_STRAIGHT = 0
    _S_CORNER   = 2
    _S_DONE     = 3
    _S_INIT     = 4

    _M_OUTER    = 0        # only maneuver kind here (forward turn), for the debug/visualiser
    # Single-step corner (match the reactive planner's numbering for the visualiser's corner_step).
    _STEP_TURN      = 1    # forward quarter-turn to target_heading (gyro-terminated)

    def __init__(
        self,
        ego_information: EgoInformation,
        track:           TrackModel,
        lidar_mount_offset: tuple[float, float] = (0.0, 0.0),
        state_hold_s:    float = 0.0,
    ) -> None:
        super().__init__()
        self._ego          = ego_information
        self._track        = track
        self._mount_offset = lidar_mount_offset
        self._offset_rad   = math.radians(self._OFFSET)

        # Debug aid: hold stopped for this many seconds at each state transition. 0 = off.
        self._state_hold_s: float = state_hold_s
        self._hold_until:   float = 0.0

        self.on_target:        Event = Event()
        self.on_stop:          Event = Event()
        self.on_debug:         Event = Event()
        self.on_track_heading: Event = Event()

        # Odometry-derived state (forward distance only; no lateral dead-reckoning is needed
        # without lanes).
        self._last_pos:    tuple[float, float] | None = None
        self._dist_signed: mm = 0.0     # running signed distance along the nose
        self._corner_mark: mm = 0.0     # signed-distance marker for the corner step

        # State machine.
        self._state:         int = self._S_INIT
        self._maneuver_step: int = 0
        self._active_front_trigger: mm = self._gate()   # trigger in force (for debug)
        self._stopped: bool = False

        # Per-scan debug scratch.
        self._dbg: dict = {}

    # ------------------------------------------------------------------ #
    #  Top-level loop                                                     #
    # ------------------------------------------------------------------ #

    def _process(self, scan: np.ndarray) -> None:
        if self._stopped or len(scan) == 0:
            return

        yaw = self._ego.yaw
        if yaw is None:
            return

        self._accumulate_odometry(yaw)

        a, r, x, y = self._body_points(scan)
        if len(a) == 0:
            return

        # PrincipalAngleDetector runs first on the same LiDAR thread and has already fit theta
        # into the shared TrackModel. Until it has a usable value keep publishing debug so the
        # planner never looks dead, but don't act.
        if not self._track.theta_seeded:
            self._dbg = self._blank_dbg()
            self._publish_debug(a, r, yaw, yaw, 0.0)
            self.on_track_heading(float(self._track.theta), float(self._track.target_heading))
            return

        # Latch the current straight's grid line the first time theta is seeded.
        if self._track.seg_k is None:
            self._track.seg_k = int(round((yaw - self._track.theta) / (math.pi / 2.0)))

        seg_heading = self._track.segment_heading(yaw)
        e_head      = self._wrap(seg_heading - yaw)

        self._dbg = self._blank_dbg()

        # Debug state-hold: sit stopped for a moment after each transition.
        if time.monotonic() < self._hold_until:
            self.on_target(0.0, 0.0, 0.0)
            self._dbg["aim"]   = 0.0
            self._dbg["speed"] = 0.0
            self._publish_debug(a, r, yaw, seg_heading, e_head)
            self.on_track_heading(float(self._track.theta), float(self._track.target_heading))
            return

        if self._state == self._S_INIT:
            self._init_phase()
        elif self._state == self._S_STRAIGHT:
            self._drive_straight(a, r, yaw, seg_heading, e_head)
        elif self._state == self._S_CORNER:
            self._corner_maneuver(a, r, yaw, e_head)

        self._publish_debug(a, r, yaw, seg_heading, e_head)
        self.on_track_heading(float(self._track.theta), float(self._track.target_heading))

    def _set_state(self, new_state: int) -> None:
        if new_state != self._state and self._state_hold_s > 0.0:
            self._hold_until = time.monotonic() + self._state_hold_s
        self._state = new_state

    def _init_phase(self) -> None:
        # First usable scan: hold stopped this cycle, then drive from the next one. turn_sign is
        # latched at the first corner (open challenge never starts parked).
        self.on_target(0.0, 0.0, 0.0)
        self._dbg["aim"]   = 0.0
        self._dbg["speed"] = 0.0
        self._set_state(self._S_STRAIGHT)

    # ------------------------------------------------------------------ #
    #  DRIVE_STRAIGHT                                                     #
    # ------------------------------------------------------------------ #

    def _drive_straight(
        self, a: np.ndarray, r: np.ndarray, yaw: radian, seg_heading: radian, e_head: radian,
    ) -> None:
        d_front = self._front_distance(a, r, e_head)

        # Finish: after the last corner, stop a fixed distance from the end wall (centre of the
        # segment), independent of the start pose.
        if (self._track.corner_count >= self._TOTAL_CORNERS
                and d_front is not None and d_front <= self._FINISH_STOP_DIST):
            self._finish()
            return

        # Corner trigger. Near the corner, latch the turn direction (deferred until the opening is
        # clearly in view), then arm the forward turn once within the per-direction trigger.
        if d_front is not None and d_front < self._gate():
            self._detect_turn_sign(a, r, e_head, d_front)
            if self._track.turn_sign is not None:
                self._active_front_trigger = self._front_trigger()
                if d_front < self._active_front_trigger:
                    self._begin_corner(yaw, seg_heading)
                    return

        # Heading-hold: pursue a point one lookahead down the segment (no lateral term), so the
        # firmware nulls e_head onto the principal angle.
        aim = max(-self._MAX_AIM, min(self._MAX_AIM, e_head))
        self._emit(aim, self._LOOKAHEAD, self._SPEED_CRUISE)
        self._dbg["d_front"] = d_front if d_front is not None else float("nan")
        self._dbg["aim"]     = aim
        self._dbg["speed"]   = self._SPEED_CRUISE

    def _front_distance(self, a: np.ndarray, r: np.ndarray, e_head: radian) -> mm | None:
        # End-wall distance: the far coherent surface in a forward cone aimed down the segment.
        # A high percentile ignores stray near returns. No obstacle split (open challenge).
        self._dbg["cone_center"] = e_head
        rel = self._wrap_array(a - e_head)
        in_cone = np.abs(rel) < self._CONE_HALF
        if not in_cone.any():
            return None
        fwd = r[in_cone] * np.cos(rel[in_cone])
        return float(np.percentile(fwd, 85))

    # ------------------------------------------------------------------ #
    #  Turn-direction latch (deferred until the opening is in view)       #
    # ------------------------------------------------------------------ #

    def _detect_turn_sign(self, a: np.ndarray, r: np.ndarray, e_head: radian, d_front: mm | None) -> None:
        if self._track.turn_sign is not None:
            return
        left  = self._wall_distance(a, r, self._wrap(e_head - math.pi / 2.0)) or 0.0
        right = self._wall_distance(a, r, self._wrap(e_head + math.pi / 2.0)) or 0.0
        hi, low = max(left, right), min(left, right)
        confident = hi >= self._TURN_OPENING_MIN and hi >= self._TURN_OPENING_RATIO * max(low, 1.0)
        forced    = d_front is not None and d_front <= self._TURN_SIGN_FORCE_DIST
        if confident or forced:
            # More open on the left -> the track opens left -> turn left (yaw lowers) -> -1.
            self._track.turn_sign = -1 if left > right else +1

    # ------------------------------------------------------------------ #
    #  Corner: forward turn only                                          #
    # ------------------------------------------------------------------ #
    # No rear-wall reverse after the turn (unlike the obstacle planner's outer maneuver): backing
    # up to a fixed standoff from the entry wall only matters when a pillar can sit at the segment
    # entry and needs room to be avoided. The open challenge has no obstacles, and the corner
    # trigger / finish both read a live absolute d_front, so the post-turn position is irrelevant.
    # The forward turn's coast/overshoot is simply absorbed by DRIVE_STRAIGHT (aim = e_head).

    def _begin_corner(self, yaw: radian, seg_heading: radian) -> None:
        self._track.target_heading = self._wrap(seg_heading + self._track.turn_sign * math.pi / 2.0)
        self._maneuver_step = self._STEP_TURN
        self._corner_mark   = self._dist_signed
        self._set_state(self._S_CORNER)

    def _corner_maneuver(self, a: np.ndarray, r: np.ndarray, yaw: radian, e_head: radian) -> None:
        # Gyro-terminate a LEAD before target so the coast lands on target_heading; DRIVE_STRAIGHT
        # then finishes any residual as it nulls e_head.
        if abs(self._wrap(yaw - self._track.target_heading)) < self._TURN_LEAD:
            # Advance the latched grid line onto the new straight so it nulls e_head correctly.
            if self._track.seg_k is not None and self._track.turn_sign is not None:
                self._track.seg_k += self._track.turn_sign
            self._complete_corner()
            return
        aim = self._track.turn_sign * self._TURN_ANGLE
        self._emit(aim, self._TURN_LOOK, self._SPEED_TURN)
        self._dbg["aim"]   = aim
        self._dbg["speed"] = self._SPEED_TURN

    def _complete_corner(self) -> None:
        self._track.corner_count += 1
        self._set_state(self._S_STRAIGHT)

    # ------------------------------------------------------------------ #
    #  Odometry / body-frame points                                      #
    # ------------------------------------------------------------------ #

    def _accumulate_odometry(self, yaw: radian) -> None:
        pos = self._ego.position
        if self._last_pos is not None:
            d_north = pos[0] - self._last_pos[0]
            d_east  = pos[1] - self._last_pos[1]
            self._dist_signed += d_north * math.cos(yaw) + d_east * math.sin(yaw)   # signed forward
        self._last_pos = pos

    def _body_points(self, scan: np.ndarray) -> tuple[np.ndarray, ...]:
        a = self._wrap_array(scan[:, 0] + self._offset_rad)
        r = scan[:, 1]
        keep = (r > 0.0) & (r < self._R_MAX)
        a, r = a[keep], r[keep]
        order = np.argsort(a)
        a, r = a[order], r[order]
        return a, r, r * np.cos(a), r * np.sin(a)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _gate(self) -> mm:
        # Coarse "near a corner" bound used before turn_sign is latched: the larger of the two
        # directions' triggers, so the corner is never missed whichever way it turns.
        return max(self._FRONT_TRIGGER_CW, self._FRONT_TRIGGER_CCW)

    def _front_trigger(self) -> mm:
        return self._FRONT_TRIGGER_CW if self._track.turn_sign > 0 else self._FRONT_TRIGGER_CCW

    def _wall_distance(self, a: np.ndarray, r: np.ndarray, center: radian) -> mm | None:
        rel = self._wrap_array(a - center)
        sel = np.abs(rel) < self._SIDE_HALF
        if not sel.any():
            return None
        return float(np.median(r[sel] * np.cos(rel[sel])))

    def _finish(self) -> None:
        self._stopped = True
        self._state   = self._S_DONE
        self.on_target(0.0, 0.0, 0.0)
        self.on_stop()

    def _emit(self, angle: radian, lookahead: mm, speed: float) -> None:
        forward = lookahead * math.cos(angle) + self._mount_offset[1]
        lateral = lookahead * math.sin(angle) + self._mount_offset[0]
        forward, lateral = self._make_reachable(forward, lateral)
        self.on_target(forward, lateral, speed)

    def _make_reachable(self, forward: float, lateral: float) -> tuple[float, float]:
        if 0.0 <= forward < self._MIN_FORWARD:
            forward = self._MIN_FORWARD
        r = self._MIN_TURN_RADIUS * self._TURN_RADIUS_MARGIN
        if abs(forward) >= r:
            return forward, lateral
        half = math.sqrt(r * r - forward * forward)
        l_lo = r - half
        l_hi = r + half
        if l_lo < abs(lateral) < l_hi:
            lateral = math.copysign(l_lo, lateral)
        return forward, lateral

    @staticmethod
    def _blank_dbg() -> dict:
        return {
            "aim":            float("nan"),
            "speed":          0.0,
            "cone_center":    float("nan"),
            "d_front":        float("nan"),
            "obstacle_x":     float("nan"),
            "obstacle_y":     float("nan"),
            "obstacle_color": 0,
            "obs_recalled":   0,
            "target_x":       float("nan"),
            "target_y":       float("nan"),
            "rear":           float("nan"),
            "park_ratio":     float("nan"),
        }

    def _publish_debug(
        self, a: np.ndarray, r: np.ndarray, yaw: radian, seg_heading: radian, e_head: radian,
    ) -> None:
        # Same payload schema as ReactiveSegmentPlanner so SegmentVisualizer / the Recorder work
        # unchanged; the obstacle / lane / parking fields are left blank (open challenge).
        d = self._dbg
        self.on_debug({
            "scan_a":         a.astype(np.float32),
            "scan_r":         r.astype(np.float32),
            "state":          np.int32(self._state),
            "corner_kind":    np.int32(self._M_OUTER),
            "corner_step":    np.int32(self._maneuver_step),
            "theta":          np.float32(self._track.theta),
            "yaw":            np.float32(yaw),
            "seg_heading":    np.float32(seg_heading),
            "target_heading": np.float32(self._track.target_heading),
            "e_head":         np.float32(e_head),
            "turn_sign":      np.int32(self._track.turn_sign if self._track.turn_sign is not None else 0),
            "corner_count":   np.int32(self._track.corner_count),
            "theta_seeded":   np.int32(1 if self._track.theta_seeded else 0),
            "theta_gyro":     np.int32(1 if self._track.theta_from_gyro else 0),
            "cone_center":    np.float32(d["cone_center"]),
            "cone_half":      np.float32(self._CONE_HALF),
            "d_front":        np.float32(d["d_front"]),
            "front_trigger":  np.float32(self._active_front_trigger),
            "obstacle_x":     np.float32(d["obstacle_x"]),
            "obstacle_y":     np.float32(d["obstacle_y"]),
            "obstacle_color": np.int32(d["obstacle_color"]),
            "obs_recalled":   np.int32(d["obs_recalled"]),
            "target_x":       np.float32(d["target_x"]),
            "target_y":       np.float32(d["target_y"]),
            "lat_dr":         np.float32(0.0),
            "target_offset":  np.float32(0.0),
            "lane_offset":    np.float32(0.0),
            "rear":           np.float32(d["rear"]),
            "rear_target":    np.float32(float("nan")),   # no rear-wall standoff step in the open planner
            "park_ratio":     np.float32(d.get("park_ratio", float("nan"))),
            "aim":            np.float32(d["aim"]),
            "lookahead":      np.float32(self._LOOKAHEAD),
            "speed":          np.float32(d["speed"]),
        })

    @staticmethod
    def _wrap(angle: radian) -> radian:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _wrap_array(angles: np.ndarray) -> np.ndarray:
        return (angles + math.pi) % (2 * math.pi) - math.pi
