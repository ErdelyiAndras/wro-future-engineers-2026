from __future__ import annotations

import math
import time

import numpy as np

from processors.Processor import Processor
from processors.BodyFrameColorSampler import BodyFrameColorSampler
from control.EgoInformation import EgoInformation
from control.TrackModel import TrackModel
from control import ObstacleColor
from utils import mm, degree, radian, Event


class ReactiveSegmentPlanner(Processor):
    """Deterministic, reactive, sensor-only planner for the WRO FE track.

    Design doc: ``docs/reactive-segment-planner.md``. Unlike the geometry-agnostic
    FTG stack this planner *assumes* the WRO layout — a rectangular loop of four
    straights joined by four 90 deg corners — and trades that assumption for
    determinism. It keeps no occupancy grid, no EKF *position*, and no obstacle map;
    every decision comes from the current LiDAR scan, the gyro heading, the wheel
    odometry, and a per-obstacle camera colour read.

    Two ideas carry the whole thing:

      * **Open-loop straights, closed-loop corners.** A straight is driven with no
        lateral wall term: the firmware's pure-pursuit holds heading toward an aim
        point placed down the segment, and lateral position is dead-reckoned from
        encoder + gyro. Every corner runs a closed-loop *reset* (settle to the new
        corridor centre on the LiDAR walls), so lateral error can accumulate over at
        most one straight and is wiped at each corner — it never compounds over the
        three laps. This is why the receding inner wall near a corner never corrupts
        anything: the straight simply doesn't look at the side walls.

      * **Gyro holds heading; LiDAR calibrates what "parallel" means.** The IMU-reset
        zero is at an arbitrary placement angle, so ``yaw = 0`` is not assumed
        parallel to the track. A principal angle ``theta`` (the grid orientation, mod
        90 deg) is fit from the wall points and held in a circular-mean EMA; the
        current segment heading is the grid direction nearest the live yaw. ``theta``
        is yaw-invariant, so it stays valid through lane switches and corners and is
        gated on fit quality rather than frozen.

    The output contract is identical to every other planner (``on_target`` /
    ``on_stop`` / ``on_debug``), and the firmware (`Navigator`) does its own
    pure-pursuit + heading-hold PID and honours a negative ``speed`` as reverse — so
    the planner's job each scan is just to choose a body-frame aim bearing, a
    lookahead, and a speed.
    """

    # --- Geometry / scan ---
    _OFFSET:            degree = -90.0     # sensor-frame -> body: a = wrap(scan_angle + offset)
    _R_MAX:             mm     = 3000.0    # range cap / drop beyond

    # --- Principal angle theta ---
    # theta (grid orientation, mod 90 deg) is fit by PrincipalAngleDetector and read
    # from the shared TrackModel; its constants live there. The planner keeps only the
    # _OFFSET / _R_MAX body-point transform above, which it also needs for the wall /
    # cone reads.

    # --- Forward cone (corner trigger + obstacle split) ---
    _CONE_HALF:         radian = math.radians(35.0)   # half-angle of the forward cone
    _CONE_LANE_SHIFT:   radian = math.radians(22.0)   # cone re-aim per full lane offset (away from near wall)
    # End-wall distance that arms the corner, PER ARRIVAL LANE and PER TRACK DIRECTION.
    # The turn maneuvers are NOT exactly symmetric (steering/servo trim differs left vs
    # right), so CW and CCW corners get independent triggers — tune each side on its own.
    # Within a direction the outer lane does a pure forward turn so it starts earliest
    # (largest trigger); centre/inner drive closer before their reverse-turn (must be < that
    # direction's outer). Selected by turn_sign: +1 = CW -> _CW, -1 = CCW -> _CCW.
    _FRONT_TRIGGER_OUTER_CW:   mm = 725.0
    _FRONT_TRIGGER_CENTER_CW:  mm = 250.0
    _FRONT_TRIGGER_INNER_CW:   mm = _FRONT_TRIGGER_CENTER_CW
    _FRONT_TRIGGER_OUTER_CCW:  mm = 800.0
    _FRONT_TRIGGER_CENTER_CCW: mm = 375.0
    _FRONT_TRIGGER_INNER_CCW:  mm = _FRONT_TRIGGER_CENTER_CCW
    # First-corner direction latch. Comparing the ±90deg side walls only tells the turn direction
    # once the CORNER OPENING is in view: far from the corner the reading is just the straight's
    # side walls, and a robot hugging one lane reads its near wall as "closed" and picks the wrong
    # way (run_172740 latched CW at 795mm where left=121/right=480, then the opening revealed left
    # at 2500mm = CCW). So defer the latch until one side is genuinely open — >= _TURN_OPENING_MIN
    # AND >= _TURN_OPENING_RATIO x the other — with a forced fallback close in so it always commits
    # before the corner triggers.
    _TURN_OPENING_MIN:   mm    = 1200.0
    _TURN_OPENING_RATIO: float = 2.0
    _TURN_SIGN_FORCE_DIST: mm  = 400.0     # <= this from the end wall: commit even if not yet "clear"
    _OBSTACLE_MARGIN:   mm     = 800.0     # a cluster this much nearer than the end wall is an obstacle
    _OBSTACLE_MAX:      mm     = 1000.0    # ignore obstacles farther than this
    _CLUSTER_RADIUS:    mm     = 200.0     # points within this of the nearest count as one obstacle
    # Free-standing test (START straight only): a real pillar is isolated — the returns on BOTH
    # angular flanks of its cluster jump at least _OBSTACLE_GAP farther (or are absent) within
    # _OBSTACLE_FLANK_WIN of the cluster edge. A parking wall is CONTINUOUS with the perimeter on
    # one flank (returns stay near ~cluster range there), so it fails and is rejected. Scan-domain,
    # so it needs no alignment / lat_dr / colour. Only in the start straight (elsewhere a pillar may
    # legitimately sit near a side wall).
    _OBSTACLE_GAP:       mm     = 300.0    # depth step (farther) that counts as open space on a flank
    _OBSTACLE_FLANK_WIN: radian = math.radians(45.0)   # bearing window just outside the cluster edge
    # Obstacle-colour memory: a pillar's colour is stored the first time it is confidently read
    # (lap 1), keyed by (segment = corner_count % 4, longitudinal distance from that segment's END
    # wall via LiDAR). On later laps a re-detected pillar within this tolerance recalls the stored
    # colour, so the lane is correct even if the camera misreads that pass. (A wrong lap-1 read is a
    # non-issue: passing an obstacle on the wrong side ends the round, so there is no later lap.)
    _OBS_MEM_TOL:       mm     = 150.0     # |end-wall distance| match window for "the same pillar"

    # --- Lanes ---
    _LANE_OFFSET:       mm     = 300.0     # |lateral| of the left / right lanes from corridor centre
    _LANE_TOL:          mm     = 40.0      # within this of target offset = "in lane" (cruise speed)
    # Parking-wall avoidance: on the START straight (corner_count % 4 == 0) the parking lot
    # sits against the outer (perimeter) wall, so the OUTER lane is pulled this much toward
    # centre to clear the magenta parking barriers. Applied only to the outer lane there;
    # centre/inner untouched. MUST be < _LANE_OFFSET (else the lane sign is lost). 0 = off.
    _PARKING_LANE_SHIFT: mm    = 220.0
    # Lane switch: the target point handed to the firmware is the pillar's forward
    # (longitudinal) distance paired with the mandated lane's lateral position, so pure
    # pursuit brings the robot into the lane right beside the pillar, on the correct
    # side. The forward distance is latched when the pillar is detected and coasted on
    # odometry once it leaves the forward cone (you cannot see a pillar you are next to).
    # The bearing to that point is capped well short of +-45deg (where segment_heading
    # would snap to the next grid line and corrupt e_head / lat_dr, making the switch
    # oscillate); a point sharper than the cap is pushed farther ahead instead, which
    # also preserves forward progress.
    _SWITCH_MAX_YAW:    radian = math.radians(35.0)  # cap on the switch bearing off the segment
    _SWITCH_STANDOFF:   mm     = 400.0     # aim the switch target this much NEARER than the pillar,
                                           # so the lateral move sharpens and finishes BEFORE the
                                           # obstacle (0 = aim at the pillar; larger = sharper /
                                           # earlier, up to the _SWITCH_MAX_YAW cap)

    # --- Aim / pursuit ---
    _LOOKAHEAD:         mm     = 400.0     # forward distance of the aim point on a straight
    _MAX_AIM:           radian = math.radians(40.0)   # cap on the commanded aim bearing

    # --- Corner maneuvers (open-loop, scripted; see _MANEUVERS / _begin_corner) ---
    # The corner is one state that plays back a predefined, encoder/gyro-metered recipe
    # chosen by arrival lane. No LiDAR wall settle — the maneuver constants below are the
    # only tuning surface. (This drops the old closed-loop lateral reset; the recipes are
    # tuned to land centred on the new straight instead.)
    _TURN_ANGLE:        radian = math.radians(38.0)   # steady aim bearing during the FORWARD turn (outer)
    _TURN_LOOK:         mm     = 350.0
    # Each arc is gyro-terminated a LEAD angle before target_heading so the coast (the robot
    # keeps rotating after the command is cut — the firmware never decelerates under per-scan
    # re-emit) lands it on target. Tune the lead ~= the observed overshoot; the reverse arc
    # coasts more (tighter/faster arc) so its lead is larger. Whatever residual remains is
    # then cleaned up by _align_to_segment. These are the arc-precision knobs.
    _TURN_LEAD:          radian = math.radians(15.0)  # forward turn (outer)
    _REVERSE_TURN_LEAD:  radian = math.radians(22.0)  # reverse-arc turn (centre / inner)
    # Reverse-arc turn (centre & inner): a lateral-offset reverse target that rotates the
    # robot continuously toward target_heading while it backs up. Larger angle = tighter arc.
    _REVERSE_TURN_ANGLE: radian = math.radians(38.0)
    _REVERSE_TURN_MAX:   mm     = 700.0    # safety cap on reverse-arc travel (a 90deg min-radius
                                           # arc is ~380mm; a wrong-way arc bails out here)
    _SIDE_HALF:         radian = math.radians(30.0)   # window around +-90 deg used to read a side wall
                                                       # (only for latching the turn direction)
    _REVERSE_LOOK:      mm     = 300.0     # reverse target distance (behind, for the straight reverses)
    # CLOSED-LOOP rear-wall positioning + heading ALIGNMENT (ALL THREE maneuvers): after the
    # arc, drive to set the wall BEHIND the robot (LiDAR) to _REAR_WALL_DIST WHILE steering to
    # null e_head onto the new segment. This is the only step that corrects the turn's overshoot
    # (the arc always overshoots; the old SETTLE used to fix it — this restores that role). It
    # moves along the segment: forward if too close to the rear wall (the usual case after a
    # reverse arc), reverse if too far; done when positioned AND aligned.
    _REAR_WALL_DIST:      mm   = 400.0     # the standoff knob
    _REAR_WALL_TOL:       mm   = 40.0      # within this of the target = positioned
    _REAR_WALL_ALIGN_TOL: radian = math.radians(4.0)   # |e_head| below this = aligned to the segment
    _REAR_WALL_MAX_TRAVEL:  mm = 500.0     # backstop: give up (net travel) after this far

    # --- Speeds ---
    _SPEED_CRUISE:      float  = 80.0
    _SPEED_SWITCH:      float  = 80.0
    _SPEED_CORNER:      float  = 80.0
    _SPEED_REVERSE:     float  = 80.0
    _SPEED_TURN:        float  = 80.0      # constant arc speed (both arcs). NOTE: speeds are PWM units
                                           # (0-255) and the firmware MIN_SPEED (deadband floor) is 80,
                                           # so you CANNOT go below 80 — it stalls. 80 is the floor.

    # --- Parking-slot start (obstacle challenge) ---
    # At startup the robot may be parked between the two parking walls, facing down-track
    # (walls fore & aft, perimeter wall on one side, open track on the other). Whether the run
    # starts there is an EXPLICIT flag (`start_in_parking` ctor param) set by the operator — not
    # auto-detected (detection was the recurring source of "did the corner instead of the exit"
    # failures). When set, _init_phase plays the leave maneuver; the exit/open side (turn_sign)
    # is still read from the LiDAR by _ensure_turn_sign.
    # Dedicated 3-step leave maneuver (see _park_leave), ALL tuned on hardware:
    #   1 KTURN       — rotate 90deg toward the open side to face the inner wall, as a forward/
    #                   reverse K-turn (little net forward travel, so it never reaches the fore
    #                   wall). Legs alternate: a forward leg drives up to _PARK_KTURN_FWD_LEG, a reverse
    #                   leg up to _PARK_KTURN_REV_LEG (separate so the reverse can travel less), each
    #                   ending early once it has gone at least _PARK_KTURN_MIN AND a wall is within
    #                   _PARK_KTURN_CLEAR ahead/behind. The MIN floor is essential — the robot starts
    #                   BOXED between the fore & aft walls (both already within CLEAR), so without it
    #                   every leg would flip on scan 0 without ever driving (deadlock, run_155113). Done
    #                   _PARK_PERP_LEAD before 90.
    #   2 APPROACH    — drive forward/back to sit _PARK_INNER_STANDOFF_{CW,CCW} from the inner wall (LiDAR).
    #   3 REVERSE_OUT — reverse-turn back to the segment heading, then hand to DRIVE_STRAIGHT.
    _PARK_KTURN_STEER:   radian = math.radians(38.0)   # leg steer magnitude (aim toward the turn side)
    _PARK_KTURN_FWD_LEG: mm     = 180.0    # nominal FORWARD-leg travel (primary leg-end); <= fore clearance
    _PARK_KTURN_REV_LEG: mm     = 40.0    # nominal REVERSE-leg travel — smaller so the reverse moves less
    _PARK_KTURN_MIN:     mm     = 50.0     # min travel before a wall-proximity early-out (anti-deadlock)
    # Exit-side (turn direction) decision for a parking start. The side is inherently ambiguous
    # from one scan — both sides of a parallel slot are drivable, and a lone spurious return on the
    # open (off-field) side can flip a single-frame latch (run_162325 chose CCW off a stray 1058mm
    # return). When park_exit_side is "auto" we vote the LiDAR comparison over this many stationary
    # scans and commit the majority, instead of latching the first frame.
    _PARK_EXIT_VOTES:    int    = 5
    _PARK_KTURN_CLEAR:   mm     = 150.0    # end a leg early (after MIN) if a wall is this near ahead/behind
    _PARK_PERP_LEAD:     radian = math.radians(15.0)   # stop the K-turn this far before 90deg (coast)
    # Inner-wall standoff (the "front-trigger distance" approached before the reverse-turn-out),
    # split by track direction: the following reverse arc turns left for CCW / right for CW, and the
    # robot is not exactly symmetric, so each side needs its own standoff. Selected by turn_sign.
    _PARK_INNER_STANDOFF_CCW: mm = 200.0   # CCW = left reverse-out (currently tuned value)
    _PARK_INNER_STANDOFF_CW:  mm = 330.0   # CW  = right reverse-out (tune separately)
    _PARK_INNER_TOL:     mm     = 40.0     # within this of the standoff = positioned
    _PARK_APPROACH_MAX:  mm     = 600.0    # approach travel cap (backstop)
    _PARK_OUT_ANGLE:     radian = math.radians(38.0)   # reverse-turn arc tightness (step 3)
    _PARK_OUT_LEAD:      radian = math.radians(22.0)    # stop the reverse-turn this far before the segment heading
    _PARK_OUT_MAX:       mm     = 500.0    # reverse-turn travel cap
    _SPEED_PARK:         float  = 80.0
    # After the exit (robot aligned with the segment) dwell stopped this long, sampling the
    # obstacle ahead, so its colour is read on clean motion-free frames (vote buffer fills,
    # camera settles, theta re-converges) before driving. 0 = no dwell.
    _PARK_SETTLE_S:      float  = 0.5

    # --- Final parking (end of run; park_mode selects the maneuver, closed-loop on the walls) ---
    # At the finish (back at the slot) the robot parks into the bay instead of just stopping.
    # PERPENDICULAR: turn 90deg to face the OUTER/perimeter wall, then drive in between the two
    # parking walls and STOP as soon as both flank the robot (it is between them). It does NOT drive
    # to / push toward the outer wall: the outer wall drops below the LiDAR min range up close, and a
    # forward push metered by the encoder loops forever if the robot stalls against the wall (the
    # run_20260913_153744 crash). A wall-clock timeout is the hard backstop so a stall can never loop.
    # Centres on left-right the whole time. (Parallel variants to come.) Reuses `_kturn_toward`.
    _PARK_IN_STANDOFF:   mm     = 180.0    # stop this far from the outer wall (dead ahead when facing it); >
                                           # LiDAR min range, so it is still rangeable at the stop
    _PARK_IN_LOST_MARGIN: mm    = 150.0    # if the outer wall (front) is LOST after being within standoff+this,
                                           # treat as "at the wall" and stop (handles it vanishing up close)
    _PARK_IN_FLANK_MAX:  mm     = 250.0    # (alternate seat) both parking walls within this = flanked
    _PARK_IN_MAX:        mm     = 700.0    # drive-in distance cap (backstop; also stops a stall via odometry)
    _PARK_IN_TIMEOUT:    float  = 2.5      # HARD wall-clock cap on the drive-in (s) — stall-proof
    # Anti-clip: while driving in, steer AWAY from a side parking wall only when it is genuinely
    # CLOSE (< _PARK_SIDE_MIN), by a proportional amount capped at _PARK_AVOID_MAX. Unlike a
    # "centre-between" term this reacts only to a near wall, so a far/open reading can't saturate it
    # (that was the run_20260913_155013 veer). Prevents clipping when the K-turn leaves it off-centre.
    _PARK_SIDE_MIN:      mm     = 140.0    # a side wall this near => nudge away from it
    _PARK_AVOID_GAIN:    float  = 0.006    # rad per mm inside _PARK_SIDE_MIN
    _PARK_AVOID_MAX:     radian = math.radians(28.0)   # cap on the avoid nudge

    # Parking-leave sub-steps (recorded as corner_step during _S_PARK_LEAVE).
    _PK_KTURN       = 0
    _PK_APPROACH    = 1
    _PK_REVERSE_OUT = 2

    # --- Laps / finish ---
    _TOTAL_CORNERS:     int    = 4        # 3 laps x 4 corners
    _FINISH_TOL:        mm     = 120.0     # PARKING modes: stop when the end-wall distance returns to within this of start
    # NON-parking (park_mode "none"): stop a fixed distance from the end wall ahead, so the robot
    # halts near the CENTRE of the start straight regardless of where the run began. The old
    # start-relative trigger fired too early when the run started near the segment's entry line
    # (d_front_start was ~the whole segment), so the robot never actually entered the end segment.
    _FINISH_STOP_DIST:  mm     = 1500.0

    # --- Steering feasibility (same clamp as FTG / TrackPathPlanner) ---
    _MIN_TURN_RADIUS:   mm     = 211.0
    _TURN_RADIUS_MARGIN: float = 1.15
    _MIN_FORWARD:       mm     = 50.0

    # --- Colour vote buffer ---
    _VOTE_LEN:          int    = 5

    # Top-level state codes (recorded numerically in the debug stream). The corner is a
    # SINGLE state; it plays back one of three predefined maneuvers (below).
    _S_STRAIGHT   = 0
    _S_SWITCH     = 1
    _S_CORNER     = 2    # whole corner maneuver (scripted recipe, chosen by arrival lane)
    _S_DONE       = 3
    _S_INIT       = 4    # startup: parking start (start_in_parking flag) vs normal start
    _S_PARK_LEAVE = 5    # dedicated open-loop maneuver leaving the parking slot
    _S_PARK_SETTLE = 6   # post-exit stationary dwell: read the obstacle colour before driving
    _S_PARK        = 7   # final parking maneuver at the end of the run (park_mode)

    # Which of the three maneuvers is running (recorded as corner_kind).
    _M_OUTER  = 0        # outer lane: forward turn -> rear-wall reverse
    _M_CENTER = 1        # centre lane: reverse-arc turn -> rear-wall reverse
    _M_INNER  = 2        # inner lane: reverse-arc turn -> rear-wall reverse (same as centre)

    # Maneuver step primitives.
    _STEP_TURN         = 1   # forward quarter-turn to target_heading (gyro-terminated)
    _STEP_REVERSE_TURN = 3   # reverse-arc turn: back up while rotating to target_heading (gyro-term)
    _STEP_REAR_WALL    = 2   # closed-loop straight move (reverse OR forward) to the LiDAR rear-wall standoff

    def __init__(
        self,
        ego_information: EgoInformation,
        track:           TrackModel,
        color_sampler:   BodyFrameColorSampler | None = None,
        lidar_mount_offset: tuple[float, float] = (0.0, 0.0),
        state_hold_s:    float = 0.0,
        start_in_parking: bool = False,
        park_mode:       str  = "none",
        park_exit_side:  str  = "auto",
    ) -> None:
        super().__init__()
        self._ego           = ego_information
        self._track         = track
        self._color_sampler = color_sampler
        self._mount_offset  = lidar_mount_offset
        self._offset_rad    = math.radians(self._OFFSET)
        # Whether the run starts parked between the parking walls. Set by the operator (the
        # slot start is a known race condition), replacing the fragile auto-detection: True =>
        # play the parking-leave maneuver first; False => go straight to normal driving.
        self._start_in_parking = start_in_parking
        # Final parking maneuver at the end of the run: "none" (just stop, default),
        # "perpendicular" (implemented), or the parallel variants (to come). Unknown => "none".
        self._park_mode = park_mode if park_mode in (
            "none", "perpendicular", "parallel_mirror", "parallel_reverse") else "none"
        # Which way the parking-leave exits (and thus the whole track direction). "auto" votes the
        # LiDAR opening over the first few scans (see _PARK_EXIT_VOTES); "left" forces CCW and
        # "right" forces CW, bypassing the fragile auto pick when the operator knows the layout.
        self._park_exit_side = park_exit_side if park_exit_side in ("auto", "left", "right") else "auto"

        # Debug aid: hold the robot stopped for this many seconds at every state
        # transition so a run can be examined step by step. 0 = off (normal running).
        self._state_hold_s: float = state_hold_s
        self._hold_until:   float = 0.0

        self.on_target:        Event = Event()
        self.on_stop:          Event = Event()
        self.on_debug:         Event = Event()
        self.on_track_heading: Event = Event()

        # Odometry-derived state.
        self._last_pos:  tuple[float, float] | None = None
        self._lat_dr:    mm  = 0.0     # dead-reckoned lateral offset from the straight's start centre
        self._corner_mark: mm = 0.0    # signed distance marker for corner sub-phase metering
        self._dist_signed: mm = 0.0    # running signed distance along the nose

        # Lane / obstacle. (The lane offsets and turn/grid state now live in the shared
        # TrackModel; only the planner's own metering scratch stays here.)
        self._switch_pillar_dist: mm = 0.0  # absolute nose distance of the pillar we switch beside
        self._votes:  list[ObstacleColor] = []
        # Obstacle-colour memory: [(segment, end-wall distance, colour), ...], filled on first
        # confident read (lap 1), recalled on re-detection on later laps. Persists the whole run.
        self._obstacle_memory: list[tuple[int, mm, ObstacleColor]] = []

        # State machine. (turn_sign / seg_k / target_heading / corner_count are in TrackModel.)
        # Start in _S_INIT: it decides parking-slot start vs normal start on the first
        # usable scan, then hands to _S_PARK_LEAVE or _S_STRAIGHT.
        self._state          = self._S_INIT
        # Scripted corner maneuver currently playing (set in _begin_corner).
        self._maneuver:      list[tuple] = []   # list of (step_kind, dist, steer) tuples
        self._maneuver_step: int = 0            # index into _maneuver
        self._maneuver_kind: int = self._M_CENTER
        self._active_front_trigger: mm = self._outer_gate()  # trigger in force (for debug)
        self._d_front_start: mm | None = None
        self._stopped: bool = False

        # Parking-leave maneuver scratch.
        self._exit_votes:        list[int] = []  # per-scan exit-side votes (auto pick), majority commits
        self._park_step:         int   = self._PK_KTURN  # sub-step of the leave maneuver
        self._kturn_forward:     bool  = True   # current K-turn leg direction (fwd vs reverse)
        self._park_leg_mark:     mm    = 0.0    # signed-distance marker for the current leg/step cap
        self._park_settle_until: float = 0.0    # monotonic deadline for the post-exit dwell (0 = not started)
        self._park_in_step:      int   = 0      # sub-step of the FINAL parking maneuver (_S_PARK)
        self._park_in_deadline:  float = 0.0    # monotonic hard-stop for the park drive-in (stall-proof)
        self._park_front_min:    float = float("inf")  # min outer-wall (front) distance seen during drive-in

        # Per-scan debug scratch (reset each scan, packaged by _publish_debug).
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

        # PrincipalAngleDetector (subscribed to on_scan before this planner, so it runs
        # first on the same LiDAR thread) has already fit theta from this scan into the
        # shared TrackModel. Until it has a usable value (LiDAR fit or the gyro-seed
        # fallback), keep publishing debug so the planner never looks dead, but don't act.
        if not self._track.theta_seeded:
            self._dbg = self._blank_dbg()
            self._publish_debug(a, r, yaw, yaw, 0.0)
            self.on_track_heading(float(self._track.theta), float(self._track.target_heading))
            return

        # Latch the current straight's grid line the first time theta is seeded: pick
        # the grid direction nearest the live yaw, then hold that multiple for the whole
        # straight (bumped only at corners). This is what makes the mid-straight 90deg
        # snap impossible.
        if self._track.seg_k is None:
            self._track.seg_k = int(round((yaw - self._track.theta) / (math.pi / 2.0)))

        seg_heading = self._track.segment_heading(yaw)
        e_head      = self._wrap(seg_heading - yaw)

        # Fresh per-scan debug scratch; handlers fill in what applies to them.
        self._dbg = self._blank_dbg()

        # Debug state-hold: sit stopped for a moment after each transition so the run
        # can be examined. The scan / theta / debug above keep updating, only the
        # motion is paused.
        if time.monotonic() < self._hold_until:
            self.on_target(0.0, 0.0, 0.0)
            self._dbg["aim"]   = 0.0
            self._dbg["speed"] = 0.0
            self._publish_debug(a, r, yaw, seg_heading, e_head)
            self.on_track_heading(float(self._track.theta), float(self._track.target_heading))
            return

        if self._state == self._S_INIT:
            self._init_phase(a, r, yaw, seg_heading, e_head)
        elif self._state == self._S_PARK_LEAVE:
            self._park_leave(a, r, yaw)
        elif self._state == self._S_PARK_SETTLE:
            self._park_settle(a, r, x, y, e_head)
        elif self._state == self._S_PARK:
            self._park_in(a, r, yaw, seg_heading, e_head)
        elif self._state == self._S_STRAIGHT:
            self._drive_straight(a, r, x, y, yaw, seg_heading, e_head)
        elif self._state == self._S_SWITCH:
            self._lane_switch(a, r, x, y, yaw, seg_heading, e_head)
        elif self._state == self._S_CORNER:
            self._corner_maneuver(a, r, yaw, e_head)

        self._publish_debug(a, r, yaw, seg_heading, e_head)
        self.on_track_heading(float(self._track.theta), float(self._track.target_heading))

    def _set_state(self, new_state: int) -> None:
        # All state changes route through here so a debug hold can be armed on every
        # transition (see _state_hold_s). No-op transitions do not re-arm the hold.
        if new_state != self._state and self._state_hold_s > 0.0:
            self._hold_until = time.monotonic() + self._state_hold_s
        self._state = new_state

    def _advance_maneuver(self) -> None:
        # Step to the next primitive of the scripted corner maneuver (or finish the corner
        # after the last one). Resets the per-step distance marker and arms the debug hold
        # like a transition, so a run can still be examined step by step.
        self._maneuver_step += 1
        if self._maneuver_step >= len(self._maneuver):
            self._complete_corner()
            return
        self._corner_mark = self._dist_signed
        if self._state_hold_s > 0.0:
            self._hold_until = time.monotonic() + self._state_hold_s

    # ------------------------------------------------------------------ #
    #  INIT / parking-slot start                                          #
    # ------------------------------------------------------------------ #

    def _init_phase(
        self,
        a: np.ndarray, r: np.ndarray,
        yaw: radian, seg_heading: radian, e_head: radian,
    ) -> None:
        # First scan once theta has seeded: branch on the explicit start_in_parking flag (no
        # auto-detection). Hold stopped this scan; the chosen state drives from the next one.
        self.on_target(0.0, 0.0, 0.0)
        self._dbg["aim"]   = 0.0
        self._dbg["speed"] = 0.0

        if not self._start_in_parking:
            self._set_state(self._S_STRAIGHT)    # normal start; turn_sign latched at corner 1
            return

        # Parking start. Decide the exit side (and thus the whole track direction) before the
        # maneuver begins. Operator override wins outright; otherwise vote the LiDAR opening over
        # a few stationary scans so a single stray return can't flip the pick (run_162325).
        if self._park_exit_side == "left":
            self._track.turn_sign = -1                       # exit left -> CCW
        elif self._park_exit_side == "right":
            self._track.turn_sign = +1                       # exit right -> CW
        else:
            self._exit_votes.append(self._exit_vote(a, r, e_head))
            if len(self._exit_votes) < self._PARK_EXIT_VOTES:
                return                                       # keep gathering votes; stay stopped in INIT
            self._track.turn_sign = -1 if sum(self._exit_votes) < 0 else +1

        # Step 1 target: perpendicular to the segment, facing the inner wall = the corner
        # target heading (seg_heading + turn_sign*90). Step 3 will return to seg_heading.
        self._track.target_heading = self._wrap(seg_heading + self._track.turn_sign * math.pi / 2.0)
        self._park_step     = self._PK_KTURN
        self._kturn_forward = True
        self._park_leg_mark = self._dist_signed
        # Freeze principal-angle detection for the maneuver: the robot rotates hard next to the
        # close parking walls, which would corrupt the tangent fit. theta is already seeded.
        self._track.theta_frozen = True
        self._set_state(self._S_PARK_LEAVE)

    def _park_leave(self, a: np.ndarray, r: np.ndarray, yaw: radian) -> None:
        # Dedicated 3-step exit (see the _PARK_* constants). seg_heading is yaw-independent
        # (theta + seg_k*90, both latched), so it is recomputed here for free.
        seg  = self._track.segment_heading(yaw)
        perp = self._wrap(seg + self._track.turn_sign * math.pi / 2.0)  # facing the inner wall
        self._maneuver_step = self._park_step   # surfaced as corner_step in the debug/visualiser
        if self._park_step == self._PK_KTURN:
            self._park_kturn(a, r, yaw, perp)
        elif self._park_step == self._PK_APPROACH:
            self._park_approach(a, r, yaw, perp)
        else:
            self._park_reverse_out(yaw, seg)

    def _advance_park_step(self) -> None:
        self._park_step += 1
        self._park_leg_mark = self._dist_signed
        if self._state_hold_s > 0.0:
            self._hold_until = time.monotonic() + self._state_hold_s

    def _park_kturn(self, a: np.ndarray, r: np.ndarray, yaw: radian, perp: radian) -> None:
        # Leave-maneuver step 1: rotate to face the inner wall via the shared K-turn shuffle.
        if self._kturn_toward(a, r, yaw, perp):
            self._advance_park_step()

    def _kturn_toward(self, a: np.ndarray, r: np.ndarray, yaw: radian, target: radian) -> bool:
        # Rotate yaw toward `target` by alternating forward/reverse legs so net travel stays small
        # (never reaching the fore/aft wall). `rot` = which way to turn to reach the target; a
        # forward leg steers `rot`, a reverse leg steers `-rot` (same rotation while backing). A leg
        # runs to its cap (_PARK_KTURN_FWD_LEG / _REV_LEG), ending early only after _PARK_KTURN_MIN
        # once a wall is within _PARK_KTURN_CLEAR in the travel direction (the MIN floor prevents the
        # boxed-in flip-without-moving deadlock). Returns True within _PARK_PERP_LEAD of the target.
        if abs(self._wrap(yaw - target)) < self._PARK_PERP_LEAD:
            return True
        rot      = math.copysign(1.0, self._wrap(target - yaw))
        traveled = abs(self._dist_signed - self._park_leg_mark)
        leg_cap  = self._PARK_KTURN_FWD_LEG if self._kturn_forward else self._PARK_KTURN_REV_LEG
        wall     = self._wall_distance(a, r, 0.0 if self._kturn_forward else math.pi)
        leg_done = traveled >= leg_cap or (
            traveled >= self._PARK_KTURN_MIN and wall is not None and wall < self._PARK_KTURN_CLEAR)
        if leg_done:
            self._kturn_forward = not self._kturn_forward
            self._park_leg_mark = self._dist_signed
            return False
        if self._kturn_forward:
            aim = rot * self._PARK_KTURN_STEER
            self._emit(aim, self._TURN_LOOK, self._SPEED_PARK)
            self._dbg["aim"] = aim; self._dbg["speed"] = self._SPEED_PARK
        else:
            aim = -rot * self._PARK_KTURN_STEER          # opposite steer -> same rotation, reversing
            self._emit_reverse(aim, self._REVERSE_LOOK, self._SPEED_PARK)
            self._dbg["aim"] = aim; self._dbg["speed"] = -self._SPEED_PARK
        return False

    def _park_approach(self, a: np.ndarray, r: np.ndarray, yaw: radian, perp: radian) -> None:
        # Now facing the inner wall: sit the standoff from it. Standoff is per track direction (the
        # following reverse-out turns left for CCW / right for CW; the robot isn't symmetric). Front
        # LiDAR reads the inner wall; drive forward if too far, reverse if too close, holding perp.
        standoff = self._PARK_INNER_STANDOFF_CW if self._track.turn_sign > 0 else self._PARK_INNER_STANDOFF_CCW
        front = self._wall_distance(a, r, 0.0)
        self._dbg["rear"] = front if front is not None else float("nan")
        e_perp = self._wrap(perp - yaw)
        capped = abs(self._dist_signed - self._park_leg_mark) >= self._PARK_APPROACH_MAX
        positioned = front is not None and abs(front - standoff) < self._PARK_INNER_TOL
        if capped or positioned or front is None:
            self._advance_park_step()
            return
        if front > standoff:                                     # too far -> forward toward wall
            aim = max(-self._MAX_AIM, min(self._MAX_AIM, e_perp))
            self._emit(aim, self._REVERSE_LOOK, self._SPEED_PARK)
            self._dbg["aim"] = aim; self._dbg["speed"] = self._SPEED_PARK
        else:                                                    # too close -> reverse away
            aim = max(-self._MAX_AIM, min(self._MAX_AIM, -e_perp))
            self._emit_reverse(aim, self._REVERSE_LOOK, self._SPEED_PARK)
            self._dbg["aim"] = aim; self._dbg["speed"] = -self._SPEED_PARK

    def _park_reverse_out(self, yaw: radian, seg: radian) -> None:
        # Reverse-turn from perpendicular back to the segment heading, backing away from the inner
        # wall as it rotates. Steer sign +turn_sign (opposite the step-1 reverse legs) rotates back
        # toward seg. Gyro-terminated a lead before seg so the coast lands it; travel cap backstop.
        ts = self._track.turn_sign
        reached = abs(self._wrap(yaw - seg)) < self._PARK_OUT_LEAD
        capped  = abs(self._dist_signed - self._park_leg_mark) >= self._PARK_OUT_MAX
        if reached or capped:
            self._complete_park_leave()
            return
        angle = ts * self._PARK_OUT_ANGLE
        self._emit_reverse(angle, self._REVERSE_LOOK, self._SPEED_PARK)
        self._dbg["aim"] = angle; self._dbg["speed"] = -self._SPEED_PARK

    def _complete_park_leave(self) -> None:
        # On the track now, aligned with the segment: reset lateral dead-reckoning to treat the
        # current position as the lane origin, centre the lane, resume theta detection. Then dwell
        # (PARK_SETTLE) to read the obstacle colour on clean frames before driving. corner_count
        # stays 0, so the start straight (and its parking-wall lane shift) is in force.
        self._lat_dr = 0.0
        self._track.target_offset = 0.0
        self._track.current_lane  = 0.0
        self._track.theta_frozen  = False   # resume principal-angle detection on the track
        if self._PARK_SETTLE_S > 0.0:
            self._park_settle_until = 0.0   # started lazily on the first PARK_SETTLE scan
            self._set_state(self._S_PARK_SETTLE)
        else:
            self._set_state(self._S_STRAIGHT)

    def _park_settle(
        self,
        a: np.ndarray, r: np.ndarray, x: np.ndarray, y: np.ndarray, e_head: radian,
    ) -> None:
        # Post-exit dwell: sit still and read the obstacle ahead so its colour resolves on clean,
        # motion-free frames (vote buffer fills, camera settles, theta re-converges) before driving.
        # Timer starts on the first scan here (after any --state-pause hold), not in _complete.
        if self._park_settle_until == 0.0:
            self._park_settle_until = time.monotonic() + self._PARK_SETTLE_S
        d_front, obstacle = self._forward_cone(a, r, x, y, e_head)
        self._apply_obstacle(obstacle, d_front, e_head)
        self.on_target(0.0, 0.0, 0.0)
        self._dbg["aim"]   = 0.0
        self._dbg["speed"] = 0.0
        if time.monotonic() >= self._park_settle_until:
            self._set_state(self._S_STRAIGHT)

    # ------------------------------------------------------------------ #
    #  Odometry (encoder + gyro dead reckoning)                           #
    # ------------------------------------------------------------------ #

    def _accumulate_odometry(self, yaw: radian) -> None:
        # Position here is pure odometry (no LiDAR pose fusion is wired in this
        # pipeline), so its increments are the encoder path. Project onto the nose to
        # get signed forward distance, and onto the segment normal to dead-reckon the
        # lateral offset within the current straight.
        pos = self._ego.position
        if self._last_pos is not None:
            d_north = pos[0] - self._last_pos[0]
            d_east  = pos[1] - self._last_pos[1]
            ds = d_north * math.cos(yaw) + d_east * math.sin(yaw)   # signed forward
            self._dist_signed += ds
            seg = self._track.segment_heading(yaw)
            self._lat_dr += ds * math.sin(self._wrap(yaw - seg))
        self._last_pos = pos

    # ------------------------------------------------------------------ #
    #  Body-frame scan points                                             #
    # ------------------------------------------------------------------ #

    def _body_points(self, scan: np.ndarray) -> tuple[np.ndarray, ...]:
        # Sensor-frame scan -> body-frame (x forward, y lateral-right), range-capped and
        # sorted by bearing. Feeds the wall-distance and forward-cone reads. (theta is fit
        # from the same transform by PrincipalAngleDetector.)
        a = self._wrap_array(scan[:, 0] + self._offset_rad)
        r = scan[:, 1]
        keep = (r > 0.0) & (r < self._R_MAX)
        a, r = a[keep], r[keep]
        order = np.argsort(a)
        a, r = a[order], r[order]
        return a, r, r * np.cos(a), r * np.sin(a)

    # ------------------------------------------------------------------ #
    #  DRIVE_STRAIGHT                                                     #
    # ------------------------------------------------------------------ #

    def _maybe_finish(self, d_front: mm | None) -> bool:
        # Finish trigger, shared by DRIVE_STRAIGHT and LANE_SWITCH so the run can end on the
        # start straight even mid lane-change (a pillar in the end segment otherwise leaves the
        # robot in LANE_SWITCH past the stop point). Only after the last corner. Returns True (and
        # begins parking) when the stop point is reached.
        if self._track.corner_count < self._TOTAL_CORNERS or d_front is None:
            return False
        if self._park_mode == "none":
            # Just stop, centred in the segment: a fixed standoff from the end wall,
            # independent of the start pose (see _FINISH_STOP_DIST).
            reached = d_front <= self._FINISH_STOP_DIST
        else:
            # Parking modes must reach the actual slot (where the run began), so keep the
            # start-relative trigger.
            reached = (self._d_front_start is not None
                       and d_front <= self._d_front_start + self._FINISH_TOL)
        if reached:
            self._begin_park()
        return reached

    def _drive_straight(
        self,
        a: np.ndarray, r: np.ndarray, x: np.ndarray, y: np.ndarray,
        yaw: radian, seg_heading: radian, e_head: radian,
    ) -> None:
        d_front, obstacle = self._forward_cone(a, r, x, y, e_head)

        if self._d_front_start is None and d_front is not None:
            self._d_front_start = d_front

        # Finish: after the last corner we are back on the start straight.
        if self._maybe_finish(d_front):
            return

        # Corner trigger — per arrival lane. Once within the largest (outer) trigger we
        # are near a corner: latch the turn direction so the lane can be classified as
        # outer / centre / inner, then arm the corner at that lane's own trigger (outer
        # starts earliest, centre/inner drive closer first).
        if d_front is not None and d_front < self._outer_gate():
            self._detect_turn_sign(a, r, e_head, d_front)
            # Only arm the corner once the direction is known; while it is still deferred (the
            # opening not yet clearly in view) keep driving straight down the centre.
            if self._track.turn_sign is not None:
                self._active_front_trigger = self._front_trigger()
                if d_front < self._active_front_trigger:
                    self._begin_corner(a, r, yaw, seg_heading, e_head)
                    return

        # Obstacle -> mandated lane (sticky).
        self._apply_obstacle(obstacle, d_front, e_head)

        # A mandated lane different from the one we are settled in hands off to the
        # dedicated LANE_SWITCH state (where the corner trigger is disabled).
        if self._track.target_offset != self._track.current_lane:
            self._set_state(self._S_SWITCH)
            return

        # Hold the current lane: pursue a point one lookahead down the segment, with
        # only the small drift correction toward the lane. No wall term. The heading
        # offset from the segment (beta) is capped so a large residual lane error (e.g.
        # left over a lane-to-lane switch) cannot steer the robot's yaw toward the 45deg
        # boundary; the residual is then trimmed over more forward travel instead.
        lateral_error = self._track.target_offset - self._lat_dr
        beta = math.atan2(lateral_error, self._LOOKAHEAD)
        beta = max(-self._SWITCH_MAX_YAW, min(self._SWITCH_MAX_YAW, beta))
        aim = self._wrap(beta + e_head)
        aim = max(-self._MAX_AIM, min(self._MAX_AIM, aim))

        near_corner = d_front is not None and d_front < 2.0 * self._outer_gate()
        speed = self._SPEED_CORNER if near_corner else self._SPEED_CRUISE

        self._emit(aim, self._LOOKAHEAD, speed)
        self._dbg["d_front"] = d_front if d_front is not None else float("nan")
        self._dbg["aim"]     = aim
        self._dbg["speed"]   = speed

    # ------------------------------------------------------------------ #
    #  LANE_SWITCH — dedicated lane change (corner trigger disabled)      #
    # ------------------------------------------------------------------ #

    def _lane_switch(
        self,
        a: np.ndarray, r: np.ndarray, x: np.ndarray, y: np.ndarray,
        yaw: radian, seg_heading: radian, e_head: radian,
    ) -> None:
        # Drive to a point beside the pillar in the mandated lane: forward = the
        # pillar's longitudinal distance, lateral = the lane offset. The corner (front)
        # trigger is intentionally NOT checked here, so a switch cannot be preempted by
        # the corner part-way through — the switch always finishes first. The forward
        # cone is still computed to keep the pillar-distance latch fresh (and to feed
        # the visualiser).
        d_front, obstacle = self._forward_cone(a, r, x, y, e_head)
        if obstacle is not None:
            self._dbg["obstacle_x"] = obstacle[0]
            self._dbg["obstacle_y"] = obstacle[1]

        # Finish can land in the end segment mid-switch (a pillar there keeps us in LANE_SWITCH);
        # stop at the same end-wall standoff as on the straight rather than driving the switch out.
        if self._maybe_finish(d_front):
            return

        # How far ahead the pillar's lateral line still is: refreshed from the live
        # detection while it is visible, otherwise coasted on odometry.
        if obstacle is not None and obstacle[0] > self._MIN_FORWARD:
            self._switch_pillar_dist = self._dist_signed + obstacle[0]
        # Aim _SWITCH_STANDOFF nearer than the pillar so the lateral move completes before
        # the obstacle (a sharper diagonal). The bearing cap below still bounds how sharp.
        forward_seg = (self._switch_pillar_dist - self._SWITCH_STANDOFF) - self._dist_signed

        lateral_error = self._track.target_offset - self._lat_dr

        # Stay in the switch until the lane is actually REACHED. (Previously it also
        # completed on drawing level with the pillar, `forward_seg < _SWITCH_MIN_FORWARD`,
        # dumping a big lateral residual onto the straight — very visible on the second,
        # lane-to-lane obstacle. That early-out only existed as the 45deg grid-flip guard,
        # which the latched segment heading now handles structurally, so it is gone.) Once
        # the pillar is passed `forward_seg` shrinks / goes negative, but the bearing
        # push-ahead below keeps the aim a gentle forward diagonal all the way to the lane.
        if abs(lateral_error) < self._LANE_TOL:
            self._track.current_lane = self._track.target_offset
            self._set_state(self._S_STRAIGHT)
            return

        # Target point in the segment frame: (forward_seg, lateral_error). Cap its
        # bearing so the switch never yaws toward the 45deg grid-flip; if the point
        # would be sharper than the cap, push it farther ahead (larger forward) rather
        # than aiming sideways — this keeps forward progress and the same arrival lane.
        bearing = math.atan2(lateral_error, forward_seg)
        if abs(bearing) > self._SWITCH_MAX_YAW:
            forward_seg = abs(lateral_error) / math.tan(self._SWITCH_MAX_YAW)
            bearing = math.copysign(self._SWITCH_MAX_YAW, lateral_error)

        lookahead = math.hypot(forward_seg, lateral_error)
        aim = self._wrap(bearing + e_head)     # segment-frame bearing -> body frame
        aim = max(-self._MAX_AIM, min(self._MAX_AIM, aim))
        self._emit(aim, lookahead, self._SPEED_SWITCH)

        self._dbg["d_front"]  = d_front if d_front is not None else float("nan")
        self._dbg["aim"]      = aim
        self._dbg["speed"]    = self._SPEED_SWITCH
        self._dbg["target_x"] = lookahead * math.cos(aim)
        self._dbg["target_y"] = lookahead * math.sin(aim)

    def _forward_cone(
        self,
        a: np.ndarray, r: np.ndarray, x: np.ndarray, y: np.ndarray,
        e_head: radian,
    ) -> tuple[mm | None, tuple[float, float] | None]:
        # Split the end wall from an obstacle in one scan. The cone is re-aimed away
        # from the near wall by an amount tied to the current lane, so a side wall we
        # are hugging is not read as a pillar.
        seg_dir = e_head
        cone_center = self._wrap(
            seg_dir - self._CONE_LANE_SHIFT * (self._track.target_offset / self._LANE_OFFSET)
        )
        self._dbg["cone_center"] = cone_center
        rel = self._wrap_array(a - cone_center)
        in_cone = np.abs(rel) < self._CONE_HALF
        if not in_cone.any():
            return None, None

        rc  = r[in_cone]
        xc  = x[in_cone]
        yc  = y[in_cone]
        ain = a[in_cone]
        # Forward distance along the segment direction.
        fwd = rc * np.cos(self._wrap_array(ain - seg_dir))

        # End wall = the far coherent surface; a high percentile ignores a nearer
        # pillar that occupies only a few rays.
        d_front = float(np.percentile(fwd, 85))

        # Obstacle = nearest compact cluster well in front of the end wall.
        near = fwd < (d_front - self._OBSTACLE_MARGIN)
        near &= rc < self._OBSTACLE_MAX
        obstacle = None
        if near.any():
            xn, yn = xc[near], yc[near]
            rn = rc[near]
            i0 = int(np.argmin(rn))
            close = np.hypot(xn - xn[i0], yn - yn[i0]) < self._CLUSTER_RADIUS
            obstacle = (float(np.mean(xn[close])), float(np.mean(yn[close])))
            # In the start straight, keep it only if it is free-standing (a pillar), not
            # continuous with the perimeter (a parking wall). Tested against the full scan.
            if self._track.corner_count % 4 == 0 and \
                    not self._is_free_standing(a, r, ain[near][close], rn[close]):
                obstacle = None
        return d_front, obstacle

    def _is_free_standing(
        self,
        a: np.ndarray, r: np.ndarray, cluster_a: np.ndarray, cluster_r: np.ndarray,
    ) -> bool:
        # A free-standing pillar has open space on BOTH angular flanks: the nearest return just
        # outside the cluster (within _OBSTACLE_FLANK_WIN) jumps >= _OBSTACLE_GAP farther than the
        # cluster, or there is none. A parking wall is continuous with the perimeter on one flank
        # (a return there at ~cluster range), so it fails. `a`/`r` are the FULL sorted scan.
        if len(cluster_a) == 0:
            return True
        b_lo   = float(np.min(cluster_a))
        b_hi   = float(np.max(cluster_a))
        obs_r  = float(np.median(cluster_r))
        thresh = obs_r + self._OBSTACLE_GAP

        def flank_open(edge: float, sign: float) -> bool:
            rel = self._wrap_array(a - edge) * sign          # >0 = outside the cluster on this side
            sel = (rel > 1e-6) & (rel < self._OBSTACLE_FLANK_WIN)
            if not np.any(sel):
                return True                                   # nothing adjacent -> open (free space)
            nearest = int(np.argmin(rel[sel]))               # closest return in bearing to the edge
            return float(r[sel][nearest]) > thresh           # far away -> gap; near ~cluster -> wall

        return flank_open(b_lo, -1.0) and flank_open(b_hi, +1.0)

    def _resolve_color(self, obstacle: tuple[float, float]) -> ObstacleColor | None:
        # obstacle is body-frame (x forward, y lateral-right). Project + sample, then
        # majority-vote over a short buffer so a single bad frame cannot flip the lane.
        if self._color_sampler is None:
            return None
        color = self._color_sampler.color_at(obstacle[0], obstacle[1])
        if color is not None:
            self._votes.append(color)
            if len(self._votes) > self._VOTE_LEN:
                self._votes.pop(0)
        if not self._votes:
            return None
        reds   = self._votes.count(ObstacleColor.RED)
        greens = self._votes.count(ObstacleColor.GREEN)
        if reds == greens:
            return None
        return ObstacleColor.RED if reds > greens else ObstacleColor.GREEN

    def _apply_obstacle(
        self,
        obstacle: tuple[float, float] | None,
        d_front:  mm | None,
        e_head:   radian,
    ) -> None:
        # Resolve the obstacle's colour and set the sticky mandated lane (with the parking-wall
        # shift), latching the pillar's nose distance for the switch. No obstacle in view clears the
        # vote buffer. Shared by DRIVE_STRAIGHT and the post-exit PARK_SETTLE dwell.
        #
        # Colour memory: a pillar is keyed by (segment, distance from the segment's END wall). If it
        # was learned on an earlier lap, RECALL that colour (so the lane is right even if the camera
        # misreads this pass); otherwise read the camera and STORE it on the first confident read.
        if obstacle is None:
            self._votes.clear()
            return

        seg   = self._track.corner_count % 4
        pos   = None
        color = None
        if d_front is not None:
            pos   = d_front - (obstacle[0] * math.cos(e_head) + obstacle[1] * math.sin(e_head))
            color = self._recall_obstacle(seg, pos)   # None if never learned
        recalled = color is not None
        if not recalled:
            color = self._resolve_color(obstacle)      # camera majority vote
            if color is not None and pos is not None:
                self._store_obstacle(seg, pos, color)  # learn it (store once)
        self._dbg["obs_recalled"] = 1 if recalled else 0

        if color == ObstacleColor.RED:
            self._track.target_offset = self._apply_parking_shift(+self._LANE_OFFSET)
            self._dbg["obstacle_color"] = 1
        elif color == ObstacleColor.GREEN:
            self._track.target_offset = self._apply_parking_shift(-self._LANE_OFFSET)
            self._dbg["obstacle_color"] = 2
        # Latch the pillar's absolute nose distance so the switch can drive to a point
        # beside it even after it leaves the forward cone.
        if self._track.target_offset != self._track.current_lane:
            self._switch_pillar_dist = self._dist_signed + obstacle[0]
        self._dbg["obstacle_x"] = obstacle[0]
        self._dbg["obstacle_y"] = obstacle[1]

    def _recall_obstacle(self, seg: int, pos: mm) -> ObstacleColor | None:
        # Stored colour of a pillar in this segment within _OBS_MEM_TOL of `pos` (end-wall distance).
        for s, p, c in self._obstacle_memory:
            if s == seg and abs(p - pos) < self._OBS_MEM_TOL:
                return c
        return None

    def _store_obstacle(self, seg: int, pos: mm, color: ObstacleColor) -> None:
        # Learn a pillar's colour once (first confident read); ignore if already known.
        if self._recall_obstacle(seg, pos) is None:
            self._obstacle_memory.append((seg, pos, color))

    # ------------------------------------------------------------------ #
    #  Corner sequence                                                    #
    # ------------------------------------------------------------------ #

    def _detect_turn_sign(self, a: np.ndarray, r: np.ndarray, e_head: radian, d_front: mm | None) -> None:
        # Latch the turn direction (once, at the first corner) — but only when the corner OPENING is
        # actually in view, so the robot's lane offset in the straight can't force a wrong early pick
        # (see _TURN_OPENING_MIN). Defers (leaves turn_sign None) until one side is clearly open, or
        # until _TURN_SIGN_FORCE_DIST forces the raw pick as a safety floor. No return on a side =>
        # 0.0 (unmeasured, treated as closed); the real corner opening reads as a large far return.
        if self._track.turn_sign is not None:
            return
        left  = self._wall_distance(a, r, self._wrap(e_head - math.pi / 2.0)) or 0.0
        right = self._wall_distance(a, r, self._wrap(e_head + math.pi / 2.0)) or 0.0
        hi, low = max(left, right), min(left, right)
        confident = hi >= self._TURN_OPENING_MIN and hi >= self._TURN_OPENING_RATIO * max(low, 1.0)
        forced    = d_front is not None and d_front <= self._TURN_SIGN_FORCE_DIST
        if confident or forced:
            self._track.turn_sign = -1 if left > right else +1

    def _ensure_turn_sign(self, a: np.ndarray, r: np.ndarray, e_head: radian) -> None:
        # Idempotent safety latch for the corner path: by the time _begin_corner runs the direction
        # is already set by _detect_turn_sign; if not (forced with no clear side), fall back to the
        # raw single-frame pick so the maneuver always has a direction.
        if self._track.turn_sign is not None:
            return
        self._track.turn_sign = self._exit_vote(a, r, e_head)

    def _exit_vote(self, a: np.ndarray, r: np.ndarray, e_head: radian) -> int:
        # One scan's vote for the turn direction: -1 (open/track to the left -> CCW) or +1 (right ->
        # CW). A wall we can measure marks the track side; an unmeasured side (None -> 0.0) reads as
        # off-field open and loses. A left turn lowers yaw, a right turn raises it, hence the signs.
        left  = self._wall_distance(a, r, self._wrap(e_head - math.pi / 2.0))
        right = self._wall_distance(a, r, self._wrap(e_head + math.pi / 2.0))
        return -1 if (left or 0.0) > (right or 0.0) else +1

    def _arrival_kind(self) -> int:
        # Which maneuver the current lane calls for. Inner = the turn side (lateral sign
        # equals turn_sign); outer = the opposite side; centre = on the corridor centre.
        if self._track.target_offset == 0.0 or self._track.turn_sign is None:
            return self._M_CENTER
        if math.copysign(1.0, self._track.target_offset) == self._track.turn_sign:
            return self._M_INNER
        return self._M_OUTER

    def _apply_parking_shift(self, offset: mm) -> mm:
        # Parking-wall avoidance (see _PARKING_LANE_SHIFT): on the START straight, pull the
        # OUTER lane toward centre so it clears the parking barriers on the perimeter wall.
        # Outer = the side opposite the turn (sign == -turn_sign). Only bites when a pillar
        # actually mandates the outer lane there; centre/inner and every other straight are
        # untouched. turn_sign None (first straight, pre-corner, non-parking start) => no shift.
        ts = self._track.turn_sign
        if offset == 0.0 or ts is None or self._PARKING_LANE_SHIFT <= 0.0:
            return offset
        in_start = self._track.corner_count % 4 == 0
        is_outer = math.copysign(1.0, offset) == -ts
        if in_start and is_outer:
            return math.copysign(abs(offset) - self._PARKING_LANE_SHIFT, offset)
        return offset

    def _front_trigger(self) -> mm:
        # Per arrival lane AND per track direction. Only called after _ensure_turn_sign has
        # latched turn_sign (in _drive_straight), so turn_sign is never None here.
        if self._track.turn_sign > 0:      # CW
            return {
                self._M_OUTER:  self._FRONT_TRIGGER_OUTER_CW,
                self._M_CENTER: self._FRONT_TRIGGER_CENTER_CW,
                self._M_INNER:  self._FRONT_TRIGGER_INNER_CW,
            }[self._arrival_kind()]
        return {                            # CCW
            self._M_OUTER:  self._FRONT_TRIGGER_OUTER_CCW,
            self._M_CENTER: self._FRONT_TRIGGER_CENTER_CCW,
            self._M_INNER:  self._FRONT_TRIGGER_INNER_CCW,
        }[self._arrival_kind()]

    def _outer_gate(self) -> mm:
        # Coarse "near a corner" bound used BEFORE turn_sign is latched (the first corner):
        # the larger of the two directions' outer triggers, so the corner is never missed
        # whichever way it turns. The precise per-direction arming still uses _front_trigger().
        return max(self._FRONT_TRIGGER_OUTER_CW, self._FRONT_TRIGGER_OUTER_CCW)

    def _begin_corner(
        self,
        a: np.ndarray, r: np.ndarray,
        yaw: radian, seg_heading: radian, e_head: radian,
    ) -> None:
        self._ensure_turn_sign(a, r, e_head)
        self._track.target_heading = self._wrap(seg_heading + self._track.turn_sign * math.pi / 2.0)

        # Build the predefined maneuver for this arrival lane. ALL THREE end with the same
        # closed-loop straight rear-wall reverse, so whatever the arrival, the robot finishes
        # facing the new heading at _REAR_WALL_DIST from the wall behind it. Outer has room
        # for a pure FORWARD turn; centre/inner back AROUND the corner with a reverse-arc turn
        # (no forward motion). Centre and inner share the recipe — they differ only in their
        # front trigger.
        self._maneuver_kind = self._arrival_kind()
        if self._maneuver_kind == self._M_OUTER:
            self._maneuver = [
                (self._STEP_TURN,         0.0, 0.0),
                (self._STEP_REAR_WALL,    0.0, 0.0),
            ]
        else:
            self._maneuver = [
                (self._STEP_REVERSE_TURN, 0.0, 0.0),
                (self._STEP_REAR_WALL,    0.0, 0.0),
            ]

        self._maneuver_step = 0
        self._corner_mark   = self._dist_signed
        self._set_state(self._S_CORNER)

    def _corner_maneuver(
        self, a: np.ndarray, r: np.ndarray, yaw: radian, e_head: radian,
    ) -> None:
        # Play back the current step of the scripted maneuver. REVERSE is metered by
        # encoder distance; TURN is gyro-terminated; REVERSE_WALL is closed-loop on the
        # LiDAR rear-wall distance. _advance_maneuver steps to the next primitive and
        # finishes the corner after the last one.
        kind, dist, steer = self._maneuver[self._maneuver_step]

        if kind == self._STEP_TURN:
            # Terminate a LEAD before target so the coast lands on target (see _TURN_LEAD).
            if abs(self._wrap(yaw - self._track.target_heading)) < self._TURN_LEAD:
                # Advance the latched grid line onto the new straight (target_heading =
                # theta + (k + turn_sign)*90) so the next straight nulls e_head correctly.
                if self._track.seg_k is not None and self._track.turn_sign is not None:
                    self._track.seg_k += self._track.turn_sign
                self._advance_maneuver()
                return
            aim = self._track.turn_sign * self._TURN_ANGLE
            self._emit(aim, self._TURN_LOOK, self._SPEED_TURN)
            self._dbg["aim"]   = aim
            self._dbg["speed"] = self._SPEED_TURN

        elif kind == self._STEP_REVERSE_TURN:
            # Back AROUND the corner: a lateral-offset reverse target makes the robot rotate
            # continuously toward target_heading while reversing. The offset sign is set so the
            # rotation goes the turn direction (a reverse target to the turn side rotates the
            # nose the opposite way). Gyro-terminated like the forward turn; a distance cap is a
            # safety backstop so a wrong-way arc can never reverse unbounded.
            # Terminate a LEAD before target so the coast lands on target (see _REVERSE_TURN_LEAD).
            reached = abs(self._wrap(yaw - self._track.target_heading)) < self._REVERSE_TURN_LEAD
            capped  = abs(self._dist_signed - self._corner_mark) >= self._REVERSE_TURN_MAX
            if reached or capped:
                if self._track.seg_k is not None and self._track.turn_sign is not None:
                    self._track.seg_k += self._track.turn_sign
                self._advance_maneuver()
                return
            angle = -self._track.turn_sign * self._REVERSE_TURN_ANGLE
            self._emit_reverse(angle, self._REVERSE_LOOK, self._SPEED_TURN)
            self._dbg["aim"]   = angle
            self._dbg["speed"] = -self._SPEED_TURN

        elif kind == self._STEP_REAR_WALL:
            if self._align_to_segment(a, r, e_head):
                self._advance_maneuver()

    def _align_to_segment(self, a: np.ndarray, r: np.ndarray, e_head: radian) -> bool:
        """Align the robot with the new segment AND set its standoff from the wall behind.

        Closed loop on both heading and the rear-wall distance — this is what corrects the
        turn/arc overshoot (the arc always overshoots; without this the corner ends
        mis-aligned). Drives ALONG the segment and decides forward vs reverse internally to
        hold `_REAR_WALL_DIST` from the wall behind: forward if too close (the usual case
        after a reverse arc — forward fixes position AND heading together), reverse if too
        far. Steering nulls `e_head` — `aim = e_head` forward, `-e_head` in reverse (reversing
        inverts the steering sign). Shared by all three corner maneuvers.

        Returns True when aligned AND positioned (or a safety guard fires) so the caller can
        move on; emits the drive target and returns False while still working.
        """
        rear = self._wall_distance(a, r, self._wrap(e_head + math.pi))
        self._dbg["rear"] = rear if rear is not None else float("nan")

        aligned    = abs(e_head) < self._REAR_WALL_ALIGN_TOL
        positioned = rear is not None and abs(rear - self._REAR_WALL_DIST) < self._REAR_WALL_TOL
        capped     = abs(self._dist_signed - self._corner_mark) >= self._REAR_WALL_MAX_TRAVEL
        if capped or (aligned and positioned) or (aligned and rear is None):
            return True

        if rear is not None and rear > self._REAR_WALL_DIST:   # too far -> reverse along segment
            aim = max(-self._MAX_AIM, min(self._MAX_AIM, -e_head))
            self._emit_reverse(aim, self._REVERSE_LOOK, self._SPEED_REVERSE)
            self._dbg["aim"]   = aim
            self._dbg["speed"] = -self._SPEED_REVERSE
        else:                                                  # too close / aligning -> forward
            aim = max(-self._MAX_AIM, min(self._MAX_AIM, e_head))
            self._emit(aim, self._REVERSE_LOOK, self._SPEED_CORNER)
            self._dbg["aim"]   = aim
            self._dbg["speed"] = self._SPEED_CORNER
        return False

    def _complete_corner(self) -> None:
        self._track.corner_count += 1
        self._track.target_offset = 0.0        # every straight begins centred
        self._track.current_lane  = 0.0
        self._lat_dr        = 0.0
        self._votes.clear()
        self._set_state(self._S_STRAIGHT)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _wall_distance(self, a: np.ndarray, r: np.ndarray, center: radian) -> mm | None:
        rel = self._wrap_array(a - center)
        sel = np.abs(rel) < self._SIDE_HALF
        if not sel.any():
            return None
        # Perpendicular distance to the wall: project onto the window centre normal.
        return float(np.median(r[sel] * np.cos(rel[sel])))

    # ------------------------------------------------------------------ #
    #  Final parking (end of run)                                         #
    # ------------------------------------------------------------------ #

    def _begin_park(self) -> None:
        # Reached the slot on the finish straight. Park per park_mode instead of just stopping.
        if self._park_mode == "none":
            self._finish()
            return
        self._park_in_step   = 0
        self._kturn_forward  = True
        self._park_leg_mark  = self._dist_signed
        self._park_front_min = float("inf")
        self._track.theta_frozen = True     # rotating next to the parking walls
        self._set_state(self._S_PARK)

    def _park_in(
        self, a: np.ndarray, r: np.ndarray, yaw: radian, seg: radian, e_head: radian,
    ) -> None:
        self._maneuver_step = self._park_in_step   # surfaced as corner_step in debug/visualiser
        if self._park_mode == "perpendicular":
            self._park_perpendicular(a, r, yaw, seg)
        else:
            # Parallel variants not implemented yet -> stop safely at the slot.
            self._finish()

    def _park_perpendicular(self, a: np.ndarray, r: np.ndarray, yaw: radian, seg: radian) -> None:
        # Nose-in perpendicular park, referenced to the two SIDE parking walls (the outer wall
        # vanishes below the LiDAR min range as the nose approaches). Steps:
        #   0 turn  — K-turn 90deg to face the OUTER wall.
        #   1 drive — drive in (facing outer, centring between the two parking walls) and STOP as
        #             soon as both flank the robot (it is between them). It never drives to the outer
        #             wall. Distance cap + a HARD wall-clock timeout make a stall-against-the-wall
        #             impossible to loop (the crash in run_20260913_153744).
        ts = self._track.turn_sign
        face_outer = self._wrap(seg - ts * math.pi / 2.0)   # world heading pointing at the outer wall
        if self._park_in_step == 0:
            if self._kturn_toward(a, r, yaw, face_outer):
                self._park_in_step     = 1
                self._park_leg_mark    = self._dist_signed
                self._park_in_deadline = time.monotonic() + self._PARK_IN_TIMEOUT
            return

        # Facing the outer wall: it is dead ahead (bearing 0); the two parking walls are now
        # left/right (bearings -+90deg).
        front = self._wall_distance(a, r, 0.0)
        left  = self._wall_distance(a, r, -math.pi / 2.0)
        right = self._wall_distance(a, r,  math.pi / 2.0)
        if front is not None:
            self._park_front_min = min(self._park_front_min, front)
        self._dbg["rear"] = front if front is not None else float("nan")

        near    = front is not None and front <= self._PARK_IN_STANDOFF          # at the outer wall
        lost    = front is None and self._park_front_min <= self._PARK_IN_STANDOFF + self._PARK_IN_LOST_MARGIN
        flanked = (left is not None and right is not None                        # alternate seat
                   and left  < self._PARK_IN_FLANK_MAX
                   and right < self._PARK_IN_FLANK_MAX)
        capped  = abs(self._dist_signed - self._park_leg_mark) >= self._PARK_IN_MAX
        timeout = time.monotonic() >= self._park_in_deadline
        if near or lost or flanked or capped or timeout:
            self._finish()
            return

        # Hold the outer-facing heading, plus an ANTI-CLIP nudge: steer away from a side parking wall
        # only when it is genuinely CLOSE (< _PARK_SIDE_MIN). Reacting only to a near wall (not a
        # "centre between the two" term) means a far/open reading can't saturate the steer — that was
        # the veer in run_20260913_155013. Left wall (-90deg) too close -> steer +y (right); right too
        # close -> steer -y.
        aim = self._wrap(face_outer - yaw)
        near_left  = left  is not None and left  < self._PARK_SIDE_MIN
        near_right = right is not None and right < self._PARK_SIDE_MIN
        if near_left and (not near_right or left <= right):
            aim += min(self._PARK_AVOID_MAX, self._PARK_AVOID_GAIN * (self._PARK_SIDE_MIN - left))
        elif near_right:
            aim -= min(self._PARK_AVOID_MAX, self._PARK_AVOID_GAIN * (self._PARK_SIDE_MIN - right))
        aim = max(-self._MAX_AIM, min(self._MAX_AIM, aim))
        self._emit(aim, self._TURN_LOOK, self._SPEED_PARK)
        self._dbg["aim"] = aim; self._dbg["speed"] = self._SPEED_PARK

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

    def _emit_reverse(self, angle: radian, lookahead: mm, speed: float) -> None:
        # Reverse target: point behind the robot; the firmware reverses toward it when
        # speed is negative. Run it through _make_reachable so it lands OUTSIDE the firmware
        # min-turn-radius circles — otherwise the firmware rejects it (-> IDLE, motor holds
        # its last forward command, robot coasts into the wall). For a reverse-arc turn the
        # raw target is inside the circle; the clamp snaps it onto the min-turn circle, which
        # is the tightest reverse arc the car can hold.
        forward = -(lookahead * math.cos(angle)) + self._mount_offset[1]
        lateral = lookahead * math.sin(angle) + self._mount_offset[0]
        forward, lateral = self._make_reachable(forward, lateral)
        self.on_target(forward, lateral, -abs(speed))

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
        self,
        a: np.ndarray, r: np.ndarray,
        yaw: radian, seg_heading: radian, e_head: radian,
    ) -> None:
        # One payload per scan: the scan (body frame), the geometry the planner acted
        # on this cycle (cone, walls, aim, obstacle), and the scalar state. Consumed
        # live by SegmentVisualizer and archived by the Recorder (attach_npz).
        d = self._dbg
        self.on_debug({
            # scan (body frame) so the visualiser can redraw walls + cone membership
            "scan_a":         a.astype(np.float32),
            "scan_r":         r.astype(np.float32),
            # heading / grid
            "state":          np.int32(self._state),
            "corner_kind":    np.int32(self._maneuver_kind),
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
            # forward cone / end wall / obstacle
            "cone_center":    np.float32(d["cone_center"]),
            "cone_half":      np.float32(self._CONE_HALF),
            "d_front":        np.float32(d["d_front"]),
            "front_trigger":  np.float32(self._active_front_trigger),
            "obstacle_x":     np.float32(d["obstacle_x"]),
            "obstacle_y":     np.float32(d["obstacle_y"]),
            "obstacle_color": np.int32(d["obstacle_color"]),
            "obs_recalled":   np.int32(d["obs_recalled"]),
            # switch target point (forward beside pillar, lateral = lane) in body frame
            "target_x":       np.float32(d["target_x"]),
            "target_y":       np.float32(d["target_y"]),
            # lane / lateral
            "lat_dr":         np.float32(self._lat_dr),
            "target_offset":  np.float32(self._track.target_offset),
            "lane_offset":    np.float32(self._LANE_OFFSET),
            # closed-loop rear-wall standoff (post-turn reverse)
            "rear":           np.float32(d["rear"]),
            "rear_target":    np.float32(self._REAR_WALL_DIST),
            # parking-slot start: frame magenta fraction (NaN except during INIT)
            "park_ratio":     np.float32(d.get("park_ratio", float("nan"))),
            # command
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
