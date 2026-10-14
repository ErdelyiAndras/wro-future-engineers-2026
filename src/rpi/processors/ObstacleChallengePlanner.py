from __future__ import annotations

import math
import time

import numpy as np

from processors.Processor import Processor
from processors.CameraProcessor import CameraProcessor
from control.EgoInformation import EgoInformation
from control.TrackModel import TrackModel
from control import ObstacleColor
from utils import mm, degree, radian, Event


class ObstacleChallengePlanner(Processor):
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
    # The flank window must be NARROW: rays beside a pillar often return nothing (beyond _R_MAX), and a
    # 45deg window then reached the side wall ~37deg away and called the pillar wall-attached
    # (run_005259/005346: pillar ~550mm ahead from the settle state, not detected until ~330mm).
    # Narrowed, a parking wall can also pass when the rays toward the outer wall drop out, so a size
    # cap backs it up: a pillar is 50x50 (cluster span < ~80mm), a parking wall's visible span is
    # 150-190mm. Checked on 30 runs: 0 parking walls accepted, pillar seen from the first settle scan.
    _OBSTACLE_GAP:       mm     = 300.0    # depth step (farther) that counts as open space on a flank
    _OBSTACLE_FLANK_WIN: radian = math.radians(15.0)   # bearing window just outside the cluster edge
    _OBSTACLE_MAX_EXTENT: mm    = 120.0    # cluster span at/above this = a wall, not a pillar
    # Obstacle-colour memory: a pillar's colour is stored the first time it is confidently read
    # (lap 1), keyed by (segment = corner_count % 4, longitudinal distance from that segment's END
    # wall via LiDAR). On later laps a re-detected pillar within this tolerance recalls the stored
    # colour, so the lane is correct even if the camera misreads that pass. (A wrong lap-1 read is a
    # non-issue: passing an obstacle on the wrong side ends the round, so there is no later lap.)
    _OBS_MEM_TOL:       mm     = 150.0     # |end-wall distance| match window for "the same pillar"
    # Default colour: a pillar whose colour is still unresolved (no recall, no camera majority) once
    # it is this close ahead is ASSUMED to be _DEFAULT_COLOR, so the robot still switches lanes (50%
    # chance of the correct side) instead of driving straight into it. Not applied earlier, because
    # LANE_SWITCH never re-reads the colour: a guess on the first frame would lock out a later real
    # read. The guess IS stored in the colour memory like a camera read: if the robot survives the
    # pass the guess was right (wrong side ends the round), so later laps recall it.
    _DEFAULT_COLOR:      ObstacleColor = ObstacleColor.RED
    _DEFAULT_COLOR_DIST: mm            = 800.0   # pillar forward distance at/below which the default applies

    # --- Lanes ---
    _LANE_OFFSET:       mm     = 300.0     # |lateral| of the left / right lanes from corridor centre
    _LANE_TOL:          mm     = 40.0      # within this of target offset = "in lane" (cruise speed)
    # Parking-wall avoidance: on the START straight (corner_count % 4 == 0) the parking lot
    # sits against the outer (perimeter) wall, so the OUTER lane is pulled this much toward
    # centre to clear the magenta parking barriers. Applied only to the outer lane there;
    # centre/inner untouched. MUST be < _LANE_OFFSET (else the lane sign is lost). 0 = off.
    _PARKING_LANE_SHIFT: mm    = 190.0
    # Lateral re-anchor after a parking start (see _lateral_fix): the slot-leave does NOT end on the
    # centre line (CW: 35-107mm toward the outer wall, CCW up to ~100mm; run_222909 drove the outer
    # lane ~50mm too close to the outer wall). On the first start-straight scan where both side walls
    # read and add up to the corridor width, lat_dr is set from the LiDAR: (left - right) / 2.
    _CORRIDOR_WIDTH:    mm     = 1000.0
    _LAT_FIX_TOL:       mm     = 60.0      # |left + right - _CORRIDOR_WIDTH| <= this = clean wall pair
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
    # Two straight reverses wrap the reverse turn (see _park_sequence), per direction:
    #   KTURN -> APPROACH -> BACK_PRE (hold perp) -> REVERSE_OUT -> BACK_POST (hold seg).
    # BACK_POST backs the robot up the track, away from a pillar close ahead of the slot, so the
    # PARK_SETTLE colour read and the first lane switch have room. A 0 distance skips that reverse.
    _PARK_KTURN_STEER:   radian = math.radians(38.0)   # leg steer magnitude (aim toward the turn side)
    _PARK_KTURN_FWD_LEG: mm     = 180.0    # nominal FORWARD-leg travel (primary leg-end); <= fore clearance
    _PARK_KTURN_REV_LEG: mm     = 40.0    # nominal REVERSE-leg travel — smaller so the reverse moves less
    # FINAL-park K-turn (parallel STRAIGHTEN) uses SHORTER legs than the
    # slot-leave: the end slot is tight, so long legs swing the nose/tail into the parking walls
    # (run_173210 touched the walls). Smaller than the leave's 180/40, but KEEP the reverse-dominant
    # ratio (fwd_leg > rev_leg): in the inverted STRAIGHTEN the long leg is the REVERSE one, which
    # drifts the robot INTO the slot. Equal legs (70/70) lost that tuck-in and it drifted OUT (run_174542).
    _PARK_END_KTURN_FWD_LEG: mm = 90.0
    _PARK_END_KTURN_REV_LEG: mm = 40.0
    # Wall-proximity early-out for the FINAL-park K-turns (larger than the leave's _PARK_KTURN_CLEAR,
    # which stays small so the boxed slot-leave can escape). The LiDAR is ~160mm ahead of the TAIL, so
    # a reverse leg that stopped at the leave's 150 put the tail ~29mm THROUGH the rear wall and tipped
    # it (run_131140). With the _PARK_KTURN_MIN(50) overshoot, tail clearance ~= CLEAR - 50 - 160, so
    # 250 leaves ~40mm worst-case. Shared by the forward (nose) check too.
    _PARK_END_KTURN_CLEAR: mm = 100.0
    # parallel-park STRAIGHTEN final heading: this far PAST parallel toward the OUTER/perimeter wall
    # (nose tilted toward the outer wall). 0 = parallel to the segment (parks IN the slot). WARNING:
    # positive rotates PAST parallel, swinging the tail inward and walking the robot OUT of the bay
    # (run_174542 at 30deg stopped way outside) — set small and test. This is the "final facing" knob.
    _PARK_STRAIGHTEN_OUT_BIAS: radian = math.radians(10.0)
    _PARK_KTURN_MIN:     mm     = 50.0     # min travel before a wall-proximity early-out (anti-deadlock)
    # Exit-side (turn direction) decision for a parking start. The side is inherently ambiguous
    # from one scan — both sides of a parallel slot are drivable, and a lone spurious return on the
    # open (off-field) side can flip a single-frame latch (run_162325 chose CCW off a stray 1058mm
    # return). When park_exit_side is "auto" we vote the LiDAR comparison over this many stationary
    # scans and commit the majority, instead of latching the first frame.
    _PARK_EXIT_VOTES:    int    = 5
    _PARK_KTURN_CLEAR:   mm     = 150.0    # end a leg early (after MIN) if a wall is this near ahead/behind
    # A K-turn wall reading counts only with at least this many rays in the +-30deg window. Up close the
    # wall drops out of the LiDAR and the window keeps 0-3 stray rays (a meaningless "far" median); real
    # readings — the wall, or open track once the robot turned away — had 5+ (runs 150338, 203536).
    _PARK_KTURN_WALL_RAYS: int  = 5
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
    _PARK_LEAVE_BACK_PRE_CCW:  mm = 50.0  # CCW: straight reverse after APPROACH, before the reverse turn
    _PARK_LEAVE_BACK_POST_CCW: mm = 200.0  # CCW: straight reverse after the reverse turn (facing seg)
    _PARK_LEAVE_BACK_PRE_CW:   mm = 10.0   # CW: straight reverse after APPROACH, before the reverse turn
    _PARK_LEAVE_BACK_POST_CW:  mm = 0.0   # CW: straight reverse after the reverse turn (facing seg)
    # Backstop for those straight reverses: end early if the wall behind is this close (LiDAR range;
    # the tail is ~160mm behind the LiDAR, so 250 leaves ~90mm). 0 / unset distance = step skipped.
    _PARK_LEAVE_BACK_CLEAR:    mm = 250.0
    _SPEED_PARK:         float  = 80.0   # slot-LEAVE maneuver speed (start-of-run exit)
    _SPEED_PARK_END:     float  = 65.0   # FINAL park maneuver speed (end of run: perpendicular / parallel)
    # After the exit (robot aligned with the segment) dwell stopped this long, sampling the
    # obstacle ahead, so its colour is read on clean motion-free frames (vote buffer fills,
    # camera settles, theta re-converges) before driving. 0 = no dwell.
    _PARK_SETTLE_S:      float  = 0.5

    # --- Final parking (end of run; park_mode selects the maneuver, closed-loop on the walls) ---
    # At the finish (back at the slot) the robot parks into the bay instead of just stopping. Both modes
    # share the same first two steps (see _park_mirror): REV_ARC (reverse arc to face the inner wall)
    # then BACK_IN (reverse into the slot).
    # PARALLEL: the slot-LEAVE maneuver run in reverse — then STRAIGHTEN (K-turn back to
    # parallel with the segment).
    # PERPENDICULAR: stops after BACK_IN, left perpendicular in the slot (nose to the inner wall).
    _PARK_IN_FLANK_MAX:  mm     = 250.0    # (backstop seat) both parking walls within this = flanked
    # BACK_IN (final-park step 1, both modes) reverses the WHOLE track width into the slot, facing
    # the inner wall — so the OUTER wall is directly behind (bearing 180deg) and is the seat reference
    # (the +-90deg flank sees the far side walls during the approach, not the parking walls). It needs
    # a big distance / time budget.
    # BACK_IN seats when the INNER wall (dead ahead while facing perp) is at least this far: the robot
    # has then reversed the slot depth into the bay. The inner wall is far and always cleanly rangeable,
    # so it is far more robust than the near outer wall (which drops below the LiDAR min range up close
    # and is offset by the ~173mm LiDAR-to-tail distance). A fixed constant because the field dimensions
    # are standard; this is the primary depth tuning knob (larger => stops shallower, more rotation room).
    _PARK_BACK_FRONT:    mm     = 715.0
    # HARD safety floor on the wall BEHIND: if _PARK_BACK_FRONT is set larger than the corridor allows
    # (front can only grow to ~width - this), the inner-wall seat is unreachable and the robot reverses
    # into the outer wall (run_164406: front capped at 815, tail touching at rear=181). Seat as soon as
    # the rear wall is this close so the tail (~173mm behind the rear beam) can never hit it.
    _PARK_BACK_REAR_MIN: mm     = 0.0
    _PARK_BACK_MAX:      mm     = 1200.0   # distance-cap backstop for the long cross-track reverse
    _PARK_BACK_TIMEOUT:  float  = 6.0      # wall-clock backstop (s) for the long reverse
    # Anti-clip: while driving in, steer AWAY from a side parking wall only when it is genuinely
    # CLOSE (< _PARK_SIDE_MIN), by a proportional amount capped at _PARK_AVOID_MAX. Unlike a
    # "centre-between" term this reacts only to a near wall, so a far/open reading can't saturate it
    # (that was the run_20260913_155013 veer). Prevents clipping when the K-turn leaves it off-centre.
    _PARK_SIDE_MIN:      mm     = 140.0    # a side wall this near => nudge away from it
    _PARK_AVOID_GAIN:    float  = 0.006    # rad per mm inside _PARK_SIDE_MIN
    _PARK_AVOID_MAX:     radian = math.radians(28.0)   # cap on the avoid nudge
    # How far (end-wall distance) BEFORE the slot the final-park maneuver STARTS, relative to the slot
    # (_d_front_start = the end-wall distance of the pose the run STARTED in — for a parking start it is
    # corrected for the leave's along-track travel, see _drive_straight). The REV_ARC backs the robot ~a turn-radius AWAY from the
    # end wall as it rotates to perpendicular, and BACK_IN then moves mostly laterally — so to land the
    # park at the slot the arc must begin PAST the slot (smaller d_front => a negative value). Larger =>
    # the arc starts sooner / farther from the end wall. (parallel/perpendicular modes; "none" is separate.)
    # Split by turn direction: the arc curves toward the inner wall (perp = seg + turn_sign*90), so the
    # geometry — and thus the lead distance — differs between CCW (turn_sign -1) and CW (turn_sign +1).
    # Converted when the reference moved from the post-leave pose to the slot pose (2026-10-11): old
    # -440 / -360 minus the average leave travel of the good runs (CCW -350mm, CW -175mm), so the
    # arc point is unchanged for those runs.
    _PARK_ARC_START_AHEAD_CCW: mm = -90.0
    _PARK_ARC_START_AHEAD_CW:  mm = -220.0
    # Approach slow-down (parking modes, after the last corner): within _PARK_SLOWDOWN_DIST of the
    # arc-start point the drive speed ramps linearly from its normal value down to
    # _PARK_SLOWDOWN_MIN_SPEED at the arc start, so the maneuver begins from a slow, repeatable speed.
    # Below the firmware MIN_SPEED (80) works while already moving (the kickstart only applies from rest).
    _PARK_SLOWDOWN_DIST:      mm    = 200.0
    _PARK_SLOWDOWN_MIN_SPEED: float = 65.0
    # Final-pass pillar cutoff (_past_park_cutoff): pillars beyond the arc-start point are ignored, but
    # only if they are more than this PAST it — the nose sits ahead of the LiDAR, so a pillar just past
    # the arc start would still be reached. NOT a measured value: set to LiDAR-to-nose + clearance.
    _PARK_CUTOFF_NOSE_MARGIN: mm    = 150.0

    # Parking-leave sub-steps (recorded as corner_step during _S_PARK_LEAVE).
    _PK_KTURN       = 0
    _PK_APPROACH    = 1
    _PK_REVERSE_OUT = 2
    _PK_BACK_PRE    = 3   # straight reverse holding perp (before REVERSE_OUT)
    _PK_BACK_POST   = 4   # straight reverse holding seg  (after REVERSE_OUT)

    # --- Laps / finish ---
    _TOTAL_CORNERS:     int    = 12        # 3 laps x 4 corners
    _FINISH_TOL:        mm     = 120.0     # PARKING modes: stop when the end-wall distance returns to within this of start
    # NON-parking (park_mode "none"): stop a fixed distance from the end wall ahead, so the robot
    # halts near the CENTRE of the start straight regardless of where the run began. The old
    # start-relative trigger fired too early when the run started near the segment's entry line
    # (d_front_start was ~the whole segment), so the robot never actually entered the end segment.
    _FINISH_STOP_DIST:  mm     = 1500.0
    # Finish-trigger filtering (both modes). d_front is the 85th percentile of a sparse forward cone, so
    # one scan where parking walls / a pillar fill the cone reads far too short (run_20261010_210715:
    # 2578 -> 890 -> 2354, park started in the corner square). Two guards:
    #   jump check — reject a reading that DROPS more than the odometry travel since the last accepted
    #                reading + _FINISH_JUMP_TOL. After _FINISH_JUMP_RESET consecutive rejections the new
    #                level is accepted (a persistent drop is real; also stops a stale-high reference
    #                from locking the finish out forever).
    #   debounce   — the stop condition must hold on _FINISH_DEBOUNCE consecutive accepted scans.
    _FINISH_JUMP_TOL:   mm     = 150.0
    _FINISH_JUMP_RESET: int    = 3
    _FINISH_DEBOUNCE:   int    = 3

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
        color_sampler:   CameraProcessor | None = None,
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
        # "perpendicular" or "parallel" (see _park_mirror). Unknown => "none".
        self._park_mode = park_mode if park_mode in (
            "none", "perpendicular", "parallel") else "none"
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
        self._lat_fix_pending: bool = False   # re-anchor lat_dr from the side walls (after a parking start)
        self._leave_start_pos: tuple[mm, mm] | None = None   # odometry pose where the slot-leave began
        self._stopped: bool = False
        # Finish-trigger filter state (see _FINISH_JUMP_* / _FINISH_DEBOUNCE).
        self._finish_ref:     tuple[mm, mm] | None = None   # (last accepted d_front, _dist_signed then)
        self._finish_rejects: int = 0
        self._finish_hits:    int = 0

        # Parking-leave maneuver scratch.
        self._exit_votes:        list[int] = []  # per-scan exit-side votes (auto pick), majority commits
        self._park_step:         int   = 0  # index into _park_sequence()
        self._park_back_pre_travel: mm = 0.0  # distance BACK_PRE actually reversed (lateral offset)
        self._kturn_forward:     bool  = True   # current K-turn leg direction (fwd vs reverse)
        self._kturn_wall_last:   mm | None = None  # last counted wall reading in the travel direction this leg
        self._park_leg_mark:     mm    = 0.0    # signed-distance marker for the current leg/step cap
        self._park_settle_until: float = 0.0    # monotonic deadline for the post-exit dwell (0 = not started)
        self._park_in_step:      int   = 0      # sub-step of the FINAL parking maneuver (_S_PARK)
        self._park_in_deadline:  float = 0.0    # monotonic hard-stop for the park drive-in (stall-proof)

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

        if self._lat_fix_pending and self._state in (self._S_PARK_SETTLE, self._S_STRAIGHT, self._S_SWITCH):
            self._lateral_fix(a, r, e_head)

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
        self._park_step     = 0   # index into _park_sequence()
        self._park_back_pre_travel = 0.0
        self._kturn_forward = True
        self._kturn_wall_last = None
        self._park_leg_mark = self._dist_signed
        # Freeze principal-angle detection for the maneuver: the robot rotates hard next to the
        # close parking walls, which would corrupt the tangent fit. theta is already seeded.
        self._track.theta_frozen = True
        self._leave_start_pos = self._ego.position   # the slot pose (reference for _d_front_start)
        self._set_state(self._S_PARK_LEAVE)

    def _park_leave(self, a: np.ndarray, r: np.ndarray, yaw: radian) -> None:
        # Dedicated 3-step exit (see the _PARK_* constants). seg_heading is yaw-independent
        # (theta + seg_k*90, both latched), so it is recomputed here for free.
        seg  = self._track.segment_heading(yaw)
        perp = self._wrap(seg + self._track.turn_sign * math.pi / 2.0)  # facing the inner wall
        kind = self._park_sequence()[self._park_step]
        self._maneuver_step = kind              # surfaced as corner_step in the debug/visualiser
        if kind == self._PK_KTURN:
            self._park_kturn(a, r, yaw, perp)
        elif kind == self._PK_APPROACH:
            self._park_approach(a, r, yaw, perp)
        elif kind == self._PK_BACK_PRE:
            self._park_back_straight(a, r, yaw, perp, self._park_back_dists()[0])
        elif kind == self._PK_REVERSE_OUT:
            self._park_reverse_out(yaw, seg)
        else:
            self._park_back_straight(a, r, yaw, seg, self._park_back_dists()[1])

    def _park_sequence(self) -> tuple[int, ...]:
        # Leave steps (both directions; a 0 BACK_* distance makes that step a no-op).
        return (self._PK_KTURN, self._PK_APPROACH, self._PK_BACK_PRE,
                self._PK_REVERSE_OUT, self._PK_BACK_POST)

    def _park_back_dists(self) -> tuple[mm, mm]:
        # (BACK_PRE, BACK_POST) straight-reverse distances for the current track direction.
        if self._track.turn_sign < 0:
            return self._PARK_LEAVE_BACK_PRE_CCW, self._PARK_LEAVE_BACK_POST_CCW
        return self._PARK_LEAVE_BACK_PRE_CW, self._PARK_LEAVE_BACK_POST_CW

    def _advance_park_step(self) -> None:
        if self._park_sequence()[self._park_step] == self._PK_BACK_PRE:
            self._park_back_pre_travel = abs(self._dist_signed - self._park_leg_mark)
        self._park_step += 1
        self._park_leg_mark = self._dist_signed
        if self._park_step >= len(self._park_sequence()):
            self._complete_park_leave()
            return
        if self._state_hold_s > 0.0:
            self._hold_until = time.monotonic() + self._state_hold_s

    def _park_kturn(self, a: np.ndarray, r: np.ndarray, yaw: radian, perp: radian) -> None:
        # Leave-maneuver step 1: rotate to face the inner wall via the shared K-turn shuffle.
        if self._kturn_toward(a, r, yaw, perp):
            self._advance_park_step()

    def _kturn_toward(self, a: np.ndarray, r: np.ndarray, yaw: radian, target: radian,
                      invert: bool = False, speed: float | None = None,
                      fwd_leg: float | None = None, rev_leg: float | None = None,
                      clear: float | None = None) -> bool:
        # Rotate yaw toward `target` by alternating forward/reverse legs so net travel stays small
        # (never reaching the fore/aft wall). `rot` = which way to turn to reach the target; a
        # forward leg steers `rot`, a reverse leg steers `-rot` (same rotation while backing). A leg
        # runs to its cap (_PARK_KTURN_FWD_LEG / _REV_LEG), ending early only after _PARK_KTURN_MIN
        # once a wall is within _PARK_KTURN_CLEAR in the travel direction (the MIN floor prevents the
        # boxed-in flip-without-moving deadlock). Returns True within _PARK_PERP_LEAD of the target.
        #
        # `invert` plays the K-turn in REVERSE: the long leg becomes the REVERSE one (so the net drift
        # flips direction). The caller must also start with `_kturn_forward = False`. This is what the
        # parallel-park STRAIGHTEN uses to undo the leave's forward-dominant KTURN, step for step.
        if abs(self._wrap(yaw - target)) < self._PARK_PERP_LEAD:
            return True
        rot      = math.copysign(1.0, self._wrap(target - yaw))
        traveled = abs(self._dist_signed - self._park_leg_mark)
        fl = self._PARK_KTURN_FWD_LEG if fwd_leg is None else fwd_leg
        rl = self._PARK_KTURN_REV_LEG if rev_leg is None else rev_leg
        if invert:
            leg_cap = rl if self._kturn_forward else fl
        else:
            leg_cap = fl if self._kturn_forward else rl
        cl = self._PARK_KTURN_CLEAR if clear is None else clear
        # The wall early-out uses the LAST COUNTED reading (>= _PARK_KTURN_WALL_RAYS rays), so a wall that
        # drops out of the LiDAR up close keeps its last close value: in CW slot-leaves the front parking
        # wall read 131-139mm before _PARK_KTURN_MIN was reached and was gone (0-3 rays) by the time it
        # was, so the first leg ran its full cap into the wall (run_150338 and 9 others). Not the
        # NEAREST reading: when the robot turns AWAY from a wall it starts the leg near (CCW 2nd forward
        # leg, ~140-150mm), a real far reading must replace it or the leg is cut short (run_203536).
        center = 0.0 if self._kturn_forward else math.pi
        n_rays = int(np.count_nonzero(np.abs(self._wrap_array(a - center)) < self._SIDE_HALF))
        if n_rays >= self._PARK_KTURN_WALL_RAYS:
            self._kturn_wall_last = self._wall_distance(a, r, center)
        leg_done = traveled >= leg_cap or (
            traveled >= self._PARK_KTURN_MIN
            and self._kturn_wall_last is not None and self._kturn_wall_last < cl)
        if leg_done:
            self._kturn_forward = not self._kturn_forward
            self._park_leg_mark = self._dist_signed
            self._kturn_wall_last = None
            return False
        spd = self._SPEED_PARK if speed is None else speed
        if self._kturn_forward:
            aim = rot * self._PARK_KTURN_STEER
            self._emit(aim, self._TURN_LOOK, spd)
            self._dbg["aim"] = aim; self._dbg["speed"] = spd
        else:
            aim = -rot * self._PARK_KTURN_STEER          # opposite steer -> same rotation, reversing
            self._emit_reverse(aim, self._REVERSE_LOOK, spd)
            self._dbg["aim"] = aim; self._dbg["speed"] = -spd
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
            self._advance_park_step()   # last step for CW; CCW continues with BACK_POST
            return
        angle = ts * self._PARK_OUT_ANGLE
        self._emit_reverse(angle, self._REVERSE_LOOK, self._SPEED_PARK)
        self._dbg["aim"] = angle; self._dbg["speed"] = -self._SPEED_PARK

    def _park_back_straight(
        self, a: np.ndarray, r: np.ndarray, yaw: radian, heading: radian, dist: mm,
    ) -> None:
        # Leave: straight reverse holding `heading` for `dist` (odometry). Ends early if the wall
        # behind is within _PARK_LEAVE_BACK_CLEAR. dist <= 0 skips the step.
        rear = self._wall_distance(a, r, math.pi)
        self._dbg["rear"] = rear if rear is not None else float("nan")
        # Count only REVERSE travel, from the most-forward point reached in this step. BACK_PRE starts
        # while the robot is still coasting forward from APPROACH (~35-40mm before the motor reverses);
        # an abs() distance counted that coast as travel: 30mm ended on the coast alone without ever
        # reversing (run_004440), 40mm had to undo the coast first and reversed ~85mm (run_004348).
        self._park_leg_mark = max(self._park_leg_mark, self._dist_signed)
        traveled = self._park_leg_mark - self._dist_signed
        if traveled >= dist or (rear is not None and rear < self._PARK_LEAVE_BACK_CLEAR):
            self._advance_park_step()
            return
        aim = max(-self._MAX_AIM, min(self._MAX_AIM, -self._wrap(heading - yaw)))   # reverse heading hold
        self._emit_reverse(aim, self._REVERSE_LOOK, self._SPEED_PARK)
        self._dbg["aim"] = aim; self._dbg["speed"] = -self._SPEED_PARK

    def _lateral_fix(self, a: np.ndarray, r: np.ndarray, e_head: radian) -> None:
        # One-shot after a parking start: set lat_dr (+ = right of the corridor centre) from the two
        # side walls, read perpendicular to the SEGMENT (so robot yaw doesn't matter). Only a clean pair
        # counts — a parking wall, pillar or the open corner square beside the robot breaks the sum.
        # Abandoned once the start straight is left.
        if self._track.corner_count != 0:
            self._lat_fix_pending = False
            return
        left  = self._wall_distance(a, r, self._wrap(e_head - math.pi / 2.0))
        right = self._wall_distance(a, r, self._wrap(e_head + math.pi / 2.0))
        if left is None or right is None or abs(left + right - self._CORRIDOR_WIDTH) > self._LAT_FIX_TOL:
            return
        self._lat_dr = (left - right) / 2.0
        self._lat_fix_pending = False

    def _complete_park_leave(self) -> None:
        # On the track now, aligned with the segment: reset lateral dead-reckoning to treat the
        # current position as the lane origin, centre the lane, resume theta detection. Then dwell
        # (PARK_SETTLE) to read the obstacle colour on clean frames before driving. corner_count
        # stays 0, so the start straight (and its parking-wall lane shift) is in force.
        # Seed the lateral offset assuming the leave ended on the centre line, plus BACK_PRE (which
        # backs that much further from the inner wall, toward the outer side = -turn_sign). This is
        # only a fallback: the leave's real end varies (35-107mm off in CW), so _lateral_fix replaces it
        # with the LiDAR-measured offset on the first clean side-wall pair.
        self._lat_dr = -self._track.turn_sign * self._park_back_pre_travel
        self._lat_fix_pending = True
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
        if not self._plausible_front(d_front):
            self._finish_hits = 0
            return False
        if self._park_mode == "none":
            # Just stop, centred in the segment: a fixed standoff from the end wall,
            # independent of the start pose (see _FINISH_STOP_DIST).
            reached = d_front <= self._FINISH_STOP_DIST
        else:
            # Parking modes start the maneuver relative to the slot (where the run began). The FWD_ARC
            # pulls forward as it rotates, so begin a turn-radius early (see _PARK_ARC_START_AHEAD_*).
            arc_front = self._park_arc_start_front()
            reached = arc_front is not None and d_front <= arc_front
        self._finish_hits = self._finish_hits + 1 if reached else 0
        if self._finish_hits >= self._FINISH_DEBOUNCE:
            self._begin_park()
            return True
        return False

    def _park_arc_start_front(self) -> mm | None:
        # End-wall distance at which the final-park arc starts (relative to the slot = run start).
        if self._d_front_start is None:
            return None
        ahead = (self._PARK_ARC_START_AHEAD_CW if self._track.turn_sign > 0
                 else self._PARK_ARC_START_AHEAD_CCW)
        return self._d_front_start + ahead

    def _approach_speed(self, speed: float) -> float:
        # Ramp the drive speed down toward the final-park arc start (see _PARK_SLOWDOWN_*). Uses the
        # last jump-checked end-wall distance, so a cluttered-cone glitch cannot cause a slow-down.
        if (self._park_mode == "none" or self._track.corner_count < self._TOTAL_CORNERS
                or self._finish_ref is None):
            return speed
        arc_front = self._park_arc_start_front()
        if arc_front is None:
            return speed
        remaining = self._finish_ref[0] - arc_front
        frac = max(0.0, min(1.0, remaining / self._PARK_SLOWDOWN_DIST))
        return min(speed, self._PARK_SLOWDOWN_MIN_SPEED + frac * (speed - self._PARK_SLOWDOWN_MIN_SPEED))

    def _plausible_front(self, d_front: mm) -> bool:
        # Jump check for the finish trigger: the end wall can only get nearer by about the distance
        # driven. A larger drop is a cone filled by near clutter (parking walls / pillar) -> reject.
        # Increases are always accepted. _FINISH_JUMP_RESET consecutive rejections accept the new level.
        if self._finish_ref is not None:
            ref_front, ref_dist = self._finish_ref
            travel = self._dist_signed - ref_dist
            if d_front < ref_front - travel - self._FINISH_JUMP_TOL:
                self._finish_rejects += 1
                if self._finish_rejects < self._FINISH_JUMP_RESET:
                    return False
        self._finish_rejects = 0
        self._finish_ref = (d_front, self._dist_signed)
        return True

    def _drive_straight(
        self,
        a: np.ndarray, r: np.ndarray, x: np.ndarray, y: np.ndarray,
        yaw: radian, seg_heading: radian, e_head: radian,
    ) -> None:
        d_front, obstacle = self._forward_cone(a, r, x, y, e_head)

        if self._d_front_start is None and d_front is not None:
            # The final park is placed relative to the SLOT. After a parking start, d_front is read only
            # once the leave is done, and the leave's along-track travel varies: in CW the K-turn's first
            # forward leg sometimes stops early at the parking wall ahead (~45mm instead of ~175mm), so
            # the leave ended ~130mm further back and the arc started ~130mm early, reversing into the
            # first parking wall (run_141221 ... run_151532). Add back the odometry travel along the
            # segment since the leave began, so _d_front_start is the slot's end-wall distance.
            self._d_front_start = d_front
            if self._leave_start_pos is not None:
                pos = self._ego.position
                along = ((pos[0] - self._leave_start_pos[0]) * math.cos(seg_heading)
                         + (pos[1] - self._leave_start_pos[1]) * math.sin(seg_heading))
                self._d_front_start = d_front + along

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
        speed = self._approach_speed(speed)

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
        if obstacle is not None and self._past_park_cutoff(obstacle, d_front, e_head):
            obstacle = None   # beyond the slot on the final pass: don't re-aim the switch at it
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
        speed = self._approach_speed(self._SPEED_SWITCH)
        self._emit(aim, lookahead, speed)

        self._dbg["d_front"]  = d_front if d_front is not None else float("nan")
        self._dbg["aim"]      = aim
        self._dbg["speed"]    = speed
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
            rn, an = rc[near], ain[near]
            left = np.ones(len(rn), dtype=bool)          # points not yet assigned to a cluster
            while left.any():
                idx = np.flatnonzero(left)
                i0 = int(idx[np.argmin(rn[idx])])
                close = left & (np.hypot(xn - xn[i0], yn - yn[i0]) < self._CLUSTER_RADIUS)
                # In the start straight, keep it only if it is free-standing (a pillar), not
                # continuous with the perimeter (a parking wall). Tested against the full scan.
                # A rejected parking wall must not hide a pillar behind it, so try the next-nearest
                # cluster (run_005259: the parking wall was nearer, the pillar never got tested).
                if self._track.corner_count % 4 != 0 or \
                        self._is_free_standing(a, r, an[close], rn[close]):
                    obstacle = (float(np.mean(xn[close])), float(np.mean(yn[close])))
                    break
                left &= ~close
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
        cx, cy = cluster_r * np.cos(cluster_a), cluster_r * np.sin(cluster_a)
        span = float(np.max(np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])))
        if span >= self._OBSTACLE_MAX_EXTENT:
            return False                                      # too long for a 50x50 pillar -> parking wall
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
        if obstacle is not None and self._past_park_cutoff(obstacle, d_front, e_head):
            obstacle = None
        if obstacle is None:
            self._votes.clear()
            return

        seg   = self._track.corner_count % 4
        pos   = None
        color = None
        if d_front is not None:
            pos   = d_front - (obstacle[0] * math.cos(e_head) + obstacle[1] * math.sin(e_head))
            color = self._recall_obstacle(seg, pos)   # None if never learned
        recalled  = color is not None
        defaulted = False
        if not recalled:
            color = self._resolve_color(obstacle)      # camera majority vote
            defaulted = color is None and obstacle[0] <= self._DEFAULT_COLOR_DIST
            if defaulted:
                color = self._DEFAULT_COLOR            # last-moment guess
            if color is not None and pos is not None:
                self._store_obstacle(seg, pos, color)  # learn it (store once)
        # Colour source: 0 = camera, 1 = recalled from memory, 2 = default guess.
        self._dbg["obs_recalled"] = 1 if recalled else (2 if defaulted else 0)

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

    def _past_park_cutoff(self, obstacle: tuple[float, float], d_front: mm | None, e_head: radian) -> bool:
        # Final segment of a parking run: the robot never drives past the final-park arc start, so a
        # pillar between that point and the end wall can't block it. Discard it — otherwise it starts a
        # lane switch that is still running when the park begins. Cutoff = the arc-start point
        # (_d_front_start + _PARK_ARC_START_AHEAD_*) minus _PARK_CUTOFF_NOSE_MARGIN toward the end wall,
        # so a pillar the nose would reach at the arc start is still avoided. The end-wall distance is the jump-checked one
        # (propagated by odometry) so a cluttered-cone glitch can't misplace the pillar.
        if self._park_mode == "none" or self._track.corner_count < self._TOTAL_CORNERS:
            return False
        arc_front = self._park_arc_start_front()
        if arc_front is None:
            return False
        front = d_front
        if self._finish_ref is not None:
            ref_front, ref_dist = self._finish_ref
            front = ref_front - (self._dist_signed - ref_dist)
        if front is None:
            return False
        pos = front - (obstacle[0] * math.cos(e_head) + obstacle[1] * math.sin(e_head))
        return pos < arc_front - self._PARK_CUTOFF_NOSE_MARGIN

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
        self._kturn_wall_last = None
        self._park_leg_mark  = self._dist_signed
        self._track.theta_frozen = True     # rotating next to the parking walls
        self._set_state(self._S_PARK)

    def _park_in(
        self, a: np.ndarray, r: np.ndarray, yaw: radian, seg: radian, e_head: radian,
    ) -> None:
        self._maneuver_step = self._park_in_step   # surfaced as corner_step in debug/visualiser
        self._park_mirror(a, r, yaw, seg)

    def _park_mirror(self, a: np.ndarray, r: np.ndarray, yaw: radian, seg: radian) -> None:
        # Final park, both modes. Parallel park = the slot-LEAVE maneuver run in reverse, step for step,
        # reusing its tuned primitives. The leave goes (slot, facing seg) -> KTURN to perp -> APPROACH to
        # the inner-wall standoff -> REVERSE_OUT arc back onto the lane facing seg. Inverting each step,
        # in reverse order, drives (lane, facing seg) back into the slot:
        #   0 REV_ARC   — REVERSE arc seg -> perp: back up while rotating to face the inner wall, the
        #                 tail swinging toward the slot. Reverse (not forward) so it works from the inner
        #                 lane too, where a forward arc has no room. Gyro-terminated; travel cap backstop.
        #   1 BACK_IN   — reverse (facing perp) toward the outer wall / slot, holding perp with an
        #                 anti-clip nudge off the side parking walls, until seated (inner wall at
        #                 _PARK_BACK_FRONT). Flank / distance cap / wall-clock timeout are stall-proof
        #                 backstops. "perpendicular" STOPS here, perpendicular in the slot.
        #   2 STRAIGHTEN— (parallel only) K-turn perp -> seg in place (inverse of KTURN):
        #                 rotate to parallel, then stop.
        ts   = self._track.turn_sign
        perp = self._wrap(seg + ts * math.pi / 2.0)   # facing the inner wall (same perp as the leave)

        if self._park_in_step == 0:
            self._track.target_heading = perp
            reached = abs(self._wrap(yaw - perp)) < self._PARK_OUT_LEAD
            capped  = abs(self._dist_signed - self._park_leg_mark) >= self._PARK_OUT_MAX
            if reached or capped:
                self._park_in_step     = 1
                self._park_leg_mark    = self._dist_signed
                self._park_in_deadline = time.monotonic() + self._PARK_BACK_TIMEOUT
                return
            # REVERSE arc toward perp: reversing, the opposite steer (-ts) rotates the nose toward the
            # inner wall while the tail swings toward the outer wall / slot. Works from any lane — a
            # forward arc from the inner lane has no room before the inner wall.
            aim = -ts * self._PARK_OUT_ANGLE
            self._emit_reverse(aim, self._REVERSE_LOOK, self._SPEED_PARK_END)
            self._dbg["aim"] = aim; self._dbg["speed"] = -self._SPEED_PARK_END
            return

        if self._park_in_step == 1:
            self._track.target_heading = perp
            # Facing the inner wall, it is dead ahead (bearing 0); reversing drives toward the outer
            # wall / into the slot, so the inner-wall distance GROWS. Seat when it exceeds the fixed
            # _PARK_BACK_FRONT (the robot has reversed the slot depth in). The inner wall is far and
            # always cleanly rangeable; flank / cap / timeout are backstops.
            front = self._wall_distance(a, r, 0.0)
            rear  = self._wall_distance(a, r, math.pi)
            left  = self._wall_distance(a, r, -math.pi / 2.0)
            right = self._wall_distance(a, r,  math.pi / 2.0)
            self._dbg["rear"] = rear if rear is not None else float("nan")
            seated  = front is not None and front >= self._PARK_BACK_FRONT
            rear_min = rear is not None and rear <= self._PARK_BACK_REAR_MIN   # hard floor: never hit the wall
            flanked = (left is not None and right is not None                # backstop: wedged between walls
                       and left  < self._PARK_IN_FLANK_MAX
                       and right < self._PARK_IN_FLANK_MAX)
            capped  = abs(self._dist_signed - self._park_leg_mark) >= self._PARK_BACK_MAX
            timeout = time.monotonic() >= self._park_in_deadline
            if seated or rear_min or flanked or capped or timeout:
                if self._park_mode == "perpendicular":
                    self._finish()            # perpendicular park: no STRAIGHTEN
                    return
                self._park_in_step  = 2
                self._park_leg_mark = self._dist_signed
                self._kturn_forward = False   # STRAIGHTEN plays the leave's KTURN in reverse (reverse-first)
                self._kturn_wall_last = None
                return
            # Hold perp while reversing (steer -e_perp, as the leave's APPROACH reverse branch does),
            # plus an anti-clip nudge: back AWAY from a side wall only when it is genuinely CLOSE, so
            # a far/open reading can't saturate the steer (the run_20260913_155013 veer).
            e_perp = self._wrap(perp - yaw)
            aim = -e_perp
            near_left  = left  is not None and left  < self._PARK_SIDE_MIN
            near_right = right is not None and right < self._PARK_SIDE_MIN
            if near_left and (not near_right or left <= right):
                aim += min(self._PARK_AVOID_MAX, self._PARK_AVOID_GAIN * (self._PARK_SIDE_MIN - left))
            elif near_right:
                aim -= min(self._PARK_AVOID_MAX, self._PARK_AVOID_GAIN * (self._PARK_SIDE_MIN - right))
            aim = max(-self._MAX_AIM, min(self._MAX_AIM, aim))
            self._emit_reverse(aim, self._REVERSE_LOOK, self._SPEED_PARK_END)
            self._dbg["aim"] = aim; self._dbg["speed"] = -self._SPEED_PARK_END
            return

        # step 2: rotate from perp toward (just past) parallel by playing the leave's KTURN in reverse
        # (reverse-first, long leg reversed), tucking into the slot. The final heading is tilted a bit
        # PAST parallel toward the outer wall (nose toward the perimeter) by _PARK_STRAIGHTEN_OUT_BIAS.
        target = self._wrap(seg - ts * self._PARK_STRAIGHTEN_OUT_BIAS)
        self._track.target_heading = target
        if self._kturn_toward(a, r, yaw, target, invert=True, speed=self._SPEED_PARK_END,
                              fwd_leg=self._PARK_END_KTURN_FWD_LEG, rev_leg=self._PARK_END_KTURN_REV_LEG,
                              clear=self._PARK_END_KTURN_CLEAR):
            self._finish()

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
        # live by PathPlanningVisualizer and archived by the Recorder (attach_npz).
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
