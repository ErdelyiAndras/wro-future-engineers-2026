from __future__ import annotations

import math
import time

import numpy as np

from processors.Processor import Processor
from control.FieldMap import FieldMap, ObstacleColor
from control.EgoInformation import EgoInformation
from utils import mm, degree, radian, second, Point, Event


class FollowTheGapPlanner(Processor):
    """Reactive scan-space planner: Follow-the-Gap with a disparity extender.

    Phase 1 of the geometry-agnostic stack (docs/path-planning-research.md §3.A,
    §6). The planner assumes only that the track is a corridor bounded by a left
    and a right wall and that it closes into a loop — nothing about its shape.
    Steering never touches the world map or the EKF pose; each cycle works on the
    latest raw LiDAR scan in the robot frame:

      1. Bin the scan into fixed angular bins (min range per bin). Holes are
         never free space by default: on a walled field a missing return is
         usually absorption (matte black walls, grazing angles), so a hole
         bounded by returns at similar range is bridged as a continuous
         surface, and only a hole bordering a genuine disparity stays open —
         flagged *unknown* so it cannot outrank a corroborated gap.
      2. Mask everything outside a forward window centred on the *travel
         direction* — the tangent of the recently driven odometry trail, not the
         instantaneous nose — so the sector the robot came from can never win,
         even mid-corner with the nose swung around.
      3. Extend disparities: every range discontinuity (wall corner, pillar edge)
         is widened by the robot's half width in scan space, so any ray that
         survives is one the whole robot body can follow.
      4. Pick the best gap (contiguous run of sufficiently deep bins) and the
         best ray within it.
      5. Commit with hysteresis: the steering angle follows a low-pass filter and
         a large jump must win two consecutive scans before it is accepted, so
         near-ties between gaps cannot thrash the target.
      6. Emit a body-frame lookahead target, clamped to the car's minimum turning
         radius, with speed scaled by how deep and straight the chosen gap is.

    World-frame state is used only for memory duties: lap counting via
    ``FieldMap.update_lap_progress`` (stop after ``TOTAL_LAPS``) and the odometry
    breadcrumbs behind the travel tangent. Both tolerate EKF drift.
    """

    # --- Scan binning ---
    _BIN_COUNT:       int = 180                    # 2° bins over the full circle
    _R_MAX:           mm  = 3000.0                 # range cap; also "no return" fill
    _MAX_HOLE_BINS:   int = 3                      # sparse-sampling holes up to this many
                                                   # bins take the nearer neighbour

    # --- Occupancy-grid memory fusion (VFH-style; research doc §3.A) ---
    # A single scan misses whole sectors off matte black walls (the IR pulse is
    # absorbed), so a wall renders as a deep phantom gap and the disparity
    # extender, having no near return to inflate, steers the car into it. The
    # accumulated occupancy grid saw those walls on earlier scans from closer,
    # squarer angles; raycasting it each cycle restores them. Only the near field
    # is trusted (_MEMORY_MAX) so EKF drift over a lap cannot inject stale walls
    # at long range, and a bin is corroborated only above _MEMORY_OCC log-odds
    # (~one confident hit) so single-scan noise cannot fabricate one.
    _MEMORY_MAX:      mm    = 2000.0               # cap on the memory raycast
    _MEMORY_STEP:     mm    = FieldMap.CELL_SIZE   # march one cell at a time (no tunnelling)
    _MEMORY_OCC:      float = 0.0                  # log-odds above which a cell counts as wall.
                                                   # 0.0 = any net-positive evidence (one hit is
                                                   # +0.95, a free crossing -0.40); catches the
                                                   # sparsely-hit walls left by heavy absorption
                                                   # without counting net-free cells. Going
                                                   # negative fabricates phantom walls.

    # --- Pass-side rule (obstacle challenge; research doc §3.A / §4.2) ---
    # Coloured pillars mandate a passing side (red -> pass on its right, keeping
    # the pillar on the left; green -> pass on its left). Instead of blocking the
    # wrong side in the world grid (the old PathPlanningProcessor way, which
    # assumed a rectilinear track), we close the wrong-side gap *in scan space*:
    # fill the bins between the pillar and the wall on the forbidden side with the
    # pillar's range, so the deepest surviving gap — and thus the gap FTG steers
    # into — is on the mandated side. The side is taken relative to the travel
    # tangent (drift-stable), and the pillar colour is FieldMap's vote-thresholded
    # `obs.color`, so no per-pillar latching is needed. Geometry-agnostic: nothing
    # about the track shape is assumed.
    _PASS_HALF_WIDTH:    mm     = 60.0             # pillar half-width + margin (scan-space edge)
    _PASS_RANGE:         mm     = 1500.0           # only mask pillars within this range
    _PASS_MASK_MAX_SPAN: radian = math.radians(75.0)  # cap on how far the fill reaches from a pillar

    # --- Colour inspection (drive at an unknown pillar so the camera can read it) ---
    # The camera only classifies a pillar near the image centre (~±26° FOV), so an
    # uncoloured pillar ahead is actively inspected: face it and creep straight at
    # it, which centres it in the frame; stop at a standoff and hold until the
    # colour is decided (or a timeout gives up so we never hang forever).
    _INSPECT_RANGE:      mm     = 1400.0           # start inspecting an uncoloured pillar within this
    _INSPECT_CONE:       radian = math.radians(70.0)  # only inspect a pillar roughly ahead
    _INSPECT_STANDOFF:   mm     = 450.0            # creep to here, then stop and stare
    _INSPECT_SPEED:      float  = 60.0             # slow approach speed while inspecting
    _INSPECT_TIMEOUT:    second = 4.0              # give up and drive on if colour never resolves

    # --- Forward window (rear mask, §5 of the research doc) ---
    _FORWARD_HALF:    radian = math.radians(100.0) # keep bins within ± of travel dir
    _TRAVEL_STEP:     mm     = 30.0                # breadcrumb spacing
    _TRAVEL_SPAN:     mm     = 300.0               # tangent measured over this distance
    _TRAVEL_MIN:      mm     = 100.0               # below this, fall back to the nose

    # --- Disparity extension ---
    _DISPARITY_MM:    mm = 200.0                   # range jump treated as an edge
    _HALF_WIDTH:      mm = 130.0                   # robot half width + safety margin

    # --- Gap selection ---
    _CLEAR_MIN:       mm     = 350.0               # bin depth needed to count as free
    _PLATEAU_RATIO:   float  = 0.90                # bins within this of the gap's max
                                                   # depth form the plateau; aim at its middle

    # --- Commitment / hysteresis ---
    _ANGLE_ALPHA:     float  = 0.5                 # low-pass factor toward the new angle
    _JUMP_MAX:        radian = math.radians(35.0)  # larger jumps need confirmation
    _PENDING_TOL:     radian = math.radians(10.0)  # jump candidate match tolerance

    # --- Target / speed ---
    _LOOKAHEAD_K:     float = 0.5                  # lookahead as fraction of gap depth
    _L_MIN:           mm    = 200.0
    _L_MAX:           mm    = 500.0
    _SPEED_MIN:       float = 80.0
    _SPEED_MAX:       float = 90.0

    # --- Steering feasibility ---
    # Same clamp as TrackPathPlanner: the firmware rejects targets inside the
    # minimum turning-radius circles. r_min = WHEELBASE / tan(MAX_STEER); keep in
    # sync with src/arduino/Config.h (89.6251 mm / tan(0.4014 rad) = 211 mm).
    _MIN_TURN_RADIUS:    mm    = 211.0
    _TURN_RADIUS_MARGIN: float = 1.15
    _MIN_FORWARD:        mm    = 50.0

    # --- Collision guard ---
    _COLLISION_ACTIVE_S: second = 0.8

    def __init__(
        self,
        field_map:          FieldMap,
        ego_information:    EgoInformation,
        lidar_offset_angle: degree,
        lidar_mount_offset: Point = (0.0, 0.0),
        use_memory:         bool  = True,
        use_pass_side:      bool  = True,
    ) -> None:
        super().__init__()
        self._field_map:       FieldMap       = field_map
        self._ego_information: EgoInformation = ego_information
        self._offset_rad:      float          = math.radians(lidar_offset_angle)
        self._mount_offset:    Point          = lidar_mount_offset
        self._use_memory:      bool           = use_memory
        self._use_pass_side:   bool           = use_pass_side

        # Sample distances for the memory raycast, one grid cell apart.
        self._mem_dists: np.ndarray = np.arange(
            self._MEMORY_STEP, self._MEMORY_MAX + self._MEMORY_STEP, self._MEMORY_STEP
        )

        self.on_target: Event = Event()
        self.on_stop:   Event = Event()
        self.on_debug:  Event = Event()

        self._bin_width:   float      = 2.0 * math.pi / self._BIN_COUNT
        self._bin_angles:  np.ndarray = (np.arange(self._BIN_COUNT) + 0.5) \
                                        * self._bin_width - math.pi

        self._breadcrumbs: list[Point] = []
        self._committed:   float | None = None   # committed steering angle (body)
        self._pending:     float | None = None   # unconfirmed large-jump candidate

        self._stopped: bool = False
        self._last_collision_t: float = -math.inf

        self._inspect_start: float | None = None   # monotonic time inspection began

    # ------------------------------------------------------------------ #
    #  Top-level loop                                                     #
    # ------------------------------------------------------------------ #

    def _process(self, scan: np.ndarray) -> None:
        if self._stopped or len(scan) == 0:
            return

        pos, yaw = self._ego_information.get_ego_information()
        if pos is None or yaw is None:
            return

        laps = self._field_map.update_lap_progress(pos)
        if laps >= FieldMap.TOTAL_LAPS:
            self._stopped = True
            self.on_target(0.0, 0.0, 0.0)
            self.on_stop()
            return

        if self._collision_active():
            self.on_target(0.0, 0.0, 0.0)
            return

        travel_body = self._travel_direction(pos, yaw)

        ranges, unknown = self._bin_scan(scan)

        # Fuse the accumulated occupancy grid: walls the current scan dropped are
        # restored from memory (taking the nearer of live/memory per bin), and a
        # bin memory confirms is no longer "unknown" — so a dropout phantom can
        # no longer masquerade as the deepest gap.
        memory = None
        if self._use_memory:
            memory  = self._memory_scan(pos, yaw)
            ranges  = np.minimum(ranges, memory)
            unknown = unknown & ~np.isfinite(memory)

        # Close the wrong side of every coloured pillar ahead so the gap search can
        # only route the mandated way.
        pass_masked = None
        obstacles   = None
        if self._use_pass_side:
            ranges, unknown, pass_masked, obstacles = \
                self._pass_side_mask(ranges, unknown, pos, yaw, travel_body)

            # If an uncoloured pillar is coming up, break off and inspect it: face
            # it and creep so the camera (narrow FOV) frames it head-on, stopping
            # at a standoff until the colour resolves. Overrides gap-following.
            action = self._inspection_action(obstacles)
            if action is not None:
                aim, look, spd = action
                if spd <= 0.0:
                    self.on_target(0.0, 0.0, 0.0)
                else:
                    self._emit(aim, look, spd)
                self.on_debug(self._debug_dict(
                    ranges, memory, unknown, pass_masked, obstacles,
                    rel = np.empty(0), ext = np.empty(0), travel_body = travel_body,
                    chosen = aim, lookahead = look, speed = spd, laps = laps,
                    inspecting = True,
                ))
                return

        # Roll the circular bins so the forward window is one contiguous slice
        # centred on the travel direction. rel[i] is each kept bin's angle
        # relative to travel_body, ascending.
        rel_all  = self._wrap_array(self._bin_angles - travel_body)
        order    = np.argsort(rel_all)
        rel      = rel_all[order]
        win      = np.abs(rel) <= self._FORWARD_HALF
        rel      = rel[win]
        win_rng  = ranges[order][win]
        known    = ~unknown[order][win]

        if len(win_rng) == 0:
            self.on_target(0.0, 0.0, 0.0)
            return

        ext = self._extend_disparities(win_rng)

        best_idx = self._select_ray(rel, ext, known)
        a_new    = travel_body + rel[best_idx]          # body-frame steering angle
        a_cmd    = self._commit(a_new, rel, ext, travel_body)

        # Depth actually available in the committed direction.
        cmd_idx  = int(np.argmin(np.abs(rel - (a_cmd - travel_body))))
        depth    = float(ext[cmd_idx])

        lookahead = min(max(self._LOOKAHEAD_K * depth, self._L_MIN), self._L_MAX)
        speed     = self._speed(depth, a_cmd)

        self._emit(a_cmd, lookahead, speed)

        self.on_debug(self._debug_dict(
            ranges, memory, unknown, pass_masked, obstacles,
            rel = rel, ext = ext, travel_body = travel_body,
            chosen = a_cmd, lookahead = lookahead, speed = speed, laps = laps,
            inspecting = False,
        ))

    def _debug_dict(self, ranges, memory, unknown, pass_masked, obstacles,
                    *, rel, ext, travel_body, chosen, lookahead, speed, laps,
                    inspecting) -> dict:
        return {
            "bin_angles":  self._bin_angles,
            "ranges":      ranges,
            "memory":      memory,
            "unknown":     unknown,
            "pass_masked": pass_masked,
            "obstacles":   obstacles,
            "rel":         rel,
            "extended":    ext,
            "travel_body": travel_body,
            "chosen":      chosen,
            "lookahead":   lookahead,
            "speed":       speed,
            "laps":        laps,
            "inspecting":  inspecting,
        }

    def notify_collision(self) -> None:
        self._last_collision_t = time.monotonic()

    def _collision_active(self) -> bool:
        return (time.monotonic() - self._last_collision_t) < self._COLLISION_ACTIVE_S

    # ------------------------------------------------------------------ #
    #  Travel direction (rear mask reference)                             #
    # ------------------------------------------------------------------ #

    def _travel_direction(self, pos: Point, yaw: radian) -> radian:
        # Body-frame bearing of the short-horizon travel tangent: where the robot
        # has actually been going, from odometry breadcrumbs. Robust to the nose
        # swinging mid-corner, and to slow EKF drift (yaw and breadcrumbs drift
        # together, so their difference stays sound). Before the robot has moved
        # _TRAVEL_MIN it falls back to the nose (the robot is placed pointing
        # down the corridor at start).
        crumbs = self._breadcrumbs
        if not crumbs or math.hypot(pos[0] - crumbs[-1][0],
                                    pos[1] - crumbs[-1][1]) >= self._TRAVEL_STEP:
            crumbs.append(pos)
            max_len = int(self._TRAVEL_SPAN / self._TRAVEL_STEP) + 1
            if len(crumbs) > max_len:
                del crumbs[: len(crumbs) - max_len]

        dn = pos[0] - crumbs[0][0]
        de = pos[1] - crumbs[0][1]
        if math.hypot(dn, de) < self._TRAVEL_MIN:
            return 0.0
        return self._wrap(math.atan2(de, dn) - yaw)

    # ------------------------------------------------------------------ #
    #  Scan processing                                                    #
    # ------------------------------------------------------------------ #

    def _bin_scan(self, scan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Fixed angular bins over the full circle, min range per bin (the
        # conservative choice). Returns (ranges, unknown): unknown marks bins
        # whose depth is a guess (no return, not corroborated by neighbours).
        angles = self._wrap_array(scan[:, 0] + self._offset_rad)
        dists  = scan[:, 1]

        valid  = dists > 0.0
        angles = angles[valid]
        dists  = np.minimum(dists[valid], self._R_MAX)

        idx = ((angles + math.pi) / self._bin_width).astype(int) % self._BIN_COUNT

        ranges = np.full(self._BIN_COUNT, np.inf)
        np.minimum.at(ranges, idx, dists)

        empty = ~np.isfinite(ranges)
        if empty.any():
            return self._fill_holes(ranges, empty)
        return ranges, np.zeros(self._BIN_COUNT, dtype = bool)

    def _fill_holes(
        self,
        ranges: np.ndarray,
        empty:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Every ray on a walled field hits *something* within LiDAR range, so a
        # missing return is a sensing failure, not open space. Three cases:
        #   - short run (sparse sampling): take the nearer neighbour, as before;
        #   - long run bounded by returns at similar range: a poorly reflecting
        #     continuous surface (matte black wall) — bridge it by interpolating
        #     between the bounds;
        #   - long run bordering a genuine disparity: possibly a real opening —
        #     keep it at _R_MAX but mark it unknown, so gap selection prefers
        #     corroborated gaps over this phantom.
        n = self._BIN_COUNT
        if empty.all():
            return np.full(n, self._R_MAX), np.ones(n, dtype = bool)

        # Rotate so index 0 is a measured bin: every empty run is then interior
        # and circularly bounded by returns on both sides (no seam special case).
        start = int(np.argmax(~empty))
        out   = np.roll(ranges, -start)
        gaps  = np.roll(empty, -start)
        holes = np.zeros(n, dtype = bool)

        i = 0
        while i < n:
            if not gaps[i]:
                i += 1
                continue
            j = i
            while j < n and gaps[j]:
                j += 1
            run   = j - i
            left  = out[i - 1]
            right = out[j % n]
            if run <= self._MAX_HOLE_BINS:
                out[i:j] = min(left, right)
            elif abs(left - right) < self._DISPARITY_MM:
                out[i:j] = np.linspace(left, right, run + 2)[1:-1]
            else:
                out[i:j]   = self._R_MAX
                holes[i:j] = True
            i = j

        return np.roll(out, start), np.roll(holes, start)

    def _memory_scan(self, pos: Point, yaw: radian) -> np.ndarray:
        # Raycast the accumulated occupancy grid into the same body-frame bins as
        # the live scan: per bin, the distance to the first cell above _MEMORY_OCC
        # log-odds, or +inf if none within _MEMORY_MAX. Uses the world transform
        # inverse to LidarProcessor's (east = E + r·sin(a+yaw), north = N +
        # r·cos(a+yaw)), so bin i here and bin i of _bin_scan share a body angle.
        bearings = self._bin_angles + yaw                      # (B,) world bearings
        east  = pos[1] + np.outer(np.sin(bearings), self._mem_dists)   # (B, S)
        north = pos[0] + np.outer(np.cos(bearings), self._mem_dists)

        cols = ((east  + FieldMap.ORIGIN[0]) // FieldMap.CELL_SIZE).astype(int)
        rows = ((north + FieldMap.ORIGIN[1]) // FieldMap.CELL_SIZE).astype(int)

        occ = self._field_map.sample_cells(rows, cols)         # (B, S), NaN = unknown/oob
        hit = occ > self._MEMORY_OCC                           # NaN compares False

        any_hit = hit.any(axis = 1)
        first   = hit.argmax(axis = 1)
        return np.where(any_hit, self._mem_dists[first], np.inf)

    def _pass_side_mask(
        self,
        ranges:      np.ndarray,
        unknown:     np.ndarray,
        pos:         Point,
        yaw:         radian,
        travel_body: radian,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # For every coloured pillar in the forward window, fill the gap on its
        # forbidden side (between the pillar and the wall there) with the pillar's
        # range, so the deepest surviving gap is on the mandated side and the gap
        # search steers that way. Sides are taken relative to the travel tangent,
        # so they don't flip as the nose swings past the pillar. Returns the
        # possibly-modified (ranges, unknown), the masked-bin mask (for debug), and
        # an (N, 3) debug array of every known obstacle as (bearing, range,
        # colour_code) — 0 none / 1 red / 2 green — so the visualizer and recording
        # show what detection produced. Uncoloured pillars are left to the caller's
        # inspection behaviour.
        masked    = np.zeros(self._BIN_COUNT, dtype = bool)
        rel_all   = self._wrap_array(self._bin_angles - travel_body)
        obs_debug: list[tuple[float, float, int]] = []

        for obs in self._field_map.get_obstacles():
            # Body-frame bearing and range of the pillar centroid. centroid is
            # world [east, north]; ego pos is (north, east). Inverse of the
            # LidarProcessor transform: bearing = atan2(dEast, dNorth) - yaw.
            d_east  = float(obs.centroid[0]) - pos[1]
            d_north = float(obs.centroid[1]) - pos[0]
            r_obs   = math.hypot(d_east, d_north)
            bearing = self._wrap(math.atan2(d_east, d_north) - yaw)
            code    = 0 if obs.color is None else (1 if obs.color == ObstacleColor.RED else 2)
            obs_debug.append((bearing, r_obs, code))

            if obs.color is None or r_obs < 1.0 or r_obs > self._PASS_RANGE:
                continue

            rel_c = self._wrap(bearing - travel_body)
            if abs(rel_c) > self._FORWARD_HALF:      # not ahead of us
                continue

            half        = math.atan2(self._PASS_HALF_WIDTH, r_obs)
            forbid_left = obs.color == ObstacleColor.RED   # red -> pass right -> block left of pillar

            # Cap every bin on the forbidden side of the pillar (within the span)
            # that is *deeper* than the pillar down to the pillar's range. No gap
            # on the wrong side can then be the deepest, so the mandated side wins
            # the gap search. We cap (not block) and never touch nearer bins, so a
            # genuinely closer wall stays put and the wrong side is only
            # de-prioritised, not sealed. The earlier version scanned outward and
            # stopped at the first nearer bin, which fired almost never on a real
            # track where the side wall is closer than the pillar.
            from_pillar = np.abs(self._wrap_array(rel_all - rel_c))
            side = (rel_all < rel_c - half) if forbid_left else (rel_all > rel_c + half)
            sel  = side & (from_pillar <= self._PASS_MASK_MAX_SPAN) & (ranges > r_obs)
            ranges[sel]  = r_obs
            unknown[sel] = False
            masked |= sel

        obstacles = np.array(obs_debug, dtype = float) if obs_debug \
                    else np.empty((0, 3), dtype = float)
        return ranges, unknown, masked, obstacles

    def _inspection_action(
        self,
        obstacles: np.ndarray | None,
    ) -> tuple[radian, mm, float] | None:
        # Decide whether to inspect an uncoloured pillar coming up, and how. Returns
        # (aim bearing, lookahead, speed) — speed 0 meaning "stop and stare" — or
        # None to let normal gap-following run. Only the nearest pillar ahead within
        # the inspection cone/range counts: if it already has a colour there is
        # nothing to inspect, and a per-pillar timeout stops us hanging forever if
        # the colour never resolves.
        if obstacles is None or len(obstacles) == 0:
            self._inspect_start = None
            return None

        ahead = obstacles[(np.abs(obstacles[:, 0]) < self._INSPECT_CONE) &
                          (obstacles[:, 1] < self._INSPECT_RANGE)]
        if len(ahead) == 0:
            self._inspect_start = None
            return None

        bearing, r_obs, code = ahead[int(np.argmin(ahead[:, 1]))]
        if int(code) != 0:                      # nearest pillar ahead is already coloured
            self._inspect_start = None
            return None

        now = time.monotonic()
        if self._inspect_start is None:
            self._inspect_start = now
        elif now - self._inspect_start > self._INSPECT_TIMEOUT:
            return None                         # gave up; drive on (keep timer so we don't retrigger)

        if r_obs <= self._INSPECT_STANDOFF:
            return (float(bearing), self._L_MIN, 0.0)
        lookahead = min(max(r_obs - self._INSPECT_STANDOFF, self._L_MIN), self._L_MAX)
        return (float(bearing), float(lookahead), self._INSPECT_SPEED)

    def _extend_disparities(self, ranges: np.ndarray) -> np.ndarray:
        # At every range discontinuity, overwrite the far-side bins with the near
        # range across the angular width the robot's half width subtends at that
        # range — scan-space obstacle inflation. A surviving deep ray is then a
        # direction the whole robot fits through, so aiming at the deepest point
        # is safe without any grid.
        ext = ranges.copy()
        n   = len(ext)
        for i in range(1, n):
            near = min(ranges[i - 1], ranges[i])
            if abs(ranges[i] - ranges[i - 1]) < self._DISPARITY_MM or near <= 0.0:
                continue
            span = int(math.ceil(math.atan2(self._HALF_WIDTH, near) / self._bin_width))
            if ranges[i] > ranges[i - 1]:          # edge on the left of bin i
                hi = min(n, i + span)
                np.minimum(ext[i:hi], near, out = ext[i:hi])
            else:                                  # edge on the right of bin i-1
                lo = max(0, i - span)
                np.minimum(ext[lo:i], near, out = ext[lo:i])
        return ext

    def _select_ray(self, rel: np.ndarray, ext: np.ndarray, known: np.ndarray) -> int:
        # Split free bins (deep enough after extension) into contiguous gaps and
        # take the gap with the deepest *corroborated* reach — depth in unknown
        # bins (dropout sectors) counts for nothing, so a phantom gap can only
        # win when no measured gap qualifies at all. Within the winning gap, aim
        # at the middle of the near-max plateau of measured bins — the deepest
        # corroborated direction without hugging either edge. With no qualifying
        # gap, creep toward the single deepest bin (the corridor is never truly
        # closed; this rides out a bad scan without inventing a reverse behavior).
        free = ext >= self._CLEAR_MIN
        if not free.any():
            return int(np.argmax(ext))

        edges  = np.flatnonzero(np.diff(free.astype(int)))
        starts = np.concatenate(([0] if free[0] else [], edges[~free[edges]] + 1)).astype(int)
        ends   = np.concatenate((edges[free[edges]] + 1, [len(free)] if free[-1] else [])).astype(int)

        def corroborated_depth(se: tuple[int, int]) -> float:
            gap_known = known[se[0]:se[1]]
            if not gap_known.any():
                return 0.0
            return float(ext[se[0]:se[1]][gap_known].max())

        best = max(zip(starts, ends), key = corroborated_depth)
        lo, hi = best

        gap       = ext[lo:hi]
        gap_known = known[lo:hi]
        depths    = np.where(gap_known, gap, -np.inf) if gap_known.any() else gap
        plateau   = np.flatnonzero(depths >= self._PLATEAU_RATIO * depths.max())
        return lo + int(plateau[(len(plateau) - 1) // 2])

    # ------------------------------------------------------------------ #
    #  Commitment                                                         #
    # ------------------------------------------------------------------ #

    def _commit(
        self,
        a_new:       radian,
        rel:         np.ndarray,
        ext:         np.ndarray,
        travel_body: radian,
    ) -> radian:
        # Small changes are low-passed; a large jump (a different gap winning)
        # must win two consecutive scans before it is accepted. If the committed
        # direction itself has become blocked, safety overrides commitment and
        # the new winner is taken immediately.
        if self._committed is None:
            self._committed = a_new
            return a_new

        committed_idx   = int(np.argmin(np.abs(rel - (self._committed - travel_body))))
        committed_depth = float(ext[committed_idx])
        blocked         = committed_depth < 0.8 * self._CLEAR_MIN

        delta = self._wrap(a_new - self._committed)
        if blocked or abs(delta) <= self._JUMP_MAX:
            self._committed = self._wrap(
                self._committed + self._ANGLE_ALPHA * delta
                if not blocked else a_new
            )
            self._pending = None
        elif self._pending is not None \
                and abs(self._wrap(a_new - self._pending)) <= self._PENDING_TOL:
            self._committed = a_new
            self._pending   = None
        else:
            self._pending = a_new
        return self._committed

    # ------------------------------------------------------------------ #
    #  Emit                                                               #
    # ------------------------------------------------------------------ #

    def _speed(self, depth: mm, angle: radian) -> float:
        straight = max(0.0, math.cos(angle))
        deep     = min(max((depth - self._CLEAR_MIN)
                           / (self._R_MAX / 2.0 - self._CLEAR_MIN), 0.0), 1.0)
        return self._SPEED_MIN + (self._SPEED_MAX - self._SPEED_MIN) * straight * deep

    def _emit(self, angle: radian, lookahead: mm, speed: float) -> None:
        # Target is measured from the LiDAR; shift by the mount offset so the
        # emitted point is in the robot body frame the firmware expects.
        forward = lookahead * math.cos(angle) + self._mount_offset[1]
        lateral = lookahead * math.sin(angle) + self._mount_offset[0]
        forward, lateral = self._make_reachable(forward, lateral)
        self.on_target(forward, lateral, speed)

    def _make_reachable(self, forward: float, lateral: float) -> tuple[float, float]:
        # Same clamp as TrackPathPlanner._make_reachable: pull targets that would
        # demand a turn tighter than the car's minimum radius onto the near
        # turning circle, and keep a little forward so (0, 0) is never sent by
        # accident (the firmware reads that as "arrived").
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

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wrap(angle: radian) -> radian:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _wrap_array(angles: np.ndarray) -> np.ndarray:
        return (angles + math.pi) % (2 * math.pi) - math.pi
