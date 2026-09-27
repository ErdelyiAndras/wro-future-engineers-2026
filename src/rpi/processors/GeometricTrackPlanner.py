from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from processors.Processor import Processor
from control.FieldMap import FieldMap, CellLabel, Direction, Obstacle, ObstacleColor
from control.EgoInformation import EgoInformation
from utils import mm, radian, second, Point, Event

Cell = tuple[int, int]


@dataclass
class _RectModel:
    """Partially-known axis-aligned rectangle in the track's aligned frame.

    All coordinates are in mm, measured in the rotated frame where u runs
    along the track's dominant axis (theta) and v runs perpendicular.
    Each bound is None until that wall has been observed or inferred.
    """
    u_lo: float | None = None
    u_hi: float | None = None
    v_lo: float | None = None
    v_hi: float | None = None

    def is_complete(self) -> bool:
        return all(x is not None for x in (self.u_lo, self.u_hi, self.v_lo, self.v_hi))

    def center(self) -> tuple[float, float] | None:
        if not self.is_complete():
            return None
        return ((self.u_lo + self.u_hi) / 2.0,   # type: ignore[operator]
                (self.v_lo + self.v_hi) / 2.0)    # type: ignore[operator]


class GeometricTrackPlanner(Processor):
    """Path planner that estimates the full track layout from partial observations.

    The WRO track is two concentric rectangles:
        outer – always 3000 × 3000 mm
        inner – one of {1000, 1400, 1800} mm per side (variable per round)

    From partial wall observations the planner:
      1. Estimates the track's axis orientation via a structure tensor.
      2. Separates observed wall cells into two perpendicular families (H / V).
      3. Within each family clusters cells into line segments and reads their
         perpendicular offsets.
      4. Pairs offsets separated by ~3000 mm → outer rectangle sides; pairs
         separated by a valid inner size → inner rectangle sides.
      5. With one outer pair known the opposite sides are completed from the
         known 3000 mm extent; a single observed side is anchored against the
         robot's current position to infer the opposite side.
      6. Rasterises the estimated walls onto the coarse grid, masked to
         unobserved cells only, so real measurements always take priority.
      7. Plans on the union of measured and estimated walls (same directed
         ray-cast + A* as DirectionalPathPlanner).
    """

    # ------------------------------------------------------------------ #
    #  Feature switches                                                   #
    # ------------------------------------------------------------------ #
    _OBSTACLE_RULES_ENABLED: bool = False
    _INSPECT_ENABLED:        bool = False
    _REVERSE_ENABLED:        bool = False

    # ------------------------------------------------------------------ #
    #  Coarse grid / clearance                                            #
    # ------------------------------------------------------------------ #
    _K:                   int   = 5
    _OCCUPANCY_THRESHOLD: float = 0.0
    _WALL_INFLATION:      mm    = 170.0
    _OBSTACLE_INFLATION:  mm    = 50.0

    # ------------------------------------------------------------------ #
    #  Goal-selection ray cast                                            #
    # ------------------------------------------------------------------ #
    _HALF_ANGLE:  radian = math.radians(80.0)
    _N_RAYS:      int    = 17
    _L_MIN:       mm     = 100.0
    _L_MAX:       mm     = 500.0
    _W_DIST:      float  = 2.0
    _W_CLEAR:     float  = 1.0
    _K_ADAPTIVE:  float  = 3.0
    # Rays within this band of forward may turn either way; beyond it they
    # must turn in the allowed direction.
    _CENTER_BAND: radian = math.radians(25.0)

    # ------------------------------------------------------------------ #
    #  A*                                                                 #
    # ------------------------------------------------------------------ #
    _TARGET_CLEARANCE: mm    = 400.0
    _CLEARANCE_WEIGHT: float = 13.0
    _TURN_PENALTY:     float = 30.0

    # ------------------------------------------------------------------ #
    #  Wrong-side obstacle block                                          #
    # ------------------------------------------------------------------ #
    _BLOCK_REACH:           mm  = 700.0
    _BLOCK_WINDOW:          mm  = 100.0
    _BLOCK_BACK:            mm  = 50.0
    _BLOCK_ARM_THICKNESS:   mm  = 150.0
    _BLOCK_CLEARANCE:       mm  = 30.0
    _WALL_ORIENT_MIN_CELLS: int = 60

    # ------------------------------------------------------------------ #
    #  Route / speed                                                      #
    # ------------------------------------------------------------------ #
    _L_LOOKAHEAD:           mm     = 450.0
    _SPEED:                 float  = 70.0
    _FOLLOW_SPEED:          float  = 100.0
    _ROUTE_MIN_SPACING:     mm     = 50.0
    _MIN_ROUTE_POINTS:      int    = 20
    _STOP_RADIUS:           mm     = 200.0
    _REVERSE_SPEED:         float  = 60.0
    _COLLISION_ACTIVE_S:    second = 0.8
    _TRAVEL_TANGENT_POINTS: int    = 5
    _RECOVER_TURN:          radian = math.radians(45.0)
    _RECOVER_SPEED:         float  = 50.0

    # ------------------------------------------------------------------ #
    #  Active colour inspection                                           #
    # ------------------------------------------------------------------ #
    _INSPECT_RANGE:      mm     = 1500.0
    _INSPECT_HALF_ANGLE: radian = math.radians(60.0)
    _INSPECT_STANDOFF:   mm     = 400.0
    _INSPECT_SPEED:      float  = 60.0
    _INSPECT_TIMEOUT:    second = 3.0
    _INSPECT_MATCH:      mm     = 100.0

    # ------------------------------------------------------------------ #
    #  Track geometry estimation                                          #
    # ------------------------------------------------------------------ #
    OUTER_SIZE:      mm    = 3000.0                    # outer rectangle side length
    INNER_SIZES:     tuple = (1000.0, 1400.0, 1800.0) # valid inner wall sizes
    _PAIR_TOL:          mm    = 200.0  # tolerance when matching a pair's separation
    _LINE_MIN_CELLS:    int   = 8      # min coarse cells to form a wall-line cluster
    _CLUSTER_GAP:       mm    = 180.0  # projection gap larger than this splits clusters
    _EST_WALL_HW:       mm    = 12.5   # half-width of a rasterised estimated wall
    _OUTER_STABLE_N:    int   = 5      # consecutive stable frames before locking
    _OUTER_STABLE_TOL:  mm    = 50.0   # per-bound tolerance for "same" estimate
    _OUTER_FILTER_TOL:  mm    = 300.0  # cells within this distance of outer wall are excluded from inner estimation
    _EST_ABSORB_TOL:    mm    = 150.0  # real wall cells within this distance of any estimated wall are suppressed in planning
    _W_OUTER_WALL:      float = 2.0   # phase-1 A* outer-wall bias (distance penalty)
    _OUTER_WALL_OFFSET: mm    = 150.0  # phase-2 distance from outer wall to follow

    # ------------------------------------------------------------------ #
    #  Init                                                               #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        field_map:       FieldMap,
        ego_information: EgoInformation,
    ) -> None:
        super().__init__()
        self._field_map       = field_map
        self._ego_information = ego_information
        self.on_target:        Event = Event()
        self.on_next_obstacle: Event = Event()
        self.on_route:         Event = Event()
        self.on_stop:          Event = Event()
        self.on_plan_debug:    Event = Event()

        self._cached_blocked_walls: np.ndarray | None = None
        self._cached_D_mm:          np.ndarray | None = None
        self._cached_obstacle:      np.ndarray | None = None
        self._cached_D_obs:         np.ndarray | None = None
        self._cached_D_outer:       np.ndarray | None = None

        # Track orientation (shared with the obstacle-block wall-snap logic)
        self._wall_theta: float | None = None

        self._following:    bool        = False
        self._stopped:      bool        = False
        self._route:        list[Point] = []
        self._route_cursor: int         = 0

        self._last_collision_t: float = -math.inf

        self._inspect_start:  float | None = None
        self._inspect_anchor: Point | None = None

        # Persistent geometry model, updated incrementally each planning cycle
        self._track_theta:        float | None            = None
        self._outer_model:        _RectModel              = _RectModel()
        self._outer_locked:       bool                    = False
        self._outer_stable_count: int                     = 0
        self._inner_model:        _RectModel              = _RectModel()
        self._last_estimated:     np.ndarray | None       = None
        self._last_intersections: list[tuple[float, float]] = []

    # ------------------------------------------------------------------ #
    #  Top-level state machine                                            #
    # ------------------------------------------------------------------ #

    def _process(self) -> None:
        if self._stopped:
            return

        pos, yaw = self._ego_information.get_ego_information()
        if pos is None or yaw is None:
            return

        laps = self._field_map.update_lap_progress(pos)

        if laps >= FieldMap.TOTAL_LAPS:
            self._home_and_stop(pos, yaw)
            return

        if self._following:
            target = self._follow_route(pos)
            if target is not None:
                self._emit_target(pos, yaw, target)
            return

        if laps >= 1:
            self._finalize_route(pos)
            self._following = True
            target = self._follow_route(pos)
            if target is not None:
                self._emit_target(pos, yaw, target)
            return

        self._directed_step(pos, yaw)

    def notify_collision(self) -> None:
        self._last_collision_t = time.monotonic()

    def _collision_active(self) -> bool:
        return (time.monotonic() - self._last_collision_t) < self._COLLISION_ACTIVE_S

    def _allowed_sign(self) -> float | None:
        direction = self._field_map.direction
        if direction == Direction.CW:
            return +1.0
        if direction == Direction.CCW:
            return -1.0
        return None

    def _directed_step(self, pos: Point, yaw: radian) -> None:
        if self._INSPECT_ENABLED and self._inspect_obstacle(pos, yaw):
            return

        target, next_obstacle, status = self._plan(pos, yaw)
        self.on_next_obstacle(next_obstacle)

        collision = self._REVERSE_ENABLED and self._collision_active()
        if target is not None and not collision:
            self._record_pose(pos)
            self._emit_target(pos, yaw, target)
            return

        if status == "recover" or collision:
            if self._REVERSE_ENABLED:
                rev_target = self._reverse_target(pos)
                if rev_target is not None:
                    self._emit_target(pos, yaw, rev_target, reverse=True)
                    return
            self._recover(pos, yaw)
            return

        self.on_target(0.0, 0.0, 0.0)

    def _recover(self, pos: Point, yaw: radian) -> None:
        sign = self._allowed_sign()
        if sign is None:
            self.on_target(0.0, 0.0, 0.0)
            return
        aim_yaw = yaw + sign * self._RECOVER_TURN
        aim     = (pos[0] + self._L_MIN * math.cos(aim_yaw),
                   pos[1] + self._L_MIN * math.sin(aim_yaw))
        self._emit_target(pos, yaw, aim, speed=self._RECOVER_SPEED)

    def _reverse_target(self, pos: Point) -> Point | None:
        while len(self._route) > 1 and \
              self._dist(pos, self._route[-1]) < self._ROUTE_MIN_SPACING:
            self._route.pop()
        return self._route[-1] if self._route else None

    # ------------------------------------------------------------------ #
    #  Track geometry estimation                                          #
    # ------------------------------------------------------------------ #

    def _update_track_model(
        self,
        semantic: np.ndarray,
        pos:      Point,
        yaw:      float = 0.0,
    ) -> None:
        """Update the outer/inner rectangle models from the current semantic grid.

        Orientation is estimated from a Gaussian-blurred coarse wall mask so
        that the structure tensor sees a smooth diagonal ramp rather than a
        staircase.  Any discrete binary wall mask — coarse or fine — has only
        axis-aligned edge steps, which all give exp(4j·φ) = 1 and force θ = 0.
        Blurring the mask first converts the staircase into a smooth gradient
        field whose direction correctly tracks the true wall angle.

        H/V classification and offset clustering use the sharp unblurred mask
        for precise wall positions.
        """
        K    = self._K
        rows = semantic.shape[0] // K
        cols = semantic.shape[1] // K
        sem  = semantic[:rows * K, :cols * K]

        coarse_wall = (sem == int(CellLabel.WALL)).reshape(
            rows, K, cols, K
        ).any(axis=(1, 3))

        # --- Orientation from blurred mask ---
        blurred        = gaussian_filter(coarse_wall.astype(np.float32), sigma=3.0)
        gr_b, gc_b     = np.gradient(blurred)
        mag2_b         = gr_b ** 2 + gc_b ** 2
        edge_b         = mag2_b > 1e-6
        if int(edge_b.sum()) < self._WALL_ORIENT_MIN_CELLS:
            return

        phi = np.arctan2(gc_b[edge_b], gr_b[edge_b])
        z   = np.sum(mag2_b[edge_b] * np.exp(4j * phi))
        if abs(z) < 1e-9:
            return
        theta = float(np.angle(z) / 4.0)
        self._track_theta = theta
        self._wall_theta  = theta   # shared with wrong-side-block snap

        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # --- H/V classification and offset clustering from sharp mask ---
        gr, gc = np.gradient(coarse_wall.astype(np.float32))
        mag2   = gr * gr + gc * gc
        edge   = mag2 > 1e-6

        edge_rows, edge_cols = np.where(edge)
        eg = gr[edge].copy()
        ec = gc[edge].copy()
        emag = np.sqrt(mag2[edge])
        eg /= emag
        ec /= emag

        # u-axis unit vector in (row, col): (cos_t, sin_t)
        # v-axis unit vector in (row, col): (-sin_t, cos_t)
        dot_u = eg * cos_t + ec * sin_t
        dot_v = eg * (-sin_t) + ec * cos_t
        is_v_wall = np.abs(dot_u) >= np.abs(dot_v)

        cell_mm = K * FieldMap.CELL_SIZE
        north_e = edge_rows * cell_mm - FieldMap.ORIGIN[1]
        east_e  = edge_cols * cell_mm - FieldMap.ORIGIN[0]

        u_e =  north_e * cos_t + east_e * sin_t
        v_e = -north_e * sin_t + east_e * cos_t

        v_wall_offsets = self._find_line_offsets(u_e[is_v_wall])
        h_wall_offsets = self._find_line_offsets(v_e[~is_v_wall])

        # For each wall, compute the mean coordinate of its cells in the
        # perpendicular direction.  This tells us which side of each intersection
        # corner the wall actually runs toward — used instead of robot position.
        assign_tol = self._CLUSTER_GAP
        v_wall_centers: dict[float, float] = {}
        for u0 in v_wall_offsets:
            sel = is_v_wall & (np.abs(u_e - u0) < assign_tol)
            if sel.any():
                v_wall_centers[u0] = float(np.mean(v_e[sel]))

        h_wall_centers: dict[float, float] = {}
        for v0 in h_wall_offsets:
            sel = (~is_v_wall) & (np.abs(v_e - v0) < assign_tol)
            if sel.any():
                h_wall_centers[v0] = float(np.mean(u_e[sel]))

        # Robot position kept as a fallback when no cell-center data is available.
        robot_u =  pos[0] * cos_t + pos[1] * sin_t
        robot_v = -pos[0] * sin_t + pos[1] * cos_t

        intersections = [(u, v) for u in v_wall_offsets for v in h_wall_offsets]
        self._last_intersections = intersections
        self._fit_outer(intersections, robot_u, robot_v, v_wall_centers, h_wall_centers, yaw)

        # Inner wall estimation: only after outer is locked so filtering is reliable
        if self._outer_locked:
            om  = self._outer_model
            tol = self._OUTER_FILTER_TOL
            is_outer = np.zeros(len(u_e), dtype=bool)
            if om.u_lo is not None: is_outer |= np.abs(u_e - om.u_lo) < tol
            if om.u_hi is not None: is_outer |= np.abs(u_e - om.u_hi) < tol
            if om.v_lo is not None: is_outer |= np.abs(v_e - om.v_lo) < tol
            if om.v_hi is not None: is_outer |= np.abs(v_e - om.v_hi) < tol
            iu_e  = u_e[~is_outer]
            iv_e  = v_e[~is_outer]
            iis_v = is_v_wall[~is_outer]

            iv_offsets = self._find_line_offsets(iu_e[iis_v])
            ih_offsets = self._find_line_offsets(iv_e[~iis_v])

            iv_centers: dict[float, float] = {}
            for u0 in iv_offsets:
                sel = iis_v & (np.abs(iu_e - u0) < assign_tol)
                if sel.any():
                    iv_centers[u0] = float(np.mean(iv_e[sel]))

            ih_centers: dict[float, float] = {}
            for v0 in ih_offsets:
                sel = (~iis_v) & (np.abs(iv_e - v0) < assign_tol)
                if sel.any():
                    ih_centers[v0] = float(np.mean(iu_e[sel]))

            self._fit_inner(iv_offsets, ih_offsets, iv_centers, ih_centers)

    def _find_line_offsets(
        self,
        projections: np.ndarray,
        min_cells:   int | None = None,
    ) -> list[float]:
        """Cluster a 1D array of projected wall-cell positions into line offsets.

        Splits at gaps larger than _CLUSTER_GAP; keeps only clusters with at
        least min_cells cells (defaults to _LINE_MIN_CELLS); returns the median
        of each cluster.
        """
        if len(projections) == 0:
            return []
        min_c    = min_cells if min_cells is not None else self._LINE_MIN_CELLS
        sorted_p = np.sort(projections)
        gaps     = np.diff(sorted_p)
        splits   = np.where(gaps > self._CLUSTER_GAP)[0] + 1
        offsets  = []
        for cluster in np.split(sorted_p, splits):
            if len(cluster) >= min_c:
                offsets.append(float(np.median(cluster)))
        return offsets

    def _fit_outer(
        self,
        intersections:  list[tuple[float, float]],
        robot_u:        float,
        robot_v:        float,
        v_wall_centers: dict[float, float] | None = None,
        h_wall_centers: dict[float, float] | None = None,
        yaw:            float = 0.0,
    ) -> None:
        """Estimate the outer rectangle (3000×3000 mm) from observed wall intersections.

        Only runs after track direction is determined.

        Anchor (outer corner) selection priority:
          1. Confirmed pair: two intersections sharing a wall, 3000 mm apart.
          2. Direction-based: score each candidate by its displacement in the
             outer-corner direction (right-forward for CW, left-forward for CCW).
             In the aligned (u, v) frame with heading (hu, hv):
               right = (−hv, hu)  →  outer_dir_CW  = (hu−hv, hu+hv)
               left  = (hv, −hu)  →  outer_dir_CCW = (hu+hv, hv−hu)

        Wall extension direction:
          - Mean v-position of V-wall cells at u=u0 → which side of v0 to extend.
          - Mean u-position of H-wall cells at v=v0 → which side of u0 to extend.
          Robot position is a fallback when no cell-center data is available.
        """
        if self._field_map.direction is None or self._outer_locked:
            return

        size = self.OUTER_SIZE
        tol  = self._PAIR_TOL
        vc   = v_wall_centers or {}
        hc   = h_wall_centers or {}

        # Strategy 0: per-axis pair matching — works on straights where only one
        # wall family is visible (no cross-product intersections are needed).
        # Sets individual bounds immediately without requiring a full anchor so that
        # _outer_dist_map can return a partial distance even before a corner is seen.
        v_offs = sorted(vc.keys())
        for i, u1 in enumerate(v_offs):
            for u2 in v_offs[i + 1:]:
                if abs(abs(u1 - u2) - size) < tol:
                    new_ulo, new_uhi = min(u1, u2), max(u1, u2)
                    if (self._outer_model.u_lo is None
                            or abs(new_ulo - self._outer_model.u_lo) > 1.0):
                        self._outer_model.u_lo = new_ulo
                        self._outer_model.u_hi = new_uhi
                        self._cached_D_outer   = None
                    break
        h_offs = sorted(hc.keys())
        for i, v1 in enumerate(h_offs):
            for v2 in h_offs[i + 1:]:
                if abs(abs(v1 - v2) - size) < tol:
                    new_vlo, new_vhi = min(v1, v2), max(v1, v2)
                    if (self._outer_model.v_lo is None
                            or abs(new_vlo - self._outer_model.v_lo) > 1.0):
                        self._outer_model.v_lo = new_vlo
                        self._outer_model.v_hi = new_vhi
                        self._cached_D_outer   = None
                    break

        anchor: tuple[float, float] | None = None

        # Strategy 1: confirmed pair (two intersections 3000 mm apart on same wall)
        for i, (u1, v1) in enumerate(intersections):
            for u2, v2 in intersections[i + 1:]:
                if abs(u1 - u2) < tol and abs(abs(v1 - v2) - size) < tol:
                    anchor = (u1, min(v1, v2))
                    break
                if abs(v1 - v2) < tol and abs(abs(u1 - u2) - size) < tol:
                    anchor = (min(u1, u2), v1)
                    break
            if anchor is not None:
                break

        # Strategy 2: direction-based scoring when confirmed pair not available
        if anchor is None and intersections and self._track_theta is not None:
            theta = self._track_theta
            hu = math.cos(yaw - theta)
            hv = math.sin(yaw - theta)
            direction = self._field_map.direction
            if direction == Direction.CW:
                # outer is to the right: outer_dir = heading + right
                ou, ov = hu - hv, hu + hv
            else:  # CCW
                # outer is to the left: outer_dir = heading + left
                ou, ov = hu + hv, hv - hu
            anchor = max(intersections,
                         key=lambda c: (c[0] - robot_u) * ou + (c[1] - robot_v) * ov)

        if anchor is None:
            return

        u0, v0 = anchor

        # u direction: extend toward where H-wall cells are concentrated.
        u_center = hc.get(v0)
        if u_center is not None:
            u1 = u0 + size if u_center > u0 else u0 - size
        else:
            u1 = u0 - size if u0 > robot_u else u0 + size

        # v direction: extend toward where V-wall cells are concentrated.
        v_center = vc.get(u0)
        if v_center is not None:
            v1 = v0 + size if v_center > v0 else v0 - size
        else:
            v1 = v0 - size if v0 > robot_v else v0 + size

        u_lo_new = min(u0, u1)
        u_hi_new = max(u0, u1)
        v_lo_new = min(v0, v1)
        v_hi_new = max(v0, v1)

        prev = self._outer_model
        tol  = self._OUTER_STABLE_TOL
        if (prev.u_lo is not None and prev.u_hi is not None
                and prev.v_lo is not None and prev.v_hi is not None
                and abs(u_lo_new - prev.u_lo) < tol
                and abs(u_hi_new - prev.u_hi) < tol
                and abs(v_lo_new - prev.v_lo) < tol
                and abs(v_hi_new - prev.v_hi) < tol):
            self._outer_stable_count += 1
        else:
            self._outer_stable_count = 1

        self._outer_model.u_lo = u_lo_new
        self._outer_model.u_hi = u_hi_new
        self._outer_model.v_lo = v_lo_new
        self._outer_model.v_hi = v_hi_new
        self._cached_D_outer   = None  # invalidate distance map when bounds change

        if self._outer_stable_count >= self._OUTER_STABLE_N:
            self._outer_locked = True

    def _fit_inner(
        self,
        v_offsets: list[float],
        h_offsets: list[float],
        v_centers: dict[float, float],
        h_centers: dict[float, float],
    ) -> None:
        """Estimate the inner rectangle from pre-filtered (non-outer) wall cells.

        Size discovery:
          - Search both axes for a confirmed pair (two walls separated by a valid
            inner size).  The first match fixes inner_size.

        Square invariant (size_u == size_v == inner_size):
          - Once inner_size is known, a single observed wall on a missing axis is
            enough to place the opposite wall.  The missing wall is placed at
            distance inner_size on the side of the observed wall that faces the
            outer-rectangle centre (so the inner rectangle ends up centred on
            the track, not drifting outward).
        """
        outer_center = self._outer_model.center()
        if outer_center is None:
            return
        cu, cv = outer_center
        tol = self._PAIR_TOL

        inner_size: float | None = None
        found_u: tuple[float, float] | None = None
        found_v: tuple[float, float] | None = None

        # Confirmed pairs take precedence: two walls with separation ≈ valid inner size
        for s in self.INNER_SIZES:
            if found_u is None:
                for i, u1 in enumerate(v_offsets):
                    for u2 in v_offsets[i + 1:]:
                        if abs(abs(u1 - u2) - s) < tol:
                            found_u = (min(u1, u2), max(u1, u2))
                            inner_size = s
                            break
                    if found_u:
                        break

            if found_v is None:
                for i, v1 in enumerate(h_offsets):
                    for v2 in h_offsets[i + 1:]:
                        if abs(abs(v1 - v2) - s) < tol:
                            found_v = (min(v1, v2), max(v1, v2))
                            if inner_size is None:
                                inner_size = s
                            break
                    if found_v:
                        break

        if inner_size is None:
            return

        # Square invariant: use outer centre to place the missing opposite wall
        if found_u is None and v_offsets:
            u0 = min(v_offsets, key=lambda u: abs(u - cu))
            u1 = u0 + inner_size if u0 < cu else u0 - inner_size
            found_u = (min(u0, u1), max(u0, u1))

        if found_v is None and h_offsets:
            v0 = min(h_offsets, key=lambda v: abs(v - cv))
            v1 = v0 + inner_size if v0 < cv else v0 - inner_size
            found_v = (min(v0, v1), max(v0, v1))

        if found_u is not None:
            self._inner_model.u_lo, self._inner_model.u_hi = found_u
        if found_v is not None:
            self._inner_model.v_lo, self._inner_model.v_hi = found_v

    # ------------------------------------------------------------------ #
    #  Estimated wall rasterisation                                       #
    # ------------------------------------------------------------------ #

    def _make_estimated_wall_mask(
        self,
        coarse_shape:     tuple[int, int],
        locked_outer_only: bool = False,
    ) -> np.ndarray:
        """Return a coarse boolean mask of all estimated wall cells.

        Each known rectangle side is rasterised as an infinite line (thickness
        2×_EST_WALL_HW) in the track-aligned frame.  Lines are clipped to the
        perpendicular extent of their rectangle when both bounds are known.
        The mask covers the full line regardless of observed/unobserved status;
        the visualiser paints it on top of real data so it is always visible.

        locked_outer_only=True: outer walls are only painted when _outer_locked
        is True (used for path planning so an uncertain estimate never blocks the
        corridor before it has converged).
        """
        if self._track_theta is None or self._field_map.direction is None:
            return np.zeros(coarse_shape, dtype=bool)

        cos_t, sin_t = math.cos(self._track_theta), math.sin(self._track_theta)
        rows, cols   = coarse_shape
        K            = self._K
        cell_mm      = K * FieldMap.CELL_SIZE

        # World coords of every coarse cell centre
        r_idx = np.arange(rows)
        c_idx = np.arange(cols)
        CC, RR = np.meshgrid(c_idx, r_idx)
        north = RR * cell_mm - FieldMap.ORIGIN[1] + cell_mm / 2.0
        east  = CC * cell_mm - FieldMap.ORIGIN[0] + cell_mm / 2.0

        u_grid =  north * cos_t + east * sin_t
        v_grid = -north * sin_t + east * cos_t

        hw   = self._EST_WALL_HW
        mask = np.zeros(coarse_shape, dtype=bool)

        def paint_v_wall(u0: float, v_lo: float | None, v_hi: float | None) -> None:
            on_line = np.abs(u_grid - u0) <= hw
            if v_lo is not None:
                on_line &= v_grid >= v_lo
            if v_hi is not None:
                on_line &= v_grid <= v_hi
            mask.__ior__(on_line)

        def paint_h_wall(v0: float, u_lo: float | None, u_hi: float | None) -> None:
            on_line = np.abs(v_grid - v0) <= hw
            if u_lo is not None:
                on_line &= u_grid >= u_lo
            if u_hi is not None:
                on_line &= u_grid <= u_hi
            mask.__ior__(on_line)

        # Outer walls: clipped to the outer rectangle's perpendicular extent
        om = self._outer_model
        if not locked_outer_only or self._outer_locked:
            if om.u_lo is not None:
                paint_v_wall(om.u_lo, om.v_lo, om.v_hi)
            if om.u_hi is not None:
                paint_v_wall(om.u_hi, om.v_lo, om.v_hi)
            if om.v_lo is not None:
                paint_h_wall(om.v_lo, om.u_lo, om.u_hi)
            if om.v_hi is not None:
                paint_h_wall(om.v_hi, om.u_lo, om.u_hi)

        # Inner walls: clipped to the inner rectangle's perpendicular extent
        im = self._inner_model
        if im.u_lo is not None:
            paint_v_wall(im.u_lo, im.v_lo, im.v_hi)
        if im.u_hi is not None:
            paint_v_wall(im.u_hi, im.v_lo, im.v_hi)
        if im.v_lo is not None:
            paint_h_wall(im.v_lo, im.u_lo, im.u_hi)
        if im.v_hi is not None:
            paint_h_wall(im.v_hi, im.u_lo, im.u_hi)

        return mask

    # ------------------------------------------------------------------ #
    #  Phase-2 outer-wall perimeter follower                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _perimeter_walk(
        u:    float,
        v:    float,
        ulo:  float,
        uhi:  float,
        vlo:  float,
        vhi:  float,
        walk: float,
        cw:   bool,
    ) -> tuple[float, float]:
        """Project (u, v) onto the nearest point of the rectangle perimeter
        and advance ``walk`` mm along it in the CW or CCW direction.

        Clockwise arc-length (north = +u, east = +v, viewed from above):
            s = 0 at NW corner (uhi, vlo)
            [0,   A)   North  u = uhi,       v = vlo + s
            [A,  A+B)  East   v = vhi,       u = uhi − (s − A)
            [A+B, 2A+B) South  u = ulo,       v = vhi − (s − A − B)
            [2A+B, P)  West   v = vlo,       u = ulo + (s − 2A − B)
            A = vhi−vlo, B = uhi−ulo, P = 2(A+B)
        """
        A = vhi - vlo
        B = uhi - ulo
        P = 2.0 * (A + B)
        if P <= 0.0:
            return u, v

        u_c = min(max(u, ulo), uhi)
        v_c = min(max(v, vlo), vhi)

        # (side-label, |perp-distance|, projection-coordinate)
        sides = (
            ('n', abs(uhi - u), v_c),  # north: project v, arc along v+
            ('e', abs(vhi - v), u_c),  # east:  project u, arc along u-
            ('s', abs(u - ulo), v_c),  # south: project v, arc along v-
            ('w', abs(v - vlo), u_c),  # west:  project u, arc along u+
        )
        side, _, proj = min(sides, key=lambda x: x[1])

        if side == 'n':
            s = proj - vlo
        elif side == 'e':
            s = A + (uhi - proj)
        elif side == 's':
            s = A + B + (vhi - proj)
        else:
            s = 2 * A + B + (proj - ulo)

        s_t = (s + (walk if cw else -walk)) % P

        if s_t < A:
            return uhi, vlo + s_t
        if s_t < A + B:
            return uhi - (s_t - A), vhi
        if s_t < 2 * A + B:
            return ulo, vhi - (s_t - A - B)
        return ulo + (s_t - 2 * A - B), vlo

    def _outer_wall_follow_target(self, pos: Point) -> Point | None:
        """Return the phase-2 lookahead point: the position on the outer-wall
        offset rectangle that is _L_LOOKAHEAD mm ahead of the robot."""
        om    = self._outer_model
        theta = self._track_theta
        if not om.is_complete() or theta is None:
            return None
        direction = self._field_map.direction
        if direction is None:
            return None

        off = self._OUTER_WALL_OFFSET
        ulo = om.u_lo + off
        uhi = om.u_hi - off
        vlo = om.v_lo + off
        vhi = om.v_hi - off
        if ulo >= uhi or vlo >= vhi:
            return None

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        robot_u =  pos[0] * cos_t + pos[1] * sin_t
        robot_v = -pos[0] * sin_t + pos[1] * cos_t

        target_u, target_v = self._perimeter_walk(
            robot_u, robot_v, ulo, uhi, vlo, vhi,
            self._L_LOOKAHEAD,
            cw=(direction == Direction.CW),
        )

        # Invert track-aligned → world: pos[0]=u·cos−v·sin, pos[1]=u·sin+v·cos
        return (
            target_u * cos_t - target_v * sin_t,
            target_u * sin_t + target_v * cos_t,
        )

    # ------------------------------------------------------------------ #
    #  Two-stage plan                                                     #
    # ------------------------------------------------------------------ #

    def _plan(
        self,
        pos: Point,
        yaw: radian,
    ) -> tuple[Point | None, Obstacle | None, str]:
        occupancy, semantic, obstacles = self._field_map.snapshot()

        self._update_track_model(semantic, pos, yaw)
        blocked_walls, obstacle, unknown = self._coarse_masks(occupancy, semantic)

        current_cell = self._world_to_coarse(pos)
        if not self._in_bounds(current_cell, blocked_walls):
            return None, None, "stop"

        allowed_sign = self._allowed_sign()
        upcoming     = self._find_upcoming_obstacle(obstacles, current_cell, yaw)

        D_mm        = self._clearance(blocked_walls)
        traversable = (D_mm >= self._WALL_INFLATION) & ~unknown

        # ── Phase 2: outer wall locked → follow the offset perimeter ────────
        if self._outer_locked:
            target = self._outer_wall_follow_target(pos)
            if target is not None:
                self._emit_plan_debug(traversable, D_mm, current_cell,
                                      None, None, None, None)
                return target, upcoming, "ok"

        # ── Phase 1: directed A* (used until outer wall is locked) ──────────
        if not traversable[current_cell]:
            r0, c0 = current_cell
            r_cells = int(math.ceil(self._WALL_INFLATION / (self._K * FieldMap.CELL_SIZE)))
            rows_t, cols_t = traversable.shape
            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    nr, nc = r0 + dr, c0 + dc
                    if (0 <= nr < rows_t and 0 <= nc < cols_t
                            and not blocked_walls[nr, nc]):
                        traversable[nr, nc] = True

        D_outer   = self._outer_dist_map(blocked_walls.shape)
        goal_cell = self._select_directed_goal(
            traversable, D_mm, current_cell, yaw, allowed_sign, D_outer,
        )
        if goal_cell is None:
            self._emit_plan_debug(traversable, D_mm, current_cell,
                                  None, None, None, None)
            return None, upcoming, ("recover" if allowed_sign is not None else "stop")

        trav_obs = traversable.copy()
        D_obs, _ = self._obstacle_fields(obstacle)
        if D_obs is not None:
            trav_obs &= (D_obs >= self._OBSTACLE_INFLATION)

        block:  np.ndarray | None = None
        region: np.ndarray | None = None
        if self._OBSTACLE_RULES_ENABLED:
            wtheta = self._estimate_wall_orientation(semantic)
            if wtheta is not None:
                self._wall_theta = wtheta

            region = self._latched_blocks(obstacles, blocked_walls.shape, current_cell)

            if upcoming is not None and upcoming.color is not None:
                travel_yaw     = self._travel_yaw(yaw)
                normal, anchor = self._latched_pass(upcoming, travel_yaw)
                up_region      = self._wrong_side_region(blocked_walls.shape, anchor, normal, current_cell)
                region         = up_region if region is None else (region | up_region)

            if region is not None:
                block    = trav_obs & region
                trav_obs = trav_obs & ~region

        goal_cell = self._pull_goal_in(goal_cell, current_cell, trav_obs)
        if goal_cell is None:
            self._emit_plan_debug(traversable, D_mm, current_cell,
                                  None, None, None, block)
            return None, upcoming, ("recover" if allowed_sign is not None else "stop")

        path = self._astar(trav_obs, D_mm, current_cell, goal_cell, yaw, D_outer)
        if path is None:
            self._emit_plan_debug(traversable, D_mm, current_cell,
                                  goal_cell, None, None, block)
            return None, upcoming, ("recover" if allowed_sign is not None else "stop")

        thinned = self._thin_path(path, trav_obs)
        gx, gy  = self._lookahead_point(thinned)
        self._emit_plan_debug(traversable, D_mm, current_cell,
                              goal_cell, path, thinned, block)
        return (gx, gy), upcoming, "ok"

    def _select_directed_goal(
        self,
        traversable:  np.ndarray,
        D_mm:         np.ndarray,
        current_cell: Cell,
        heading:      radian,
        allowed_sign: float | None,
        D_outer_mm:   np.ndarray | None = None,
    ) -> Cell | None:
        D_current = float(D_mm[current_cell])
        L_max_mm  = float(np.clip(self._K_ADAPTIVE * D_current, self._L_MIN, self._L_MAX))
        coarse_mm = self._K * FieldMap.CELL_SIZE
        L_min_c   = self._L_MIN / coarse_mm
        L_max_c   = L_max_mm / coarse_mm

        angles = np.linspace(
            heading - self._HALF_ANGLE,
            heading + self._HALF_ANGLE,
            self._N_RAYS,
        )

        rows, cols = traversable.shape
        best_goal:  Cell | None = None
        best_score: float       = -math.inf

        for theta in angles:
            delta = self._wrap(theta - heading)
            if allowed_sign is not None and \
               abs(delta) > self._CENTER_BAND and (delta * allowed_sign) < 0.0:
                continue

            dx, dy    = math.cos(theta), math.sin(theta)
            last_free: Cell | None = None

            d = L_min_c
            while d <= L_max_c:
                r = int(current_cell[0] + dx * d)
                c = int(current_cell[1] + dy * d)
                if not (0 <= r < rows and 0 <= c < cols):
                    break
                if not traversable[r, c]:
                    break
                last_free = (r, c)
                d += 1.0

            if last_free is None:
                continue
            reach = math.hypot(
                last_free[0] - current_cell[0],
                last_free[1] - current_cell[1],
            )
            outer_pen = self._W_OUTER_WALL * float(D_outer_mm[last_free]) \
                        if D_outer_mm is not None else 0.0
            score = self._W_DIST * reach + self._W_CLEAR * float(D_mm[last_free]) - outer_pen
            if score > best_score:
                best_score = score
                best_goal  = last_free

        return best_goal

    @staticmethod
    def _pull_goal_in(
        goal:        Cell,
        current:     Cell,
        traversable: np.ndarray,
    ) -> Cell | None:
        if traversable[goal]:
            return goal
        r0, c0 = current
        r1, c1 = goal
        n      = max(abs(r1 - r0), abs(c1 - c0))
        if n == 0:
            return None
        for k in range(n, -1, -1):
            t = k / n
            r = int(round(r0 + (r1 - r0) * t))
            c = int(round(c0 + (c1 - c0) * t))
            if traversable[r, c]:
                return (r, c)
        return None

    # ------------------------------------------------------------------ #
    #  Coarse grids (augmented with estimated walls)                      #
    # ------------------------------------------------------------------ #

    def _coarse_masks(
        self,
        occupancy: np.ndarray,
        semantic:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        K    = self._K
        rows = occupancy.shape[0] // K
        cols = occupancy.shape[1] // K

        occ = occupancy[:rows * K, :cols * K]
        sem = semantic [:rows * K, :cols * K]

        coarse_occ = occ.reshape(rows, K, cols, K)
        occ_hit    = (coarse_occ > self._OCCUPANCY_THRESHOLD).any(axis=(1, 3))
        # A coarse cell is unknown if no fine cell within it has ever received a
        # free-space vote (log-odds < 0).  Returned separately so the clearance
        # EDT is computed from real walls only (unknown cells nearby would
        # artificially shrink the clearance of the robot's own cell).
        unknown    = ~(coarse_occ < 0.0).any(axis=(1, 3))
        wall       = np.isin(sem, [int(CellLabel.WALL), int(CellLabel.PARKING_WALL)]) \
                       .reshape(rows, K, cols, K).any(axis=(1, 3))
        obstacle   = (sem == int(CellLabel.OBSTACLE)) \
                       .reshape(rows, K, cols, K).any(axis=(1, 3))

        # Visualization mask: outer walls shown even before locking
        estimated            = self._make_estimated_wall_mask((rows, cols))
        self._last_estimated = estimated

        # Planning mask: locked outer walls + inner estimate
        planning_est = self._make_estimated_wall_mask((rows, cols), locked_outer_only=True)

        # Observed hits: raw occupancy (not obstacles) + semantic wall cells
        observed = (occ_hit & ~obstacle) | wall

        # Suppress observed cells that are already explained by the estimated geometry
        if planning_est.any():
            D_from_est  = distance_transform_edt(~planning_est).astype(np.float32) \
                          * (self._K * FieldMap.CELL_SIZE)
            unexplained = observed & (D_from_est > self._EST_ABSORB_TOL)
        else:
            unexplained = observed

        blocked_walls = planning_est | unexplained
        return blocked_walls, obstacle, unknown

    def _clearance(self, blocked_walls: np.ndarray) -> np.ndarray:
        if self._cached_blocked_walls is not None and \
           np.array_equal(blocked_walls, self._cached_blocked_walls):
            return self._cached_D_mm   # type: ignore[return-value]
        D_mm = distance_transform_edt(~blocked_walls).astype(np.float32) \
               * (self._K * FieldMap.CELL_SIZE)
        self._cached_blocked_walls = blocked_walls
        self._cached_D_mm          = D_mm
        return D_mm

    def _obstacle_fields(
        self,
        obstacle: np.ndarray,
    ) -> tuple[np.ndarray | None, None]:
        if not obstacle.any():
            return None, None
        if self._cached_obstacle is not None and \
           np.array_equal(obstacle, self._cached_obstacle):
            return self._cached_D_obs, None
        D_obs = distance_transform_edt(~obstacle).astype(np.float32) \
                * (self._K * FieldMap.CELL_SIZE)
        self._cached_obstacle = obstacle
        self._cached_D_obs    = D_obs
        return D_obs, None

    def _outer_dist_map(self, coarse_shape: tuple[int, int]) -> np.ndarray | None:
        """Analytical per-cell distance (mm) to the nearest outer wall segment.

        Computed in the track-aligned frame: for each coarse cell the perpendicular
        distances to the four outer walls are computed and the minimum is taken.
        Returns None until the outer model has all four bounds.
        Cached — invalidated whenever _fit_outer updates the bounds.
        """
        if self._track_theta is None:
            return None
        om = self._outer_model
        if om.u_lo is None and om.u_hi is None and om.v_lo is None and om.v_hi is None:
            return None
        if self._cached_D_outer is not None and \
                self._cached_D_outer.shape == coarse_shape:
            return self._cached_D_outer

        cell_mm  = self._K * FieldMap.CELL_SIZE
        cos_t, sin_t = math.cos(self._track_theta), math.sin(self._track_theta)
        rows, cols   = coarse_shape

        north = np.arange(rows) * cell_mm - FieldMap.ORIGIN[1]
        east  = np.arange(cols) * cell_mm - FieldMap.ORIGIN[0]
        north_g = north[:, None]
        east_g  = east[None, :]
        u_g =  north_g * cos_t + east_g * sin_t
        v_g = -north_g * sin_t + east_g * cos_t

        _BIG = np.float32(1e9)
        d = np.full(coarse_shape, _BIG, dtype=np.float32)
        if om.u_lo is not None:
            d = np.minimum(d, np.abs(u_g - om.u_lo))
        if om.u_hi is not None:
            d = np.minimum(d, np.abs(u_g - om.u_hi))
        if om.v_lo is not None:
            d = np.minimum(d, np.abs(v_g - om.v_lo))
        if om.v_hi is not None:
            d = np.minimum(d, np.abs(v_g - om.v_hi))
        self._cached_D_outer = d.astype(np.float32)
        return self._cached_D_outer

    # ------------------------------------------------------------------ #
    #  A*                                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _astar(
        traversable:   np.ndarray,
        D_mm:          np.ndarray,
        start:         Cell,
        goal:          Cell,
        robot_heading: radian,
        D_outer_mm:    np.ndarray | None = None,
    ) -> list[Cell] | None:
        SQRT2 = math.sqrt(2)

        rows, cols = traversable.shape

        neighbours = [
            (-1,  0, 1.0),   ( 1,  0, 1.0),
            ( 0, -1, 1.0),   ( 0,  1, 1.0),
            (-1, -1, SQRT2), (-1,  1, SQRT2),
            ( 1, -1, SQRT2), ( 1,  1, SQRT2),
        ]

        # Precompute turn-angle lookup table [incoming_dir][outgoing_dir] → radians in [0, π]
        move_angles = [math.atan2(float(dc), float(dr)) for dr, dc, _ in neighbours]
        turn_table = [
            [abs(((move_angles[j] - move_angles[i] + math.pi) % (2.0 * math.pi)) - math.pi)
             for j in range(8)]
            for i in range(8)
        ]

        # Quantize robot heading to the nearest of the 8 move directions
        d_init = min(range(8), key=lambda i: abs(
            ((robot_heading - move_angles[i] + math.pi) % (2.0 * math.pi)) - math.pi
        ))

        # State: (r, c, d) where d is the index of the direction we arrived from
        start_state = (start[0], start[1], d_init)
        g_score:   dict[tuple[int, int, int], float] = {start_state: 0.0}
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        open_heap: list[tuple[float, float, int, int, int]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start[0], start[1], d_init))

        while open_heap:
            _, g, r, c, d = heapq.heappop(open_heap)
            state = (r, c, d)
            if (r, c) == goal:
                path = [(r, c)]
                cur  = state
                while cur in came_from:
                    cur = came_from[cur]
                    path.append((cur[0], cur[1]))
                path.reverse()
                return path
            if g > g_score.get(state, math.inf):
                continue
            for nd, (dr, dc, move_cost) in enumerate(neighbours):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if not traversable[nr, nc]:
                    continue
                D_n      = float(D_mm[nr, nc])
                cl_pen   = (max(0.0, GeometricTrackPlanner._TARGET_CLEARANCE - D_n) /
                            GeometricTrackPlanner._TARGET_CLEARANCE *
                            GeometricTrackPlanner._CLEARANCE_WEIGHT)
                turn_pen = GeometricTrackPlanner._TURN_PENALTY * turn_table[d][nd]
                out_pen  = (GeometricTrackPlanner._W_OUTER_WALL * float(D_outer_mm[nr, nc])
                            if D_outer_mm is not None else 0.0)
                new_g      = g + move_cost + cl_pen + turn_pen + out_pen
                next_state = (nr, nc, nd)
                if new_g < g_score.get(next_state, math.inf):
                    g_score[next_state]   = new_g
                    came_from[next_state] = state
                    f = new_g + math.hypot(nr - goal[0], nc - goal[1])
                    heapq.heappush(open_heap, (f, new_g, nr, nc, nd))

        return None

    # ------------------------------------------------------------------ #
    #  Obstacle selection + wrong-side block (same as DirectionalPlanner) #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_upcoming_obstacle(
        obstacles:    list[Obstacle],
        current_cell: Cell,
        heading:      radian,
    ) -> Obstacle | None:
        hx = math.cos(heading)
        hy = math.sin(heading)
        best_obs:  Obstacle | None = None
        best_dist: float           = math.inf
        for obs in obstacles:
            if obs.color is None:
                continue
            cc = GeometricTrackPlanner._world_to_coarse(
                (float(obs.centroid[1]), float(obs.centroid[0]))
            )
            vx = cc[1] - current_cell[1]
            vy = cc[0] - current_cell[0]
            dist = math.hypot(vx, vy)
            if dist == 0:
                continue
            dot   = (hy * vx + hx * vy) / dist
            angle = math.acos(max(-1.0, min(1.0, dot)))
            if angle <= GeometricTrackPlanner._HALF_ANGLE and dist < best_dist:
                best_dist = dist
                best_obs  = obs
        return best_obs

    def _latched_pass(
        self,
        obstacle: Obstacle,
        heading:  radian,
    ) -> tuple[np.ndarray, np.ndarray]:
        color = obstacle.color
        if (obstacle.pass_normal is not None
                and obstacle.pass_anchor is not None
                and obstacle.pass_color == color):
            return obstacle.pass_normal, obstacle.pass_anchor
        normal = self._wrong_side_normal(heading, color)
        if self._wall_theta is not None:
            normal = self._snap_to_walls(normal, self._wall_theta)
        anchor = np.asarray(obstacle.centroid, dtype=np.float32)
        self._field_map.latch_pass_side(obstacle, normal, color, anchor)
        return normal, anchor

    @staticmethod
    def _wrong_side_normal(heading: radian, color: ObstacleColor) -> np.ndarray:
        s, c = math.sin(heading), math.cos(heading)
        if color == ObstacleColor.RED:
            return np.array([ s, -c], dtype=np.float32)
        return np.array([-s,  c], dtype=np.float32)

    @staticmethod
    def _snap_to_walls(vec: np.ndarray, theta: radian) -> np.ndarray:
        phi = math.atan2(float(vec[1]), float(vec[0]))
        k   = round((phi - theta) / (math.pi / 2.0))
        a   = theta + k * (math.pi / 2.0)
        return np.array([math.cos(a), math.sin(a)], dtype=np.float32)

    def _estimate_wall_orientation(self, semantic: np.ndarray) -> float | None:
        K    = self._K
        rows = semantic.shape[0] // K
        cols = semantic.shape[1] // K
        sem  = semantic[:rows * K, :cols * K]
        wall = (sem == int(CellLabel.WALL)).reshape(rows, K, cols, K).any(axis=(1, 3))
        gr, gc = np.gradient(wall.astype(np.float32))
        mag2   = gr * gr + gc * gc
        edge   = mag2 > 1e-6
        if int(edge.sum()) < self._WALL_ORIENT_MIN_CELLS:
            return None
        phi = np.arctan2(gc[edge], gr[edge])
        z   = np.sum(mag2[edge] * np.exp(4j * phi))
        if abs(z) < 1e-9:
            return None
        return float(np.angle(z) / 4.0)

    def _latched_blocks(
        self,
        obstacles:    list[Obstacle],
        shape:        tuple[int, int],
        current_cell: Cell,
    ) -> np.ndarray | None:
        region: np.ndarray | None = None
        for obs in obstacles:
            if obs.pass_normal is None or obs.pass_anchor is None:
                continue
            r = self._wrong_side_region(shape, obs.pass_anchor, obs.pass_normal, current_cell)
            region = r if region is None else (region | r)
        return region

    def _wrong_side_region(
        self,
        shape:        tuple[int, int],
        anchor:       np.ndarray,
        normal:       np.ndarray,
        current_cell: Cell,
    ) -> np.ndarray:
        rows, cols = shape
        cr, cc     = self._world_to_coarse((float(anchor[1]), float(anchor[0])))
        coarse_mm  = self._K * FieldMap.CELL_SIZE
        reach  = self._BLOCK_REACH        / coarse_mm
        window = self._BLOCK_WINDOW       / coarse_mm
        back   = self._BLOCK_BACK         / coarse_mm
        arm    = self._BLOCK_ARM_THICKNESS / coarse_mm
        clr    = self._BLOCK_CLEARANCE    / coarse_mm

        nr, nc = float(normal[0]), float(normal[1])
        fr, fc = -nc, nr

        RR, CC = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        dR, dC = RR - cr, CC - cc
        lateral = dR * nr + dC * nc
        along   = dR * fr + dC * fc

        leg1 = (lateral >= -back - clr) & (lateral <= reach + clr) & \
               (np.abs(along) <= window + clr)

        along_robot = (current_cell[0] - cr) * fr + (current_cell[1] - cc) * fc
        a_lo = min(along_robot, -window) - clr
        a_hi = max(along_robot,  window) + clr
        leg2 = (lateral >= reach - arm - clr) & (lateral <= reach + clr) & \
               (along >= a_lo) & (along <= a_hi)

        return leg1 | leg2

    # ------------------------------------------------------------------ #
    #  Active colour inspection                                           #
    # ------------------------------------------------------------------ #

    def _inspect_obstacle(self, pos: Point, yaw: radian) -> bool:
        obstacles    = self._field_map.get_obstacles()
        current_cell = self._world_to_coarse(pos)
        obs          = self._obstacle_to_inspect(obstacles, current_cell, yaw)

        if obs is None:
            self._inspect_start  = None
            self._inspect_anchor = None
            return False

        obs_pt = (float(obs.centroid[1]), float(obs.centroid[0]))
        now    = time.monotonic()
        if self._inspect_anchor is None or \
           self._dist(obs_pt, self._inspect_anchor) > self._INSPECT_MATCH:
            self._inspect_start  = now
            self._inspect_anchor = obs_pt

        if now - self._inspect_start >= self._INSPECT_TIMEOUT:  # type: ignore[operator]
            return False

        self.on_next_obstacle(obs)
        self._aim_at_obstacle(pos, yaw, obs_pt)
        return True

    def _obstacle_to_inspect(
        self,
        obstacles:    list[Obstacle],
        current_cell: Cell,
        heading:      radian,
    ) -> Obstacle | None:
        hx, hy    = math.cos(heading), math.sin(heading)
        coarse_mm = self._K * FieldMap.CELL_SIZE
        best:      Obstacle | None = None
        best_dist: float           = math.inf
        for obs in obstacles:
            if obs.color is not None:
                continue
            cell = self._world_to_coarse((float(obs.centroid[1]), float(obs.centroid[0])))
            vx   = cell[1] - current_cell[1]
            vy   = cell[0] - current_cell[0]
            dist = math.hypot(vx, vy)
            if dist == 0 or dist * coarse_mm > self._INSPECT_RANGE:
                continue
            angle = math.acos(max(-1.0, min(1.0, (hy * vx + hx * vy) / dist)))
            if angle <= self._INSPECT_HALF_ANGLE and dist < best_dist:
                best_dist = dist
                best      = obs
        return best

    def _aim_at_obstacle(self, pos: Point, yaw: radian, obs_pt: Point) -> None:
        dist = self._dist(pos, obs_pt)
        if dist <= self._INSPECT_STANDOFF:
            self.on_target(0.0, 0.0, 0.0)
            return
        t   = (dist - self._INSPECT_STANDOFF) / dist
        aim = (pos[0] + t * (obs_pt[0] - pos[0]),
               pos[1] + t * (obs_pt[1] - pos[1]))
        self._emit_target(pos, yaw, aim, speed=self._INSPECT_SPEED)

    # ------------------------------------------------------------------ #
    #  Targets / route / lap                                              #
    # ------------------------------------------------------------------ #

    def _home_and_stop(self, pos: Point, yaw: radian) -> None:
        home = self._field_map.start_position
        if home is not None and self._dist(pos, home) > self._STOP_RADIUS:
            self._emit_target(pos, yaw, home)
            return
        self._stopped = True
        self.on_target(0.0, 0.0, 0.0)
        self.on_stop()

    def _emit_target(
        self,
        pos:     Point,
        yaw:     radian,
        target:  Point,
        reverse: bool         = False,
        speed:   float | None = None,
    ) -> None:
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        forward_mm =  dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral_mm = -dx * math.sin(yaw) + dy * math.cos(yaw)
        if reverse:
            out_speed = -self._REVERSE_SPEED
        elif speed is not None:
            out_speed = speed
        else:
            out_speed = self._FOLLOW_SPEED if self._following else self._SPEED
        self.on_target(forward_mm, lateral_mm, out_speed)

    def _record_pose(self, pos: Point) -> None:
        if not self._route or self._dist(pos, self._route[-1]) >= self._ROUTE_MIN_SPACING:
            self._route.append(pos)

    def _finalize_route(self, pos: Point) -> None:
        if self._route:
            self._route.append(self._route[0])
        self._route_cursor = self._closest_route_index(pos)
        self.on_route(list(self._route))

    def _follow_route(self, pos: Point) -> Point | None:
        if len(self._route) < 2:
            return None
        self._route_cursor = self._closest_route_index(pos)
        n           = len(self._route)
        accumulated = 0.0
        i           = self._route_cursor
        for _ in range(n):
            a = self._route[i % n]
            b = self._route[(i + 1) % n]
            seg_len = self._dist(a, b)
            if seg_len > 1e-6 and accumulated + seg_len >= self._L_LOOKAHEAD:
                t = (self._L_LOOKAHEAD - accumulated) / seg_len
                return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            accumulated += seg_len
            i += 1
        return self._route[self._route_cursor]

    def _closest_route_index(self, pos: Point) -> int:
        n      = len(self._route)
        window = min(n, max(self._MIN_ROUTE_POINTS, n // 4))
        best_i, best_d = self._route_cursor, math.inf
        for k in range(window):
            i = (self._route_cursor + k) % n
            d = self._dist(pos, self._route[i])
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _travel_yaw(self, fallback_yaw: radian) -> radian:
        n = self._TRAVEL_TANGENT_POINTS
        if len(self._route) >= n:
            a = self._route[-n]
            b = self._route[-1]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            if math.hypot(dx, dy) > 1e-6:
                return math.atan2(dy, dx)
        return fallback_yaw

    # ------------------------------------------------------------------ #
    #  Debug                                                              #
    # ------------------------------------------------------------------ #

    def _emit_plan_debug(
        self,
        traversable:  np.ndarray,
        D_mm:         np.ndarray,
        current_cell: Cell,
        goal_cell:    Cell | None,
        path:         list[Cell] | None,
        thinned:      list[Cell] | None,
        block:        np.ndarray | None = None,
    ) -> None:
        self.on_plan_debug({
            "K":                self._K,
            "traversable":      traversable,
            "D_mm":             D_mm,
            "trail":            np.zeros_like(D_mm),
            "block":            block,
            "estimated":        self._last_estimated,
            "intersections":    self._last_intersections,
            "track_theta":      self._track_theta,
            "current":          current_cell,
            "goal":             goal_cell,
            "path":             path,
            "thinned":          thinned,
            "robot_radius":     self._WALL_INFLATION,
            "target_clearance": self._TARGET_CLEARANCE,
        })

    # ------------------------------------------------------------------ #
    #  Path post-processing + geometry helpers                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _thin_path(path: list[Cell], traversable: np.ndarray) -> list[Cell]:
        if len(path) <= 2:
            return path
        anchor         = path[0]
        control_points = [anchor]
        for i in range(1, len(path)):
            if not GeometricTrackPlanner._line_of_sight(anchor, path[i], traversable):
                control_points.append(path[i - 1])
                anchor = path[i - 1]
        control_points.append(path[-1])
        return control_points

    @staticmethod
    def _lookahead_point(thinned: list[Cell]) -> Point:
        waypoints = [GeometricTrackPlanner._coarse_to_world(c) for c in thinned]
        accumulated = 0.0
        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i]
            x1, y1 = waypoints[i + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if accumulated + seg_len >= GeometricTrackPlanner._L_LOOKAHEAD:
                t = (GeometricTrackPlanner._L_LOOKAHEAD - accumulated) / seg_len
                return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            accumulated += seg_len
        return waypoints[-1]

    @staticmethod
    def _world_to_coarse(pos: tuple[float, float]) -> Cell:
        K   = GeometricTrackPlanner._K
        col = int((pos[1] + FieldMap.ORIGIN[0]) / (FieldMap.CELL_SIZE * K))
        row = int((pos[0] + FieldMap.ORIGIN[1]) / (FieldMap.CELL_SIZE * K))
        return (row, col)

    @staticmethod
    def _coarse_to_world(cell: Cell) -> tuple[float, float]:
        K      = GeometricTrackPlanner._K
        offset = (K * FieldMap.CELL_SIZE) / 2.0
        north  = cell[0] * K * FieldMap.CELL_SIZE - FieldMap.ORIGIN[1] + offset
        east   = cell[1] * K * FieldMap.CELL_SIZE - FieldMap.ORIGIN[0] + offset
        return (north, east)

    @staticmethod
    def _in_bounds(cell: Cell, grid: np.ndarray) -> bool:
        return 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]

    @staticmethod
    def _reconstruct_path(came_from: dict[Cell, Cell], goal: Cell) -> list[Cell]:
        path    = [goal]
        current = goal
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _line_of_sight(a: Cell, b: Cell, traversable: np.ndarray) -> bool:
        r0, c0 = a
        r1, c1 = b
        dr  = abs(r1 - r0)
        dc  = abs(c1 - c0)
        sr  = 1 if r1 > r0 else -1
        sc  = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        rows, cols = traversable.shape
        while True:
            if not (0 <= r < rows and 0 <= c < cols):
                return False
            if not traversable[r, c]:
                return False
            if r == r1 and c == c1:
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r   += sr
            if e2 < dr:
                err += dr
                c   += sc

    @staticmethod
    def _dist(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi
