from __future__ import annotations

import math

import numpy as np

from processors.Processor import Processor
from control.EgoInformation import EgoInformation
from control.TrackModel import TrackModel
from utils import mm, degree, radian


class PrincipalAngleDetector(Processor):
    """Fits the track's principal angle ``theta`` (grid orientation, mod 90 deg) from
    each LiDAR scan and publishes it into the shared ``TrackModel``.

    This is pure perception, split out of ``ReactiveSegmentPlanner`` so the planner
    keeps no theta state of its own. The estimate is yaw-invariant: wall tangents give
    the grid orientation in the body frame, the gyro yaw lifts it into the absolute
    frame, and a circular-mean EMA on the mod-90 quantity holds it. Because ``theta``
    is only refined on clean straights (a strict fit gate rejects messy / mid-turn
    scans), a gyro-referenced fallback seeds it after a short wait so the robot never
    deadlocks waiting for a fit that a corner start can't provide.

    Wiring: subscribe this to ``lidar.on_scan`` **before** the planner, so the planner
    reads a freshly updated ``theta`` the same scan (both run on the LiDAR thread).
    """

    # Sensor-frame -> body-frame offset and range cap. Must match the planner's own
    # body-point transform (same physical mount); kept here so the detector is
    # self-contained.
    _OFFSET: degree = -90.0
    _R_MAX:  mm     = 3000.0

    # --- Principal angle theta (grid orientation, mod 90 deg) ---
    _THETA_ALPHA:         float = 0.05    # circular-mean EMA rate on the mod-90 quantity
    _SAME_SURFACE:        mm    = 120.0   # consecutive returns within this = same wall (tangent OK)
    _MIN_WALL_LEN:        mm    = 400.0   # total tangent length needed to trust a scan's fit
    _MIN_RESULTANT:       float = 0.6     # mod-90 resultant length needed (0..1) = fit quality gate
    _SEED_FALLBACK_SCANS: int   = 12      # seed theta from the gyro after this many scans without a
                                          # LiDAR fit (e.g. a corner start), so motion can begin

    def __init__(
        self,
        ego_information: EgoInformation,
        track:           TrackModel,
    ) -> None:
        super().__init__()
        self._ego        = ego_information
        self._track      = track
        self._offset_rad = math.radians(self._OFFSET)

        # Principal angle as a mod-90 unit vector on 4*angle (circular-mean EMA).
        self._theta_vec: np.ndarray | None = None   # (cos 4t, sin 4t); None until seeded
        self._scans_no_theta: int  = 0              # scans elapsed before theta first seeds
        self._from_gyro:      bool = False          # latched True once the gyro fallback seeds

    def _process(self, scan: np.ndarray) -> None:
        if len(scan) == 0:
            return

        # Held by the planner during the parking-leave arc (rotating next to close walls),
        # where the fit would be corrupted. theta is already seeded by then, so we simply
        # keep the last value until the maneuver clears the flag.
        if self._track.theta_frozen:
            return

        yaw = self._ego.yaw
        if yaw is None:
            return

        a, r, x, y = self._body_points(scan)
        if len(a) == 0:
            return

        self._update_theta(a, r, x, y, yaw)

        if self._theta_vec is None:
            # No LiDAR fit yet. Wait a few scans, then seed from the gyro so the planner
            # (which gates on TrackModel.theta_seeded) can start; the EMA gate then
            # refines theta once a clean straight is in view.
            self._scans_no_theta += 1
            if self._scans_no_theta >= self._SEED_FALLBACK_SCANS:
                m4 = 4.0 * yaw
                self._theta_vec = np.array([math.cos(m4), math.sin(m4)])
                self._from_gyro = True
                self._track.update_theta(self._theta(), from_gyro = True)
            return

        self._track.update_theta(self._theta(), from_gyro = self._from_gyro)

    # ------------------------------------------------------------------ #
    #  Principal angle theta                                              #
    # ------------------------------------------------------------------ #

    def _body_points(self, scan: np.ndarray) -> tuple[np.ndarray, ...]:
        a = self._wrap_array(scan[:, 0] + self._offset_rad)
        r = scan[:, 1]
        keep = (r > 0.0) & (r < self._R_MAX)
        a, r = a[keep], r[keep]
        order = np.argsort(a)
        a, r = a[order], r[order]
        return a, r, r * np.cos(a), r * np.sin(a)

    def _update_theta(
        self,
        a: np.ndarray, r: np.ndarray, x: np.ndarray, y: np.ndarray,
        yaw: radian,
    ) -> None:
        # Estimate the grid orientation (mod 90 deg) from wall tangents: between
        # consecutive returns on the same surface the segment direction is a wall
        # tangent. Both corridor side walls and the perpendicular end wall lie on the
        # same 90 deg grid, so every clean tangent reinforces the same mod-90 angle.
        # Accumulate as unit vectors on 4*tangent so the mod-90 wrap is handled and
        # the resultant length doubles as a fit-quality measure.
        dx  = np.diff(x)
        dy  = np.diff(y)
        seg = np.hypot(dx, dy)
        good = (np.abs(np.diff(r)) < self._SAME_SURFACE) & (seg > 1.0)
        if not good.any():
            return

        t = np.arctan2(dy[good], dx[good])
        w = seg[good]
        c = float(np.sum(w * np.cos(4.0 * t)))
        s = float(np.sum(w * np.sin(4.0 * t)))
        total = float(np.sum(w))
        if total < self._MIN_WALL_LEN:
            return
        resultant = math.hypot(c, s) / total
        if resultant < self._MIN_RESULTANT:         # messy / mid-turn scan: skip, don't trust
            return

        # Body-frame grid orientation -> absolute (gyro) frame: shift the 4-angle by
        # 4*yaw. theta is yaw-invariant because yaw enters here exactly as it leaves
        # via phi_body.
        phi_body = math.atan2(s, c) / 4.0
        m4 = 4.0 * (yaw + phi_body)
        meas = np.array([math.cos(m4), math.sin(m4)])

        if self._theta_vec is None:
            self._theta_vec = meas
        else:
            v = (1.0 - self._THETA_ALPHA) * self._theta_vec + self._THETA_ALPHA * meas
            n = np.linalg.norm(v)
            if n > 1e-9:
                self._theta_vec = v / n

    def _theta(self) -> radian:
        # Representative grid orientation in (-45, 45] deg (absolute gyro frame).
        if self._theta_vec is None:
            return 0.0
        return math.atan2(self._theta_vec[1], self._theta_vec[0]) / 4.0

    @staticmethod
    def _wrap_array(angles: np.ndarray) -> np.ndarray:
        return (angles + math.pi) % (2 * math.pi) - math.pi
