from __future__ import annotations

import math
from threading import Lock

from utils import mm, radian


class TrackModel:
    """Thread-safe, mutable shared state describing the track as the robot
    currently understands it.

    This is a blackboard in the same spirit as ``EgoInformation`` -- a
    lock-protected bag of state that several processors read and write -- holding
    **no maps or grids**, only the scalar track descriptors the reactive stack
    reasons about:

      * **Principal angle ``theta``** (grid orientation, mod 90 deg, absolute gyro
        frame) plus its seed status. Written by ``PrincipalAngleDetector``; every
        other field is written by the planner.
      * **Track structure** -- the latched grid line of the current straight
        (``seg_k``), the corner turn direction (``turn_sign``), and how many corners
        have been completed (``corner_count``).
      * **Lane / heading** -- the lane the pursuit aims at (``target_offset``), the
        lane the robot is settled in (``current_lane``), and the corner target
        heading (``target_heading``).

    ``segment_heading(yaw)`` is derived (both its inputs, ``theta`` and ``seg_k``,
    live here) so it belongs on the model rather than any one processor.

    Every accessor takes the lock. In the live pipeline the detector and planner run
    back-to-back on the LiDAR thread (the detector is registered first, so the
    planner always reads a freshly updated ``theta``); the lock guards cross-thread
    readers such as a visualiser or the recorder.
    """

    def __init__(self) -> None:
        self._lock: Lock = Lock()

        # --- Principal angle (written by PrincipalAngleDetector) ---
        self._theta:           radian = 0.0    # grid orientation in (-45, 45] deg, absolute frame
        self._theta_seeded:    bool   = False  # True once a usable theta exists (LiDAR fit or gyro seed)
        self._theta_from_gyro: bool   = False  # True if the initial seed came from the gyro fallback
        self._theta_frozen:    bool   = False  # when True the detector holds theta (skips updates) — set
                                               # by the planner during the parking-leave arc, where the
                                               # rotating close-wall scan would corrupt the fit

        # --- Track structure (written by the planner) ---
        self._seg_k:        int | None = None  # grid multiple of the CURRENT straight
                                               # (segment_heading = theta + k*90); latched per
                                               # straight, bumped by turn_sign at each corner
        self._turn_sign:    int | None = None  # +1 CW / -1 CCW, latched at the first corner
        self._corner_count: int        = 0     # corners completed so far

        # --- Lane / heading (written by the planner) ---
        self._current_lane:   mm     = 0.0     # lane the robot is settled in
        self._target_offset:  mm     = 0.0     # lane the pursuit aims at
        self._target_heading: radian = 0.0     # corner target heading

    # ------------------------------------------------------------------ #
    #  Principal angle                                                    #
    # ------------------------------------------------------------------ #

    def update_theta(self, theta: radian, from_gyro: bool) -> None:
        """Publish a fresh principal angle; marks the model seeded.

        ``from_gyro`` records whether the value ultimately derives from the gyro
        fallback seed. Once seeded from the gyro it stays flagged for the run (the
        detector passes its own latched flag), matching the old planner behaviour.
        """
        with self._lock:
            self._theta           = theta
            self._theta_seeded    = True
            self._theta_from_gyro = from_gyro

    @property
    def theta(self) -> radian:
        with self._lock:
            return self._theta

    @property
    def theta_seeded(self) -> bool:
        with self._lock:
            return self._theta_seeded

    @property
    def theta_from_gyro(self) -> bool:
        with self._lock:
            return self._theta_from_gyro

    @property
    def theta_frozen(self) -> bool:
        with self._lock:
            return self._theta_frozen

    @theta_frozen.setter
    def theta_frozen(self, value: bool) -> None:
        with self._lock:
            self._theta_frozen = value

    # ------------------------------------------------------------------ #
    #  Track structure                                                    #
    # ------------------------------------------------------------------ #

    @property
    def seg_k(self) -> int | None:
        with self._lock:
            return self._seg_k

    @seg_k.setter
    def seg_k(self, value: int | None) -> None:
        with self._lock:
            self._seg_k = value

    @property
    def turn_sign(self) -> int | None:
        with self._lock:
            return self._turn_sign

    @turn_sign.setter
    def turn_sign(self, value: int | None) -> None:
        with self._lock:
            self._turn_sign = value

    @property
    def corner_count(self) -> int:
        with self._lock:
            return self._corner_count

    @corner_count.setter
    def corner_count(self, value: int) -> None:
        with self._lock:
            self._corner_count = value

    # ------------------------------------------------------------------ #
    #  Lane / heading                                                     #
    # ------------------------------------------------------------------ #

    @property
    def current_lane(self) -> mm:
        with self._lock:
            return self._current_lane

    @current_lane.setter
    def current_lane(self, value: mm) -> None:
        with self._lock:
            self._current_lane = value

    @property
    def target_offset(self) -> mm:
        with self._lock:
            return self._target_offset

    @target_offset.setter
    def target_offset(self, value: mm) -> None:
        with self._lock:
            self._target_offset = value

    @property
    def target_heading(self) -> radian:
        with self._lock:
            return self._target_heading

    @target_heading.setter
    def target_heading(self, value: radian) -> None:
        with self._lock:
            self._target_heading = value

    # ------------------------------------------------------------------ #
    #  Derived                                                            #
    # ------------------------------------------------------------------ #

    def segment_heading(self, yaw: radian) -> radian:
        """Heading of the current straight: the grid direction theta + k*90 deg for
        the LATCHED multiple k.

        k is fixed for the whole straight and changes only by turn_sign at each
        corner, so the heading tracks any theta refinement while the grid LINE never
        snaps -- a yaw wander past +-45 deg (e.g. a big lane switch) can no longer
        flip e_head / lat_dr by 90 deg mid-straight. Before the first latch (theta not
        yet seeded) it falls back to the grid nearest the live yaw, which is only used
        by pre-seed odometry.
        """
        with self._lock:
            base = self._theta
            k = self._seg_k if self._seg_k is not None \
                else round((yaw - base) / (math.pi / 2.0))
            return self._wrap(base + k * (math.pi / 2.0))

    @staticmethod
    def _wrap(angle: radian) -> radian:
        return (angle + math.pi) % (2 * math.pi) - math.pi
