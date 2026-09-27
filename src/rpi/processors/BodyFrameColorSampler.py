from __future__ import annotations

import math
import time
from threading import Lock

import cv2
import numpy as np

from processors.Processor import Processor
from processors.CameraProcessor import (
    CameraIntrinsics,
    CameraExtrinsics,
)
from control import ColorRange, ObstacleColor
from utils import mm


class BodyFrameColorSampler(Processor):
    """Stateless body-frame pillar-colour reader for the reactive planner (option B).

    The reactive planner has no world map and no `SemanticClassifier`, so it cannot
    reuse `CameraProcessor`'s world->camera projection (that needs a world obstacle
    centroid + ego pose). Instead the planner clusters an obstacle in the *current*
    LiDAR scan and hands its **body-frame** point straight to this sampler, which:

      1. keeps the latest camera frame (updated on `camera.on_frame`, thread-safe);
      2. projects the body-frame point into the image via the same camera
         intrinsics/extrinsics `CameraProcessor` uses (the extrinsics are already
         body-relative, so no ego pose is involved);
      3. samples a small HSV patch and classifies it against the red / green
         `ColorRange`s (each a union of HSV bands).

    The parking wall (magenta) is a third `ColorRange`, checked whole-frame by
    `parking_wall_ratio` for the parking-slot start.

    Body-frame convention matches the planner / firmware: ``forward`` is straight
    ahead, ``lateral`` is positive to the robot's right. This maps to the ego frame
    `CameraProcessor._world_to_camera` produces as ``P_ego = [lateral, forward, 0]``:
    for a robot facing north (yaw 0), a point to its right is due east, which that
    transform places at ``P_ego[0] = +lateral`` (camera +X, i.e. the right of the
    image). So a right-of-robot pillar lands right-of-centre in the frame.
    """

    def __init__(
        self,
        *,
        intrinsics:      CameraIntrinsics,
        extrinsics:      CameraExtrinsics,
        red:             ColorRange,
        green:           ColorRange,
        parking:         ColorRange,
        sample_radius:   int   = 20,
        min_color_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self._intrinsics      = intrinsics
        self._extrinsics      = extrinsics
        self._red             = red
        self._green           = green
        self._parking         = parking
        self._sample_radius   = sample_radius
        self._min_color_ratio = min_color_ratio

        self._lock:  Lock                 = Lock()
        self._frame: np.ndarray | None    = None

        # Last color_at query, recorded for ColorVisualizer (thread-safe). None until
        # the planner first queries a colour.
        self._last: dict | None = None

    # ------------------------------------------------------------------ #
    #  Frame intake (camera thread)                                       #
    # ------------------------------------------------------------------ #

    def _process(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    # ------------------------------------------------------------------ #
    #  Debug accessors (visualizer thread)                                #
    # ------------------------------------------------------------------ #

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def last_detection(self) -> dict | None:
        with self._lock:
            return None if self._last is None else dict(self._last)

    # ------------------------------------------------------------------ #
    #  Parking-wall detection (planner thread, at startup)                #
    # ------------------------------------------------------------------ #

    def parking_wall_ratio(self) -> float | None:
        """Fraction of the current frame that is parking-wall magenta, or None if no
        frame yet.

        Used once at startup: a high ratio means the robot is boxed in the parking slot
        (a magenta wall fills the forward view). Whole-frame, not a patch — the planner
        only needs "is most of what I see a parking wall".
        """
        with self._lock:
            frame = None if self._frame is None else self._frame
        if frame is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return self._parking.ratio(hsv)

    # ------------------------------------------------------------------ #
    #  Colour query (planner thread)                                      #
    # ------------------------------------------------------------------ #

    def color_at(self, forward: mm, lateral: mm) -> ObstacleColor | None:
        """Colour of the pillar at body-frame ``(forward, lateral)``, or None.

        None means either no frame yet, the point projects behind/off the image,
        or neither colour clears the ratio threshold in the sampled patch. Every
        call records what it saw (projection, patch ratios, verdict, status) for
        ColorVisualizer via `last_detection`.
        """
        with self._lock:
            frame = None if self._frame is None else self._frame

        if frame is None:
            self._record(forward, lateral, None, None, 0.0, 0.0, None, "no-frame")
            return None

        # Body-frame point -> ego frame the extrinsics expect (see class docstring),
        # then into the camera frame.
        p_ego = np.array([lateral, forward, 0.0], dtype = np.float64)
        p_cam = self._extrinsics.R @ (p_ego - self._extrinsics.translation)
        if p_cam[2] <= 0.0:                       # behind the camera
            self._record(forward, lateral, None, None, 0.0, 0.0, None, "behind")
            return None

        pts, _ = cv2.projectPoints(
            p_cam.reshape(1, 3).astype(np.float32),
            np.zeros(3), np.zeros(3),
            self._intrinsics.K,
            self._intrinsics.dist_coeffs,
        )
        u = int(pts[0, 0, 0])
        v = int(pts[0, 0, 1])

        h, w = frame.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            self._record(forward, lateral, u, v, 0.0, 0.0, None, "off-frame")
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color, red_ratio, green_ratio = self._classify(hsv, u, v, h, w)
        self._record(forward, lateral, u, v, red_ratio, green_ratio, color, "ok")
        return color

    def _record(
        self,
        forward: mm, lateral: mm,
        u: int | None, v: int | None,
        red_ratio: float, green_ratio: float,
        color: ObstacleColor | None, status: str,
    ) -> None:
        with self._lock:
            self._last = {
                "t":       time.monotonic(),
                "forward": float(forward),
                "lateral": float(lateral),
                "u":       u,
                "v":       v,
                "red":     float(red_ratio),
                "green":   float(green_ratio),
                "color":   None if color is None else color.value,
                "status":  status,
                "radius":  self._sample_radius,
            }

    def _classify(
        self,
        hsv: np.ndarray,
        u:   int,
        v:   int,
        h:   int,
        w:   int,
    ) -> tuple[ObstacleColor | None, float, float]:
        r     = self._sample_radius
        patch = hsv[max(0, v - r):min(h, v + r), max(0, u - r):min(w, u + r)]
        if patch.size == 0:
            return None, 0.0, 0.0

        total = patch.shape[0] * patch.shape[1]

        red_ratio   = np.count_nonzero(self._red.mask(patch))   / total
        green_ratio = np.count_nonzero(self._green.mask(patch)) / total

        if red_ratio   >= self._min_color_ratio and red_ratio   > green_ratio:
            return ObstacleColor.RED, red_ratio, green_ratio
        if green_ratio >= self._min_color_ratio and green_ratio > red_ratio:
            return ObstacleColor.GREEN, red_ratio, green_ratio
        return None, red_ratio, green_ratio
