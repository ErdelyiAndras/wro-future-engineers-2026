from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen = True)
class ColorRange:
    """An HSV colour defined by one or more ``(lower, upper)`` bands.

    OpenCV HSV convention (H 0-180, S/V 0-255). A colour is the UNION of its bands,
    so a hue that straddles the 0/180 seam — red, or a magenta parking wall — is
    represented as two bands rather than being clipped to one. A colour that needs no
    wrap (e.g. green) is simply one band. The same type represents every colour the
    reactive planner cares about (red / green pillars and the parking wall), so there
    is no per-colour fixed field count to outgrow.

    ``bands`` is a sequence of ``(lower, upper)`` uint8 HSV arrays. Build it straight
    from the calibration script's output, e.g.::

        ColorRange(bands = [
            (np.array([  0, 131,  56], dtype = np.uint8), np.array([ 10, 255, 165], dtype = np.uint8)),
            (np.array([170, 131,  56], dtype = np.uint8), np.array([180, 255, 165], dtype = np.uint8)),
        ])
    """

    bands: Sequence[tuple[np.ndarray, np.ndarray]]

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        """Binary mask of the pixels in ``hsv`` (an HSV image/patch) inside any band."""
        out = np.zeros(hsv.shape[:2], dtype = np.uint8)
        for lower, upper in self.bands:
            out |= cv2.inRange(hsv, lower, upper)
        return out

    def ratio(self, hsv: np.ndarray) -> float:
        """Fraction of ``hsv`` inside the colour (0..1); 0 for an empty image."""
        total = hsv.shape[0] * hsv.shape[1]
        if total == 0:
            return 0.0
        return float(np.count_nonzero(self.mask(hsv))) / float(total)
