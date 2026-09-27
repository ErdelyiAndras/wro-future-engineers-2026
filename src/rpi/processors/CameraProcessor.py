from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
import math
import cv2
from processors.Processor import Processor
from control import FieldMap, EgoInformation, ObstacleColor
from utils import mm, degree

@dataclass(frozen = True)
class CameraIntrinsics:
    fx:          float
    fy:          float
    cx:          float
    cy:          float
    dist_coeffs: np.ndarray

    @staticmethod
    def from_file(intrinsics_path: str, distortion_path: str) -> CameraIntrinsics:
        K    = np.load(intrinsics_path)
        dist = np.load(distortion_path)
        return CameraIntrinsics(
            fx          = float(K[0, 0]),
            fy          = float(K[1, 1]),
            cx          = float(K[0, 2]),
            cy          = float(K[1, 2]),
            dist_coeffs = dist.ravel().astype(np.float64),
        )

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx,       0, self.cx],
            [      0, self.fy, self.cy],
            [      0,       0,       1],
        ], dtype = np.float64)


@dataclass(frozen = True)
class CameraExtrinsics:
    x:     mm
    y:     mm
    z:     mm
    pitch: degree = 0.0
    yaw:   degree = 0.0
    roll:  degree = 0.0

    @property
    def translation(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype = np.float64)

    @property
    def R(self) -> np.ndarray:
        R_base = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0],
        ], dtype = np.float64)

        R_extra = Rotation.from_euler(
            'xyz', [self.pitch, self.yaw, self.roll], degrees = True
        ).as_matrix()

        return R_extra @ R_base


@dataclass(frozen = True)
class ObstacleColorRanges:
    red_lower_1: np.ndarray
    red_upper_1: np.ndarray
    red_lower_2: np.ndarray
    red_upper_2: np.ndarray
    green_lower: np.ndarray
    green_upper: np.ndarray


class CameraProcessor(Processor):
    def __init__(
        self,
        field_map:             FieldMap,
        ego_information:       EgoInformation,
        *,
        intrinsics:            CameraIntrinsics,
        extrinsics:            CameraExtrinsics,
        obstacle_color_ranges: ObstacleColorRanges,
        sample_radius:         int   = 20,
        min_color_ratio:       float = 0.2,
    ) -> None:
        super().__init__()

        self._field_map:             FieldMap            = field_map
        self._ego_information:       EgoInformation      = ego_information
        self._intrinsics:            CameraIntrinsics    = intrinsics
        self._extrinsics:            CameraExtrinsics    = extrinsics
        self._obstacle_color_ranges: ObstacleColorRanges = obstacle_color_ranges
        self._sample_radius:         int                 = sample_radius
        self._min_color_ratio:       float               = min_color_ratio

        self._frame_count: int = 0
        self._debug_every: int = 30   # print a diagnostic line every N frames (0 = off)
        self._save_dir:    str = 'recordings/cam_debug'  # annotated frames land here (None = off)
        self._save_every:  int = 15   # save a frame every N frames that have an obstacle
        self._save_cap:    int = 40   # stop after this many, so disk doesn't fill
        self._saved:       int = 0

    def _process(self, frame: np.ndarray) -> None:
        self._frame_count += 1
        report = self._debug_every > 0 and self._frame_count % self._debug_every == 0

        ego_pos, yaw = self._ego_information.get_ego_information()
        if ego_pos is None or yaw is None:
            if report:
                print(f"[cam] frame {self._frame_count}: no ego pose yet")
            return

        ego_pos   = np.asarray(ego_pos, dtype = np.float64)
        obstacles = self._field_map.get_obstacles()
        if not obstacles:
            if report:
                print(f"[cam] frame {self._frame_count}: {frame.shape[1]}x{frame.shape[0]}, "
                      f"0 obstacles to classify")
            return

        h, w = frame.shape[:2]
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        notes:  list[str]   = []
        marks:  list[tuple] = []   # (u, v, in_frame, red, green, color) for annotation
        for obs in obstacles:
            P_cam = self._world_to_camera(obs.centroid.reshape(1, 2), ego_pos, yaw)
            if P_cam[0, 2] <= 0:
                notes.append("behind-camera")
                continue

            pts, _ = cv2.projectPoints(
                P_cam.astype(np.float32),
                np.zeros(3), np.zeros(3),
                self._intrinsics.K,
                self._intrinsics.dist_coeffs,
            )
            u = int(pts[0, 0, 0])
            v = int(pts[0, 0, 1])

            if not (0 <= u < w and 0 <= v < h):
                notes.append(f"off-frame(u={u},v={v})")
                continue

            color, red_ratio, green_ratio = self._classify_color(hsv, u, v, h, w)
            if color is not None:
                self._field_map.vote_obstacle_color(obs, color)
            notes.append(f"u={u},v={v},red={red_ratio:.2f},green={green_ratio:.2f}"
                         f"->{color.value if color else 'none'}")
            marks.append((u, v, True, red_ratio, green_ratio, color))

        if report:
            print(f"[cam] frame {self._frame_count}: {w}x{h}, {len(obstacles)} obs | "
                  + " ; ".join(notes))

        self._save_debug_frame(frame, marks)

    def _save_debug_frame(self, frame: np.ndarray, marks: list[tuple]) -> None:
        # Write an annotated frame to disk so the camera view can be inspected
        # offline: the sample point, the patch box, and the red/green ratios. Only
        # every _save_every-th frame, capped, so it can't fill the disk.
        if self._save_dir is None or self._saved >= self._save_cap:
            return
        if self._frame_count % self._save_every != 0:
            return

        import os
        os.makedirs(self._save_dir, exist_ok = True)
        img = frame.copy()
        r   = self._sample_radius
        for u, v, _in, red_ratio, green_ratio, color in marks:
            hit = (0, 0, 255) if color and color.value == 'red' else \
                  (0, 255, 0) if color and color.value == 'green' else (0, 255, 255)
            cv2.rectangle(img, (u - r, v - r), (u + r, v + r), hit, 2)
            cv2.drawMarker(img, (u, v), hit, cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, f"R{red_ratio:.2f} G{green_ratio:.2f} {color.value if color else 'none'}",
                        (max(0, u - 60), max(15, v - r - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hit, 1, cv2.LINE_AA)
        cv2.putText(img, f"frame {self._frame_count}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        path = f"{self._save_dir}/frame_{self._frame_count:05d}.jpg"
        if cv2.imwrite(path, img):
            self._saved += 1

    def _world_to_camera(
        self,
        world_points: np.ndarray,
        ego_pos:      np.ndarray,
        yaw:          float,
    ) -> np.ndarray:
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        R_w2e = np.array([
            [ cos_yaw, -sin_yaw],
            [ sin_yaw,  cos_yaw],
        ])

        d_xy     = world_points - ego_pos[[1, 0]]
        P_ego_xy = (R_w2e @ d_xy.T).T

        N     = len(world_points)
        P_ego = np.column_stack([P_ego_xy, np.zeros(N)])

        P_rel = P_ego - self._extrinsics.translation
        P_cam = (self._extrinsics.R @ P_rel.T).T

        return P_cam

    def _classify_color(
        self,
        hsv: np.ndarray,
        u:   int,
        v:   int,
        h:   int,
        w:   int,
    ) -> tuple[ObstacleColor | None, float, float]:
        r     = self._sample_radius
        patch = hsv[
            max(0, v - r) : min(h, v + r),
            max(0, u - r) : min(w, u + r),
        ]
        if patch.size == 0:
            return None, 0.0, 0.0

        total = patch.shape[0] * patch.shape[1]

        red_mask = (
            cv2.inRange(patch, self._obstacle_color_ranges.red_lower_1,
                               self._obstacle_color_ranges.red_upper_1) |
            cv2.inRange(patch, self._obstacle_color_ranges.red_lower_2,
                               self._obstacle_color_ranges.red_upper_2)
        )
        green_mask = cv2.inRange(patch, self._obstacle_color_ranges.green_lower,
                                        self._obstacle_color_ranges.green_upper)

        red_ratio   = np.count_nonzero(red_mask)   / total
        green_ratio = np.count_nonzero(green_mask) / total

        if red_ratio   >= self._min_color_ratio and red_ratio   > green_ratio:
            return ObstacleColor.RED, red_ratio, green_ratio
        if green_ratio >= self._min_color_ratio and green_ratio > red_ratio:
            return ObstacleColor.GREEN, red_ratio, green_ratio
        return None, red_ratio, green_ratio
