from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import map_coordinates

from control import FieldMap
from utils   import Point, mm, radian


class LidarPoseEstimator:
    _PYRAMID_LEVELS:        int    = 4
    _GN_ITERATIONS:         int    = 10
    _GN_CONVERGENCE_T:      mm     = 0.1
    _GN_CONVERGENCE_R:      radian = math.radians(0.05)
    _MIN_CONFIDENT_CELLS:   int    = 30
    _MIN_MATCH_SCORE:       float  = 1.0
    _MIN_SCORE_IMPROVEMENT: float  = 0.3
    _HIGH_CONFIDENCE_RATIO: float  = 0.73

    _HIGH_CONFIDENCE_THRESHOLD: float = (
        FieldMap.L_MIN + _HIGH_CONFIDENCE_RATIO * (FieldMap.L_MAX - FieldMap.L_MIN)
    )

    def __init__(
        self,
        field_map:    FieldMap,
        offset_angle: radian,
        mount_offset: Point,
    ) -> None:
        self._field_map:    FieldMap = field_map
        self._offset_angle: radian   = offset_angle
        self._mount_offset: Point    = mount_offset

    def estimate(
        self,
        scan:           np.ndarray,
        prior_position: Point  | None,
        prior_yaw:      radian | None,
    ) -> tuple[Point, radian] | None:
        if prior_position is None or prior_yaw is None:
            return None

        occupancy, _, _ = self._field_map.snapshot()

        scan_ego = self._scan_to_ego_cartesian(scan)
        if len(scan_ego) == 0:
            # print("[LidarPoseEstimator] No valid scan points, skipping.")
            return None

        confident_cells = int(np.sum(occupancy > self._HIGH_CONFIDENCE_THRESHOLD))
        if confident_cells < self._MIN_CONFIDENT_CELLS:
            # print(f"[LidarPoseEstimator] Bootstrapping: {confident_cells} / {self._MIN_CONFIDENT_CELLS} confident cells.")
            return None

        pyramid = self._build_pyramid(occupancy)

        px, py      = prior_position
        pose        = (px, py, prior_yaw)
        final_score = 0.0

        for level in range(self._PYRAMID_LEVELS - 1, -1, -1):
            pose, final_score = self._optimize_level(pyramid[level], pose, scan_ego, level)

        prior_pts   = self._ego_to_world(scan_ego, prior_position, prior_yaw)
        prior_row_f = (prior_pts[:, 1] + FieldMap.ORIGIN[1]) / FieldMap.CELL_SIZE
        prior_col_f = (prior_pts[:, 0] + FieldMap.ORIGIN[0]) / FieldMap.CELL_SIZE
        M_prior     = map_coordinates(
            pyramid[0],
            np.stack([prior_row_f, prior_col_f]),
            order = 1, mode = 'constant', cval = FieldMap.L_MIN, prefilter = False,
        )
        prior_score = float(np.mean(M_prior))
        improvement = final_score - prior_score

        est_px, est_py, est_yaw = pose
        dx     = est_px - px
        dy     = est_py - py
        dtheta = (est_yaw - prior_yaw + math.pi) % (2 * math.pi) - math.pi

        # print(
        #     f"[LidarPoseEstimator] "
        #     f"prior_score={prior_score:.3f}  "
        #     f"final_score={final_score:.3f}  "
        #     f"improvement={improvement:.3f}  "
        #     f"dx={dx:.1f}mm  dy={dy:.1f}mm  dθ={math.degrees(dtheta):.2f}°  "
        #     f"confident_cells={confident_cells}"
        # )

        if final_score < self._MIN_MATCH_SCORE:
            # print(
            #     f"[LidarPoseEstimator] REJECTED — final_score {final_score:.3f} "
            #     f"< MIN_MATCH_SCORE {self._MIN_MATCH_SCORE:.3f}"
            # )
            return None

        if improvement < self._MIN_SCORE_IMPROVEMENT:
            # print(
            #     f"[LidarPoseEstimator] REJECTED — improvement {improvement:.3f} "
            #     f"< MIN_SCORE_IMPROVEMENT {self._MIN_SCORE_IMPROVEMENT:.3f}"
            # )
            return None

        print(f"[LidarPoseEstimator] ACCEPTED")
        return (est_px, est_py), est_yaw

    @staticmethod
    def _build_pyramid(occupancy: np.ndarray) -> list[np.ndarray]:
        levels = [occupancy]
        prev   = occupancy
        for _ in range(3):
            h     = prev.shape[0] // 2
            w     = prev.shape[1] // 2
            level = prev[:h * 2, :w * 2].reshape(h, 2, w, 2).mean(axis = (1, 3)).astype(np.float32)
            levels.append(level)
            prev  = level
        return levels

    @staticmethod
    def _optimize_level(
        occ:      np.ndarray,
        pose:     tuple[float, float, float],
        scan_ego: np.ndarray,
        level:    int,
    ) -> tuple[tuple[float, float, float], float]:
        cell_size          = FieldMap.CELL_SIZE * (2 ** level)
        grad_row, grad_col = np.gradient(occ)
        mc  = dict(order = 1, mode = 'constant', cval = FieldMap.L_MIN, prefilter = False)
        sx  = scan_ego[:, 0]
        sy  = scan_ego[:, 1]

        px, py, yaw = pose

        for _ in range(LidarPoseEstimator._GN_ITERATIONS):
            sin_yaw = math.sin(yaw)
            cos_yaw = math.cos(yaw)

            x_world =  sx * cos_yaw + sy * sin_yaw + px
            y_world = -sx * sin_yaw + sy * cos_yaw + py

            row_f  = (y_world + FieldMap.ORIGIN[1]) / cell_size
            col_f  = (x_world + FieldMap.ORIGIN[0]) / cell_size
            coords = np.stack([row_f, col_f])

            M      = map_coordinates(occ,      coords, **mc)
            dM_row = map_coordinates(grad_row, coords, **mc)
            dM_col = map_coordinates(grad_col, coords, **mc)

            dM_dy    = dM_row / cell_size
            dM_dx    = dM_col / cell_size
            dTx_dyaw = -sx * sin_yaw + sy * cos_yaw
            dTy_dyaw = -sx * cos_yaw - sy * sin_yaw

            J     = np.column_stack([dM_dx, dM_dy, dM_dx * dTx_dyaw + dM_dy * dTy_dyaw])
            delta = np.linalg.lstsq(J.T @ J, J.T @ M, rcond = None)[0]

            px  += delta[0]
            py  += delta[1]
            yaw += delta[2]

            if abs(delta[0]) < LidarPoseEstimator._GN_CONVERGENCE_T and \
               abs(delta[1]) < LidarPoseEstimator._GN_CONVERGENCE_T and \
               abs(delta[2]) < LidarPoseEstimator._GN_CONVERGENCE_R:
                break

        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        x_world =  sx * cos_yaw + sy * sin_yaw + px
        y_world = -sx * sin_yaw + sy * cos_yaw + py
        row_f   = (y_world + FieldMap.ORIGIN[1]) / cell_size
        col_f   = (x_world + FieldMap.ORIGIN[0]) / cell_size
        M_final = map_coordinates(occ, np.stack([row_f, col_f]), **mc)

        return (px, py, yaw), float(np.mean(M_final))

    def _scan_to_ego_cartesian(self, scan: np.ndarray) -> np.ndarray:
        if len(scan) == 0:
            return np.empty((0, 2))

        raw       = np.array(scan, dtype = np.float64)
        angles    = raw[:, 0] + self._offset_angle
        distances = raw[:, 1]

        valid     = distances > 0.0
        angles    = angles[valid]
        distances = distances[valid]

        mount_x, mount_y = self._mount_offset
        x = distances * np.sin(angles) + mount_x
        y = distances * np.cos(angles) + mount_y

        return np.column_stack((x, y))

    @staticmethod
    def _ego_to_world(
        pts_ego:  np.ndarray,
        position: Point,
        yaw:      radian,
    ) -> np.ndarray:
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        px, py  = position

        x_world =  pts_ego[:, 0] * cos_yaw + pts_ego[:, 1] * sin_yaw + px
        y_world = -pts_ego[:, 0] * sin_yaw + pts_ego[:, 1] * cos_yaw + py

        return np.column_stack((x_world, y_world))
