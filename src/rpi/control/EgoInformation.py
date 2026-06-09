from __future__ import annotations

import math
from threading import Lock

import numpy as np

from utils import Point, mm, radian

class EgoInformation:
    _INITIAL_UNCERTAINTY:  float  = 1e4
    _PROCESS_NOISE_V:      mm     = 5.0
    _PROCESS_NOISE_W:      radian = 0.02
    _MEAS_NOISE_LIDAR_XY:  mm     = 10.0
    _MEAS_NOISE_LIDAR_HDG: radian = 0.05
    _MEAS_NOISE_IMU_HDG:   radian = 0.05

    def __init__(
        self,
        wheelbase:       mm     = 100.0,
        initial_pos:     Point  = (0.0, 0.0),
        initial_heading: radian = 0.0,
    ) -> None:
        self._wheelbase: mm   = wheelbase
        self._lock:      Lock = Lock()

        self._x = np.array(
            [initial_pos[0], initial_pos[1], initial_heading],
            dtype = np.float64
        )
        self._P = np.eye(3) * self._INITIAL_UNCERTAINTY

        self._R_lidar   = np.diag([
            self._MEAS_NOISE_LIDAR_XY  ** 2,
            self._MEAS_NOISE_LIDAR_XY  ** 2,
            self._MEAS_NOISE_LIDAR_HDG ** 2,
        ])
        self._R_imu_hdg = self._MEAS_NOISE_IMU_HDG ** 2

    def reset_ego_information(self, position: Point, yaw: radian) -> None:
        with self._lock:
            self._x[:] = [position[0], position[1], yaw]
            self._P[:] = np.eye(3) * self._INITIAL_UNCERTAINTY

    def update_odometry(self, ds: mm, steering: radian) -> None:
        with self._lock:
            x, y, h = self._x

            if abs(steering) < 1e-6:
                x_n = x + ds * math.cos(h)
                y_n = y + ds * math.sin(h)
                h_n = h
                F   = np.array([
                    [1.0, 0.0, -ds * math.sin(h)],
                    [0.0, 1.0,  ds * math.cos(h)],
                    [0.0, 0.0,  1.0             ],
                ])
            else:
                R_turn = self._wheelbase / math.tan(steering)
                dh     = ds / R_turn
                x_n    = x + R_turn * (math.sin(h + dh) - math.sin(h))
                y_n    = y + R_turn * (-math.cos(h + dh) + math.cos(h))
                h_n    = h + dh
                F      = np.array([
                    [1.0, 0.0, R_turn * (math.cos(h + dh) - math.cos(h))],
                    [0.0, 1.0, R_turn * (math.sin(h + dh) - math.sin(h))],
                    [0.0, 0.0, 1.0                                        ],
                ])

            abs_ds = abs(ds)
            Q = np.diag([
                self._PROCESS_NOISE_V ** 2 * abs_ds,
                self._PROCESS_NOISE_V ** 2 * abs_ds,
                self._PROCESS_NOISE_W ** 2 * abs_ds,
            ])

            self._x = np.array([x_n, y_n, self._wrap(h_n)])
            self._P = F @ self._P @ F.T + Q

    def update_lidar(self, x: mm, y: mm, heading: radian) -> None:
        with self._lock:
            z     = np.array([x, y, heading])
            innov = z - self._x
            innov[2] = self._wrap(innov[2])

            S = self._P + self._R_lidar
            K = self._P @ np.linalg.inv(S)

            self._x = self._x + K @ innov
            self._x[2] = self._wrap(self._x[2])
            self._P = (np.eye(3) - K) @ self._P

    def update_imu(self, heading: radian) -> None:
        with self._lock:
            innov = self._wrap(heading - self._x[2])
            s     = self._P[2, 2] + self._R_imu_hdg
            k     = self._P[:, 2] / s
            self._x    = self._x + k * innov
            self._x[2] = self._wrap(self._x[2])
            self._P    = (np.eye(3) - np.outer(k, [0.0, 0.0, 1.0])) @ self._P

    def get_ego_information(self) -> tuple[Point | None, radian | None]:
        with self._lock:
            return (self._x[0], self._x[1]), self._x[2]

    @property
    def position(self) -> Point:
        with self._lock:
            return (self._x[0], self._x[1])

    @property
    def yaw(self) -> radian:
        with self._lock:
            return self._x[2]

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi
