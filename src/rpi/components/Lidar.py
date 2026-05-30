from __future__ import annotations

import math
import time

import numpy as np

from adafruit_rplidar import RPLidar, RPLidarException, SCAN_TYPE_NORMAL, SCAN_TYPE_EXPRESS

from components.Component import Component
from utils import Event, mm, radian

class Lidar(Component):
    MOTOR_PWM:    int = 660
    MAX_BUF_MEAS: int = 500

    def __init__(
        self,
        port:     str = "/dev/lidar",
        baudrate: int = 115200,
        timeout:  int = 3
    ) -> None:
        super().__init__()

        self.on_scan: Event = Event()

        self._port:     str = port
        self._baudrate: int = baudrate
        self._timeout:  int = timeout

        self._lidar: RPLidar | None = None

        self._buffer: list[tuple[radian, mm, int]] = []

    def _setup(self) -> None:
        self._lidar = RPLidar(
            None,
            self._port,
            baudrate = self._baudrate,
            timeout  = self._timeout,
        )
        time.sleep(2)
        self._lidar.set_pwm(self.MOTOR_PWM)
        time.sleep(2)

    def _teardown(self) -> None:
        lidar       = self._lidar
        self._lidar = None
        if lidar:
            try:
                lidar.set_pwm(0)
            except Exception:
                pass
            try:
                lidar.stop()
            except Exception:
                pass
            try:
                lidar.disconnect()
            except Exception:
                pass

    def _on_stop(self) -> None:
        if self._lidar:
            try:
                self._lidar.stop()
            except Exception:
                pass
            try:
                self._lidar.set_pwm(0)
            except Exception:
                pass
            try:
                self._lidar.disconnect()
            except BaseException:
                pass

    def _flush_serial(self) -> None:
        try:
            self._lidar._serial_port.reset_input_buffer()
        except Exception:
            pass

    def _run(self) -> None:
        self._flush_serial()

        try:
            iterator = self._lidar.iter_measurements(
                max_buf_meas = self.MAX_BUF_MEAS,
                scan_type    = SCAN_TYPE_NORMAL
            )

            self._lidar.set_pwm(self.MOTOR_PWM)

            for new_scan, quality, angle, distance in iterator:
                if not self._is_running:
                    break

                if new_scan and self._buffer:
                    self.on_scan(np.array(self._buffer, dtype = np.float64))
                    self._buffer = []

                if quality == 0 or distance <= 0:
                    continue

                self._buffer.append((math.radians(angle), distance, quality))

        except OSError:
            pass
        except RPLidarException:
            if self._is_running:
                raise
