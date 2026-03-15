from __future__ import annotations
from pyrplidar import PyRPlidar
from utils import Event
from components.Component import Component
import time
import math

class Lidar(Component):
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, timeout: int = 3) -> None:
        super().__init__()
        self.on_point: Event = Event()

        self._port:     str = port
        self._baudrate: int = baudrate
        self._timeout:  int = timeout
        self._lidar: PyRPlidar | None = PyRPlidar()

    def _setup(self) -> None:
        self._lidar.connect(port = self._port, baudrate = self._baudrate, timeout = self._timeout)
        time.sleep(2)
        self._lidar.set_motor_pwm(200)
        time.sleep(2)

    def _teardown(self) -> None:
        lidar = self._lidar
        self._lidar = None
        if lidar:
            try:
                lidar.set_motor_pwm(0)
                lidar.disconnect()
            except Exception:
                pass

    def _on_stop(self) -> None:
        if self._lidar:
            self._lidar.stop()

    def _flush_serial(self) -> None:
        self._lidar.lidar_serial._serial.reset_input_buffer()

    def _run(self) -> None:
        self._flush_serial()
        for p in self._lidar.start_scan_express(0)():
            if not self._is_running:
                break
            if p.quality == 0 or p.distance <= 0:
                continue
            self.on_point(math.radians(p.angle), p.distance, p.quality)
