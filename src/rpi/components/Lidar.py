from pyrplidar import PyRPlidar
from utils.Event import Event
from components.Component import Component
import threading
import time

class Lidar(Component):
    def __init__(self, port = "/dev/ttyUSB0", baudrate = 115200, timeout = 3):
        self.on_point = Event()

        self._is_running = False
        self._processor_thread = None

        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._lidar = PyRPlidar()

    def _setup(self):
        self._lidar.connect(port = self._port, baudrate = self._baudrate, timeout = self._timeout)
        time.sleep(2)
        self._lidar.set_motor_pwm(660)
        time.sleep(2)

    def _teardown(self):
        self.stop()
        lidar = self._lidar
        self._lidar = None
        if lidar:
            lidar.set_motor_pwm(0)
            lidar.disconnect()

    def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._processor_thread = threading.Thread(target = self._scan, name = "Lidar::scan", daemon = True)
        self._processor_thread.start()

    def stop(self):
        if not self._is_running:
            return
        self._is_running = False
        if self._lidar:
            self._lidar.stop()
        if self._processor_thread:
            self._processor_thread.join(timeout = 3)
            self._processor_thread = None

    def _flush_serial(self):
        self._lidar.lidar_serial._serial.reset_input_buffer()

    def _scan(self):
        self._flush_serial()
        for point in self._lidar.start_scan()():
            if not self._is_running:
                break
            if point.quality == 0 or point.distance <= 0:
                continue
            self.on_point(point.angle, point.distance, point.quality)
