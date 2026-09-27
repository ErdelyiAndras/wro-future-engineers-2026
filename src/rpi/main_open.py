import time
import argparse

from gpiozero import Button

from components import Arduino, Lidar
from control import EgoInformation, FieldMap
from processors import (
    ArduinoProcessor,
    DirectionDetector,
    LidarProcessor,
    OpenChallengePathPlanner,
)
from utils import mm, degree, Point

_WHEELBASE:          mm     = 89.6251
_LIDAR_BAUDRATE:     int    = 115200
_LIDAR_OFFSET_ANGLE: degree = -90.0
_LIDAR_MOUNT_OFFSET: Point  = (0.0, 73.37)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar-port',   default = '/dev/lidar',   help = 'LiDAR serial port')
    parser.add_argument('--arduino-port', default = '/dev/arduino', help = 'Arduino serial port')
    parser.add_argument('--button-pin',   default = 17, type = int, help = 'BCM GPIO pin for the start button')
    return parser.parse_args()


def main():
    args = parse_args()

    start_button = Button(args.button_pin, pull_up = True)

    ego_information = EgoInformation(wheelbase = _WHEELBASE)
    field_map       = FieldMap()

    lidar   = Lidar(port = args.lidar_port, baudrate = _LIDAR_BAUDRATE)
    arduino = Arduino(port = args.arduino_port)

    lidar_processor = LidarProcessor(
        field_map,
        ego_information,
        offset_angle = _LIDAR_OFFSET_ANGLE,
        mount_offset = _LIDAR_MOUNT_OFFSET,
    )
    direction_detector = DirectionDetector(field_map)
    planner = OpenChallengePathPlanner(
        field_map,
        ego_information,
        lidar_offset_angle = _LIDAR_OFFSET_ANGLE,
    )
    arduino_processor = ArduinoProcessor(ego_information)

    lidar.on_scan    += lidar_processor
    lidar.on_scan    += direction_detector
    lidar.on_scan    += planner

    arduino.on_state    += arduino_processor
    planner.on_target   += arduino.set_target
    planner.on_stop     += arduino.stop

    with lidar, arduino:
        print("Ready. Press the start button to begin.")
        start_button.wait_for_press()
        print("Starting.")

        time.sleep(0.5)

        lidar.start()
        arduino.start()

        try:
            print("Running. Press the button again to stop.")
            start_button.wait_for_press()
            print("Stopping.")
        finally:
            arduino.stop()


if __name__ == '__main__':
    main()
