import time
import argparse

import numpy as np

from gpiozero import Button

from components import Arduino, Camera, Lidar, Ticker
from control import EgoInformation, FieldMap
from processors import (
    ArduinoProcessor,
    CameraProcessor,
    CollisionGuard,
    DirectionDetector,
    LidarProcessor,
    PathPlanningProcessor,
    GeometricTrackPlanner,
    SemanticClassifier,
    ObstacleColorRanges,
    CameraIntrinsics,
    CameraExtrinsics,
)
from utils import mm, degree, Point

# ego information
_WHEELBASE: mm = 89.6251

# lidar processor
_LIDAR_BAUDRATE:     int    = 115200
_LIDAR_OFFSET_ANGLE: degree = -90.0
_LIDAR_MOUNT_OFFSET: Point  = (0.0, 73.37)

# camera processor
_CAMERA_INTRINSICS = CameraIntrinsics(
    fx = 658.3516186667142,
    fy = 656.9839014361173,
    cx = 312.90122361261507,
    cy = 178.888473418321,
    dist_coeffs = np.array([
        -0.17756057290232585,
        -0.07620432808648203,
        0.0014254009734183857,
        0.0009633670165933481,
        -0.3014236479533136], dtype = np.float64
    )
)
_CAMERA_EXTRINSICS = CameraExtrinsics(
    x     = 0.0,
    y     = 134.696,
    z     = 31.0747,
    pitch = 0.0,
    yaw   = 0.0,
    roll  = 0.0
)
_OBSTACLE_COLOR_RANGES = ObstacleColorRanges(
    red_lower_1 = np.array([  0, 149, 134], dtype = np.uint8),
    red_upper_1 = np.array([ 10, 248, 228], dtype = np.uint8),
    red_lower_2 = np.array([170, 149, 134], dtype = np.uint8),
    red_upper_2 = np.array([180, 248, 228], dtype = np.uint8),
    green_lower = np.array([ 36,  40,  90], dtype = np.uint8),
    green_upper = np.array([ 97, 164, 200], dtype = np.uint8),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar-port',   default = '/dev/lidar',   help = 'LiDAR serial port')
    parser.add_argument('--arduino-port', default = '/dev/arduino', help = 'Arduino serial port')
    parser.add_argument('--camera-port',  default = 0, type = int,  help = 'Camera device index')
    parser.add_argument('--button-pin',   default = 17, type = int, help = 'BCM GPIO pin for the start button')
    return parser.parse_args()


def main():
    args = parse_args()

    start_button = Button(args.button_pin, pull_up = True)

    ego_information = EgoInformation(wheelbase = _WHEELBASE)
    field_map = FieldMap()

    lidar = Lidar(
        port     = args.lidar_port,
        baudrate = _LIDAR_BAUDRATE,
    )
    camera = Camera(
        port = args.camera_port
    )
    arduino = Arduino(
        port = args.arduino_port
    )
    classification_ticker = Ticker(interval = 0.1)
    planning_ticker       = Ticker(interval = 0.2)

    lidar_processor = LidarProcessor(
        field_map,
        ego_information,
        offset_angle = _LIDAR_OFFSET_ANGLE,
        mount_offset = _LIDAR_MOUNT_OFFSET
    )
    semantic_classifier = SemanticClassifier(
        field_map
    )
    camera_processor = CameraProcessor(
        field_map,
        ego_information,
        intrinsics            = _CAMERA_INTRINSICS,
        extrinsics            = _CAMERA_EXTRINSICS,
        obstacle_color_ranges = _OBSTACLE_COLOR_RANGES
    )
    path_planning_processor = PathPlanningProcessor(
        field_map,
        ego_information
    )
    arduino_processor = ArduinoProcessor(
        ego_information
    )
    collision_guard = CollisionGuard(
        lidar_offset_angle = _LIDAR_OFFSET_ANGLE
    )
    direction_detector = DirectionDetector(
        field_map
    )

    lidar.on_scan                     += lidar_processor
    lidar.on_scan                     += collision_guard
    lidar.on_scan                     += direction_detector

    camera.on_frame                   += camera_processor

    classification_ticker.on_tick     += semantic_classifier

    planning_ticker.on_tick           += path_planning_processor

    arduino.on_state                  += arduino_processor
    path_planning_processor.on_target += arduino.set_target
    path_planning_processor.on_stop   += arduino.stop
    collision_guard.on_collision      += path_planning_processor.notify_collision

    with lidar, camera, arduino, classification_ticker, planning_ticker:
        print("Ready. Press the start button to begin.")
        start_button.wait_for_press()
        print("Starting.")

        time.sleep(0.5)

        lidar.start()
        camera.start()
        arduino.start()
        classification_ticker.start()
        planning_ticker.start()

        try:
            print("Running. Press the button again to stop.")
            start_button.wait_for_press()
            print("Stopping.")
        finally:
            arduino.stop()


if __name__ == '__main__':
    main()
