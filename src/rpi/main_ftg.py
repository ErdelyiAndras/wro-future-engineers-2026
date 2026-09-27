import time
import argparse
import traceback
import webbrowser
from contextlib import ExitStack

import numpy as np

from gpiozero import Button

from components import Arduino, Camera, Lidar, Ticker
from control import EgoInformation, FieldMap
from processors import (
    ArduinoProcessor,
    CameraProcessor,
    CollisionGuard,
    FollowTheGapPlanner,
    LidarProcessor,
    SemanticClassifier,
    ObstacleColorRanges,
    CameraIntrinsics,
    CameraExtrinsics,
)
from recording import Recorder
from utils import mm, degree, Point

from FieldMapVisualiser import FieldMapVisualizer
from FTGVisualizer import FTGVisualizer
from LidarVisualizer import LidarVisualizer

# Single entry point for both the open and obstacle challenges. The Follow-the-Gap
# planner drives in both; the camera + SemanticClassifier obstacle subsystem is
# only meaningful when coloured pillars are present, and is otherwise a harmless
# no-op (nothing detected -> the pass-side mask does nothing). Pass --no-camera to
# skip it entirely for the open challenge or bench runs without a camera.

_WHEELBASE:          mm     = 89.6251

_LIDAR_BAUDRATE:     int    = 115200
_LIDAR_OFFSET_ANGLE: degree = -90.0
_LIDAR_MOUNT_OFFSET: Point  = (0.0, 73.37)

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
    x = 0.0, y = 134.696, z = 31.0747, pitch = 0.0, yaw = 0.0, roll = 0.0
)
_OBSTACLE_COLOR_RANGES = ObstacleColorRanges(
    red_lower_1 = np.array([  0, 158, 106], dtype = np.uint8),
    red_upper_1 = np.array([ 26, 241, 173], dtype = np.uint8),
    red_lower_2 = np.array([  0, 158, 106], dtype = np.uint8),
    red_upper_2 = np.array([ 26, 241, 173], dtype = np.uint8),
    green_lower = np.array([ 43, 132,  35], dtype = np.uint8),
    green_upper = np.array([ 93, 255, 113], dtype = np.uint8),
)


def guarded(handler, name):
    """Wrap an event handler so one exception can't silently kill its thread.

    Event dispatch and the component thread body do not catch exceptions, so a
    single bad frame/tick in the camera or classifier would otherwise take the
    whole obstacle subsystem down for the rest of the run with no trace. Print
    and carry on instead.
    """
    def wrapped(*args, **kwargs):
        try:
            handler(*args, **kwargs)
        except Exception:
            print(f"[{name}] handler error (continuing):")
            traceback.print_exc()
    return wrapped


def parse_args():
    parser = argparse.ArgumentParser(description = 'Follow-the-Gap driver (open + obstacle challenge)')
    parser.add_argument('--lidar-port',   default = '/dev/lidar',   help = 'LiDAR serial port')
    parser.add_argument('--arduino-port', default = '/dev/arduino', help = 'Arduino serial port')
    parser.add_argument('--camera-port',  default = 0, type = int,  help = 'Camera device index')
    parser.add_argument('--button-pin',   default = 17, type = int, help = 'BCM GPIO pin for the start button')
    parser.add_argument('--no-button',    action = 'store_true', help = 'auto-start ~2 s after launch instead of waiting for the button')
    parser.add_argument('--no-camera',    action = 'store_true', help = 'skip the camera + obstacle detection (open challenge / no camera)')
    parser.add_argument('--no-memory',    action = 'store_true', help = 'disable occupancy-grid memory fusion')
    parser.add_argument('--no-pass-side', action = 'store_true', help = 'disable the colour pass-side mask')
    parser.add_argument('--no-record',    action = 'store_true', help = 'disable signal recording')
    parser.add_argument('--visu',         action = 'store_true', help = 'start the field-map / FTG / LiDAR visualizers')
    parser.add_argument('--no-browser',   action = 'store_true', help = 'with --visu, do not open the visualizers in a browser')
    return parser.parse_args()


def main():
    args = parse_args()

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
    planner = FollowTheGapPlanner(
        field_map,
        ego_information,
        lidar_offset_angle = _LIDAR_OFFSET_ANGLE,
        lidar_mount_offset = _LIDAR_MOUNT_OFFSET,
        use_memory         = not args.no_memory,
        use_pass_side      = not args.no_pass_side,
    )
    collision_guard   = CollisionGuard(lidar_offset_angle = _LIDAR_OFFSET_ANGLE)
    arduino_processor = ArduinoProcessor(ego_information)

    # LidarProcessor must run before the planner: it accumulates the occupancy
    # grid the planner reads for memory fusion and (via SemanticClassifier) for
    # obstacles.
    lidar.on_scan += lidar_processor
    lidar.on_scan += collision_guard
    lidar.on_scan += planner

    arduino.on_state             += arduino_processor
    planner.on_target            += arduino.set_target
    planner.on_stop              += arduino.stop
    collision_guard.on_collision += planner.notify_collision

    # Obstacle subsystem: LiDAR clusters -> obstacles (SemanticClassifier),
    # camera -> colour votes (CameraProcessor). The planner reads obs.color to
    # decide the pass side. Skipped entirely with --no-camera.
    camera:                Camera | None = None
    classification_ticker: Ticker | None = None
    if not args.no_camera:
        camera                = Camera(port = args.camera_port)
        classification_ticker = Ticker(interval = 0.1)
        semantic_classifier   = SemanticClassifier(field_map)
        camera_processor      = CameraProcessor(
            field_map,
            ego_information,
            intrinsics            = _CAMERA_INTRINSICS,
            extrinsics            = _CAMERA_EXTRINSICS,
            obstacle_color_ranges = _OBSTACLE_COLOR_RANGES,
        )
        camera.on_frame               += guarded(camera_processor, 'camera')
        classification_ticker.on_tick += guarded(semantic_classifier, 'classifier')

    # Recorder taps go after the wiring above so ego_pose is sampled after
    # ArduinoProcessor has ingested the same state message.
    recorder = Recorder.create_default(enabled = not args.no_record)
    recorder.attach_scan(lidar.on_scan, ego = ego_information)
    recorder.attach_array(arduino.on_state, 'arduino_state',
                          fields = ('heading_rad', 'speed_mm_s', 'steering_rad', 'distance_mm'))
    recorder.attach_pose(arduino.on_state, ego_information)
    recorder.attach_marker(collision_guard.on_collision, 'collision')
    recorder.attach_planner(planner)

    visualizers: list = []
    if args.visu:
        field_map_visualizer = FieldMapVisualizer(field_map, ego_information)
        field_map_visualizer.start()
        planner.on_target += field_map_visualizer.set_target
        visualizers.append(field_map_visualizer)

        ftg_visualizer = FTGVisualizer()
        ftg_visualizer.start()
        planner.on_debug += ftg_visualizer.set_debug
        visualizers.append(ftg_visualizer)

        lidar_visualizer = LidarVisualizer(offset_angle = _LIDAR_OFFSET_ANGLE)
        lidar_visualizer.start()
        lidar.on_scan += lidar_visualizer
        visualizers.append(lidar_visualizer)

        if not args.no_browser:
            for vis in (field_map_visualizer, ftg_visualizer, lidar_visualizer):
                webbrowser.open(f"http://localhost:{vis._DEFAULT_PORT}")

    start_button = None if args.no_button else Button(args.button_pin, pull_up = True)

    with ExitStack() as stack:
        stack.enter_context(recorder)
        stack.enter_context(lidar)
        stack.enter_context(arduino)
        if camera is not None:
            stack.enter_context(camera)
            stack.enter_context(classification_ticker)

        if args.no_button:
            time.sleep(2)
        else:
            print("Ready. Press the start button to begin.")
            start_button.wait_for_press()
            print("Starting.")
            time.sleep(0.5)

        lidar.start()
        arduino.start()
        if camera is not None:
            camera.start()
            classification_ticker.start()

        try:
            if args.no_button:
                print("Running. Ctrl-C to stop.")
                while True:
                    time.sleep(1)
            else:
                print("Running. Press the button again to stop.")
                start_button.wait_for_press()
                print("Stopping.")
        except KeyboardInterrupt:
            pass
        finally:
            arduino.stop()

    for vis in visualizers:
        vis.stop()


if __name__ == '__main__':
    main()
