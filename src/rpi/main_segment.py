import time
import argparse
import traceback
import webbrowser
from contextlib import ExitStack

import numpy as np

from gpiozero import Button

from components import Arduino, Camera, Lidar
from control import EgoInformation, TrackModel, ColorRange
from processors import (
    ArduinoProcessor,
    BodyFrameColorSampler,
    CollisionGuard,
    PrincipalAngleDetector,
    ReactiveSegmentPlanner,
    CameraIntrinsics,
    CameraExtrinsics,
)
from recording import Recorder
from utils import mm, degree, Point

from LidarVisualizer import LidarVisualizer
from SegmentVisualizer import SegmentVisualizer
from ColorVisualizer import ColorVisualizer

# Single entry point for the reactive segment planner (open + obstacle challenge).
# The planner drives from the raw LiDAR scan, the gyro heading, and the wheel
# odometry; the camera only supplies pillar colour via the stateless body-frame
# sampler and is a harmless no-op when no obstacle is queried. Pass --no-camera to
# skip it for the open challenge or bench runs without a camera.
#
# Parameters below mirror main_ftg.py exactly (wheelbase, LiDAR, camera).

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
# Obstacle + parking-wall colours. Each is a ColorRange = union of HSV bands (OpenCV
# H 0-180); a hue that wraps the 0/180 seam (red, or the magenta parking wall) uses two
# bands. Paste calibration output straight into these lists.
_RED_COLOR = ColorRange(bands = [
    (np.array([  0,  43,   4], dtype = np.uint8), np.array([ 10, 255, 255], dtype = np.uint8)),
    (np.array([170,  43,   4], dtype = np.uint8), np.array([180, 255, 255], dtype = np.uint8)),
])
_GREEN_COLOR = ColorRange(bands = [
    (np.array([ 33,  24,  28], dtype = np.uint8), np.array([ 97, 255, 231], dtype = np.uint8)),
])
_PARKING_COLOR = ColorRange(bands = [
    (np.array([  0, 150,  77], dtype = np.uint8), np.array([ 10, 217, 165], dtype = np.uint8)),
    (np.array([170, 150,  77], dtype = np.uint8), np.array([180, 217, 165], dtype = np.uint8)),
])


def guarded(handler, name):
    """Wrap an event handler so one exception can't silently kill its thread."""
    def wrapped(*args, **kwargs):
        try:
            handler(*args, **kwargs)
        except Exception:
            print(f"[{name}] handler error (continuing):")
            traceback.print_exc()
    return wrapped


def parse_args():
    parser = argparse.ArgumentParser(description = 'Reactive segment planner (open + obstacle challenge)')
    parser.add_argument('--lidar-port',   default = '/dev/lidar',   help = 'LiDAR serial port')
    parser.add_argument('--arduino-port', default = '/dev/arduino', help = 'Arduino serial port')
    parser.add_argument('--camera-port',  default = 0, type = int,  help = 'Camera device index')
    parser.add_argument('--button-pin',   default = 17, type = int, help = 'BCM GPIO pin for the start button')
    parser.add_argument('--no-button',    action = 'store_true', help = 'auto-start ~2 s after launch instead of waiting for the button')
    parser.add_argument('--no-camera',    action = 'store_true', help = 'skip the camera + colour reads (open challenge / no camera)')
    parser.add_argument('--parking-start', action = 'store_true', help = 'the run starts parked in the parking slot (play the leave maneuver first)')
    parser.add_argument('--park-mode', default = 'none',
                        choices = ('none', 'perpendicular', 'parallel_mirror', 'parallel_reverse'),
                        help = 'final parking maneuver at the end of the run (default: none = just stop)')
    parser.add_argument('--park-exit-side', default = 'auto', choices = ('auto', 'left', 'right'),
                        help = 'with --parking-start, which way the leave maneuver exits: '
                               'left=CCW, right=CW, auto=vote the LiDAR opening (default)')
    parser.add_argument('--no-record',    action = 'store_true', help = 'disable signal recording')
    parser.add_argument('--visu',         action = 'store_true', help = 'start the segment-planner + LiDAR visualizers')
    parser.add_argument('--no-browser',   action = 'store_true', help = 'with --visu, do not open the visualizers in a browser')
    parser.add_argument('--state-pause',  nargs = '?', const = 2.0, default = 0.0, type = float,
                        metavar = 'SEC',
                        help = 'debug: hold the robot stopped for SEC seconds at each state '
                               'transition (default 2 s when the flag is given, off otherwise)')
    return parser.parse_args()


def main():
    args = parse_args()

    ego_information = EgoInformation(wheelbase = _WHEELBASE)
    track           = TrackModel()

    lidar   = Lidar(port = args.lidar_port, baudrate = _LIDAR_BAUDRATE)
    arduino = Arduino(port = args.arduino_port)

    arduino_processor = ArduinoProcessor(ego_information)

    # Colour subsystem: the camera keeps the latest frame in the sampler; the planner
    # queries it on demand with a body-frame obstacle point. Skipped with --no-camera.
    camera:        Camera | None                = None
    color_sampler: BodyFrameColorSampler | None = None
    if not args.no_camera:
        camera        = Camera(port = args.camera_port)
        color_sampler = BodyFrameColorSampler(
            intrinsics = _CAMERA_INTRINSICS,
            extrinsics = _CAMERA_EXTRINSICS,
            red        = _RED_COLOR,
            green      = _GREEN_COLOR,
            parking    = _PARKING_COLOR,
        )
        camera.on_frame += guarded(color_sampler, 'camera')

    # PrincipalAngleDetector fits the track's principal angle into the shared TrackModel;
    # the planner reads it from there. It MUST be subscribed to on_scan before the planner
    # so it runs first on the LiDAR thread and the planner sees a fresh theta each scan.
    angle_detector = PrincipalAngleDetector(ego_information, track)

    planner = ReactiveSegmentPlanner(
        ego_information,
        track,
        color_sampler      = color_sampler,
        lidar_mount_offset = _LIDAR_MOUNT_OFFSET,
        state_hold_s       = args.state_pause,
        start_in_parking   = args.parking_start,
        park_mode          = args.park_mode,
        park_exit_side     = args.park_exit_side,
    )
    # collision_guard = CollisionGuard(lidar_offset_angle = _LIDAR_OFFSET_ANGLE)

    # lidar.on_scan += collision_guard
    lidar.on_scan += angle_detector
    lidar.on_scan += planner

    arduino.on_state  += arduino_processor
    planner.on_target += arduino.set_target
    planner.on_stop   += arduino.stop

    # Recorder taps go after the wiring above so ego_pose is sampled after
    # ArduinoProcessor has ingested the same state message.
    recorder = Recorder.create_default(enabled = not args.no_record)
    recorder.attach_scan(lidar.on_scan, ego = ego_information)
    recorder.attach_array(arduino.on_state, 'arduino_state',
                          fields = ('heading_rad', 'speed_mm_s', 'steering_rad', 'distance_mm'))
    recorder.attach_pose(arduino.on_state, ego_information)
    # recorder.attach_marker(collision_guard.on_collision, 'collision')
    recorder.attach_planner(planner)

    visualizers: list = []
    if args.visu:
        segment_visualizer = SegmentVisualizer()
        segment_visualizer.start()
        planner.on_debug += segment_visualizer.set_debug
        visualizers.append(segment_visualizer)

        lidar_visualizer = LidarVisualizer(offset_angle = _LIDAR_OFFSET_ANGLE)
        lidar_visualizer.start()
        lidar.on_scan += lidar_visualizer
        visualizers.append(lidar_visualizer)

        # Obstacle-colour view: only meaningful when the camera + sampler are wired.
        if color_sampler is not None:
            color_visualizer = ColorVisualizer(color_sampler)
            color_visualizer.start()
            camera.on_frame += color_visualizer
            visualizers.append(color_visualizer)

        if not args.no_browser:
            for vis in visualizers:
                webbrowser.open(f"http://localhost:{vis._DEFAULT_PORT}")

    start_button = None if args.no_button else Button(args.button_pin, pull_up = True)

    with ExitStack() as stack:
        stack.enter_context(recorder)
        stack.enter_context(lidar)
        stack.enter_context(arduino)
        if camera is not None:
            stack.enter_context(camera)

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
