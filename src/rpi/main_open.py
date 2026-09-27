import time
import argparse
import webbrowser
from contextlib import ExitStack

from gpiozero import Button

from components import Arduino, Lidar
from control import EgoInformation, TrackModel
from processors import (
    ArduinoProcessor,
    PrincipalAngleDetector,
    OpenChallengePlanner,
)
from recording import Recorder
from utils import mm, degree, Point

from visu.LidarVisualizer import LidarVisualizer
from visu.PathPlanningVisualizer import PathPlanningVisualizer

# Entry point for the OPEN CHALLENGE using the minimal segment-style planner
# (OpenChallengePlanner): follow the current segment's principal angle, and do a forward
# turn when the end wall is close. No camera, no obstacles, no parking. The planner drives
# from the raw LiDAR scan, the gyro heading, and the wheel odometry only.
#
# Parameters below mirror main_obstacle.py (wheelbase, LiDAR).

_WHEELBASE:          mm     = 89.6251

_LIDAR_BAUDRATE:     int    = 115200
_LIDAR_OFFSET_ANGLE: degree = -90.0
_LIDAR_MOUNT_OFFSET: Point  = (0.0, 73.37)


def parse_args():
    parser = argparse.ArgumentParser(description = 'Open-challenge planner (principal-angle follow + forward turns)')
    parser.add_argument('--lidar-port',   default = '/dev/lidar',   help = 'LiDAR serial port')
    parser.add_argument('--arduino-port', default = '/dev/arduino', help = 'Arduino serial port')
    parser.add_argument('--button-pin',   default = 17, type = int, help = 'BCM GPIO pin for the start button')
    parser.add_argument('--no-button',    action = 'store_true', help = 'auto-start ~2 s after launch instead of waiting for the button')
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

    # PrincipalAngleDetector fits the track's principal angle into the shared TrackModel; the
    # planner reads it from there. It MUST be subscribed to on_scan before the planner so it runs
    # first on the LiDAR thread and the planner sees a fresh theta each scan.
    angle_detector = PrincipalAngleDetector(ego_information, track)

    planner = OpenChallengePlanner(
        ego_information,
        track,
        lidar_mount_offset = _LIDAR_MOUNT_OFFSET,
        state_hold_s       = args.state_pause,
    )

    lidar.on_scan += angle_detector
    lidar.on_scan += planner

    arduino.on_state  += arduino_processor
    planner.on_target += arduino.set_target
    planner.on_stop   += arduino.stop

    # Recorder taps go after the wiring above so ego_pose is sampled after ArduinoProcessor has
    # ingested the same state message.
    recorder = Recorder.create_default(enabled = not args.no_record)
    recorder.attach_scan(lidar.on_scan, ego = ego_information)
    recorder.attach_array(arduino.on_state, 'arduino_state',
                          fields = ('heading_rad', 'speed_mm_s', 'steering_rad', 'distance_mm'))
    recorder.attach_pose(arduino.on_state, ego_information)
    recorder.attach_planner(planner)

    visualizers: list = []
    if args.visu:
        planner_visualizer = PathPlanningVisualizer()
        planner_visualizer.start()
        planner.on_debug += planner_visualizer.set_debug
        visualizers.append(planner_visualizer)

        lidar_visualizer = LidarVisualizer(offset_angle = _LIDAR_OFFSET_ANGLE)
        lidar_visualizer.start()
        lidar.on_scan += lidar_visualizer
        visualizers.append(lidar_visualizer)

        if not args.no_browser:
            for vis in visualizers:
                webbrowser.open(f"http://localhost:{vis._DEFAULT_PORT}")

    start_button = None if args.no_button else Button(args.button_pin, pull_up = True)

    with ExitStack() as stack:
        stack.enter_context(recorder)
        stack.enter_context(lidar)
        stack.enter_context(arduino)

        if args.no_button:
            time.sleep(2)
        else:
            print("Ready. Press the start button to begin.")
            start_button.wait_for_press()
            print("Starting.")
            time.sleep(0.5)

        lidar.start()
        arduino.start()

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
