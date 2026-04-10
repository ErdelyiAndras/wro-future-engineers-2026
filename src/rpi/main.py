from components import Camera, Lidar, Ticker
from control import EgoInformation, FieldMap, Direction
from processors import (
    CameraProcessor,
    LidarProcessor,
    SemanticClassifier,
    ObstacleColorRanges,
    CameraIntrinsics,
    CameraExtrinsics
)

from FieldMapVisualiser import FieldMapVisualizer
from CameraVisualizer import CameraVisualizer

import time
import webbrowser
import numpy as np

def main():
    ego_information = EgoInformation()
    ego_information.set_ego_information((0.0, 0.0), 0.0)

    field_map = FieldMap()
    field_map.initialize_direction(Direction.CW)

    camera_intrinsics = CameraIntrinsics(
        fx = 658.3516186667142,
        fy = 656.9839014361173,
        cx = 312.90122361261507,
        cy = 178.888473418321,
        dist_coeffs = np.array([
            -0.17756057290232585,
            -0.07620432808648203,
            0.0014254009734183857,
            0.0009633670165933481,
            -0.3014236479533136], dtype=np.float64
        )
    )
    camera_extrinsics = CameraExtrinsics(
        x     = 0.0,
        y     = 136.6,
        z     = 42.47,
        pitch = 3.2,
        yaw   = 1.5,
        roll  = 0.0
    )

    lidar_processor = LidarProcessor(
        field_map, ego_information,
        offset_angle = -90.0,
        mount_offset = (0.0, 73.37)
    )
    semantic_classifier = SemanticClassifier(field_map)
    camera_processor = CameraProcessor(
        field_map,
        ego_information,
        obstacle_color_ranges = ObstacleColorRanges(
            red_lower_1 = np.array([  0,  91,  92], dtype = np.uint8),
            red_upper_1 = np.array([ 10, 248, 222], dtype = np.uint8),
            red_lower_2 = np.array([170,  91,  92], dtype = np.uint8),
            red_upper_2 = np.array([180, 248, 222], dtype = np.uint8),
            green_lower = np.array([ 53,  84,  56], dtype = np.uint8),
            green_upper = np.array([ 87, 215, 157], dtype = np.uint8),
        ),
        intrinsics    = camera_intrinsics,
        extrinsics    = camera_extrinsics,
        sample_radius = 10,
    )

    field_map_visualizer = FieldMapVisualizer(
        field_map,
        ego_information
    )
    field_map_visualizer.start()
    webbrowser.open(f"http://localhost:{FieldMapVisualizer._DEFAULT_PORT}")

    camera_visualizer = CameraVisualizer(
        field_map,
        ego_information,
        camera_intrinsics,
        camera_extrinsics,
    )
    camera_visualizer.start()
    webbrowser.open(f"http://localhost:{CameraVisualizer._DEFAULT_PORT}")

    with Lidar() as lidar, \
         Camera() as camera, \
         Ticker(interval = 0.1) as classification_ticker:

        lidar.on_scan                 += lidar_processor
        camera.on_frame               += camera_processor
        classification_ticker.on_tick += semantic_classifier

        camera.on_frame               += camera_visualizer

        lidar.start()
        camera.start()
        classification_ticker.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    field_map_visualizer.stop()
    camera_visualizer.stop()

if __name__ == '__main__':
    main()
