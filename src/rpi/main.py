from components import Lidar, Ticker
from control import EgoInformation, FieldMap, Direction
from processors import LidarProcessor, SemanticClassifier

from FieldMapVisualiser import FieldMapVisualizer

import time

def main():
    ego_information = EgoInformation()
    ego_information.set_ego_information((0.0, 0.0), 0.0)

    field_map = FieldMap()
    field_map.initialize_direction(Direction.CW)

    lidar_processor = LidarProcessor(field_map, ego_information, lidar_offset_angle = 0)

    semantic_classifier = SemanticClassifier(field_map)

    visualiser = FieldMapVisualizer(field_map, ego_information)
    visualiser.start()

    with Lidar() as lidar, \
         Ticker(interval = 1.0) as classification_ticker:

        lidar.on_point                += lidar_processor
        classification_ticker.on_tick += semantic_classifier

        lidar.start()
        classification_ticker.start()

        end = time.time() + 60
        while time.time() < end:
            visualiser.update()
            time.sleep(1 / 10)

    visualiser.stop()

if __name__ == '__main__':
    main()
