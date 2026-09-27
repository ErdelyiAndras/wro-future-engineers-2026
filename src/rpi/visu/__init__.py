"""Live web-based visualizers (LiDAR scan, planner state, camera colour).

Kept import-light on purpose: import the specific module you need
(e.g. ``from visu.LidarVisualizer import LidarVisualizer``) so a consumer
that only wants the LiDAR view does not pull in the camera/OpenCV stack.
"""
