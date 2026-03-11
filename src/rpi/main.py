from components.Lidar import Lidar
import time
import math
from threading import Lock
import matplotlib.pyplot as plt

def process(lock, points_x, points_y, angle, distance, quality):
    angle_rad = math.radians(angle)
    x = distance * math.sin(angle_rad)
    y = distance * math.cos(angle_rad)
    with lock:
        points_x.append(x)
        points_y.append(y)

def visualise(x, y):
    if x:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.set_aspect('equal')
        ax.scatter(x, y, s=2, c='lime')
        ax.scatter([0], [0], s=30, c='red')
        ax.tick_params(colors='white')
        ax.set_title("Detected points", color='white')
        ax.grid(color='#222222')
        plt.tight_layout()
        plt.savefig('scan.png', dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)

def main():
    print("Constructing Lidar object")
    with Lidar() as lidar:
        print("Done constructing")
        time.sleep(2)
        print("Done sleeping 2 secs")

        lock = Lock()
        points_x = []
        points_y = []

        lidar.on_point += lambda a, d, q: process(lock, points_x, points_y, a, d, q)

        print("Start sleeping 3 secs")
        time.sleep(3)
        print("Done sleeping 3 secs")

        print("Start processing")
        lidar.start()
        time.sleep(10)
        lidar.stop()
        print("Stop processing")

        print("Start sleeping 5 secs")
        time.sleep(5)
        print("Done sleeping 5 secs")

        print("Start processing")
        lidar.start()
        time.sleep(3)

    visualise(points_x, points_y)

if __name__ == '__main__':
    main()
