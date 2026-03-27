import cv2
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("image_dir",    type=str,   help="Directory containing calibration images")
parser.add_argument("output_dir",   type=str,   help="Directory to save intrinsics and distortion files")
parser.add_argument("--square_size",type=float, default=24.5, help="Checkerboard square size in mm")
parser.add_argument("--cols",       type=int,   default=9,    help="Number of inner corners per row")
parser.add_argument("--rows",       type=int,   default=6,    help="Number of inner corners per column")
args = parser.parse_args()

image_dir  = Path(args.image_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

images = sorted(image_dir.glob("*.jpg"))
if not images:
    print(f"No images found in {image_dir}")
    exit(1)

objp          = np.zeros((args.rows * args.cols, 3), np.float32)
objp[:, :2]   = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_size
obj_points    = []
img_points    = []
gray          = None
failed        = []

for path in images:
    img  = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)
    if not ret:
        failed.append(path.name)
        continue

    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    obj_points.append(objp)
    img_points.append(corners)
    print(f"OK      {path.name}")

for name in failed:
    print(f"FAILED  {name}")

print(f"\n{len(obj_points)}/{len(images)} images used")

if len(obj_points) < 10:
    print("Too few valid images, take more photos")
    exit(1)

ret, K, dist, _, _ = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

print(f"\nReprojection error : {ret:.4f} px  (good if < 0.5)")
print(f"\nfx = {fx:.2f}")
print(f"fy = {fy:.2f}")
print(f"cx = {cx:.2f}")
print(f"cy = {cy:.2f}")
print(f"\nDistortion coefficients: {dist.ravel()}")
print(f"\nFull intrinsic matrix:\n{K}")

intrinsics_path = output_dir / "camera_intrinsics.npy"
distortion_path = output_dir / "camera_distortion.npy"
np.save(intrinsics_path, K)
np.save(distortion_path, dist)
print(f"\nSaved {intrinsics_path}")
print(f"Saved {distortion_path}")
