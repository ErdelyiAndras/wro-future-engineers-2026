import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, help="Path to the image")
parser.add_argument("--tolerance", type=int, default=10, help="HSV range tolerance margin")
args = parser.parse_args()

img = cv2.imread(args.image)
if img is None:
    print(f"Could not load {args.image}")
    exit(1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def select_region(label: str) -> np.ndarray | None:
    print(f"\nSelect the {label} obstacle — draw a rectangle, press ENTER or SPACE to confirm, C to cancel")
    roi = cv2.selectROI(f"Select {label}", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(f"Select {label}")

    x, y, w, h = roi
    if w == 0 or h == 0:
        print(f"No region selected for {label}")
        return None

    return hsv[y:y+h, x:x+w].reshape(-1, 3)

def compute_range(pixels: np.ndarray, tolerance: int) -> tuple:
    h_vals = pixels[:, 0]
    s_vals = pixels[:, 1]
    v_vals = pixels[:, 2]

    # Detect red hue wrap-around (hue values on both sides of 0/180)
    if h_vals.max() - h_vals.min() > 90:
        # Wrap: shift values above 90 down by 180 for range computation
        h_shifted         = h_vals.copy().astype(int)
        h_shifted[h_shifted > 90] -= 180

        h_center = int(np.mean(h_shifted))
        h_radius = int(np.std(h_shifted) * 2) + tolerance

        lower_1 = np.array([max(  0, h_center - h_radius),
                             max(  0, int(s_vals.min()) - tolerance),
                             max(  0, int(v_vals.min()) - tolerance)], dtype=np.uint8)
        upper_1 = np.array([min( 10, h_center + h_radius),
                             min(255, int(s_vals.max()) + tolerance),
                             min(255, int(v_vals.max()) + tolerance)], dtype=np.uint8)

        lower_2 = np.array([max(170, 180 + h_center - h_radius),
                             max(  0, int(s_vals.min()) - tolerance),
                             max(  0, int(v_vals.min()) - tolerance)], dtype=np.uint8)
        upper_2 = np.array([180,
                             min(255, int(s_vals.max()) + tolerance),
                             min(255, int(v_vals.max()) + tolerance)], dtype=np.uint8)

        return (lower_1, upper_1), (lower_2, upper_2)
    else:
        lower = np.array([max(  0, int(h_vals.min()) - tolerance),
                          max(  0, int(s_vals.min()) - tolerance),
                          max(  0, int(v_vals.min()) - tolerance)], dtype=np.uint8)
        upper = np.array([min(180, int(h_vals.max()) + tolerance),
                          min(255, int(s_vals.max()) + tolerance),
                          min(255, int(v_vals.max()) + tolerance)], dtype=np.uint8)
        return (lower, upper),

def preview_mask(label: str, ranges: tuple) -> None:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, lower, upper)

    highlight = img.copy()
    highlight[mask == 0] = (highlight[mask == 0] * 0.3).astype(np.uint8)
    cv2.putText(highlight, f"{label} mask — press any key", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(f"{label} mask", highlight)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_ranges(label: str, ranges: tuple) -> None:
    print(f"\n--- {label} ---")
    if len(ranges) == 2:
        (l1, u1), (l2, u2) = ranges
        print(f"_{label.upper()}_LOWER_1 = np.array([{l1[0]:3d}, {l1[1]:3d}, {l1[2]:3d}], dtype=np.uint8)")
        print(f"_{label.upper()}_UPPER_1 = np.array([{u1[0]:3d}, {u1[1]:3d}, {u1[2]:3d}], dtype=np.uint8)")
        print(f"_{label.upper()}_LOWER_2 = np.array([{l2[0]:3d}, {l2[1]:3d}, {l2[2]:3d}], dtype=np.uint8)")
        print(f"_{label.upper()}_UPPER_2 = np.array([{u2[0]:3d}, {u2[1]:3d}, {u2[2]:3d}], dtype=np.uint8)")
    else:
        (l, u), = ranges
        print(f"_{label.upper()}_LOWER = np.array([{l[0]:3d}, {l[1]:3d}, {l[2]:3d}], dtype=np.uint8)")
        print(f"_{label.upper()}_UPPER = np.array([{u[0]:3d}, {u[1]:3d}, {u[2]:3d}], dtype=np.uint8)")

for label in ("red", "green"):
    pixels = select_region(label)
    if pixels is None:
        continue
    ranges = compute_range(pixels, args.tolerance)
    preview_mask(label, ranges)
    print_ranges(label, ranges)

print("\nDone — paste the ranges above into CameraProcessor")
