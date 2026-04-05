import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image",        type=str, help="Path to the image")
parser.add_argument("--tolerance",  type=int, default=10, help="HSV range tolerance margin")
parser.add_argument("--headless",   action="store_true", help="No GUI: provide ROIs via --red-roi and --green-roi")
parser.add_argument("--red-roi",    type=str, default=None, metavar="X,Y,W,H", help="ROI for red obstacle (headless mode)")
parser.add_argument("--green-roi",  type=str, default=None, metavar="X,Y,W,H", help="ROI for green obstacle (headless mode)")
args = parser.parse_args()

img = cv2.imread(args.image)
if img is None:
    print(f"Could not load {args.image}")
    exit(1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def parse_roi(s: str) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = map(int, s.split(","))
        return x, y, w, h
    except ValueError:
        print(f"Invalid ROI format '{s}' — expected X,Y,W,H")
        exit(1)

def select_region_gui(label: str) -> np.ndarray | None:
    print(f"\nSelect the {label} obstacle — draw a rectangle, press ENTER or SPACE to confirm, C to cancel")
    roi = cv2.selectROI(f"Select {label}", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(f"Select {label}")
    x, y, w, h = roi
    if w == 0 or h == 0:
        print(f"No region selected for {label}")
        return None
    return hsv[y:y+h, x:x+w].reshape(-1, 3)

def select_region_headless(label: str, roi_str: str | None) -> np.ndarray | None:
    if roi_str is None:
        print(f"--{label}-roi not provided, skipping {label}")
        return None
    x, y, w, h = parse_roi(roi_str)
    print(f"{label}: using ROI x={x} y={y} w={w} h={h}")
    return hsv[y:y+h, x:x+w].reshape(-1, 3)

def compute_range(pixels: np.ndarray, tolerance: int) -> tuple:
    h_vals = pixels[:, 0]
    s_vals = pixels[:, 1]
    v_vals = pixels[:, 2]

    # Detect red hue wrap-around (hue values on both sides of 0/180)
    if h_vals.max() - h_vals.min() > 90:
        # Wrap: shift values above 90 down by 180 for range computation
        h_shifted = h_vals.copy().astype(int)
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
        print(f"{label}_lower_1 = np.array([{l1[0]:>3}, {l1[1]:>3}, {l1[2]:>3}], dtype = np.uint8),")
        print(f"{label}_upper_1 = np.array([{u1[0]:>3}, {u1[1]:>3}, {u1[2]:>3}], dtype = np.uint8),")
        print(f"{label}_lower_2 = np.array([{l2[0]:>3}, {l2[1]:>3}, {l2[2]:>3}], dtype = np.uint8),")
        print(f"{label}_upper_2 = np.array([{u2[0]:>3}, {u2[1]:>3}, {u2[2]:>3}], dtype = np.uint8),")
    else:
        (l, u), = ranges
        print(f"{label}_lower = np.array([{l[0]:>3}, {l[1]:>3}, {l[2]:>3}], dtype = np.uint8),")
        print(f"{label}_upper = np.array([{u[0]:>3}, {u[1]:>3}, {u[2]:>3}], dtype = np.uint8),")

roi_args = {"red": args.red_roi, "green": args.green_roi}

for label in ("red", "green"):
    if args.headless:
        pixels = select_region_headless(label, roi_args[label])
    else:
        pixels = select_region_gui(label)

    if pixels is None:
        continue

    ranges = compute_range(pixels, args.tolerance)

    if not args.headless:
        preview_mask(label, ranges)

    print_ranges(label, ranges)

print("\nDone — paste the ranges above into CameraProcessor")
