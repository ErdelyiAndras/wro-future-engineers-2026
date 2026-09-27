import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="Calibrate obstacle HSV ranges. Pass several images (e.g. one per "
                "segment) to pool their ROI pixels into a single range that covers all.")
parser.add_argument("images",       type=str, nargs="+", help="Path(s) to the image(s)")
parser.add_argument("--tolerance",  type=int, default=10, help="HSV range tolerance margin")
parser.add_argument("--headless",    action="store_true", help="No GUI: provide ROIs via --red-roi / --green-roi / --magenta-roi (single image only)")
parser.add_argument("--red-roi",     type=str, default=None, metavar="X,Y,W,H", help="ROI for red obstacle (headless mode)")
parser.add_argument("--green-roi",   type=str, default=None, metavar="X,Y,W,H", help="ROI for green obstacle (headless mode)")
parser.add_argument("--magenta-roi", type=str, default=None, metavar="X,Y,W,H", help="ROI for the magenta parking wall (headless mode)")
args = parser.parse_args()

if args.headless and len(args.images) > 1:
    print("--headless takes a single image (ROIs differ per image); pass one image or drop --headless")
    exit(1)

def parse_roi(s: str) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = map(int, s.split(","))
        return x, y, w, h
    except ValueError:
        print(f"Invalid ROI format '{s}' — expected X,Y,W,H")
        exit(1)

def select_region_gui(img: np.ndarray, hsv: np.ndarray, label: str, tag: str) -> np.ndarray | None:
    print(f"\n[{tag}] Select the {label} obstacle — draw a rectangle, press ENTER or SPACE to confirm, C to cancel (empty = skip this colour in this image)")
    win = f"[{tag}] Select {label}"
    roi = cv2.selectROI(win, img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)
    x, y, w, h = roi
    if w == 0 or h == 0:
        print(f"No region selected for {label} in {tag}")
        return None
    return hsv[y:y+h, x:x+w].reshape(-1, 3)

def select_region_headless(hsv: np.ndarray, label: str, roi_str: str | None) -> np.ndarray | None:
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

def preview_mask(img: np.ndarray, hsv: np.ndarray, label: str, ranges: tuple, tag: str) -> None:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, lower, upper)

    highlight = img.copy()
    highlight[mask == 0] = (highlight[mask == 0] * 0.3).astype(np.uint8)
    cv2.putText(highlight, f"[{tag}] {label} merged mask — press any key", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(f"{label} mask", highlight)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_ranges(label: str, ranges: tuple, name: str, n_images: int, n_pixels: int) -> None:
    # Emit a ready-to-paste ColorRange (one (lower, upper) band, or two if the hue wraps
    # the 0/180 seam). `name` is the variable prefix, e.g. _RED -> `_RED_COLOR`.
    print(f"\n--- {label} (pooled from {n_images} image(s), {n_pixels} px) ---")
    print(f"{name}_COLOR = ColorRange(bands = [")
    for lower, upper in ranges:
        print(f"    (np.array([{lower[0]:>3}, {lower[1]:>3}, {lower[2]:>3}], dtype = np.uint8), "
              f"np.array([{upper[0]:>3}, {upper[1]:>3}, {upper[2]:>3}], dtype = np.uint8)),")
    print("])")

# Load every image up front so previews can revisit them after ranges are computed.
loaded: list[tuple[str, np.ndarray, np.ndarray]] = []
for path in args.images:
    img = cv2.imread(path)
    if img is None:
        print(f"Could not load {path}")
        exit(1)
    loaded.append((path, img, cv2.cvtColor(img, cv2.COLOR_BGR2HSV)))

roi_args     = {"red": args.red_roi, "green": args.green_roi, "magenta": args.magenta_roi}
# Printed variable name per label (matches the segment mains' _RED_COLOR / _GREEN_COLOR /
# _PARKING_COLOR that feed BodyFrameColorSampler's red / green / parking args).
output_names = {"red": "_RED", "green": "_GREEN", "magenta": "_PARKING"}

# Pool ROI pixels per colour across every image, then compute one range over the union.
pooled: dict[str, list[np.ndarray]] = {"red": [], "green": [], "magenta": []}
contributing: dict[str, int] = {"red": 0, "green": 0, "magenta": 0}

for path, img, hsv in loaded:
    for label in ("red", "green", "magenta"):
        if args.headless:
            pixels = select_region_headless(hsv, label, roi_args[label])
        else:
            pixels = select_region_gui(img, hsv, label, path)
        if pixels is not None and pixels.size:
            pooled[label].append(pixels)
            contributing[label] += 1

for label in ("red", "green", "magenta"):
    if not pooled[label]:
        continue

    pixels = np.concatenate(pooled[label], axis=0)
    ranges = compute_range(pixels, args.tolerance)

    if not args.headless:
        # Show the merged range against every image so you can confirm it covers each segment.
        for path, img, hsv in loaded:
            preview_mask(img, hsv, label, ranges, path)

    print_ranges(label, ranges, output_names[label], contributing[label], pixels.shape[0])

print("\n(_RED_COLOR / _GREEN_COLOR / _PARKING_COLOR, fed to BodyFrameColorSampler's red / green / parking)")
