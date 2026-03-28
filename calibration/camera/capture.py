import cv2
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("output_dir", type=str, help="Directory to save images to")
parser.add_argument("--device",   type=int,  default=0)
parser.add_argument("--width",    type=int,  default=640)
parser.add_argument("--height",   type=int,  default=480)
parser.add_argument("--headless", action="store_true", help="No GUI: press ENTER to capture, 'q'+ENTER to quit")
parser.add_argument("--roll",     type=float, default=0.0,  help="Rotate frame by this many degrees (counter-clockwise)")
args = parser.parse_args()

def apply_roll(frame, degrees):
    if degrees == 0.0:
        return frame
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(frame, M, (w, h))

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(args.device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

if not cap.isOpened():
    print(f"Could not open device {args.device}")
    exit(1)

print(f"Saving to: {output_dir.resolve()}")

index = 0

if args.headless:
    print("ENTER: capture | q+ENTER: quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        try:
            line = input()
        except EOFError:
            break
        if line.strip().lower() == "q":
            break
        path = output_dir / f"img_{index:03d}.jpg"
        cv2.imwrite(str(path), apply_roll(frame, args.roll))
        print(f"Saved {path}  (total: {index + 1})")
        index += 1
else:
    print("SPACE: save image | Q: quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        frame = apply_roll(frame, args.roll)
        display = frame.copy()
        cv2.putText(display, f"Saved: {index}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            path = output_dir / f"img_{index:03d}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Saved {path}")
            index += 1
    cv2.destroyAllWindows()

cap.release()
print(f"Captured {index} images")
