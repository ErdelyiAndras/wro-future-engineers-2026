import cv2
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("output_dir", type=str, help="Directory to save images to")
parser.add_argument("--device", type=int, default=1)
parser.add_argument("--width",  type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(args.device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

if not cap.isOpened():
    print(f"Could not open device {args.device}")
    exit(1)

print(f"Saving to: {output_dir.resolve()}")
print("SPACE: save image | Q: quit")

index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    display = frame.copy()
    cv2.putText(display, f"Saved: {index}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Capture", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        path = output_dir / f"img_{index:03d}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"Saved {path}")
        index += 1

cap.release()
cv2.destroyAllWindows()
print(f"Captured {index} images")
