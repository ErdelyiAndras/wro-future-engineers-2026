import cv2
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("output_dir", type=str, help="Directory to save images to")
parser.add_argument("--device",   type=int,  default=0)
parser.add_argument("--width",    type=int,  default=640)
parser.add_argument("--height",   type=int,  default=480)
parser.add_argument("--headless", action="store_true", help="No GUI: press ENTER to capture, 'q'+ENTER to quit")
parser.add_argument("--http",     action="store_true", help="No GUI: serve a live MJPEG preview with a capture button at http://<host>:<port>")
parser.add_argument("--port",     type=int,  default=8086, help="Port for --http (default: 8086)")
parser.add_argument("--no-browser", action="store_true", help="With --http, do not open the preview in a browser")
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


_PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Capture</title>
<style>
  body { margin: 0; background: #111; color: #eee; font-family: sans-serif; text-align: center; }
  img  { max-width: 100%; margin-top: 8px; }
  button { font-size: 1.2em; padding: 0.5em 1.5em; margin: 8px; }
</style></head><body>
<div><img src="/stream"></div>
<button id="b">Capture (SPACE)</button> <span id="s"></span>
<script>
  async function capture() {
    const r = await fetch('/capture', {method: 'POST'});
    const j = await r.json();
    document.getElementById('s').textContent = 'Saved ' + j.path + ' (total ' + j.count + ')';
  }
  document.getElementById('b').onclick = capture;
  document.addEventListener('keydown', e => { if (e.code === 'Space') { e.preventDefault(); capture(); } });
</script></body></html>"""

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
elif args.http:
    lock    = threading.Condition()
    latest  = {"frame": None, "id": 0}

    def save_latest() -> dict | None:
        global index
        with lock:
            frame = latest["frame"]
        if frame is None:
            return None
        path = output_dir / f"img_{index:03d}.jpg"
        cv2.imwrite(str(path), frame)
        index += 1
        print(f"Saved {path}  (total: {index})")
        return {"path": path.name, "count": index}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_id = 0
                try:
                    while True:
                        with lock:
                            lock.wait_for(lambda: latest["id"] != last_id, timeout=2.0)
                            frame, last_id = latest["frame"], latest["id"]
                        if frame is None:
                            continue
                        display = frame.copy()
                        cv2.putText(display, f"Saved: {index}", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        ok, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if not ok:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(_PAGE)

        def do_POST(self):
            result = save_latest() if self.path == "/capture" else None
            self.send_response(200 if result else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result or {"error": "no frame"}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", args.port), _Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Preview: http://localhost:{args.port}   (Ctrl+C to quit)")
    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            with lock:
                latest["frame"] = apply_roll(frame, args.roll)
                latest["id"] += 1
                lock.notify_all()
    except KeyboardInterrupt:
        pass
    server.shutdown()
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
