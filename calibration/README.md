# Calibration

Scripts that measure the camera constants the software depends on. Run each **once on the
assembled robot** — and re-run the colour calibration at every venue, because lighting
changes everything. Each output is pasted into the robot code as noted. This chapter is part
of the [developer & setup guide](../docs/software/setup-guide.md).

Calibration has its own lightweight environment (`calibration/requirements.txt`: `numpy`,
`opencv-python`) — set it up the same way as the
[Pi environment](../docs/software/setup-guide.md#2-raspberry-pi-python-environment), inside
`calibration/`.

| What | Script | Output goes to |
|---|---|---|
| Camera intrinsics | `camera/camera-intrinsics.py` (+ `camera/capture.py`) | `_CAMERA_INTRINSICS` in `main_obstacle.py` |
| Camera extrinsics | *manual measurement* | `_CAMERA_EXTRINSICS` in `main_obstacle.py` |
| Obstacle colours | `camera/obstacle-colors.py` | `_RED_COLOR` / `_GREEN_COLOR` / `_PARKING_COLOR` in `main_obstacle.py` |

## Running the GUI tools over SSH (XLaunch / VcXsrv)

The camera scripts open OpenCV windows — `capture.py` shows a live preview, and
`obstacle-colors.py` lets you draw selection boxes. When you run them **on the Pi over SSH**
from a Windows laptop, those windows have to be forwarded to an X server on the laptop. We use
**VcXsrv**, launched through **XLaunch**:

1. Install **VcXsrv** on the Windows machine and start **XLaunch**:
   - *Multiple windows*, **Display number `0`**
   - *Start no client*
   - tick **Disable access control** (so the Pi is allowed to connect)
2. SSH in with X11 forwarding and run the tool:
   ```bash
   ssh -Y <pi-user>@<robot-host>          # -Y = trusted X11 forwarding
   cd <repo>/calibration
   python camera/capture.py <out_dir>     # the preview window opens on your laptop
   ```
3. If no window appears, point the display at the laptop explicitly (VcXsrv listens on
   display 0) and re-run:
   ```bash
   export DISPLAY=<windows-host>:0.0      # <windows-host> = the laptop's Tailscale name / IP
   ```

Trusted forwarding needs `X11Forwarding yes` in the Pi's `sshd_config`. If you can't (or don't
want to) forward a display, every camera script also has a `--headless` mode that captures /
selects without a GUI — noted in each section below.

## Camera intrinsics

1. Print a **9 × 6** inner-corner checkerboard with **24.5 mm** squares.
2. Capture ≥ 15 shots from varied angles/distances:
   `python camera/capture.py <out_dir>` (`--headless` for no-GUI capture).
3. Solve: `python camera/camera-intrinsics.py <image_dir> <output_dir>`. It detects corners
   sub-pixel, discards images where detection fails, and runs `cv2.calibrateCamera`. Aim for
   a reprojection error < 0.5 px.
4. Paste `fx, fy, cx, cy, dist_coeffs` into `_CAMERA_INTRINSICS` in `main_obstacle.py`.

## Camera extrinsics (manual)

The camera's pose relative to the rear-axle (body) origin — measure with a ruler on the
assembled robot and update `_CAMERA_EXTRINSICS` in `main_obstacle.py`: `x` (lateral),
`y` (forward from rear axle), `z` (height), and `pitch / yaw / roll`. Verify it with the
[`ColorVisualizer`](../src/rpi/visu/README.md): a LiDAR-detected pillar must project onto the
pillar in the image. If it lands off the pillar, the extrinsics are wrong.

## Obstacle colours

Lighting varies between venues *and between corners*, so capture one photo per
segment/lighting condition where a pillar appears, and calibrate over all of them at once:

```bash
python camera/obstacle-colors.py <image1> <image2> ...
```

For each image, draw a box over the red pillar, then the green, then the magenta parking wall
(confirm an empty selection to skip a colour absent from that image). The script pools the
selected pixels per colour across all images into **one HSV range** per colour (red and the
magenta wall are split into two bands to handle the 0/180° hue wrap) and previews the merged
mask against each image. Paste the printed `ColorRange` blocks into `main_obstacle.py`. The
colour type is documented in the [control chapter](../src/rpi/control/README.md).

> The ROI picker is a GUI tool — run it over SSH with the XLaunch setup above, or pass
> `--headless` with an explicit `--red-roi` / `--green-roi` per image.
