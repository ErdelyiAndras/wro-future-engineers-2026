# Developer & setup guide

How to set up, flash, run, and debug the robot from scratch. Placeholders like
`<pi-user>` / `<robot-host>` are things specific to your machine — fill them in for your
setup.

> Angle brackets = **fill this in**. Commands assume you are in the repository root unless a
> `cd` says otherwise.

## Prerequisites

- **Raspberry Pi 5** running Raspberry Pi OS (64-bit), with the LiDAR, camera, and Arduino
  connected over USB.
- **Python 3.13** on the Pi (the code uses `X | None` syntax and `from __future__`).
- **`arduino-cli`** (on whatever machine flashes the Nano), with the AVR core.
- **Git**.

## Repository layout

```
src/rpi/          Raspberry Pi software (Python)
  components/     hardware drivers            (README)
  control/        blackboard state            (README)
  processors/     algorithms + planner        (README)
  recording/      HDF5 signal recorder        (README)
  visu/           live web visualizers        (README)
  main_open.py    open-challenge entry point
  main_obstacle.py obstacle-challenge entry point
src/arduino/      Arduino Nano firmware (C++)      (README)
calibration/      one-time calibration scripts     (README)
recordings/       run_*.h5 files land here (git-ignored)
Makefile          flash the Arduino
```

The architecture is in [`docs/software/architecture.md`](architecture.md); each folder's
`README.md` documents its code.

## 1. Flash the Arduino

The [`Makefile`](../../Makefile) wraps `arduino-cli` (FQBN `arduino:avr:nano:cpu=atmega328`,
port `/dev/arduino`). One-time setup:

```bash
arduino-cli core install arduino:avr
arduino-cli lib install "Adafruit BNO055"        # pulls in Adafruit Unified Sensor + BusIO
```

Then, whenever the firmware changes:

```bash
make flash          # = compile + upload to /dev/arduino
# or separately:
make compile
make upload
```

If the Nano is a clone with the old bootloader, change the FQBN in the `Makefile` to
`...:cpu=atmega328old`. Firmware details: [Arduino firmware chapter](../../src/arduino/README.md).

## 2. Raspberry Pi Python environment

```bash
cd src/rpi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the hardware libraries (`pyserial`,
`adafruit-circuitpython-rplidar`, `adafruit-blinka`, `lgpio`, `gpiozero`) and the compute
stack (`numpy`, `scipy`, `scikit-learn`, `h5py`, `opencv-python`, `matplotlib`). `gpiozero`
+ `lgpio` drive the start button on BCM pin 17.

The [calibration scripts](../../calibration/README.md) use a **separate** environment
(`calibration/requirements.txt`) — set it up the same way inside `calibration/`.

## 3. Device symlinks (`/dev/lidar`, `/dev/arduino`)

The code opens the LiDAR and Arduino by **stable names** (`/dev/lidar`, `/dev/arduino`)
rather than `/dev/ttyUSB*`, which reorder on reboot. These are udev symlinks created on the
Pi (they are **not** in the repo). Find each device's attributes:

```bash
udevadm info -a -n /dev/ttyUSB0 | grep -E 'idVendor|idProduct|serial' | head
```

then create `/etc/udev/rules.d/99-robot.rules` (fill in from the output above):

```udev
# RPLIDAR (CP210x USB-serial)
SUBSYSTEM=="tty", ATTRS{idVendor}=="<lidar-vid>", ATTRS{idProduct}=="<lidar-pid>", SYMLINK+="lidar"
# Arduino Nano
SUBSYSTEM=="tty", ATTRS{idVendor}=="<arduino-vid>", ATTRS{idProduct}=="<arduino-pid>", SYMLINK+="arduino"
```

Reload with `sudo udevadm control --reload && sudo udevadm trigger`. (If both adapters share
a VID/PID, key the rule on `ATTRS{serial}` instead.) Either port can also be overridden on
the command line with `--lidar-port` / `--arduino-port`.

## 4. Run a challenge

From `src/rpi/` with the venv active:

```bash
python main_open.py                 # open challenge
python main_obstacle.py             # obstacle challenge
python main_obstacle.py --no-camera # obstacle main, LiDAR-only (bench / open)
```

By default the robot **waits for the start button** (press once to start, again to stop).
Useful flags (both mains share most of them):

| Flag | Effect |
|---|---|
| `--no-button` | auto-start ~2 s after launch (no button) |
| `--visu` | start the [visualizers](../../src/rpi/visu/README.md) (LiDAR 8082, planner 8084, colour 8085) |
| `--no-browser` | with `--visu`, don't auto-open a browser |
| `--no-record` | disable [recording](../../src/rpi/recording/README.md) |
| `--state-pause [SEC]` | debug: hold stopped for SEC s (default 2) at every state transition |
| `--lidar-port` / `--arduino-port` / `--camera-port` | override device paths |

`main_obstacle.py` adds parking options: `--parking-start` (start parked in the slot),
`--park-mode {none,perpendicular,...}`, `--park-exit-side {auto,left,right}`. See the
[processors chapter](../../src/rpi/processors/README.md#parking).

## 5. Remote access with Tailscale

We SSH into the Pi over a **Tailscale** private network, so the robot is reachable by a
stable hostname from anywhere, without port-forwarding or being on the same LAN.

One-time, on the Pi:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up            # authenticate in the browser link it prints
```

Do the same on your laptop and log into the **same** Tailscale account. Then:

```bash
ssh <pi-user>@<robot-host>   # <robot-host> = the Pi's Tailscale name, e.g. from `tailscale status`
```

> Fill in `<pi-user>` (the Pi login) and `<robot-host>` (the Tailscale machine name — see
> `tailscale status`, or use the `100.x.y.z` Tailscale IP). The
> [visualizers](../../src/rpi/visu/README.md) are then reachable in a browser at
> `http://<robot-host>:8082` / `:8084` / `:8085` while a run with `--visu` is live.

For an untethered run: SSH in, start the main with `--no-button` (or press the physical
button), and watch the visualizers in the browser.

## 6. Record & replay a run

Every run writes `recordings/run_<timestamp>.h5` unless `--no-record` is given. To inspect one
(from `src/rpi/`):

```bash
python -m recording summary recordings/run_<ts>.h5    # streams, rates, sizes, drops
python -m recording scans   recordings/run_<ts>.h5    # per-scan LiDAR dropout
python -m recording dump     recordings/run_<ts>.h5 planner_debug --count 40
```

Copy a file off the robot with `scp <pi-user>@<robot-host>:~/…/recordings/run_<ts>.h5 .`.
The recorder and the replay-oriented `Recording` reader are documented in the
[recording chapter](../../src/rpi/recording/README.md); the planner's `on_debug` stream holds
everything the [planner visualizer](../../src/rpi/visu/README.md) draws, so a run can be
reconstructed after the fact.

## 7. Calibration

Before a competition, (re)run the relevant [calibration scripts](../../calibration/README.md)
— **always** the obstacle colours at the venue, and the camera intrinsics after any change to
the camera mount. Each script's output and where it is pasted is in that chapter.

## Typical bring-up checklist

1. `make flash` the firmware; confirm the `STATE` stream arrives (e.g. `python -m recording`
   on a short run, or the LiDAR/planner visualizers showing live data).
2. Create the `/dev/lidar` and `/dev/arduino` symlinks (step 3).
3. `pip install -r requirements.txt` in `src/rpi/.venv`.
4. `python main_obstacle.py --no-button --visu --no-camera` on the bench; watch 8082/8084.
5. Calibrate camera + colours; run the full obstacle main on the track.
