#!/usr/bin/env bash
# Launcher run by wro.service at boot. Edit this file to switch challenge or flags;
# no `systemctl daemon-reload` needed, just `sudo systemctl restart wro.service`.

# Which challenge to run: open or obstacle
CHALLENGE=obstacle

# Extra flags per challenge (see `python3 src/rpi/main_<challenge>.py --help`)
OPEN_ARGS=()
OBSTACLE_ARGS=(--parking-start --park-mode perpendicular)

case "$CHALLENGE" in
    open)     MAIN=main_open.py;     ARGS=("${OPEN_ARGS[@]}") ;;
    obstacle) MAIN=main_obstacle.py; ARGS=("${OBSTACLE_ARGS[@]}") ;;
    *)        echo "start.sh: unknown CHALLENGE '$CHALLENGE' (use open or obstacle)" >&2; exit 1 ;;
esac

cd "$(dirname "$0")" || exit 1
exec src/rpi/.venv/bin/python3 "src/rpi/$MAIN" "${ARGS[@]}"
