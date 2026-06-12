#!/usr/bin/env bash
# Launch the base-platform headless DMD demo recorder (tools/demo/run_demo.py)
# in a container with the right devices, mounts, X11, and IDS SDK wired up.
# Extra args pass through to run_demo.py.
#
# Examples:
#   ./scripts/run_demo.sh --no-camera --hold-scale 0.5   # projection-only smoke
#   ./scripts/run_demo.sh --hold-scale 0.5               # full run (camera)
#   OUT_DIR=/mnt/nvme/demo ./scripts/run_demo.sh         # write to fast storage
#   ./scripts/run_demo.sh --dry-run --out-dir /tmp/dry   # no hardware
#
# Prereqs (host): an X server on $DISPLAY, the second monitor / DMD powered,
# and the IDS Peak SDK installed at $IDS_PEAK_PATH (default /opt/ids-peak) for
# camera mode. You are in the docker group (no sudo needed).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${DEMO_IMAGE:-crispi:latest}"
IDS_PEAK="${IDS_PEAK_PATH:-/opt/ids-peak}"
TS="$(date +%Y%m%d_%H%M%S)"
# OUT_DIR is a HOST path (default under the repo), mounted at /out so external
# storage (e.g. OUT_DIR=/mnt/nvme/demo) works too. We do NOT mkdir it on the
# host: the container runs as root and a prior root-owned Saved_Media would
# block a host-side mkdir. Docker creates the bind-mount source dir (as root)
# if it's missing — so creation always succeeds.
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/Saved_Media/demo_${TS}}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: image '${IMAGE}' not found. Build it with ./build.sh" >&2
    exit 1
fi

export DISPLAY="${DISPLAY:-:0}"
xhost +local:docker >/dev/null 2>&1 || true

echo "[run_demo] image=${IMAGE}  out=${OUT_DIR}  display=${DISPLAY}"
echo "[run_demo] args: $*"

# Use the image entrypoint (sets up IDS Peak env + LD_LIBRARY_PATH from the
# mounted SDK, then exec python3 "$@").
exec docker run --rm --privileged --network=host \
    -e DISPLAY="${DISPLAY}" \
    -e GENICAM_GENTL64_PATH=/opt/ids-peak/lib/aarch64-linux-gnu/ids-peak/cti \
    -e STIM_HW_EXP_US="${STIM_HW_EXP_US:-15000}" \
    -e STIM_TRIG_DELAY_US="${STIM_TRIG_DELAY_US:-0}" \
    -e STIM_GAIN="${STIM_GAIN:-1.0}" \
    -e PYTHONUNBUFFERED=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${IDS_PEAK}:/opt/ids-peak:ro" \
    --device=/dev/bus/usb:/dev/bus/usb \
    --device=/dev/i2c-1:/dev/i2c-1 \
    --device=/dev/gpiochip1:/dev/gpiochip1 \
    -v "${REPO_ROOT}:/repo" \
    -v "${OUT_DIR}:/out" -w /repo \
    "${IMAGE}" \
    /repo/tools/demo/run_demo.py --out-dir /out "$@"
