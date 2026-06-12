#!/bin/bash
set -e

# Auto-detect JetPack version from L4T release info
if [ ! -f /etc/nv_tegra_release ]; then
    echo "ERROR: /etc/nv_tegra_release not found. Is this a Jetson?"
    exit 1
fi

# Friendly warning if user is not in the docker group (build will work via sudo,
# but the iterative dev cycle is much smoother without sudo)
if ! id -nG "$USER" 2>/dev/null | grep -qw docker; then
    echo "NOTE: '$USER' is not in the 'docker' group."
    echo "      Add yourself with:  sudo usermod -aG docker $USER  (then log out / back in)"
    echo "      Otherwise you'll need 'sudo' for every docker command."
    echo ""
fi

# IDS Peak SDK .deb is needed at BUILD time. It is gitignored (license restricted —
# you must download it yourself from https://en.ids-imaging.com/download-peak.html).
# If missing, we create a 0-byte stub so the COPY layer succeeds. The Dockerfile's
# `dpkg -i ... || true` handles the failed install, and the pipeline will run in
# simulation mode (hardware camera mode disabled until you re-build with the real .deb).
IDS_DEB="ids-peak_2.17.0.0-488_arm64.deb"
if [ ! -s "$IDS_DEB" ]; then
    echo "WARNING: ${IDS_DEB} not found (or empty)."
    echo "         Building without IDS Peak SDK — hardware camera mode will be disabled."
    echo "         To enable hardware mode: download the SDK from"
    echo "           https://en.ids-imaging.com/download-peak.html"
    echo "         (pick 'IDS peak' for Linux ARM 64-bit, version 2.17.0)"
    echo "         Place the .deb at:  $(pwd)/${IDS_DEB}"
    echo "         Then re-run ./build.sh"
    echo ""
    : > "$IDS_DEB"   # create empty placeholder so the Dockerfile COPY succeeds
fi

L4T_MAJOR=$(sed -n 's/^# R\([0-9]*\).*/\1/p' /etc/nv_tegra_release)
L4T_REVISION=$(sed -n 's/.*REVISION: \([0-9.]*\).*/\1/p' /etc/nv_tegra_release)

echo "Detected L4T R${L4T_MAJOR}.${L4T_REVISION}"

if [[ "$L4T_MAJOR" -ge 36 ]]; then
    L4T_JETPACK_VERSION="r36.2.0"
    CUDA_VERSION="12.2"
    CUPY_PACKAGE="cupy-cuda12x"
    echo "JetPack 6 detected -> l4t-jetpack:${L4T_JETPACK_VERSION}, CUDA ${CUDA_VERSION}"
elif [[ "$L4T_MAJOR" -ge 35 ]]; then
    L4T_JETPACK_VERSION="r35.2.1"
    CUDA_VERSION="11.4"
    CUPY_PACKAGE="cupy-cuda11x"
    echo "JetPack 5 detected -> l4t-jetpack:${L4T_JETPACK_VERSION}, CUDA ${CUDA_VERSION}"
else
    echo "ERROR: Unsupported JetPack version (L4T R${L4T_MAJOR}.${L4T_REVISION})"
    echo "This container supports JetPack 5 (L4T R35.x) and JetPack 6 (L4T R36.x)."
    exit 1
fi

echo ""
echo "Building crispi:latest ..."
echo ""

GIT_SHA=$(git -C "$(dirname "$0")" rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

docker build \
    --build-arg L4T_JETPACK_VERSION="${L4T_JETPACK_VERSION}" \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg CUPY_PACKAGE="${CUPY_PACKAGE}" \
    --build-arg GIT_SHA="${GIT_SHA}" \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    -t crispi:latest \
    .

echo ""
echo "Build complete! Run with:"
echo "  export DISPLAY=:0"
echo "  xhost +local:docker"
echo "  docker-compose up gui       # STIMscope / CRISPI GUI"
echo ""
echo "If you are not in the docker group, prefix commands with 'sudo -E'."
