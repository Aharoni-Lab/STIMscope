#!/bin/bash
set -e

# Make outputs deletable by the host user. The container runs as root
# (hardware access — GPIO/USB/IDS camera — needs it), so every file it writes
# into the bind-mounted /data would otherwise be root-owned and undeletable
# on the host without sudo. A permissive umask makes root-created files/dirs
# world-writable so the host user (a different UID) can manage/delete them.
umask 0000

# Designated writable output tree (bind-mounted from the host as /data).
# Ensure it exists so first-run saves never fall back to the read-only/source
# locations. STIM_SAVE_DIR / STIM_DATA_ROOT are set by the launchers.
# chmod the mount root world-writable so the host user (a different UID) can
# create/delete entries even if Docker auto-created /data as root:root. New
# files/dirs below it inherit deletability from the umask above.
mkdir -p "${STIM_SAVE_DIR:-/data}" 2>/dev/null || true
chmod 0777 "${STIM_SAVE_DIR:-/data}" 2>/dev/null || true

# Add CUDA libraries to path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH:-}

# Add conda to PATH if present (JP5)
if [ -d /opt/conda ]; then
    export PATH=/opt/conda/bin:${PATH}
fi

# Add IDS Peak libraries if mounted from host
if [ -d /opt/ids-peak ]; then
    # Find all directories containing .so files and add to LD_LIBRARY_PATH
    for dir in $(find /opt/ids-peak -name "*.so" -exec dirname {} \; 2>/dev/null | sort -u); do
        export LD_LIBRARY_PATH="${dir}:${LD_LIBRARY_PATH}"
    done
    # Set GenICam transport layer path for camera discovery
    CTI_DIR=$(find /opt/ids-peak -name "*.cti" -exec dirname {} \; 2>/dev/null | head -1)
    if [ -n "$CTI_DIR" ]; then
        export GENICAM_GENTL64_PATH="${CTI_DIR}"
    fi
    # Install Python bindings if not already present
    python3 -c "import ids_peak" 2>/dev/null || \
        pip install --quiet ids_peak ids_peak_ipl ids_peak_afl 2>/dev/null || true
fi

exec python3 "$@"
