#!/bin/bash
set -e
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
