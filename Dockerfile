ARG L4T_JETPACK_VERSION=r36.2.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_JETPACK_VERSION}

ARG CUDA_VERSION=12.2
ARG CUPY_PACKAGE=cupy-cuda12x

ENV CUDA_VERSION=${CUDA_VERSION}
ENV DEBIAN_FRONTEND=noninteractive

# Layer 1a: On JP5 (Ubuntu 20.04, Python 3.8), install Python 3.10 + PyQt5 via miniforge
# On JP6 (Ubuntu 22.04), Python 3.10 is already the system default — skip this
RUN if [ "$(python3 --version | cut -d' ' -f2 | cut -d. -f1-2)" != "3.10" ]; then \
        apt-get update && apt-get install -y --no-install-recommends wget && \
        wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/miniforge.sh && \
        bash /tmp/miniforge.sh -b -p /opt/conda && \
        rm /tmp/miniforge.sh && \
        /opt/conda/bin/conda install -y python=3.10 pyqt numpy scipy pip && \
        /opt/conda/bin/conda clean -afy && \
        ln -sf /opt/conda/bin/python3 /usr/local/bin/python3 && \
        ln -sf /opt/conda/bin/python3 /usr/bin/python3 && \
        ln -sf /opt/conda/bin/pip /usr/local/bin/pip && \
        ln -sf /opt/conda/bin/pip /usr/bin/pip && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Layer 1b: System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # X11 / GUI
    libx11-dev libxext-dev libxrender-dev libxcb1-dev \
    libgl1-mesa-glx libgl1-mesa-dev libglib2.0-0 \
    libfontconfig1 libdbus-1-3 \
    # C++ projector build
    libglfw3-dev libglew-dev libzmq3-dev libgpiod-dev \
    g++ pkg-config \
    # Python tools (JP6 needs these; JP5 has them from miniforge)
    python3-pip python3-dev \
    # Camera / USB
    libusb-1.0-0 \
    # External TIFF viewer for the "Open in External Viewer" button
    # (xdg-open handler + a TIFF-capable viewer). Operators can install
    # Fiji/ImageJ for full stack tools; eog covers single/first-frame view.
    xdg-utils eog \
    && rm -rf /var/lib/apt/lists/*

# Layer 1c: PyQt5 — apt on JP6 (matches Python 3.10), already installed via conda on JP5
RUN python3 -c "from PyQt5 import QtWidgets" 2>/dev/null || \
    (apt-get update && apt-get install -y --no-install-recommends python3-pyqt5 && \
     rm -rf /var/lib/apt/lists/*)

# Layer 2: IDS Peak camera SDK (optional — see IDS-PEAK-SDK.md for the install
# flow). The .deb requires Ubuntu 22.04 deps. If installation fails (e.g. on
# JetPack 5), the platform falls back gracefully to camera-absent mode.
COPY ids-peak_2.17.0.0-488_arm64.deb /tmp/
RUN dpkg -i /tmp/ids-peak_2.17.0.0-488_arm64.deb || true; \
    apt-get update && apt-get install -f -y --no-install-recommends || true; \
    rm -f /tmp/ids-peak_2.17.0.0-488_arm64.deb; \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir ids_peak ids_peak_ipl ids_peak_afl || \
    echo "WARNING: IDS Peak not available — camera hardware mode disabled, simulation works fine"

# Layer 3: C++ projector binary
COPY STIMscope/ZMQ_sender_mask/ /app/ZMQ_sender_mask/
WORKDIR /app/ZMQ_sender_mask
RUN g++ -O2 -std=c++17 main.cpp -o projector \
    -lglfw -lGL -lzmq -lgpiod -lpthread -lGLEW

# Layer 4: Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt ${CUPY_PACKAGE}

# Layer 5: Application code
COPY STIMscope/STIMViewer_CRISPI/ /app/STIMViewer_CRISPI/
# GUI is the entry point; cwd must be STIMViewer_CRISPI so main_gui.pyw's
# sibling imports (main, kill_zombies, qt_interface) resolve. The kept
# core.* shared-infra package is found via CS/ added to sys.path by
# calibration.py/camera.py (relative to __file__, not cwd).
WORKDIR /app/STIMViewer_CRISPI

# Build info — readable via `make status` or `cat /app/build_info.txt` in a container.
# Used to detect image/source skew ("is the running image actually built from
# platform-stable HEAD?"). Values are best-effort — git is available in the
# build context for JP6, projector binary hash reflects the compile step above.
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
RUN ( \
        echo "image:            crispi:latest"; \
        echo "build_date:       ${BUILD_DATE}"; \
        echo "git_sha:          ${GIT_SHA}"; \
        echo "jetpack_base:     ${L4T_JETPACK_VERSION}"; \
        echo "cuda_version:     ${CUDA_VERSION}"; \
        echo "cupy_package:     ${CUPY_PACKAGE}"; \
        printf "projector_sha256: "; sha256sum /app/ZMQ_sender_mask/projector 2>/dev/null | awk '{print $1}' || echo "missing"; \
        printf "ids_peak_ver:    "; (python3 -c "import ids_peak; print(ids_peak.__version__)" 2>/dev/null) || echo "not_installed"; \
        printf "imagecodecs_ver: "; (python3 -c "import imagecodecs; print(imagecodecs.__version__)" 2>/dev/null) || echo "not_installed"; \
    ) > /app/build_info.txt && cat /app/build_info.txt

# OCI provenance labels (metadata only — no effect on build steps or
# runtime). image.revision is the git SHA passed via the GIT_SHA
# build-arg above; verify it against /app/build_info.txt.
LABEL org.opencontainers.image.source="https://github.com/Aharoni-Lab/STIMscope"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.licenses="GPL-3.0"

# Create non-root user for running the pipeline
RUN useradd -m -u 1000 crispi
# Keep root for now since hardware access requires it (GPIO, USB, IDS camera)
# USER crispi

# Entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["main_gui.pyw"]
