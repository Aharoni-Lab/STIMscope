# Install

These steps build the CRISPI Docker image from source on an NVIDIA
Jetson. Once the upstream Docker image is publicly available, this
page will lead with `docker pull`; until then the from-source path
is the only option.

## Prerequisites

1. **NVIDIA Jetson** with JetPack 5 (L4T R35.x) or JetPack 6 (L4T R36.x).
   Tested on AGX Orin (JetPack 6); the Dockerfile also targets JetPack 5
   hosts.
2. **Docker** with the NVIDIA Container Toolkit:
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
3. **NVIDIA runtime** configured as the default Docker runtime:
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
4. *(Hardware mode only)* **IDS Peak SDK** `.deb` for ARM64. License
   forbids redistribution; download it yourself from
   <https://en.ids-imaging.com/download-peak.html> (Linux ARM 64-bit,
   version 2.17.0) and drop the `.deb` at the repo root before building.
   Simulation mode works without it.

## Clone and build

```bash
git clone https://github.com/Aharoni-Lab/STIMscope.git
cd STIMscope
./build.sh                       # auto-detects JetPack version
```

`build.sh` reads `/etc/nv_tegra_release` to pick the right base image
(`r35.x` for JP5, `r36.x` for JP6) and the right CuPy package
(`cupy-cuda11x` vs `cupy-cuda12x`). It also creates a 0-byte stub
for the IDS Peak `.deb` if you didn't supply one, so the image
builds and simulation mode still works.

## Run

X11 setup (required once per shell session for the GUI):

```bash
export DISPLAY=:0
xhost +local:docker
```

The GUI is the operator entry point:

```bash
sudo -E docker-compose up gui
```

The `-E` flag preserves your `DISPLAY` env var through sudo. The GUI
covers camera control, calibration, projector / DMD masking,
recording, and live trace extraction — see the
[GUI Reference](GUI-Reference). When no camera or projector is
present, the platform falls back to simulation-friendly behavior
(see [Portability](Portability)).

## Verifying the build

Before launching the GUI, smoke-check that the image's core modules import cleanly:

```bash
docker run --rm --entrypoint python3 crispi:latest -c \
  "import sys; sys.path.insert(0, '/app/STIMViewer_CRISPI/CS'); \
   from core import projector, structured_light, paths, logging_config; \
   print('core imports OK')"
```

If it prints `core imports OK`, the image is healthy enough to launch the GUI. GPU + IDS Peak SDK + GPIO are runtime-optional; missing pieces fall back rather than fail.

Then launch the GUI (`sudo -E docker-compose up gui`); the main window
should open on your display. If it doesn't, see
[Troubleshooting](Troubleshooting).

## Data ownership

The container runs as root, so files written into `data/` are
root-owned on the host. Reclaim with:

```bash
sudo chown -R $(id -u):$(id -g) data/
```

## Editing source code (development)

The repo's `STIMViewer_CRISPI/` and `data/` directories are bind-mounted
into the container by `docker-compose.yml`, so Python edits on the host
appear inside the running container on the next process restart — no
rebuild required for code changes. Rebuild is required for changes to
`requirements.txt`, `Dockerfile`, `entrypoint.sh`, or the C++ projector
engine.

## Build for a specific JetPack version

Bypass `build.sh` if you need explicit control:

```bash
# JetPack 6
docker build \
  --build-arg L4T_JETPACK_VERSION=r36.2.0 \
  --build-arg CUDA_VERSION=12.2 \
  --build-arg CUPY_PACKAGE=cupy-cuda12x \
  -t crispi:latest .

# JetPack 5
docker build \
  --build-arg L4T_JETPACK_VERSION=r35.2.1 \
  --build-arg CUDA_VERSION=11.4 \
  --build-arg CUPY_PACKAGE=cupy-cuda11x \
  -t crispi:latest .
```

## Next

- [Hardware Setup](Hardware-Setup) for the IDS Peak SDK install +
  projector / GPIO wiring.
- [Troubleshooting](Troubleshooting) if `docker-compose up` doesn't
  produce the expected output.
