# IDS Peak SDK — Hardware camera driver

The IDS Peak SDK is required to use an **IDS Imaging USB3 industrial camera**
(the camera the STIMscope platform was originally validated with). Without
it, the camera-acquisition path falls back to simulation mode (no live
hardware feed) — every other capability of the platform still works.

This file documents the install steps. The SDK itself is NOT redistributed
with this repository because:

1. The SDK package is large (~500 MB).
2. IDS Imaging requires a registered download and the version you need
   depends on your Jetson's L4T release (JetPack version).

## Step 1 — Download the SDK from IDS

Visit <https://en.ids-imaging.com/download-peak.html>, create a free
account, and download the appropriate Linux build:

| If your Jetson runs | Download this |
|---|---|
| JetPack 5 (L4T R35.x, Ubuntu 20.04) | `ids-peak_<version>-<build>_arm64.deb` for Linux ARM64 |
| JetPack 6 (L4T R36.x, Ubuntu 22.04) | same — Linux ARM64 build |

The exact filename will look like `ids-peak_2.17.0.0-488_arm64.deb` or
later. Newer versions are typically backward-compatible.

## Step 2 — Drop the `.deb` at the repo root

Copy or symlink the downloaded `.deb` to the **root of this repository**
(same directory as `Dockerfile`, `build.sh`, `docker-compose.yml`):

```bash
cd <repo-root>
cp ~/Downloads/ids-peak_*.deb.
# OR
ln -s ~/Downloads/ids-peak_2.17.0.0-488_arm64.deb.
```

The `*.deb` filename is gitignored, so dropping it here does not pollute
the repository.

## Step 3 — Build the image

```bash./build.sh
```

`build.sh` auto-detects the `.deb` and includes it as a Docker build
layer. The IDS Peak Python bindings + GenICam transport are installed
into the image automatically at first run via `entrypoint.sh`.

## Step 4 — Mount the SDK at run time

`docker-compose.yml` mounts whatever path you put in the `IDS_PEAK_PATH`
environment variable into `/opt/ids-peak:ro`. The default is the standard
install location:

```bash
export IDS_PEAK_PATH=/opt/ids-peak     # default
sudo -E docker-compose up gui
```

If your install lives elsewhere, point `IDS_PEAK_PATH` there before running
`docker-compose up`.

## Verifying it works

After `docker-compose up gui`, the GUI's terminal output should include:

```
INFO IDS Peak initialized
```

If the camera is connected and powered, clicking **Start Hardware
Acquisition** in the GUI brings up a live preview (latency depends on
camera USB enumeration + IDS Peak SDK init + first-frame acquisition;
typically a few seconds on a healthy USB3 connection).

## Not using an IDS Peak camera?

The platform works without IDS Peak:

- **No camera at all** — simulation paths replace `Start Hardware
  Acquisition` outputs. Off-camera features (offline ROI segmentation,
  trace replay on saved video, calibration playback, viewer tools)
  still work.
- **MIPI / generic Linux camera** — set `STIM_CAMERA_BACKEND=mipi` or
  `=generic` and follow the prompts in
  [`docs/PORTABILITY.md`](docs/PORTABILITY.md). The platform's camera
  abstraction supports v4l2 + custom backends.

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| `INFO IDS Peak init attempt 1/3... Failed` | Camera not connected or USB cable underpowered. Use a powered USB3 hub. |
| `lsusb` shows the camera but `INFO IDS Peak` never appears | `IDS_PEAK_PATH` not set or wrong. Verify the path contains `lib/aarch64-linux-gnu/ids-peak/cti/*.cti`. |
| Build fails with `dpkg: error processing ids-peak_*.deb` | Wrong architecture or corrupt download. Re-download from IDS and verify SHA. |
| GUI launches but camera dropdown empty | Reboot the Jetson with the camera connected; some USB3 hubs need cold-boot enumeration. |
