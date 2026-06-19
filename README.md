# STIMscope

![STIMscope platform in the inverted configuration](docs/figures/upstream_stimscope_inverted.jpg)

**STIMscope** (**S**patio-**T**emporal **I**llumination **M**icroscope) is
an open-source benchtop platform for simultaneous imaging and patterned
optical stimulation. A synchronized control system coordinates the camera,
the DMD-based patterned-light projector, illumination, and GPU-accelerated
analysis to support all-optical neural-interrogation experiments.

This repository packages the STIMscope platform as a Docker
distribution for NVIDIA Jetson: the Qt GUI, the C++ projector engine,
the calibration suite, the live-trace pipeline, hardware diagnostics,
and the per-feature workflows so a complete setup can be reproduced
on commodity edge hardware.

> Reference: Chorsi *et al.*, *STIMscope: A high-resolution, low-cost,
> optogenetic stimulation platform for closed-loop manipulation of neural
> activity at the centimeter scale*, bioRxiv 2026 — DOI
> [10.64898/2026.05.27.728160](https://www.biorxiv.org/content/10.64898/2026.05.27.728160v1).

![Fig 1a — STIMscope platform photo (inverted configuration)](docs/figures/fig01a_platform_photo.png)
*Fig 1a — Photo of the implemented STIMscope
platform in the inverted configuration: sample holder, objective,
GPU processing unit (NVIDIA Jetson AGX Orin), microcontroller, DMD,
and stage controller.*

## What the platform supports

Each of the following is a first-class capability of the GUI — none is a
prerequisite for the others, and the order in which an operator uses them
depends on the experiment.

| Capability | What it does |
|---|---|
| **Live camera acquisition** | IDS Peak USB3 (default), MIPI, or generic camera; software or hardware trigger; analog + digital gain; per-frame exposure control |
| **Recording** | TIFF stacks of the live feed; snapshot for single frames; in-app + external TIFF viewers |
| **DMD patterned projection** | Send static masks, mask folders, or trial-driven sequences through the C++ projector engine; per-pattern trigger out for synchronization |
| **Illumination control** | DMD-internal RED/BLUE/RGB channels (DLPC3479 Illumination Select); GPIO camera + projector trigger lines via `libgpiod` (env-configurable) |
| **Calibration suite** | ArUco/ChArUco autonomous DMD→camera homography; Affine-SIFT feature-matching; structured-light sub-pixel LUT; reload/push existing H or LUT to the engine |
| **Real-time trace extraction (RTTE)** | Per-ROI mean fluorescence per camera frame; paginated multi-ROI plots in PyQtGraph; live ΔF/F overlay + optional OASIS preview deconvolution; snapshot + comprehensive export. |
| **Offline ROI segmentation** | Otsu (with optional watershed splitting); Cellpose (`cyto2` / `cyto` / `nuclei` / custom) when installed. |
| **Hardware diagnostics** | Pixel probe, DMD R/B isolation, GPIO trigger pulse tests, engine monitor, LUT diagnostics suite (round-trip error, dot array, edge strip, calib characterization) |
| **I²C control** | Arbitrary DLPC3479 opcode bursts with templates, configurable bus + address, single-byte reads + atomic-burst writes |
| **Sensor settings** | Hardware-exposed analog gain, digital gain, exposure, contrast, gamma controls |

See [Features](wiki/Features.md) and [GUI Reference](wiki/GUI-Reference.md)
in the wiki for the full feature catalog.

## Hardware

The platform composes off-the-shelf parts into a bill of materials
under USD $5,000. Synchronization
between the image sensor, DMD projector, microcontroller, and Jetson
follows the architecture in Fig 1b.

| Component | What we use | Reference |
|---|---|---|
| Compute | NVIDIA Jetson AGX Orin (JetPack 5/L4T R35.x or JetPack 6/L4T R36.x) |  |
| Camera | Sony **IMX334** / **IMX290** small-pixel back-illuminated CMOS in an IDS Peak USB3 housing (MIPI / generic-camera paths also supported) | Fig 1b |
| Projector | TI **DLP4710** DMD driven by **DLPC3479** controller (I²C) | Fig 1b |
| Microcontroller | Microchip **ATSAMD51** (Adafruit Grand Central M4) | Fig 1b |
| Sync | `libgpiod` — gpiochip + line numbers env-configurable (`STIM_GPIO_CHIP`, `STIM_CAM_LINE`, `STIM_PROJ_LINE`) |  |

The platform falls back to simulation-friendly modes (no camera, no
projector) when hardware is absent — see
[Hardware Setup](wiki/Hardware-Setup.md) and
[Portability](wiki/Portability.md).

## Quick Start

```bash
git clone https://github.com/Aharoni-Lab/STIMscope.git
cd STIMscope./build.sh                       # auto-detects JetPack version
export DISPLAY=:0
xhost +local:docker
sudo -E docker-compose up gui    # full GUI
```

For prerequisites (NVIDIA Container Toolkit, IDS Peak SDK download path,
JetPack-specific build args), see [Install](wiki/Install.md).

## Portability

Every machine-specific value (data root, I²C bus, GPIO chip + lines,
default fps/exposure, recording format) is an environment variable
read at startup — no rebuild required to retarget a different Jetson
or carrier board. See [docs/PORTABILITY.md](docs/PORTABILITY.md) for
the full env-var surface and a sanity-check on a fresh machine.

## Performance characterization

| Metric | Value | Reference |
|---|---|---|
| Trigger-to-photodiode latency (mask → light) | **26.3 ms** (mean) | Fig 4e; 5,000-mask photodiode run |
| End-to-end closed-loop latency (project + capture + ROI extract) | **91.6 ms** | Fig 4f |
| Targeting accuracy (RMS error, ≈ 85,000 targets, 1936 × 1096 field) | **0.46 px ≈ 1.3 µm** | Fig 4c |
| Imaging FWHM (lateral) | **5.6 µm** center / **5.8 µm** edge (4 µm fluorescent beads, f/4) | Fig 2c–e |
| Excitation FWHM (lateral) | **5.8 µm** center / **6.2 µm** edge (single DMD pixel) | Fig 2f–g |
| Field of view (demagnified) | **14 × 11 mm²** | Fig 1f, Fig 3a |

The closed-loop end-to-end latency in Fig 4f explicitly **excludes** an
inference model — see [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md)
for the scope and implementation status of the platform.

## Cite

If you use STIMscope in research, see [CITATION.cff](CITATION.cff)
(GitHub renders a "Cite this repository" button from it). The
[NOTICE](NOTICE) file preserves upstream attribution. Figures
reproduced in this repository are subject to the
[CC BY-NC-ND 4.0](docs/figures/LICENSE-FIGURES.md) license,
independently of this repository's software license.

## License

GPL-3.0 — see [LICENSE](LICENSE).
