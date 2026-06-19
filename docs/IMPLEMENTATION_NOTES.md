# STIMscope — Implementation Notes

![Fig 4a — CRISPI software architecture](figures/fig04a_software_architecture.png)
*Fig 4a — Six-module CRISPI software architecture. The
**Initialization**, **Calibration**, **Central Real-Time**, **Real-Time
Trace Extraction**, and **Visualization Dashboard** modules are
implemented in this release. The **Inference Module** (Feature
Extraction → Adaptive Mask Generation + Local Memory) is **scaffolded
but not implemented in this version**. All
inter-module flow is over ZeroMQ.*

This document describes the current implementation of the platform.
Behavior is still evolving; everything below is a snapshot of what the
code does today, not a contract.

---

## Software architecture (this release)

This release ships the **interactive Qt GUI** (the operator
path) and the **C++ projector engine** (the renderer the GUI drives). The
inference module that would close the loop on activity-dependent stimulation
is **scaffolded but not implemented in this version** — this matches the
explicit statement:

> "While CRISPI provides a hardware-synchronized framework for online trace
> extraction and calibrated mask delivery, the inference module that would
> enable activity-dependent closed-loop stimulation is not implemented in
> the current version. The modular architecture defines its interfaces,
> data flow, and intended role, providing a scaffold for future
> implementations."

### What's in `STIMscope/STIMViewer_CRISPI/CS/core/`

The directory holds five files. Four are active platform code that the
rest of the GUI imports from; the package marker is the fifth.

| File | Role |
|---|---|
| `projector.py` (~401 LOC) | Canonical ZeroMQ wire constants (`DEFAULT_MASK_ENDPOINT = tcp://127.0.0.1:5558`, `DEFAULT_HOMOGRAPHY_ENDPOINT = tcp://127.0.0.1:5560`, `5562` status) used by `projector_client.py`, `qt_interface_mixins/*`, `sl_calibrate.py`, and `asift_calibration.py`. |
| `structured_light.py` (~402 LOC) | Gray-code structured-light calibration patterns invoked by the Structured-Light Calibrate flow. |
| `paths.py` (~143 LOC) | XDG path discovery (`Assets/Generated/...`, save dirs, log dirs) used across the GUI. |
| `logging_config.py` (~67 LOC) | Common logging factory. |
| `__init__.py` (~1 LOC) | Package marker. |

The `CS/` directory name is a historical artifact from an earlier
in-tree experiment; the directory was not renamed because the four
active files above are imported from many call sites in the GUI.

Closed-loop inference is the future-work extension point
and is not implemented in this release.

### Qt GUI (`STIMscope/STIMViewer_CRISPI/`)

Everyday operator path. Boots on `docker-compose up gui`.

```
main_gui.pyw                  # Bootstrap: OpenGL safety env, perf monitors,
                              # GIL-aware thread mgr, ZMQ port manager,
                              # Jetson clock/governor tweaks, X11 detection.
main.py                       # Application entry; constructs and shows the
                              # main window.
qt_interface.py               # Parent Interface(QMainWindow); composes 20
                              # mixins from qt_interface_mixins/.
qt_interface_mixins/          # Per-feature mixins: button bar, troubleshoot
                              # window, offline-setup dialog, I²C dialog,
                              # trigger controls, mask ops, calibration
                              # projector, sensor settings, etc.
camera.py                     # OptimizedCamera(QObject) — IDS Peak SDK
                              # wrapper with Qt signals. Owns acquisition,
                              # recording, calibration handshake, FPS.
calibration.py                # ArUco/ChArUco homography. Returns a typed
                              # CalibrationResult so silent fallbacks to
                              # np.eye(3) are no longer possible.
display.py                    # OpenGL display helpers.
video_recorder.py             # TIFF + mp4 writer.
projector_client.py           # Thin ZMQ wire client to the C++ projector
                              # engine on tcp://127.0.0.1:5558.
gpu_ui.py + gpu_ui_mixins/    # GPU-side viewer / export dialog mixins.
live_trace/                   # Real-time ROI trace extraction subsystem.
roi_thresh.py                 # ROI threshold helper.
roi_editor.py                 # napari "Refine ROIs" entry point —
                              # part of the planned napari removal
                              # (see "Planned removals" below).
make_mmap.py, otsu_thresh.py  # ROI generation utilities.
```

---

## C++ projector engine

Lives at `STIMscope/ZMQ_sender_mask/main.cpp`. Single translation unit
that drives the 1920×1080 DMD over OpenGL/GLFW, exposes:
- a ZMQ pull socket on `tcp://127.0.0.1:5558` for incoming mask frames
- a ZMQ REP socket on `tcp://127.0.0.1:5560` for live homography updates
- two GPIO trigger lines (projector edge + camera edge) via `libgpiod`

Built once during the Docker image build (`make projector` target).
Both stacks talk to it via the ZMQ sockets.

The DLPC3479 (DLP4710 DMD controller) I²C driver lives at
`STIMscope/ZMQ_sender_mask/dlpc_i2c.py` and encodes the TI DLPU081A
datasheet opcodes directly. Several documented quirks against the
datasheet were folded into the code — see commit history for
attribution.

---

## Subsystem file map (capability → files)

For each user-facing capability in the wiki Features page, the table below
lists which files implement it. Maintaining this map is **the single biggest
contributor to the wiki not drifting from the code** — keep it current when
you move things around.

| Capability | Python (GUI) | C++ / native |
|---|---|---|
| Main GUI shell | `qt_interface.py` + 20 mixins in `qt_interface_mixins/` | — |
| Application bootstrap | `main_gui.pyw` → `main.py` | — |
| Camera capture | `camera.py` (`OptimizedCamera(QObject)`, ~1,440 LOC) | — |
| Camera controls in GUI | `qt_interface_mixins/camera_controls.py`, `sensor_settings.py`, `triggers.py`, `trig_params.py`, `hw_acq.py` | — |
| Calibration — ArUco/ChArUco | `calibration.py` | — |
| Calibration — ASIFT | `ZMQ_sender_mask/asift_calibration.py` (CLI utility), called from GUI via `qt_interface_mixins/calib_projector.py` | — |
| Calibration — structured-light | `qt_interface_mixins/sl_calibrate.py`, `STIMViewer_CRISPI/CS/core/structured_light.py` | — |
| Calibration GUI orchestration | `qt_interface_mixins/offline_setup.py`, `calib_projector.py`, `sl_calibrate.py` | — |
| Recording | `video_recorder.py` | — |
| Projection wire (Python side) | `projector_client.py` (thin ZMQ wrapper); `STIMViewer_CRISPI/CS/core/projector.py` (canonical ZMQ endpoint constants + richer client) | — |
| Projection wire (engine side) | — | `ZMQ_sender_mask/main.cpp` (~1,927 LOC; OpenGL + GLFW + ZMQ + GPIO) |
| DMD I²C control | `qt_interface_mixins/i2c_dialog.py` (GUI front-end); `ZMQ_sender_mask/dlpc_i2c.py` (the driver) | C++ engine uses smbus directly |
| GPIO + LED control | `qt_interface_mixins/led_and_procs.py` (GUI dropdown) | C++ engine via `libgpiod` |
| Mask projection (GUI mgmt) | `qt_interface_mixins/mask_ops.py`, `projection_controls.py` | — |
| Live trace extraction (RTTE) | `live_trace/extractor.py` (~706 LOC) + 8 mixins under `live_trace/` (`ingest.py`, `processing.py`, `perf.py`, `plot_pagination.py`, `plot_aggregation.py`, `plot_modes.py`, `plot_layouts.py`, `init.py`) | — |
| GPU UI window (trace plots) | `gpu_ui.py` + mixins under `gpu_ui_mixins/` (`export_fast.py`, `export_slow.py`, `export_tabs.py`, `export_viewer.py`, `health.py`, `napari.py` (planned removal), `roi_discovery.py`, `traces.py`) | — |
| Inference module hook (future-work; not implemented in this release) | `qt_interface_mixins/cs_pipeline_dialog.py` (UI hook only) | — |
| Trace test sub-window | `qt_interface_mixins/trace_test.py` + `STIMViewer_CRISPI/test_trace_fidelity.py` (CLI) | — |
| Troubleshoot menu | `qt_interface_mixins/troubleshoot.py` (~1,463 LOC) | — |
| Pixel probe / overlay | `qt_interface_mixins/overlay_probe.py` | — |
| ROI generation helpers | `roi_thresh.py`, `otsu_thresh.py`, `make_mmap.py` | — |
| ROI editor (napari, planned removal) | `roi_editor.py`, `gpu_ui_mixins/napari.py` | — |
| Frame-receive plumbing | `qt_interface_mixins/image_received.py`, `window_lifecycle.py` | — |
| Cellpose segmentation | `cellpose_runner.py` | — |
| XDG paths + logging factory | `STIMViewer_CRISPI/CS/core/paths.py`, `logging_config.py` | — |

---

## Test layers

Tests live under `tests/` (separate from the source tree). Each layer
maps to a degree of I/O the test is willing to touch:

| Layer | What it tests | I/O | Hardware |
|---|---|---|---|
| `tests/L1_algorithms/` | Pure NumPy maths in `core/` | none | none |
| `tests/L2_orchestration/` | CLI parsing, config plumbing, dispatch | argparse only | none |
| `tests/L3_hardware/` | HAL implementations w/ fake backends | mocked | mocked |
| `tests/L3_projector/` | DLPC3479 I²C driver | mocked I²C | mocked |
| `tests/L3_5_split_first/` | Live-trace extractor mixins | Qt offscreen | none |
| `tests/L4_orchestration/` | Multi-threaded hot-path orchestration | mocked | mocked |
| `tests/L5_UI/` | Qt mixin units under offscreen platform | Qt offscreen | none |

CI runs L1 + L2 + infrastructure-smoke + bandit + ruff on the GitHub
free tier (ubuntu-latest, x86, CPU-only). Hardware-dependent layers
(L3+, L5+) run on a Jetson via `make test`, where CuPy, IDS Peak,
GPIO, and the DMD are present.

---

## Hardware overview

![Fig 1b — Hardware architecture (image sensor, DMD, MCU, Jetson)](figures/fig01b_hardware_architecture.png)
*Fig 1b — Image sensor, DMD projector, microcontroller, and
NVIDIA Jetson Orin synchronization fabric. The MCU clocks every camera
exposure (Trig-Out 1 / 2 to camera + DMD); the host configures the
DMD over I²C and streams patterns over HDMI; the host talks to the MCU
over UART.*

- **Camera.** Sony **IMX334** / **IMX290** small-pixel back-illuminated
  CMOS in an IDS Peak USB3 housing. SDK at `/opt/ids-peak`
  (bind-mounted into the container). Python bindings: `ids_peak`,
  `ids_peak_ipl`, `ids_peak_afl`. Setup is fully fallback-tolerant —
  if the SDK isn't installed the simulation path still works. (Fig 1b.)
- **Projector.** TI **DLP4710** DMD via **DLPC3479** controller. Driven
  by the C++ engine over HDMI + OpenGL; configured over I²C (addr
  `0x1B`) by `dlpc_i2c.py`. Per-pattern trigger out via `libgpiod`.
  (Fig 1b.)
- **Microcontroller.** Microchip **ATSAMD51** (Adafruit Grand Central
  M4); the slave-trigger source for camera exposures and DMD pattern
  advances. UART to host at 9600 bps (`[0x02][mode][len][data]` packet
  framing).
- **Illumination.** DMD-internal. RED / BLUE channel selection happens
  inside the DLPC3479 per pattern via I²C opcode `0x96` byte 3
  (Illumination Select). There are no separate per-LED GPIO pins on
  the host side — operator-facing surface is the `LED Color` dropdown.
- **GPIO.** Trigger lines only (camera trigger + projector trigger).
  `libgpiod` on a gpiochip / line numbers chosen via `STIM_GPIO_CHIP`
  / `STIM_CAM_LINE` / `STIM_PROJ_LINE` env vars (no hardcoded chip
  paths or line numbers in source).
- **Jetson.** NVIDIA Jetson AGX Orin (JetPack 6 / L4T R36.x); also
  tested on Xavier-class with JetPack 5 (L4T R35.x). `build.sh`
  auto-detects.

---

## Attribution

The STIMscope hardware platform is © Aharoni Lab, UCLA (GPL-3.0).
The platform is described in detail (see
[CITATION.cff](../CITATION.cff)).

