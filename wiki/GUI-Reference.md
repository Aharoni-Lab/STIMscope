# GUI Reference

Per-control reference for the STIMscope Qt interface. Organized by **the
surface the control appears on** (main button bar, then each dialog /
sub-window) — not by workflow, because operators combine these features
in whichever order their experiment requires.

For the capability framing (what each feature is for), see
[Features](Features). For the architectural view, see
[Architecture](Architecture).

> Tooltips in the running GUI are authoritative. If this page disagrees
> with a tooltip, the tooltip wins — file a doc PR to update this page.

---

## Main button bar

The always-visible control surface at the top / side of the main window.
Buttons grouped here by function. Physical layout in the GUI may differ.

### Acquisition + recording

| Control | Type | Action |
|---|---|---|
| `Camera Type` | dropdown | `IDS_Peak` / `MIPI` / `Generic Camera` |
| `Start Hardware Acquisition` | toggle button | Acquire images via hardware trigger rather than RT mode. Tooltip surfaces the hardware-trigger fps behavior; defined in [`qt_interface_mixins/button_bar.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/qt_interface_mixins/button_bar.py). |
| `Snapshot` | button | Save the next processed frame as a single image. |
| `Start Recording` | toggle button | Start/Stop recording video of the live feed to TIFF. |
| `View Recording` | button | Open a saved TIFF in an in-app viewer with frame slider + auto-contrast. |
| `Open in External Viewer` | button | Open the most recent saved TIFF in the system default image viewer. |
| `Rotate 90°` | button | Cycle camera preview rotation through 0° → 90° → 180° → 270° (display only, NOT projection). |
| Camera `Flip H` | checkbox | Mirror camera preview horizontally (affects display + recording). |
| Camera `Flip V` | checkbox | Mirror camera preview vertically (affects display + recording). |

### Projection engine + masks

| Control | Type | Action |
|---|---|---|
| `Start Projection Engine` | toggle button | Spawn/kill the C++ projector engine subprocess; binds the ZMQ ports defined in [`CS/core/projector.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/CS/core/projector.py) (mask, homography, status). |
| `Project ON` | button | Begin pattern display (engine must be running). |
| `Project OFF` | button | Stop pattern display without stopping the engine. |
| `Send Masks` | toggle button | Start/Stop streaming masks over ZMQ to the projector. |
| Mask pattern `Browse…` | button | Pick a single mask file (NPZ / PNG) to queue. |
| `Start Projector Trigger` | toggle button | Start/Stop asserting per-pattern GPIO trigger edges. |
| `HW Trigger Out` | toggle button | Per-pattern GPIO output for downstream sync. The line is set at projector-engine startup; env-configurable via `STIM_PROJ_LINE` (see [Portability](Portability)). |
| `LED Color` | dropdown | DMD Illumination Select (I²C `0x96` byte 3) for the initial pattern at `Start Projector Trigger`. Items + raw bytes defined in [`button_bar.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/qt_interface_mixins/button_bar.py) (`_led_color_dropdown`). |
| `Sequence Type` | dropdown | Pattern sequence type. |
| `Projection Mode` | dropdown | How red (stim) and blue (observe) masks are presented — `Simultaneous (Mode B)` (R+B sub-frame multiplexing) or `Temporal (Mode A)` (alternating RED ↔ BLUE per frame). |
| `Mask Flip H` | checkbox | Flip the outgoing DMD mask horizontally. Auto-restarts the mask sender. |
| `Mask Flip V` | checkbox | Flip the outgoing DMD mask vertically. Auto-restarts the mask sender. |

### Calibration

| Control | Type | Action |
|---|---|---|
| `Calibrate` | button | Autonomous DMD→camera ArUco / ChArUco homography ([`calibration.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/calibration.py)). |
| `Structured-Light Calibrate` | button | Sub-pixel LUT via sinusoidal phase patterns ([`qt_interface_mixins/sl_calibrate.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/qt_interface_mixins/sl_calibrate.py)). |
| `Project LUT-Warped` | button | Switch projection through the structured-light LUT. |
| `ASIFT Calibration` | button | Compute 3×3 H using Affine-SIFT, apply to projector. Requires `Calibrate` to have run first; warns otherwise. |
| `REQ H-Matrix` | button | Send the current 3×3 calibration to the engine over the ZMQ homography sideband (default endpoint in `CS/core/projector.py`). |
| `REQ LUT` | button | Send the structured-light LUT to the engine over ZMQ. |

### Camera tuning

| Control | Type | Action |
|---|---|---|
| `Set Trig Params` | button | Open dialog to configure `TriggerDelay` (µs) + `ExposureTime` (µs) together. |
| `Sensor Settings` | button | Open low-level GenICam node panel (gain, contrast, gamma, exposure). |

### Diagnostics

| Control | Type | Action |
|---|---|---|
| `Pixel Probe` | button | Project a single bright pixel; verify camera sees it where calibration predicts. |
| `Enable Overlay` | toggle button | Toggle the camera-on-projection overlay. |
| `I²C Burst Sender` | button | Open dialog for arbitrary DLPC3479 opcode bursts. |
| `Troubleshooting` | button | Open the troubleshooting menu (engine monitor, LUT diagnostics, single-pixel probe, etc.). |

### Workflow entry points

| Control | Opens |
|---|---|
| `Offline Setup` | The five-panel A–E offline segmentation dialog. |
| `Trace Test` | The trace-test sub-window for live ROI fluorescence testing. |
| `Real-Time Trace Extraction` | The GPU UI window with per-ROI live plots. |

### Status indicators

The button bar surfaces several non-clickable indicators with tooltips:

- "Current Acquisition Mode"
- "Projector connection status"
- Calculated FPS indicator (label `FPS: N`; defined in
  [`qt_interface_mixins/window_lifecycle.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/qt_interface_mixins/window_lifecycle.py)
  via `GUIfps_label` — rolling average over the last ~2 s, refreshed
  every 250 ms by a QTimer; read-only display label, not interactive)

---

## Sensor Settings dialog

Opened via `Sensor Settings`. Live tweaks for the camera's exposed
GenICam controls.

| Control | Type | Notes |
|---|---|---|
| Analog Gain | slider + label | "Adjust the analog gain level (brightness)." |
| Digital Gain | slider + label | "Adjust the digital gain level." |
| Exposure (µs) | slider + numeric | Exposure in microseconds. Live readback on dialog open from the camera's current `ExposureTime` node. Default range / step defined in [`qt_interface_mixins/sensor_settings.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/qt_interface_mixins/sensor_settings.py). |
| Exposure entry | line edit | "Type exposure in µs and press Enter." |
| Hardware Contrast | label + control | "Hardware Contrast (camera control). 1.0 is neutral on most cameras." (only if the camera exposes the node) |
| Hardware Gamma | label + control | "Hardware Gamma (brightness curve). 1.0 is neutral; <1 brightens, >1 darkens." (only if the camera exposes the node) |
| Contrast unavailable note | label | "Contrast not exposed by camera; consider a software preview option if needed." (shown when the node is absent) |
| `Set` | button | Commit slider values to the camera. |
| `Close` | button | Dismiss. |

---

## Set Trig Params dialog

Opened via `Set Trig Params`. Configures the camera's TriggerDelay (µs)
+ ExposureTime (µs) together for hardware-triggered acquisition.

| Control | Type | Action |
|---|---|---|
| `Blue sub-frame` | preset button | Apply preset matching a color-DMD 8-bit sub-frame. Delay/exposure values are in the button label rendered by the GUI; defined in [`qt_interface_mixins/trig_params.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/qt_interface_mixins/trig_params.py). |
| `Full frame` | preset button | Apply preset matching one full DMD frame. Delay/exposure values are in the button label rendered by the GUI; defined in `trig_params.py`. |
| `Enable TriggerDelay (µs)` | checkbox | Toggle TriggerDelay control. |
| TriggerDelay manual entry | line edit | Override preset. |
| `Enable ExposureTime (µs)` | checkbox | Toggle ExposureTime control. |
| ExposureTime manual entry | line edit | Override preset. |
| Activation dropdown | combobox | `RisingEdge` / `FallingEdge` / `LevelHigh` / `LevelLow` |
| Trigger Source dropdown | combobox | `Line0` / `Line1` / `Line2` / `Line3` |
| `Apply` | button | Commit values to camera. |
| `Close` | button | Cancel without applying. |

---

## I²C Burst Sender dialog

Opened via `I²C Burst Sender`. Send arbitrary DLPC3479 opcode bursts for
manual DMD configuration.

| Control | Type | Tooltip / Action |
|---|---|---|
| I²C bus number | spin/entry | Configurable. Default for the DMD on Jetson AGX Orin documented in [docs/PORTABILITY.md](https://github.com/Aharoni-Lab/STIMscope/blob/main/docs/PORTABILITY.md). |
| I²C 7-bit address | spin/entry | "7-bit I²C address. DLPC3479 = 0x1B." |
| Burst editor | text area | Type or load multi-byte opcode sequences. |
| Template dropdown | combobox | "Replace burst editor contents with the selected template." |
| `Load` | button | Load opcodes from a `.json` / `.txt` file. |
| Bytes to read | spin | "Bytes to read." |
| `Read Once` | button | "Read N bytes from the given opcode and append result to the log." |
| `Send All (atomic burst)` | button | Send the queued sequence in one I²C transaction. |
| `Clear Log` | button | Clear the response log panel. |
| `Close` | button | Dismiss. |

---

## Offline Setup dialog

Opened via `Offline Setup`. Five panels A–E for turning a recorded TIFF
stack into an ROI mask file.

### A. Recording Selection

| Control | Type | Action |
|---|---|---|
| `Load Recording` | button | Pick a TIFF stack. |
| Convert-to-TIFF checkbox | checkbox | "Convert loaded video to TIFF for faster reloading." |
| Projection type | dropdown | `Mean` / `Max` / `Std Dev` / `Mean + Std`. |
| `Compute Projection` | button | Run the projection. |
| `Save as TIFF` | button | Export the projection. Tooltip: "Save the current calibration preview image at original resolution in .tiff format." |

### B. Segmentation

| Control | Type | Tooltip |
|---|---|---|
| Method | dropdown | `Otsu` / `Cellpose` |
| Min area | spin | "Minimum ROI area as fraction of image (filter tiny noise)." |
| Max area | spin | "Maximum ROI area as fraction of image (filter large blobs)." |
| Blur kernel | spin | "Gaussian blur kernel size (odd number, larger = more smoothing)." |
| Blur sigma | spin | "Gaussian blur sigma (larger = more smoothing)." |
| Fill holes | spin | "Fill holes smaller than this fraction of image area." |
| `Watershed splitting` | checkbox | "Split large merged ROIs using watershed algorithm." |
| Cell diameter (Cellpose) | spin | "Expected cell diameter in pixels (0 = auto-estimate)." |
| Cellpose model | dropdown | `cyto2` / `cyto` / `nuclei` / `custom`. "Cellpose model: cyto2 (default)." |
| Flow error threshold (Cellpose) | spin | "Flow error threshold — lower = stricter segmentation (default 0.5)." |
| Cell probability threshold (Cellpose) | spin | "Cell probability threshold — lower = more permissive (default -1.0)." |
| `Browse` (custom model) | button | Pick a custom Cellpose model file. |
| Frame start | spin | "First frame to include in mean projection (skip calibration frames)." |
| Frame end | spin | "Last frame (0 = all frames)." |
| `GPU acceleration` | checkbox | "Use CuPy/CUDA for faster segmentation (falls back to CPU if unavailable)." |
| `Run Segmentation` | button | Run the chosen method. |

### C. ROI Visualization

| Control | Type | Tooltip |
|---|---|---|
| Overlay opacity | slider | "ROI overlay opacity on mean projection (0.1 = faint, 1.0 = solid)." |

### D. Target Selection

| Control | Type | Action |
|---|---|---|
| Target ROI | dropdown | Choose the ROI of interest for downstream analysis. |

### E. Export

| Control | Type | Action |
|---|---|---|
| `Save ROIs` | button | Write the `rois.npz` to the configured save directory (`STIM_SAVE_DIR`). |

---

## Trace Test dialog

Opened via `Trace Test`. Single panel for live ROI fluorescence testing.

| Control | Type | Notes |
|---|---|---|
| Radius | spin | Per-ROI radius for synthetic test ROIs. |
| `Flip H` | checkbox | Mirror horizontally. |
| `Flip V` | checkbox | Mirror vertically. |
| Rotate | spin | Rotation degrees. |
| `Clear ROI` | button | Reset ROI state. |
| `Close` | button | Dismiss. |

---

## Real-Time Trace Extraction window

Opened via `Real-Time Trace Extraction`. Hosts the live per-ROI plot
grid and the export workflow.

Source: [`gpu_ui.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/gpu_ui.py)
+ [`gpu_ui_mixins/`](https://github.com/Aharoni-Lab/STIMscope/tree/main/STIMscope/STIMViewer_CRISPI/gpu_ui_mixins).

| Control | Type | Action |
|---|---|---|
| `🖼 Select Video…` | button | Pick a TIFF stack for offline trace replay. |
| `➤ Make Memmap` | button | Memory-map a large TIFF for low-memory streaming. |
| `📂 Load ROI File…` | button | Pick a `rois.npz`. |
| `▶ Export Traces` | button | Trigger the comprehensive export (`traces_*.npz` + per-ROI metadata + optional HTML summary). |
| `👁️ View Exported Traces` | button | Open a saved export to inspect. |
| `🌐 Open Full Report in Browser` | button | Render the HTML summary from a saved export. |
| `OASIS (Online)` | checkable button | Toggle online OASIS deconvolution on the live trace stream. |
| Trace-mode dropdown | combo | `Raw` / `ΔF/F₀` / `z-score` / `Spikes` — selects the live plot transform. |
| `◀ Previous 10 ROIs` | button | Pagination back through the per-ROI checkbox list. |
| `Next 10 ROIs ▶` | button | Pagination forward. |
| Per-ROI `ROI {roi_id}` | checkbox | Toggle individual ROI plot visibility. |
| `Close` | button | Dismiss the window. |

`Clear ROI` (commonly assumed to live in this window) is actually in the
**Trace Test dialog**
([`qt_interface_mixins/trace_test.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/qt_interface_mixins/trace_test.py)).

---

## Troubleshooting menu

Opened via the main-bar `Troubleshooting` button.

### Top section

| Control | Action |
|---|---|
| `Test HW Trigger Out Pulse` | One-shot GPIO trigger pulse for scope verification. |
| `Start Engine Monitor` | Live readout of projector engine state (current pattern, GPIO state). |
| `Projector Trigger: OFF` indicator | Read-only status pill (defined disabled in [`troubleshoot.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/qt_interface_mixins/troubleshoot.py) — `setEnabled(False)`). Text + background-color update automatically when the engine asserts per-pattern triggers; not user-clickable. |

### LUT-based diagnostics

| Control | Action |
|---|---|
| `LUT Diagnostics` | Sanity-check the structured-light LUT. |
| `Project Grid (LUT)` | Project a known grid through the LUT. |
| `Capture + Evaluate` | Project + capture + measure pixel error vs. predicted. |
| `Round-Trip Error (Maps)` | Per-pixel round-trip-error heat map. |
| `Pixel Probe (1px)` | Single-pixel projection probe (full diagnostics surface). |
| `Dot Array Test` | Project + capture + localize a dot array. |
| `Round-Trip (Physical)` | Round-trip through the real optical path. |
| `Edge Strip Test` | Test sharp-edge fidelity. |
| `Calib Grid Characterization` | Detailed evaluation of calibration grid coverage. |
| `Save Current View (TIFF)` | Snapshot the troubleshooting view. |

### H-matrix-based variants

| Control | Action |
|---|---|
| `Project Grid (H)` | Project a grid through the 3×3 H matrix (instead of the LUT). |
| `Capture + Evaluate (H)` | Capture + evaluate via H matrix path. |
| `Dot Array Test (H)` | Dot array test via H matrix path. |

### Calibration projector dialog

| Control | Tooltip |
|---|---|
| Grid cell size | "Grid square size in camera pixels" |
| Grid spacing | "Center-to-center spacing of squares; must be >= Cell" |

---

## Conventions

- **Toggle buttons** show the current action in the label —
  `Start Recording` ↔ `Stop Recording`, `Start Hardware Acquisition` ↔
  `Stop Hardware Acquisition`, `Start Projection Engine` ↔
  `Stop Projection Engine`, `Send Masks` ↔ `Stop Sending Masks`,
  `Start Projector Trigger` ↔ `Stop Projector Trigger`.
- **Disabled controls** indicate a missing prerequisite (camera not
  acquiring, engine not started, ROI file not loaded, etc.). Hover for
  the tooltip surfacing the gap.
- **Independence of camera vs mask flips** — flipping the camera
  preview does NOT flip the projection mask, and vice versa. Tooltips
  make this explicit on each control.
- **Tooltips are source-of-truth.** If this page disagrees with the
  in-GUI tooltip, the tooltip wins.
- **The status bar** at the bottom of the main window shows the most
  recent operation result + any non-fatal warnings.
