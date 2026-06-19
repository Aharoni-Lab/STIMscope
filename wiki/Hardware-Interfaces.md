# Hardware Interfaces

![Fig 1b — Hardware architecture (image sensor, DMD, microcontroller, Jetson)](../docs/figures/fig01b_hardware_architecture.png)
*Fig 1b — The protocol surfaces this page documents, top to
bottom: image sensor → host over USB / MIPI-CSI; host ↔ MCU over UART;
host → DMD over HDMI (pattern stream) and I²C (control); MCU → DMD +
camera over Trig-Out 1 / 2 (synchronization).*

This page documents the **protocol layer** between the software and
the hardware: how Python talks to the camera, how Python and the C++
projector engine exchange data over ZMQ, how the DMD controller is
addressed over I²C, and how GPIO lines tie acquisition + stimulus
together. For physical wiring + SDK install, see
[Hardware Setup](Hardware-Setup).

This page intentionally avoids restating numeric constants. Pin
assignments, ZMQ endpoints, GenICam defaults, I²C opcodes, and
trigger timings live in source — restating them here invites drift.
Each section below points at the file (and where useful, the symbol)
that owns the value.

---

## Camera ↔ Python (IDS Peak SDK)

The Qt GUI wraps the IDS Peak SDK in
[`STIMscope/STIMViewer_CRISPI/camera.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/camera.py)
— `OptimizedCamera(QObject)`, emits `frame_ready` /
`recordingStarted` Qt signals.

GenICam node defaults (pixel format, frame rate, GUI FPS cap, buffer
count, trigger line, RT mode default, default fps + exposure on open)
are read from environment variables at construction time. Variable
names and defaults are defined at the top of `camera.py` — read the
source for the current values; [Portability](Portability) lists the
full env-var surface.

### Hardware trigger handshake

When trigger mode is on, the GenICam node map is configured:

```python
node_map.FindNode("TriggerMode").SetCurrentEntry("On")
node_map.FindNode("TriggerSource").SetCurrentEntry("Line0")
```

The camera waits for an edge on its physical trigger input. The
projector engine drives that edge from the camera-trigger GPIO line.
Each tick → one acquired frame.

### Frame queue model

`OptimizedCamera` owns a bounded acquisition buffer that the IDS SDK
fills, then dispatches frames to GUI consumers via a Qt signal + to
recording / live-trace via a separate sink. Buffer depth is the
trade-off between dropped frames under load and end-to-end latency;
the current default lives in `camera.py`.

---

## Projector ↔ Python ↔ C++ (ZMQ)

The DMD is driven by a custom C++ engine at
[`STIMscope/ZMQ_sender_mask/main.cpp`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/ZMQ_sender_mask/main.cpp)
that owns the OpenGL → DMD pipeline, GPIO lines, and DLPC3479 I²C
control. Python clients talk to it over **three ZMQ sockets** on
localhost.

| Pattern | Direction | Purpose |
|---|---|---|
| PUSH (Python) ↔ PULL (engine) | Python → engine | Per-frame mask data |
| REQ (Python) ↔ REP (engine) | Python ↔ engine | Homography updates (one-shot per calibration) |
| PUB (engine) ↔ SUB (Python) | engine → Python | Projector status (per-pattern `pidx` / `vis_id`), used to pace patterns |

Default endpoints are defined in
[`STIMscope/STIMViewer_CRISPI/CS/core/projector.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/CS/core/projector.py)
(`DEFAULT_MASK_ENDPOINT`, `DEFAULT_HOMOGRAPHY_ENDPOINT`, plus the
status-publisher endpoint used by the engine monitor).

### Mask frame wire format (PUSH socket)

Multipart ZMQ message, 2 parts (per
[`core/projector.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/CS/core/projector.py),
`send_mask` / `send_mask_rgb`):

```
part 1: JSON-encoded metadata dict (UTF-8 bytes)
part 2: raw mask bytes — shape (H, W) for grayscale, (H, W, 3) for color, dtype=uint8
```

The current metadata keys live in the `send_mask` / `send_mask_rgb`
implementations — read the source so this page doesn't drift if a key
is added. Frame shape is the DMD's native resolution (defined in
`main.cpp`). Channel ordering and color modes are handled by the
Python side (`send_mask` for grayscale, `send_mask_rgb` for color).
The engine does not validate — sending the wrong shape produces
undefined behavior on the DMD.

LINGER on the PUSH socket is **0** by design: the engine treats
mid-flight masks as best-effort, so client `close()` should not
block waiting to drain. If a frame is in flight when the trial
loop ends, it is dropped.

### Homography sideband (REQ/REP)

One-shot per calibration. Python sends the 3×3 homography matrix
(camera → projector) as a small binary message; the engine
acknowledges and recomputes its internal warp LUT. After a successful
reply, the engine applies the new H to every subsequent mask frame
received on the PUSH socket. Timeouts (LINGER, RCVTIMEO) are set on
the client side in `core/projector.py`.

If the engine is not running when calibrate fires, the REQ times out
and the calibration step records a "no engine" warning. This is
normal during offline / pre-launch flows — calibration is run before
the projector engine is started; the resulting homography is mediated
to the experiment phase via disk
([`Assets/Generated/homography_cam2proj.npy`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/calibration.py)).

### Status publisher (PUB socket)

The engine publishes a small status frame every time it presents a
new pattern (typically `pidx` + `vis_id`). Python clients SUBSCRIBE
to pace tightly-coupled workflows (e.g. live-trace ROI alignment
following the actual on-screen pattern, rather than the requested
one).

### Engine command-line flags

The projector engine binary exposes flags to override its
compiled-in endpoint and gpiochip defaults. The current flag list
lives in the argument parser at the top of `main.cpp` — read the
source for the exact spelling and defaults.

---

## DMD ↔ I²C (DLPC3479)

The DLP4710 DMD is configured through a DLPC3479 controller IC over
I²C. Wire-protocol details come from the **TI DLPU081A** datasheet.
The Python driver is at
[`STIMscope/ZMQ_sender_mask/dlpc_i2c.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/ZMQ_sender_mask/dlpc_i2c.py).

The driver does not implement every opcode in the datasheet — only
the subset the platform needs. Rather than re-state the opcode set
(which silently drifts when the driver adds or drops one), treat the
Python file as the authoritative list:

- Bus address constants are defined at the top of `dlpc_i2c.py`. The
  I²C bus number is env-overridable via `STIM_I2C_BUS` (see
  [Portability](Portability)).
- Each opcode has a dedicated wrapper function whose docstring cites
  the relevant DLPU081A section.
- The driver treats a non-zero error bit in the controller's
  Communication Status response as a hard failure (raises
  `DLPCError`) — silent failures on the bus are not tolerated.

### Illumination Select (opcode 0x96)

LED channel selection on this platform is **DMD-internal**: the
DLPC3479 selects which on-board LED bank illuminates each sub-frame
via opcode `0x96` byte 3 (Illumination Select). The operator-facing
surface is the `LED Color` dropdown on the main button bar; items +
raw bytes are defined in
[`qt_interface_mixins/button_bar.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/qt_interface_mixins/button_bar.py)
(`_led_color_dropdown`). There are no separate RED/BLUE GPIO lines
on the host side.

For temporal alternation between RED (stim) and BLUE (observe) during
a run, a daemon thread in
[`qt_interface_mixins/triggers.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/qt_interface_mixins/triggers.py)
(`_start_temporal_alt_thread`) repeatedly calls
`dlpc_i2c.fast_phase_switch` so the visible LED tracks the mask-side
alternation. Phase duration is tunable via `STIM_TEMPORAL_PHASE_MS`.

### Documented quirks vs. the datasheet

Several behaviors deviate from the DLPU081A documentation. Each quirk
is folded into the wrapper that hits it; comments in `dlpc_i2c.py`
explain the empirical evidence. Read the source for the current list
— the previous static enumeration on this page drifted from the
driver multiple times before being removed.

---

## GPIO (libgpiod)

GPIO is used for the camera and downstream-sync trigger lines —
**not** for LED control (LED routing is DMD-internal, see above).

Line assignments and gpiochip selection are env-overridable so the
same image runs on different Jetson carrier boards without
recompilation:

| Env var | Purpose |
|---|---|
| `STIM_GPIO_CHIP` | Which gpiochip device |
| `STIM_CAM_LINE` | Line that fires the camera trigger |
| `STIM_PROJ_LINE` | Line that drives the projector trigger out |

Defaults are defined where the engine subprocess is launched —
[`qt_interface_mixins/triggers.py`](https://github.com/Aharoni-Lab/STIMscope/blob/main/STIMscope/STIMViewer_CRISPI/qt_interface_mixins/triggers.py).
The argument parser at the top of `ZMQ_sender_mask/main.cpp` accepts
the matching flags. See [Portability](Portability) for the full
env-var surface.

The camera trigger output is wired into the GenICam input line
configured by `TriggerSource` (default `Line0`).

### Line-request lifecycle

Each GPIO line is requested with `libgpiod` at engine start, held for
the engine's lifetime, and released on shutdown. Re-requesting a line
already held by another process raises an error — if the engine
crashed without releasing, restart with `make fresh` (which brings
the container fully down and back up) to clear stale holders.

---

## When to update this page

Anything here that *describes the wire format* is part of the public
interface between Python and the engine. Changing it requires
coordinated changes to both sides + a wiki edit. If you catch a
drift, file a doc-only PR — it's the cheapest fix.

Internal implementation details (which thread holds the lock, which
queue depth is optimal) belong in
[`docs/IMPLEMENTATION_NOTES.md`](https://github.com/Aharoni-Lab/STIMscope/blob/main/docs/IMPLEMENTATION_NOTES.md),
not this page.
