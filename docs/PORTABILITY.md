# Portability — running the STIMscope Docker image on any machine

This document captures what the image *assumes* about its host and how to
adapt it to a different setup. The source uses `Path(__file__).resolve()`
for all path resolution and has no `/home/*` host-specific paths baked
in. Remaining machine-specific values are all exposed as environment
variables — no rebuild needed to retarget.

## Host requirements

| Requirement | Default | Notes |
|---|---|---|
| Docker | any recent | needs `--privileged` for I²C/GPIO access *or* targeted `--device=` mounts |
| NVIDIA Container Runtime | optional | required for GPU; image falls back to CPU if unregistered |
| X11 | host display | for the GUI; mount `/tmp/.X11-unix`; `xhost +local:docker` |
| IDS Peak SDK | `/opt/ids-peak` (mount RO) | optional — image runs in simulation mode without it |
| Jetson AGX Orin (JP5/JP6) | assumed | base image is `l4t-jetpack:r35.2.1` (JP5); set build-arg for JP6 |

## Configurable environment variables

All set in `~/run_crispi.sh` (or via `docker run -e VAR=…`).

### Persistent data
| Var | Default | Purpose |
|---|---|---|
| `STIMSCOPE_HOST_DATA` | `$HOME/stimscope-data` | host directory mounted at `/data` in the container |
| `STIM_SAVE_DIR` | `/data/recordings` | where ROIs / recordings / movie mmaps land |
| `STIM_DATA_ROOT` | `/data` | core.paths data root (config/, assets/) |

### Hardware addressing (override per Jetson variant / carrier board)
| Var | Default | Purpose |
|---|---|---|
| `STIM_I2C_BUS` | `1` | I²C bus number for the DLPC3479 (Jetson Orin = 1) |
| `STIM_GPIO_CHIP` | `/dev/gpiochip1` | GPIO chip device for projector trigger I/O |
| `STIM_CAM_LINE` | `8` | GPIO line that receives the camera trigger |
| `STIM_PROJ_LINE` | `9` | GPIO line that drives the projector trigger |

### Behavior tuning
| Var | Default | Purpose |
|---|---|---|
| `STIM_TEMPORAL_PHASE_MS` | `500` | Temporal-mode LED alternation period (ms per color) |
| `STIM_LOG_LEVEL` | `INFO` | structured logger level (`core.logging_config`) |

## Sanity-check on a fresh machine

1. **Docker runs**: `docker run --rm hello-world` succeeds.
2. **NVIDIA runtime (optional)**: `docker info | grep nvidia` shows `nvidia` in Runtimes; otherwise the launcher falls back to CPU automatically.
3. **I²C bus is right**: with the DMD connected, `sudo i2cdetect -y $STIM_I2C_BUS` lists address `1b`. If not, the bus number is wrong for this host — set `STIM_I2C_BUS` accordingly.
4. **GPIO chip is right**: `gpiodetect` lists the projector chip; lines for cam-trigger and proj-trigger correspond to the wiring.
5. **IDS Peak SDK (optional)**: `ls /opt/ids-peak` shows the install tree; otherwise the GUI starts in camera-absent mode.
6. **Display**: `echo $DISPLAY` is non-empty and you're on the graphical session; `xhost +local:docker` granted.
7. **Launch**: `~/run_crispi.sh`. Look for the launcher banner — it prints
   the chosen runtime, mount, and any missing prerequisites.

## What's *not* configurable (compile-time assumptions)

- **Camera vendor**: IDS Peak SDK. Other cameras need different driver code.
- **Projector hardware**: TI DLP4710 EVM with DLPC3479 controller and the I²C protocol implemented in `ZMQ_sender_mask/dlpc_i2c.py`. Other DLPC variants would need a different driver.
- **Architecture**: image targets ARM64 (Jetson). Cross-arch needs a rebuild.

## Verifying portability on a new machine

- Run the launcher on a *second* Jetson (or VM with the right deps), confirm
  the launcher banner reads sensible mounts/runtime and the GUI starts.
- If the second machine is a different carrier board, set `STIM_I2C_BUS` /
  `STIM_GPIO_CHIP` accordingly and confirm projector + camera operate.
- Re-run any features that touch host paths (recording → `/data/recordings`,
  ROI save → `/data/recordings/rois.npz`, calibration uses the bundled
  `Assets/calibration_board.png` ChArUco board).

If anything still looks host-specific, you can re-verify with:

```bash
grep -rnE "/home/[a-z]+jetson|/home/jetson4|/home/aharonilab|/Users/" \
  --include="*.py" --include="*.pyw" --include="*.sh" STIMscope/ scripts/
```

Result should be empty.
