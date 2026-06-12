# Portability

STIMscope is designed to move between Jetson hosts and carrier boards
without a rebuild. Every machine-specific value is read from an
environment variable at startup — the source tree carries **no
`/home/*` host paths** (paths resolve from `__file__`).

The full reference, including a fresh-machine sanity checklist and the
list of compile-time assumptions, is at
[`docs/PORTABILITY.md`](https://github.com/Aharoni-Lab/STIMscope/blob/main/docs/PORTABILITY.md).

## Environment-variable surface

Set these via `docker run -e VAR=…` or in your launch script. Defaults
work on a stock Jetson Orin; override only what your host differs on.

### Persistent data

| Var | Default | Purpose |
|---|---|---|
| `STIMSCOPE_HOST_DATA` | `$HOME/stimscope-data` | host directory mounted at `/data` in the container |
| `STIM_SAVE_DIR` | `/data/recordings` | where ROIs / recordings / movie mmaps land |
| `STIM_DATA_ROOT` | `/data` | data root for config + assets |

### Hardware addressing (per Jetson variant / carrier board)

| Var | Default | Purpose |
|---|---|---|
| `STIM_I2C_BUS` | `1` | I²C bus for the DLPC3479 (Jetson Orin = 1) |
| `STIM_GPIO_CHIP` | `/dev/gpiochip1` | GPIO chip for projector trigger I/O |
| `STIM_CAM_LINE` | `8` | GPIO line that receives the camera trigger |
| `STIM_PROJ_LINE` | `9` | GPIO line that drives the projector trigger |

### Behavior tuning

| Var | Default | Purpose |
|---|---|---|
| `STIM_TEMPORAL_PHASE_MS` | `500` | Temporal-mode LED alternation period (ms per color) |
| `STIM_LOG_LEVEL` | `INFO` | structured logger level |

## Storage throughput for sustained recording

Recording at high frame rates is write-bound. The Jetson's onboard
eMMC is fine for short clips, but **sustained high-fps recording can
outrun eMMC write throughput** and stall the recording queue. For long
runs, point `STIMSCOPE_HOST_DATA` at a fast disk — an NVMe SSD or a
USB3 SSD — so `/data/recordings` lands on storage that keeps up with
the camera:

```bash
export STIMSCOPE_HOST_DATA=/mnt/nvme/stimscope-data
```

## See also

- [`docs/PORTABILITY.md`](https://github.com/Aharoni-Lab/STIMscope/blob/main/docs/PORTABILITY.md)
  — full env-var reference, fresh-machine sanity checks, and the
  compile-time assumptions (camera vendor, DMD controller, ARM64).
- [Install](Install) — build + run on a Jetson.
- [Hardware Setup](Hardware-Setup) — SDK install and projector / GPIO
  wiring.
