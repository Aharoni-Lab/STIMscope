# CLAUDE.md — STIMscope / CRISPI

Project guidance for Claude Code (or other AI-tool) sessions on this
repository. Read this first.

## What this is

**STIMscope** is a one-photon benchtop all-optical platform for
centimeter-scale calcium imaging + DMD-patterned optogenetic stimulation
at single-cell resolution. Hardware: TI DLP4710EVM DMD via DLPC3479 (I²C);
Sony IMX334/IMX290 CMOS in an IDS Peak USB3 housing (or any GenICam
camera); Microchip ATSAMD51 MCU; NVIDIA Jetson AGX Orin. **CRISPI** is
the accompanying software stack: a Qt GUI, a C++ projection engine, the
calibration suite, real-time per-ROI trace extraction, hardware
diagnostics. Distributed as a Docker image (JetPack 5 or 6).

User-facing name: **STIMscope**. Software-stack name: **CRISPI**. Both
appear on purpose; do not collapse them.

Reference: Chorsi, Soldado-Magraner, Jin, Soltanalipouryekesammak, Zheng,
Markovic, Geschwind, Golshani, Buonomano, Aharoni (2026), bioRxiv DOI
[10.64898/2026.05.27.728160](https://www.biorxiv.org/content/10.64898/2026.05.27.728160v1).

## Scope of this release

The published preprint describes the inference module — the closed-loop
extension point that would respond to ongoing neural activity — as
"not implemented in the current version" (Discussion). In this release the **inference module is scaffolded but not implemented**. The
scaffolding interfaces are defined under
`STIMscope/STIMViewer_CRISPI/CS/core/`; the inference algorithms
themselves are out of scope for `base-platform`.

What is included: the hardware-synchronized framework the preprint
validates — Qt GUI, C++ projection engine, calibration suite, real-time
trace extraction (RTTE), hardware diagnostics, recording.

## Common commands

```bash
# Build the image (auto-detects JetPack version)./build.sh

# Launch the GUI (everyday operator path)
export DISPLAY=:0 && xhost +local:docker
sudo -E docker-compose up gui

# Run the deterministic demo recorder (tools/demo/ — the May 2026 release demo)
bash scripts/run_demo.sh

# Tests (host-side, no Docker — uses Path(__file__) resolution + Protocol fakes)
pytest -q tests/L1_algorithms/        # pure NumPy, fast
pytest -q tests/L2_orchestration/     # config + dispatch, needs PyQt5
```

CI on GitHub Actions runs L1 + L2 + infra-smoke + bandit + ruff on x86
Linux (see `.github/workflows/ci.yml`). Hardware-dependent test layers
(L3+, L3.5, L5) run on a Jetson via `make test`.

## Code conventions

- Black, 88-char line length.
- Flake8 / ruff: E, F, B, UP, SIM rules. Advisory in CI.
- Bandit medium+ severity is a gate (`make bandit`).
- Hedged documentation language: "current implementation does X", not
  "X is guaranteed."
- Production-code docstrings describe *current* behavior. Do not insert
  internal-process breadcrumbs in source — those belong in commit
  history.
- The canonical architectural reference is `docs/IMPLEMENTATION_NOTES.md`.

## Hardware-aware coding rules

- **GPU is never required.** Every CuPy code path must fall back to
  NumPy cleanly. `--no-gpu` forces CPU on subprocesses that take it.
- **Hardware is never required.** Camera / projector / GPIO each fail
  silently with a warning + no-op fallback when absent. Simulation mode
  must always work.
- **The Python ↔ C++ projector wire is ZMQ.** The default endpoints
  (`DEFAULT_MASK_ENDPOINT = tcp://127.0.0.1:5558` for masks PUSH;
  `DEFAULT_HOMOGRAPHY_ENDPOINT = tcp://127.0.0.1:5560` for H REQ;
  `5562` for status PUB) are defined in
  `STIMscope/STIMViewer_CRISPI/CS/core/projector.py`. Do not change a
  wire constant without updating both Python and C++ sides.
- **The C++ projector engine is built once into the image** from
  `STIMscope/ZMQ_sender_mask/main.cpp`. `make rebuild-projector`
  rebuilds it on the host without a full image rebuild.
- **GPIO chip + lines are env-configurable.** Defaults
  (`/dev/gpiochip1`, line 8 = camera trigger, line 9 = projector
  trigger) come from `STIM_GPIO_CHIP` / `STIM_CAM_LINE` /
  `STIM_PROJ_LINE` — read by `qt_interface_mixins/triggers.py`. Do not
  hardcode chip paths or line numbers in new code — read env, fall back
  to defaults.
- **DLPC3479 illumination control is via I²C opcode 0x96 byte 3**
  (`illum_select`), not via separate GPIO LED pins. The DMD's on-board
  LED bank is gated by the DLPC3479 per-pattern. The GUI's LED-color
  dropdown writes this byte over I²C via
  `STIMscope/ZMQ_sender_mask/dlpc_i2c.py`.

## Portability discipline

- **No hardcoded `/home/<user>` paths anywhere in source.** Host mounts
  go through `${HOME}` substitution in `docker-compose.yml`; inside the
  container they resolve to `/host_home/Desktop`, `/host_home/Videos`,
  `/host_home/Downloads`, plus the user's whole home at `/host_home`.
- **All operator-tunable runtime knobs are env vars prefixed `STIM_`**:
  `STIM_CAMERA_FPS`, `STIM_MAX_GUI_FPS`, `STIM_PIXEL_FORMAT`,
  `STIM_TRIGGER_LINE`, `STIM_PEAK_BUFFERS`, `STIM_RT_DEFAULT`,
  `STIM_ASSETS_DIR`, `STIM_SAVE_DIR`, `STIM_GPIO_CHIP`,
  `STIM_CAM_LINE`, `STIM_PROJ_LINE`, `STIM_DEFAULT_FPS_HZ`,
  `STIM_DEFAULT_EXP_US`, `STIM_RTTE_PROCESS_EVERY_N`,
  `STIM_PROJECTOR_SWAP_INTERVAL`, etc. Full surface in
  `docs/PORTABILITY.md`.
- **IDS Peak SDK path** is `IDS_PEAK_PATH` (default `/opt/ids-peak`);
  the `.deb` is gitignored — see `IDS-PEAK-SDK.md` for the install
  flow.

## Common pitfalls

- `bandit` will complain about anything but medium+ in CI; treat
  low-severity findings as advisory.
- `make test` runs inside the container; pytest at the repo root runs
  on the host. The two have different PYTHONPATH.
- Files written to `data/` are root-owned because the container runs as
  root. Reclaim ownership with
  `sudo chown -R $(id -u):$(id -g) data/`.
- `xhost +local:docker` is needed once per shell session for the GUI
  to reach the X server.

## Git remotes

| Remote | URL |
|---|---|
| `origin` | `git@github.com:Aharoni-Lab/STIMscope.git` |

## Test layer conventions

Directory names matter — they're the layer markers the CI workflow
keys off:

```
tests/L1_algorithms/      pure NumPy
tests/L2_orchestration/   config + dispatch
tests/L3_hardware/        mocked hardware HALs
tests/L3_projector/       mocked I²C / projector
tests/L3_5_split_first/   live-trace mixins
tests/L5_UI/              Qt mixins (offscreen)
```

Keep using these directory names when adding tests.
