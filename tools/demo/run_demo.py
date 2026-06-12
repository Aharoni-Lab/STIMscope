#!/usr/bin/env python3
"""Headless DMD demo recorder for the STIMscope base platform.

Drives the full DMD hardware envelope and records it end-to-end.
One command boots the DMD, launches the projector engine,
sends the calibration homography to the engine, starts the slave-triggered IDS
camera, plays a deterministic mask sequence (rapid red<->blue alternation,
simultaneous RGB mixes, varied shapes at varied intervals, many varying-size
ROIs, a density/scale ramp), then tears everything down cleanly.

Key behaviors (all code-verified against the engine, DMD, calibration, and
camera subsystems):

COLOR MODEL — rgb-cycle Mode B (8-bit RGB, R+B gated): SIMULTANEOUS red+blue.
We boot ONCE with `--rgb-cycle` (seq_type=0x03, illum=0x05) and choose color per
mask purely by WHICH RGB CHANNEL carries the grayscale mask:
    R channel only  -> red       B channel only  -> blue
    R and B both    -> red shape AND blue shape in ONE frame (simultaneous)
The DMD's 8-bit RGB sub-frame engine lights R-channel content with the red LED
and B-channel content with the blue LED within one HDMI frame. NO per-mask I²C
is needed (color is in the frame we push), so TRIG_OUT stays continuous — no
live LED switching, no trigger jitter. Boot timing is left at the proven values;
the `sequence_abort` it reports is cosmetic (bench-proven 31 fps with it present).

CALIBRATION — deterministic, GUI-identical. We send the forward H_cam2proj to the
projector engine on its REP endpoint (5560), exactly like the live GUI
(camera.py:_send_h_to_projector). The engine then displays
horizontal_flip(warpPerspective(mask, H_cam2proj, 1920x1080)) for every mask we
push raw, so the camera SEES the intended mask (RAW MASK <-> CAMERA aligned).
Masks are sent RAW (1920x1080); the engine does the warp. No Python pre-warp.

CAPTURE PHASE — the camera is slave-triggered at 30 Hz off the DMD TRIG_OUT.
Whether each exposure lands on the intended illumination sub-frame is controlled
by --exposure-us and --trig-delay-us (camera-side TriggerDelay). The exact values
are rig-specific and must be tuned on the bench (see docs §10.4).

Output bundle (under --out-dir):
    demo_frames.csv          — camera_meta rows (one per captured frame).
    demo_masklog.csv         — projection_send rows (name/led/sha256/frame_id/ts).
    tiff_frames/             — lossless per-frame camera TIFFs (LZW).
    demo_camera.mp4          — review track.
    homography_cam2proj.npy  — the exact H sent to the engine (composer reproduces
                               PROJECTION = flip(warp(mask, H)) from it).
    projector.log            — engine [PROJ]/[CAM] per-trigger log (sync backbone).
    metadata.json            — git sha, timing config, h_sent flag, host info.

Run (camera needs the host IDS SDK mounted, like the GUI) — use scripts/run_demo.sh
or `make demo`. Dry run (no hardware): `run_demo.py --dry-run --out-dir /tmp/dry`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
# Make the demo helpers + platform modules importable. Prefer the baked /app
# locations (image) and fall back to the repo paths (live mount).
for _p in (
    str(_HERE),
    "/app/STIMViewer_CRISPI",
    "/app/ZMQ_sender_mask",
    str(_REPO_ROOT / "STIMscope" / "STIMViewer_CRISPI"),
    str(_REPO_ROOT / "STIMscope" / "ZMQ_sender_mask"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mask_library as ml          # noqa: E402
from logger import DemoLogger      # noqa: E402

PROJ_W, PROJ_H = ml.PROJ_W, ml.PROJ_H
DMD_BUS = int(os.environ.get("STIM_I2C_BUS", "1"))
DMD_ADDR = 0x1B
PROJ_ENDPOINT = os.environ.get("PROJECTOR_BIND", "tcp://127.0.0.1:5558")
# Engine REP endpoint for the 3x3 homography (main.cpp ZMQ_H_BIND default).
PROJ_H_ENDPOINT = os.environ.get("PROJECTOR_H_BIND", "tcp://127.0.0.1:5560")

_PROCS: list = []  # track child processes for cleanup


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────

def _find_projector_bin() -> str | None:
    for c in ("/app/ZMQ_sender_mask/projector",
              str(_REPO_ROOT / "STIMscope" / "ZMQ_sender_mask" / "projector")):
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _find_i2c_cli() -> str | None:
    for c in ("/app/ZMQ_sender_mask/i2c_test_send_commands.py",
              str(_REPO_ROOT / "STIMscope" / "ZMQ_sender_mask" / "i2c_test_send_commands.py")):
        if os.path.isfile(c):
            return c
    return None


def _find_camera_recorder() -> str:
    return str(_HERE / "camera_recorder.py")


def _find_homography(cli: str | None = None) -> Path | None:
    cands = [Path(cli)] if cli else []
    cands += [_REPO_ROOT / "STIMscope" / "STIMViewer_CRISPI" / "Assets"
              / "Generated" / "homography_cam2proj.npy",
              Path("/app/STIMViewer_CRISPI/Assets/Generated/homography_cam2proj.npy")]
    for c in cands:
        if c and c.exists():
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Channel packing — color is chosen by which RGB channel carries the mask
# ─────────────────────────────────────────────────────────────────────────────

def _pack(red_gray: np.ndarray | None = None,
          blue_gray: np.ndarray | None = None) -> np.ndarray:
    """Pack grayscale mask(s) into an RGB frame: R-channel=red_gray,
    B-channel=blue_gray. Either may be None (that color stays dark)."""
    rgb = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
    if red_gray is not None:
        rgb[..., 0] = red_gray
    if blue_gray is not None:
        rgb[..., 2] = blue_gray
    return rgb


def _pack_for_led(mask: "ml.DemoMask") -> np.ndarray:
    return _pack(red_gray=mask.img) if mask.led.upper() == "R" else _pack(blue_gray=mask.img)


def _color_assign(ids, mode: str, seed: int = 0) -> dict:
    """Assign each ROI id a color: 'R', 'B', or 'RB' (both)."""
    if mode == "all_R":
        return {i: "R" for i in ids}
    if mode == "all_B":
        return {i: "B" for i in ids}
    if mode == "alt":
        return {i: ("R" if k % 2 == 0 else "B") for k, i in enumerate(ids)}
    # "split": deterministic random R/B (with a few RB for visual mix)
    rng = np.random.default_rng(seed)
    out = {}
    for i in ids:
        r = rng.random()
        out[i] = "RB" if r < 0.15 else ("R" if r < 0.575 else "B")
    return out


def _roi_field_frame(labels: np.ndarray, color_of: dict) -> np.ndarray:
    """Build one RGB frame from a label image + per-ROI color assignment:
    red ROIs -> R channel, blue ROIs -> B channel, 'RB' -> both.

    Vectorized via a per-label color LUT so it stays O(H*W) regardless of ROI
    count — essential for the dense pixel-level fields (thousands of ROIs)."""
    maxid = int(labels.max()) if labels.size else 0
    code = np.zeros(maxid + 1, dtype=np.uint8)   # bit0 = R, bit1 = B
    for nid, col in color_of.items():
        if 0 <= nid <= maxid:
            code[nid] = (1 if "R" in col else 0) | (2 if "B" in col else 0)
    per_px = code[labels]                         # (H,W) color codes
    red = np.where((per_px & 1) > 0, 255, 0).astype(np.uint8)
    blue = np.where((per_px & 2) > 0, 255, 0).astype(np.uint8)
    return _pack(red_gray=red, blue_gray=blue)


# ─────────────────────────────────────────────────────────────────────────────
# Hardware bring-up / teardown
# ─────────────────────────────────────────────────────────────────────────────

def _run_i2c(cli: str, *cli_args: str, out_dir: Path, tag: str) -> int:
    logf = out_dir / f"i2c_{tag}.log"
    with open(logf, "wb") as fh:
        return subprocess.run(["/usr/bin/python3", cli, *cli_args],
                              cwd=str(Path(cli).parent), stdout=fh, stderr=subprocess.STDOUT,
                              timeout=30).returncode


def boot_dmd(out_dir: Path) -> None:
    """Clean Standby then boot rgb-cycle (proven 30 fps boot). Mirrors the GUI's
    Start-Projector-Trigger path; the force-standby first avoids lingering DMD
    state (see project_dmd_lingering_state_root_cause). Boot timing is left at the
    proven values — the sequence_abort it reports is cosmetic for FPS (bench-
    proven), and the boot-timing 'auto-fit' that tried to clear it broke
    triggering (reverted a6b4e77->92bd337). Do NOT re-add timing auto-fit here."""
    cli = _find_i2c_cli()
    if cli is None:
        raise SystemExit("[demo] i2c_test_send_commands.py not found")
    print("[demo] DMD: force Standby (clean state)…")
    _run_i2c(cli, "stop", out_dir=out_dir, tag="stop")
    time.sleep(0.5)
    print("[demo] DMD: boot --rgb-cycle (8-bit RGB, R+B gated; simultaneous R+B)…")
    rc = _run_i2c(cli, "boot", "--rgb-cycle", out_dir=out_dir, tag="boot")
    if rc != 0:
        print(f"[demo] WARNING: DMD boot returned {rc} (see {out_dir}/i2c_boot.log)")
    time.sleep(1.5)  # settle


def standby_dmd(out_dir: Path) -> None:
    cli = _find_i2c_cli()
    if cli is not None:
        try:
            _run_i2c(cli, "stop", out_dir=out_dir, tag="stop_final")
        except Exception as e:
            print(f"[demo] standby failed (continuing): {e}")


def launch_projector(out_dir: Path, swap_interval: int = 0) -> subprocess.Popen:
    """Launch the C++ projector engine to the second monitor. Flags match the
    GUI's working launch line (and the calibration capture conditions, so the
    saved H_cam2proj stays valid). --horiz-flip=1 is required: the engine applies
    it after the H-warp, matching how the GUI/Calibrate path projects.

    swap_interval: 0 = vsync OFF (low-latency, but mask updates can be presented
    mid-refresh → the DMD latches a half-updated frame → tearing, which the
    slave-triggered camera then CAPTURES). 1 = vsync ON: each presented frame is
    complete (no tearing) at the cost of pacing swaps to the 60 Hz refresh. The
    engine draws in ~2 ms (well under 16.67 ms), so vsync should keep up; the DMD
    still triggers off its own 60 Hz HDMI refresh, so the camera trigger lock is
    independent of the engine swap. Use --swap-interval 1 if tearing shows in the
    capture."""
    bin_path = _find_projector_bin()
    if bin_path is None:
        raise SystemExit("[demo] projector binary not found (build the image)")
    args = [bin_path,
            f"--bind={PROJ_ENDPOINT}",
            f"--h-bind={PROJ_H_ENDPOINT}",
            f"--map-csv={out_dir / 'mask_map.csv'}",
            f"--swap-interval={int(swap_interval)}", "--visible-id=0",
            "--cam-chip=/dev/gpiochip1", "--cam-line=8", "--cam-edge=rising",
            "--proj-chip=/dev/gpiochip1", "--proj-line=9", "--proj-edge=rising",
            "--horiz-flip=1", "--force-immediate=1"]
    logf = open(out_dir / "projector.log", "wb")
    print(f"[demo] launching projector engine: {bin_path}")
    proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT)
    _PROCS.append(proc)
    time.sleep(3.0)  # let it bind ZMQ + claim the monitor
    if proc.poll() is not None:
        raise SystemExit(f"[demo] projector engine died at startup (see {out_dir}/projector.log)")
    return proc


def send_homography(out_dir: Path, cli_path: str | None = None) -> bool:
    """Send the forward H_cam2proj to the engine's REP endpoint (5560), exactly
    like the live GUI (camera.py -> core.projector._send_homography_inline).

    Wire format (verified main.cpp:1373-1387): multipart [b"H", H_row_major_f64]
    where the payload is exactly 9*8=72 bytes; the engine replies b"OK". The
    engine then displays flip(warpPerspective(mask, H_cam2proj, 1920x1080)) for
    every raw mask we push, so the camera sees the intended mask. We also copy H
    into the bundle so the composer reproduces the PROJECTION panel exactly.

    Returns True on send+ACK. Non-fatal on failure (the demo still records, but
    the camera view will be uncalibrated — flagged loudly + in metadata)."""
    hp = _find_homography(cli_path)
    if hp is None:
        print("[demo] *** WARNING: homography_cam2proj.npy not found — projecting "
              "UNCALIBRATED. Run Calibrate (or pass --homography). Camera will NOT "
              "align with the masks. ***")
        return False
    H = np.load(str(hp)).astype(np.float64)
    if H.shape != (3, 3):
        print(f"[demo] *** WARNING: homography {hp} is not 3x3 ({H.shape}); "
              "projecting UNCALIBRATED. ***")
        return False
    payload = np.ascontiguousarray(H, dtype=np.float64).tobytes()  # 72 bytes, row-major
    ok = False
    try:
        import zmq
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.setsockopt(zmq.SNDTIMEO, 3000)
        sock.connect(PROJ_H_ENDPOINT)
        try:
            sock.send_multipart([b"H", payload])
            reply = sock.recv()
            ok = (reply == b"OK")
            print(f"[demo] homography -> engine ({hp.name}, {PROJ_H_ENDPOINT}): "
                  f"reply={reply!r} -> {'OK' if ok else 'NOT OK'}")
        finally:
            sock.close(0)
    except Exception as e:
        print(f"[demo] *** WARNING: failed to send homography to engine ({e}); "
              "projecting UNCALIBRATED. ***")
        ok = False
    if ok:
        # Save the exact matrix into the bundle ONLY after a confirmed ACK, so a
        # failed send never leaves a bundle H that misrepresents the run (the
        # composer treats "bundle H present" as "this run was calibrated").
        np.save(str(out_dir / "homography_cam2proj.npy"), H)
    else:
        print("[demo] *** Camera view will NOT be calibrated for this run. ***")
    return ok


def launch_camera(out_dir: Path, fps: int, exposure_us: float, trigger_wait: float,
                  trig_delay_us: float, gain: float) -> subprocess.Popen:
    rec = _find_camera_recorder()
    env = dict(os.environ)
    env["CAMERA_EXPOSURE_US"] = str(exposure_us)
    env["STIM_TRIG_DELAY_US"] = str(trig_delay_us)
    env["STIM_GAIN"] = str(gain)
    # ── Zero-drop capture envelope (docs §10.5) ─────────────────────────────
    # Uncompressed TIFF at 1936x1096x2x30 ≈ 127 MB/s exceeds the eMMC's ~80 MB/s
    # sustained write → the write queue backs up and frames drop → the tail-
    # offset sync desyncs. LZW is LOSSLESS and demo frames are sparse, so they
    # compress ~5-50× → ~10-25 MB/s, under the disk rate. A deep buffer pool +
    # write queue absorb transient stalls so every trigger is captured.
    env.setdefault("STIM_TIFF_COMPRESSION", "lzw")
    env.setdefault("STIM_PEAK_BUFFERS", "96")
    env.setdefault("STIM_WRITE_QUEUE", "360")
    # Skip the per-frame software mp4 encode (the heaviest writer op): on long
    # runs it pushes the writer over the 33 ms budget → write-queue overflow →
    # drops + SDK starvation + trigger-interval jitter. The lossless TIFFs are
    # the output and the composer regenerates a review mp4. Set STIM_DISABLE_MP4=0
    # to keep the raw camera mp4.
    env.setdefault("STIM_DISABLE_MP4", "1")
    args = ["/usr/bin/python3", rec,
            "--out", str(out_dir / "demo_camera.mp4"),
            "--log", str(out_dir / "demo_frames.csv"),
            "--fps", str(fps),
            "--trigger-wait-sec", str(trigger_wait)]
    logf = open(out_dir / "camera.log", "wb")
    print(f"[demo] launching camera recorder (slave/HW-trigger, exposure="
          f"{exposure_us}us, trig-delay={trig_delay_us}us, LZW TIFF, "
          f"buffers={env['STIM_PEAK_BUFFERS']})…")
    proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, env=env)
    _PROCS.append(proc)
    time.sleep(2.0)
    if proc.poll() is not None:
        raise SystemExit(f"[demo] camera recorder died at startup (see {out_dir}/camera.log)")
    return proc


def _cleanup():
    # Stop camera before projector (reversed append order) so the camera drains
    # its write queue and finalizes TIFFs/mp4/CSV before we verify.
    for proc in reversed(_PROCS):
        try:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        except Exception:
            pass
    for proc in reversed(_PROCS):
        try:
            proc.wait(timeout=35)   # let the camera finalize before verify reads
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Mask sequence — the demo program (raw masks; the engine applies H + flip)
# ─────────────────────────────────────────────────────────────────────────────

def _send(client, logger, rgb: np.ndarray, name: str, color: str, sha: str,
          frame_id: int, intent: str, dry: bool) -> None:
    # Log the sha of the ACTUAL packed-RGB frame sent (not the caller's grayscale
    # sha), so the composer can verify its deterministically-regenerated mask
    # matches what was projected (catches record-vs-compose code drift).
    sha = ml._sha256(rgb)
    if not dry and client is not None:
        client.send_rgb(rgb, frame_id=frame_id, immediate=True)
    logger.projection_send(mask_name=name, mask_color=color, mask_sha256=sha,
                           frame_id=frame_id, extra=intent)


def _seq_full(client, logger, dry: bool, scale: float) -> None:
    """The 'full' demo program (shapes + alternation + ROI field + dynamic).
    `scale` multiplies hold times."""
    fid = 2000

    def hold(sec):
        if not dry:
            time.sleep(sec * scale)

    # 1) Baseline — black (DMD active so slave triggers keep firing)
    logger.segment_start("01_baseline", "black mask, DMD active")
    _send(client, logger, _pack(), "baseline_black", "OFF", "0" * 64, 1000,
          "Baseline — black", dry); fid += 1
    hold(2.0); logger.segment_end("01_baseline")

    # 2) Rapid red<->blue alternation (same shape) — fast then slow
    logger.segment_start("02_rb_alternation", "rapid red<->blue, then slow")
    circle = [m for m in ml.arbitrary_shapes() if "circle" in m.name][0].img
    for rate_label, period, n in (("fast", 0.10, 20), ("slow", 0.50, 8)):
        for i in range(n):
            red = i % 2 == 0
            rgb = _pack(red_gray=circle) if red else _pack(blue_gray=circle)
            _send(client, logger, rgb, f"alt_{rate_label}_{i:02d}",
                  "R" if red else "B", ml._sha256(rgb), fid,
                  f"R/B alternation {rate_label} {i+1}/{n}", dry); fid += 1
            hold(period)
    logger.segment_end("02_rb_alternation")

    # 3) Shape sweep (varied shapes, alternating color)
    logger.segment_start("03_shapes", "varied shapes")
    for m in ml.arbitrary_shapes():
        _send(client, logger, _pack_for_led(m), m.name, m.led, m.sha256, fid,
              m.intent, dry); fid += 1
        hold(0.8)
    logger.segment_end("03_shapes")

    # 4) RGB MIX — red shape + blue shape SIMULTANEOUSLY in ONE frame
    logger.segment_start("04_rgb_mix", "red + blue shapes in one frame")
    shapes = {m.name.split("_")[1]: m.img for m in ml.arbitrary_shapes() if m.led == "R"}
    mix_pairs = [("circle", "square"), ("triangle", "star"), ("hexagon", "irregular")]
    for rname, bname in mix_pairs:
        rgb = _pack(red_gray=shapes[rname], blue_gray=shapes[bname])
        _send(client, logger, rgb, f"mix_R-{rname}_B-{bname}", "RB",
              ml._sha256(rgb), fid, f"RGB mix: {rname} (red) + {bname} (blue)", dry); fid += 1
        hold(1.5)
    logger.segment_end("04_rgb_mix")

    # 5) ROI FIELD — many ROIs of varying sizes across the full FOV, HELD.
    #    Shows all-red, all-blue, alternating, and random R/B mixes; each frame
    #    is held a few seconds (dwell), not rapidly switched.
    logger.segment_start("05_roi_field", "many varying-size ROIs, mixed R/B, held")
    labels = ml.synthetic_roi_labels(n=40, seed=7, min_size=12, max_size=70)
    ids = sorted({int(i) for i in np.unique(labels) if i > 0})
    field_plan = [
        ("all_R", 0,  "all RED",          3.0),
        ("all_B", 0,  "all BLUE",         3.0),
        ("alt",   0,  "alternating R/B",  4.0),
        ("split", 11, "random R/B mix A", 3.5),
        ("split", 29, "random R/B mix B", 3.5),
    ]
    for mode, sd, label, dwell in field_plan:
        rgb = _roi_field_frame(labels, _color_assign(ids, mode, seed=sd))
        _send(client, logger, rgb, f"field_{mode}_{sd}", "RB", ml._sha256(rgb), fid,
              f"ROI field ({len(ids)} ROIs, varying sizes) — {label}", dry); fid += 1
        hold(dwell)
    logger.segment_end("05_roi_field")

    # 5b) DWELL — hold one rich composition for a long, steady projection.
    logger.segment_start("05b_dwell", "single mask held ~6 s (steady projection)")
    dwell_rgb = _roi_field_frame(labels, _color_assign(ids, "split", seed=3))
    _send(client, logger, dwell_rgb, "dwell_field", "RB", ml._sha256(dwell_rgb), fid,
          "Steady hold — mixed ROI field, ~6 s", dry); fid += 1
    hold(6.0)
    logger.segment_end("05b_dwell")

    # 6) Varied shapes/intervals — spiral (fast) + rings (medium)
    logger.segment_start("06_dynamic", "spiral + rings, varied intervals")
    for m in ml.spiral_sweep(n_steps=30):
        _send(client, logger, _pack_for_led(m), m.name, m.led, m.sha256, fid,
              m.intent, dry); fid += 1
        hold(0.12)
    for m in ml.concentric_rings(n_steps=12):
        _send(client, logger, _pack_for_led(m), m.name, m.led, m.sha256, fid,
              m.intent, dry); fid += 1
        hold(0.4)
    logger.segment_end("06_dynamic")


def _seq_density(client, logger, dry: bool, scale: float) -> None:
    """Density & scale ramp: hundreds of pixel-level ROIs (grid + scatter, in
    red / blue / mixes) then INCREMENTALLY LARGER ROI groups, up to a cap.
    Each tier is shown all-red, all-blue, alternating, and a random R/B mix."""
    fid = 7000

    def hold(sec):
        if not dry:
            time.sleep(sec * scale)

    # (dot_size_px, spacing_px, arrangement, shape, max_dots, label)
    tiers = [
        (1,   16, "grid",    "square", 1500, "~pixel dots, dense grid"),
        (2,   22, "scatter", "square", 600,  "hundreds of tiny scattered dots"),
        (4,   30, "grid",    "circle", 1200, "small groups, grid"),
        (8,   44, "scatter", "circle", 500,  "small-medium groups, scatter"),
        (16,  70, "grid",    "circle", 600,  "medium groups, grid"),
        (28, 110, "scatter", "circle", 250,  "large groups, scatter"),
        (44, 170, "grid",    "circle", 200,  "largest groups, grid (cap)"),
    ]
    for ds, sp, arr, shp, cap, label in tiers:
        L = ml.dot_field_labels(dot_size_px=ds, spacing_px=sp, arrangement=arr,
                                shape=shp, seed=5, max_dots=cap)
        ids = sorted({int(i) for i in np.unique(L) if i > 0})
        logger.segment_start(f"D_{ds:02d}px_{arr}", f"{label} ({len(ids)} ROIs)")
        for mode, dwell in (("all_R", 1.2), ("all_B", 1.2),
                            ("alt", 1.5), ("split", 1.5)):
            rgb = _roi_field_frame(L, _color_assign(ids, mode, seed=7))
            _send(client, logger, rgb, f"ramp_{ds:02d}px_{arr}_{mode}", "RB",
                  ml._sha256(rgb), fid,
                  f"Density ramp: {label}, {len(ids)} ROIs — {mode}", dry); fid += 1
            hold(dwell)
        logger.segment_end(f"D_{ds:02d}px_{arr}")


def run_sequence(client, logger, dry: bool, scale: float, which: str = "full") -> None:
    """Dispatch the requested sequence: 'full', 'density', or 'all'."""
    if which in ("full", "all"):
        _seq_full(client, logger, dry, scale)
    if which in ("density", "all"):
        _seq_density(client, logger, dry, scale)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true",
                   help="No hardware: build masks + write the mask log only.")
    p.add_argument("--no-camera", action="store_true",
                   help="Skip the camera recorder (projection only).")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the post-run verify.py sync/accuracy report.")
    p.add_argument("--no-warp", action="store_true",
                   help="Do NOT send the calibration homography to the engine "
                        "(project raw/uncalibrated; camera will not align).")
    p.add_argument("--homography", default=None,
                   help="Path to homography_cam2proj.npy (default: Assets/Generated).")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--exposure-us", type=float,
                   default=float(os.environ.get("STIM_HW_EXP_US", "15000")),
                   help="Camera exposure µs (float; the IDS ExposureTime node is "
                        "float-valued). Must fit one 33 ms trigger period for 30 "
                        "fps; tune with --trig-delay-us to land on the R+B "
                        "illumination — see docs §10.4).")
    p.add_argument("--trig-delay-us", type=float,
                   default=float(os.environ.get("STIM_TRIG_DELAY_US", "0")),
                   help="Camera-side TriggerDelay µs (float; the IDS TriggerDelay "
                        "node is float-valued). Delay from the trigger edge to "
                        "exposure start, to phase-align the exposure with the "
                        "DMD's R+B illumination sub-frames (bench-tuned).")
    p.add_argument("--gain", type=float,
                   default=float(os.environ.get("STIM_GAIN", "1.0")),
                   help="Camera analog gain (secondary brightness lever; the LED "
                        "is already at full PWM). Raise if captures are dark after "
                        "tuning --exposure-us / --trig-delay-us.")
    p.add_argument("--hold-scale", type=float, default=1.0,
                   help="Multiply all hold times (use <1 for a quick test).")
    p.add_argument("--trigger-wait-sec", type=float, default=10.0)
    p.add_argument("--sequence", choices=("full", "density", "all"), default="full",
                   help="Which mask program: 'full', 'density', or 'all'.")
    p.add_argument("--swap-interval", type=int, default=1, choices=(0, 1),
                   help="Projector engine vsync: 1 (default) = on (complete "
                        "frames, no engine-swap tearing; bench-verified it holds "
                        "the trigger lock + PASS); 0 = off (low-latency, but mask "
                        "updates can tear and the slave camera captures the tear). "
                        "Residual transition-blend tearing is an exposure-phase "
                        "issue — tune --trig-delay-us / --exposure-us.")
    args = p.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[demo] output bundle: {out_dir}")

    # metadata (h_sent filled in after the run)
    try:
        git_sha = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        git_sha = "unknown"
    meta = {
        "git_sha": git_sha, "fps": args.fps, "exposure_us": args.exposure_us,
        "trig_delay_us": args.trig_delay_us, "hold_scale": args.hold_scale,
        "swap_interval": args.swap_interval,
        "dry_run": args.dry_run, "sequence": args.sequence,
        "color_model": "rgb-cycle_modeB_simultaneous", "no_warp": args.no_warp,
        "proj_endpoint": PROJ_ENDPOINT, "h_endpoint": PROJ_H_ENDPOINT,
        "h_sent": False,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    if args.dry_run:
        print("[demo] DRY RUN — no hardware; writing mask log only")
        with DemoLogger(out_dir / "demo_frames.csv") as logger:
            run_sequence(client=None, logger=logger, dry=True,
                         scale=args.hold_scale, which=args.sequence)
        n = sum(1 for _ in open(out_dir / "demo_frames.csv")) - 1
        print(f"[demo] dry run complete — {n} log rows in demo_frames.csv")
        return 0

    client = None
    rc = 0
    h_sent = False
    try:
        boot_dmd(out_dir)
        launch_projector(out_dir, swap_interval=args.swap_interval)
        # Send the calibration homography to the engine BEFORE any mask, so every
        # mask we push raw is displayed warped+flipped → camera sees the intent.
        if not args.no_warp:
            h_sent = send_homography(out_dir, args.homography)
        else:
            print("[demo] --no-warp: NOT sending homography (uncalibrated projection).")
        if not args.no_camera:
            launch_camera(out_dir, args.fps, args.exposure_us, args.trigger_wait_sec,
                          args.trig_delay_us, args.gain)
            time.sleep(1.0)  # let the camera arm on the trigger
        from projector_client import ProjectorClient
        client = ProjectorClient(endpoint=PROJ_ENDPOINT, width=PROJ_W, height=PROJ_H)
        # The camera_recorder owns demo_frames.csv (camera_meta). The projection
        # log shares CLOCK_MONOTONIC ts_ns for correlation; write it to a
        # sibling file to avoid two processes writing one CSV.
        with DemoLogger(out_dir / "demo_masklog.csv") as logger:
            print(f"[demo] playing mask sequence '{args.sequence}'…")
            run_sequence(client=client, logger=logger, dry=False,
                         scale=args.hold_scale, which=args.sequence)
        print("[demo] sequence complete")
    except KeyboardInterrupt:
        print("[demo] interrupted")
        rc = 130
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        _cleanup()              # stops camera (finalizes logs/TIFFs) + projector
        standby_dmd(out_dir)

    # Record whether the projection was calibrated (composer/verify read this).
    try:
        meta["h_sent"] = bool(h_sent)
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        pass

    # Auto-verify the bundle (camera logs are finalized now) so every capture
    # comes with a sync/accuracy PASS/FAIL report + synced_frames.csv.
    if not args.no_camera and not args.no_verify:
        try:
            import verify
            print("\n[demo] ── verifying bundle ──")
            verify.main(["--bundle-dir", str(out_dir), "--fps", str(args.fps)])
        except Exception as e:
            print(f"[demo] verify skipped: {e}")
    print(f"[demo] done. Bundle: {out_dir}")
    return rc


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: (_cleanup(), sys.exit(130)))
    raise SystemExit(main())
