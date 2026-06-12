#!/usr/bin/env python3
"""Compose the demo triptych (RAW MASK | PROJECTION | CAMERA) as a multipage
TIFF (+ an mp4), to verify per-frame sync, calibration, and orientation.

DETERMINISTIC, no measuring. The demo sends RAW masks to the engine and the
engine applies the calibration: it displays flip(warpPerspective(mask,
H_cam2proj, 1920x1080)) for every mask (verified main.cpp:787,800,807). So:

Panels (left -> right):
  RAW MASK    the camera-space INTENT, drawn WHITE — should OVERLAY CAMERA
              (the engine warped the projection so the camera sees the intent).
  PROJECTION  flip(warp(mask, H_cam2proj)) tinted R/B — EXACTLY what the engine
              put on the DMD (the calibration applied). Reproduced from the
              bundle's homography_cam2proj.npy, the same H sent to the engine.
  CAMERA      the captured frame (tiff_frames/ preferred; else demo_camera.mp4).

SYNC SOURCE (authoritative): the projector engine logs, for EACH camera trigger,
which visible_id (mask frame_id) was on the DMD at that instant — the `[CAM]`
lines in projector.log. The engine sees more triggers than the camera records
(it starts first), so we tail-align and VERIFY the offset against the shared
monotonic clock (fail loud if the camera dropped frames mid-stream). Masks are
regenerated deterministically by replaying run_demo.run_sequence.

Usage:
  tools/demo/composer.py --bundle-dir <dir> [--sequence full]
      [--all | --step N] [--mask-hflip] [--mask-vflip]
      [--cam-rotate 0|90|180|270] [--cam-flip-h] [--cam-flip-v] [--out PATH]
"""

from __future__ import annotations

import argparse
import bisect
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE), "/app/STIMViewer_CRISPI",
           str(_REPO_ROOT / "STIMscope" / "STIMViewer_CRISPI")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mask_library as ml          # noqa: E402
import run_demo                    # noqa: E402  (run_sequence; import has no side effects)

PROJ_W, PROJ_H = ml.PROJ_W, ml.PROJ_H
PANEL_W, PANEL_H = 480, 270
LABEL_H = 28


class _CaptureClient:
    """Stand-in projector client used to regenerate masks deterministically.
    run_demo._send calls send_rgb(rgb, frame_id=...) with the RAW camera-intent
    RGB frame (the engine, not the demo, applies the warp), so we stash it by
    frame_id and reproduce RAW (white) + PROJECTION (warped) from it."""

    def __init__(self):
        self.frames = {}

    def send_rgb(self, rgb, frame_id=None, immediate=True, visible_overlay=None):
        if frame_id is not None:
            self.frames[int(frame_id)] = rgb.copy()

    def send_gray(self, img, frame_id=None, immediate=True, visible_overlay=None):
        # tolerate a grayscale send (older/raw paths); store as 3ch
        if frame_id is not None:
            g = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            self.frames[int(frame_id)] = g.copy()

    def close(self):
        pass


class _NullLogger:
    def set_segment(self, *a, **k): pass
    def segment_start(self, *a, **k): pass
    def segment_end(self, *a, **k): pass
    def projection_send(self, *a, **k): pass
    def metric(self, *a, **k): pass


def _regenerate_masks(which: str) -> dict:
    """Replay the deterministic sequence to recover {frame_id: raw RGB mask}.
    dry=False so run_demo._send routes the frame to our capture client; scale=0
    so there are no sleeps; the capture client touches no hardware."""
    cap = _CaptureClient()
    run_demo.run_sequence(client=cap, logger=_NullLogger(), dry=False,
                          scale=0.0, which=which)
    return cap.frames


def _parse_engine_cam(projector_log: Path):
    """Ordered list of (trigger_ts_ns, visible_id), one per camera trigger the
    engine saw. trigger_ts_ns is the engine's monotonic time of the trigger edge
    — the same system-wide clock the camera/host logs use."""
    rx = re.compile(r"\[CAM ?\].*?@(\d+) ns.*?visible_id=(-?\d+)")
    out = []
    with open(projector_log, errors="ignore") as fh:
        for ln in fh:
            m = rx.search(ln)
            if m:
                out.append((int(m.group(1)), int(m.group(2))))
    return out


def _camera_ts_first_last(bundle: Path):
    """(first, last) host monotonic ts of camera frames (from demo_frames.csv),
    used to corroborate the engine<->camera index offset at BOTH ends (a single
    global offset is only valid if the start and end offsets agree — a mid-stream
    drop shifts the end but not the start)."""
    p = bundle / "demo_frames.csv"
    if not p.exists():
        return None, None
    first = last = None
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("event") == "camera_meta" and r.get("ts_ns"):
                t = int(r["ts_ns"])
                if first is None:
                    first = t
                last = t
    return first, last


def _load_mask_meta(bundle: Path):
    """frame_id -> (name, color, segment, sha256) for page labels + the
    determinism cross-check (from masklog)."""
    meta = {}
    for fn in ("demo_masklog.csv", "demo_frames.csv"):
        p = bundle / fn
        if not p.exists():
            continue
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("event") == "projection_send" and r.get("frame_id"):
                    meta[int(r["frame_id"])] = (r.get("mask_name", ""),
                                                r.get("mask_color", ""),
                                                r.get("segment", ""),
                                                r.get("mask_sha256", ""))
    return meta


def _load_calibration(bundle: Path):
    """Return (H, calibrated, reason). H is the bundle-local matrix the run
    ACTUALLY sent to the engine (None if the run was uncalibrated). Authoritative:
    run_demo saves the bundle H only after a confirmed engine ACK, so a
    bundle-local homography_cam2proj.npy present == this run was calibrated. We do
    NOT fall back to a repo-global H the run may never have used."""
    import json
    no_warp, h_sent = False, None
    mp = bundle / "metadata.json"
    if mp.exists():
        try:
            md = json.loads(mp.read_text())
            no_warp = bool(md.get("no_warp", False))
            h_sent = md.get("h_sent", None)
        except Exception:
            pass
    if no_warp:
        return None, False, "run used --no-warp (engine projected raw)"
    if h_sent is False:
        return None, False, "metadata h_sent=False (engine did not get the homography)"
    hp = bundle / "homography_cam2proj.npy"
    if not hp.exists():
        return None, False, "no bundle homography_cam2proj.npy (run did not send H)"
    H = np.load(str(hp)).astype(np.float64)
    if H.shape != (3, 3):
        return None, False, f"bundle H not 3x3 ({H.shape})"
    return H, True, "calibrated (bundle H + engine ACK)"


def _orient_mask(rgb, hflip, vflip):
    if hflip and vflip:
        return cv2.flip(rgb, -1)
    if hflip:
        return cv2.flip(rgb, 1)
    if vflip:
        return cv2.flip(rgb, 0)
    return rgb


def _orient_cam(img, rot, fh, fv):
    if fh and fv:
        img = cv2.flip(img, -1)
    elif fh:
        img = cv2.flip(img, 1)
    elif fv:
        img = cv2.flip(img, 0)
    if rot == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rot == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rot == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _panel(img, label):
    """Fit `img` (RGB or grayscale) into a PANEL_W x PANEL_H RGB panel + label."""
    if img is None:
        img = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    s = min(PANEL_W / w, PANEL_H / h)
    rs = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                    interpolation=cv2.INTER_AREA)
    panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    yo, xo = (PANEL_H - rs.shape[0]) // 2, (PANEL_W - rs.shape[1]) // 2
    panel[yo:yo + rs.shape[0], xo:xo + rs.shape[1]] = rs
    cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 0), 1, cv2.LINE_AA)   # RGB yellow
    return panel


def _load_cam(bundle, cam_id, cap):
    """Load camera frame as RGB (mono sensor → gray → RGB)."""
    tif = bundle / "tiff_frames" / f"frame_{cam_id:06d}.tif"
    img = None
    if tif.exists():
        img = cv2.imread(str(tif), cv2.IMREAD_UNCHANGED)
    elif cap is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cam_id)
        ok, fr = cap.read()
        if ok:
            img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def _proj_from_mask(raw_rgb, H):
    """Reproduce the engine's display for this mask: flip(warp(mask, H_cam2proj,
    1920x1080)). raw_rgb is the camera-intent RGB (R=red, B=blue channels).

    INTER_LINEAR (bilinear) matches the engine's render: main.cpp uses
    warp_mask_bilinear by default (WARP_BILINEAR=1, compile-time, no CLI flag),
    so the PROJECTION panel is a faithful reproduction (nearest would harden the
    warped mask edges that the engine antialiases)."""
    if H is None:
        return raw_rgb
    warped = cv2.warpPerspective(raw_rgb, H, (PROJ_W, PROJ_H),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return cv2.flip(warped, 1)   # engine --horiz-flip=1, applied after the warp


def _white_intent(raw_rgb):
    """RAW MASK panel: the camera-space intent as WHITE (union of any lit channel)."""
    lit = raw_rgb.max(axis=2) if raw_rgb.ndim == 3 else raw_rgb
    return (lit > 0).astype(np.uint8) * 255


def compose(bundle: Path, which="full", step=None, do_all=False,
            mask_hflip=False, mask_vflip=True, lag_frames=3,
            cam_rot=0, cam_fh=False, cam_fv=False, out=None) -> Path:
    # ---- camera frames ----
    tdir = bundle / "tiff_frames"
    cap = None
    if tdir.is_dir() and any(tdir.glob("frame_*.tif")):
        cam_ids = sorted(int(p.stem.split("_")[1]) for p in tdir.glob("frame_*.tif"))
    else:
        mp4 = bundle / "demo_camera.mp4"
        if not mp4.exists():
            raise SystemExit("[composer] no tiff_frames/ and no demo_camera.mp4")
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cam_ids = list(range(n))
    n_cam = len(cam_ids)

    # ---- authoritative per-frame mask (engine [CAM] log, tail-aligned) ----
    plog = bundle / "projector.log"
    eng = _parse_engine_cam(plog) if plog.exists() else []
    if eng:
        eng_ts = [t for t, _ in eng]
        eng_vis = [v for _, v in eng]
        offset = max(0, len(eng) - n_cam)   # camera captured the LAST n_cam triggers
        # Verify the single global offset against the monotonic clock at BOTH
        # ends. A start-only drop is normal (camera arms after the engine); a
        # MID-STREAM drop shifts the end offset but not the start, so the global
        # offset silently mis-maps every frame after the drop. Comparing the
        # start-implied and end-implied offsets catches that (the start-only
        # check used before could not). Fail loud; don't claim "authoritative".
        cam_ts0, cam_ts_last = _camera_ts_first_last(bundle)
        verified = False
        if cam_ts0 is not None:
            off_start = max(0, bisect.bisect_right(eng_ts, cam_ts0) - 1)
            problems = []
            if abs(off_start - offset) > 3:
                problems.append(f"start: count-offset={offset} vs ts-offset={off_start}")
            if cam_ts_last is not None and n_cam > 1:
                off_end = max(0, (bisect.bisect_right(eng_ts, cam_ts_last) - 1) - (n_cam - 1))
                if abs(off_end - off_start) > 3:
                    problems.append(f"mid-stream drop: start-offset={off_start} "
                                    f"vs end-offset={off_end}")
            if problems:
                print("[composer] *** WARNING: per-frame sync UNRELIABLE — "
                      + "; ".join(problems) + ". Camera likely dropped frames; the "
                      "mapping is NOT trustworthy for this bundle. ***")
            else:
                verified = True
                print(f"[composer] offset {offset} VERIFIED vs monotonic clock "
                      "at start AND end (no mid-stream drops detected)")
        else:
            print("[composer] *** WARNING: no camera_meta timestamps — offset "
                  "UNVERIFIED (cannot confirm the mapping). ***")
        vis = [eng_vis[i + offset] if 0 <= i + offset < len(eng_vis) else -1
               for i in range(n_cam)]
        print(f"[composer] sync: engine [CAM]={len(eng)}  camera={n_cam}  "
              f"tail-offset={offset}" + ("  (verified)" if verified else "  (UNVERIFIED)"))
    else:
        sf = {}
        sp = bundle / "synced_frames.csv"
        if sp.exists():
            with open(sp, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("mask_frame_id"):
                        sf[int(r["cam_frame_id"])] = int(r["mask_frame_id"])
        vis = [sf.get(cid, -1) for cid in cam_ids]
        print("[composer] WARN: projector.log missing — using timestamp synced_frames.csv (less accurate)")

    if lag_frames:
        # Compensate a systematic camera-capture-vs-engine-log latency: the
        # camera integrates the mask shown ~lag_frames before the trigger the
        # engine logged, so it looks "behind" the projection. Shift the mask
        # lookup so RAW/PROJECTION match what the camera actually captured.
        # Rig-specific; tune empirically (positive = camera is behind).
        n = len(vis)
        vis = [vis[min(n - 1, max(0, i - lag_frames))] for i in range(n)]
        print(f"[composer] applied --lag-frames {lag_frames} (shifted mask lookup "
              "to match the camera's capture latency)")

    masks = _regenerate_masks(which)
    meta = _load_mask_meta(bundle)
    H, calibrated, reason = _load_calibration(bundle)
    print(f"[composer] regenerated {len(masks)} masks for sequence '{which}'")
    if calibrated:
        print(f"[composer] {reason}: PROJECTION = flip(warp(mask, H)) (bilinear).")
        proj_label = "PROJECTION (calibrated)"
    else:
        print(f"[composer] *** WARNING: UNCALIBRATED bundle — {reason}. The engine "
              "projected RAW, so PROJECTION shows the raw mask (NOT warped) and "
              "RAW-vs-CAMERA alignment is NOT expected. ***")
        proj_label = "PROJECTION (UNCALIBRATED = raw)"

    # Determinism cross-check: the regenerated mask must match the sha logged at
    # record time (run_demo logs the packed-RGB sha). A mismatch means the demo
    # code/seed changed between recording and composing → wrong PROJECTION panel.
    sha_mismatch = 0
    for fid, rgb in masks.items():
        logged = meta.get(fid, ("", "", "", ""))[3]
        if logged and ml._sha256(rgb) != logged:
            sha_mismatch += 1
    if sha_mismatch:
        print(f"[composer] *** WARNING: {sha_mismatch} regenerated masks do NOT "
              "match the logged sha256 — the demo generator changed since this "
              "bundle was recorded; PROJECTION/RAW panels may be WRONG. Recompose "
              "with the matching code revision. ***")

    # ---- select which frames to render ----
    if do_all:
        idxs = list(range(n_cam))
    elif step:
        idxs = list(range(0, n_cam, step))
    else:
        # middle frame of each contiguous run of the same visible_id
        idxs = []
        i = 0
        while i < n_cam:
            j = i
            while j + 1 < n_cam and vis[j + 1] == vis[i]:
                j += 1
            if vis[i] > 0:
                idxs.append((i + j) // 2)
            i = j + 1
    print(f"[composer] composing {len(idxs)} pages (of {n_cam} camera frames)")

    pages = []
    for i in idxs:
        cid = cam_ids[i]
        v = vis[i]
        cam = _load_cam(bundle, cid, cap)
        if cam is not None:
            cam = _orient_cam(cam, cam_rot, cam_fh, cam_fv)
        raw_rgb = masks.get(v) if v and v > 0 else None
        name, color, seg, _sha = meta.get(v, ("(none)", "-", "-", ""))
        if raw_rgb is not None:
            raw_panel = _orient_mask(_white_intent(raw_rgb), mask_hflip, mask_vflip)
            # Calibrated: reproduce the engine warp+flip. Uncalibrated: the engine
            # showed the raw mask, so the PROJECTION panel is the raw RGB.
            proj_panel = _proj_from_mask(raw_rgb, H) if calibrated else raw_rgb
        else:
            raw_panel = None
            proj_panel = None
        trip = np.hstack([_panel(raw_panel, "RAW MASK"),
                          _panel(proj_panel, proj_label),
                          _panel(cam, "CAMERA")])
        strip = np.zeros((LABEL_H, trip.shape[1], 3), np.uint8)
        cv2.putText(strip, f"cam#{cid}  vis={v}  mask={str(name)[:30]}  "
                    f"LED={color}  seg={seg}", (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        pages.append(np.vstack([strip, trip]))

    if not pages:
        raise SystemExit("[composer] no pages to write (no mapped frames)")

    out = Path(out) if out else (bundle / "demo_composite.tif")
    # ---- multipage TIFF (RGB, lossless) ----
    try:
        import tifffile
        with tifffile.TiffWriter(str(out)) as tw:
            for pg in pages:
                tw.write(pg, photometric="rgb", compression="zlib")
    except Exception as e:
        print(f"[composer] tifffile failed ({e}); using cv2 multipage")
        bgr = [cv2.cvtColor(p, cv2.COLOR_RGB2BGR) for p in pages]
        if not cv2.imwritemulti(str(out), bgr, [cv2.IMWRITE_TIFF_COMPRESSION, 5]):
            raise SystemExit("[composer] failed to write TIFF")

    # ---- companion mp4 (BGR for the player) so it can be reviewed without ImageJ
    try:
        mp4_out = out.with_suffix(".mp4")
        h, w = pages[0].shape[:2]
        vw = cv2.VideoWriter(str(mp4_out), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (w, h))
        for pg in pages:
            vw.write(cv2.cvtColor(pg, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"[composer] wrote {mp4_out} (review video, 6 fps)")
    except Exception as e:
        print(f"[composer] (mp4 companion skipped: {e})")

    if cap is not None:
        cap.release()
    print(f"[composer] wrote {out} ({len(pages)} pages, {pages[0].shape[1]}x{pages[0].shape[0]})")
    print(f"[composer] orientation: mask_hflip={mask_hflip} mask_vflip={mask_vflip} "
          f"cam_rot={cam_rot} cam_flip_h={cam_fh} cam_flip_v={cam_fv}")
    print("[composer] -> RAW MASK | PROJECTION | CAMERA.  RAW MASK (white intent) "
          "should OVERLAY CAMERA; PROJECTION is flip(warp(mask, H)) — what the DMD "
          "showed. If RAW vs CAMERA is mirrored, toggle --mask-hflip / --mask-vflip "
          "or --cam-flip-*.")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--sequence", choices=("full", "density", "all"), default="full")
    p.add_argument("--all", action="store_true")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--mask-hflip", dest="mask_hflip", action="store_true", default=False)
    p.add_argument("--no-mask-hflip", dest="mask_hflip", action="store_false")
    p.add_argument("--mask-vflip", dest="mask_vflip", action="store_true", default=True,
                   help="V-flip the RAW MASK panel to align with the camera "
                        "(default: on; this rig's camera is vertically flipped).")
    p.add_argument("--no-mask-vflip", dest="mask_vflip", action="store_false")
    p.add_argument("--lag-frames", type=int, default=3,
                   help="Shift the camera->mask mapping by N frames to compensate "
                        "capture-vs-display latency (positive = camera is behind). "
                        "Default 3 (bench-tuned at the default --swap-interval=1 / "
                        "vsync on; it was 2 at swap-interval=0 — vsync adds ~1 "
                        "frame of present latency). Re-tune if you change vsync.")
    p.add_argument("--cam-rotate", type=int, default=0, choices=[0, 90, 180, 270])
    p.add_argument("--cam-flip-h", dest="cam_fh", action="store_true", default=False)
    p.add_argument("--cam-flip-v", dest="cam_fv", action="store_true", default=False)
    p.add_argument("--out", default=None, help="output path (e.g..../demo_composite.tiff)")
    args = p.parse_args(argv)
    compose(args.bundle_dir, which=args.sequence, step=args.step, do_all=args.all,
            mask_hflip=args.mask_hflip, mask_vflip=args.mask_vflip,
            lag_frames=args.lag_frames,
            cam_rot=args.cam_rotate, cam_fh=args.cam_fh, cam_fv=args.cam_fv,
            out=args.out)


if __name__ == "__main__":
    main()
