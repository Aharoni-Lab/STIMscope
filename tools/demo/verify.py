#!/usr/bin/env python3
"""Verify a demo recording bundle — sync + accuracy report.

Cross-checks the camera capture log against the mask (projection) log, emits the
authoritative per-camera-frame -> mask mapping, and prints a PASS/FAIL summary.

Inputs (in --bundle-dir; either split across two CSVs for a real run, or both
in one CSV for a --dry-run):
  demo_frames.csv    camera_meta rows (one per captured frame: host ts_ns,
                     IDS hardware ts hw_ts_ns, camera frame_id) + metric rows
                     (write_drops, sdk_lost, total_frames, …)
  demo_masklog.csv   projection_send rows (mask name/color/sha256/frame_id,
                     host ts_ns, segment, intent)
  projector.log      optional cross-check ([CAM] frame -> PROJ visible_id lines)
  tiff_frames/       optional: counted to confirm it matches captured frames

Output:
  synced_frames.csv  one row per captured camera frame -> the mask that was on
                     the DMD when it was captured (mapped by shared
                     CLOCK_MONOTONIC host ts_ns; masks are held for many frames
                     so the mapping is unambiguous away from switch boundaries).

Accuracy is judged on: zero dropped frames, fps in band, low trigger-interval
jitter (from the IDS hardware timestamps — proof of trigger lock), and full
mask coverage (every projected mask was actually captured).

Usage:
  tools/demo/verify.py --bundle-dir <dir> [--fps 30] [--lag-ms 0]
"""

from __future__ import annotations

import argparse
import bisect
import csv
import re
import statistics
import sys
from pathlib import Path

# DemoLogger column indices
TS, WALL, EVENT, SEG, MNAME, MCOLOR, MSHA, FID, HWTS, EXTRA = range(10)


def _read_rows(path: Path):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        rows = list(r)
    return rows[1:] if rows and rows[0][:1] == ["ts_ns"] else rows


def _int(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--fps", type=float, default=30.0, help="Expected fps (band check).")
    p.add_argument("--fps-tol", type=float, default=3.0)
    p.add_argument("--lag-ms", type=float, default=0.0,
                   help="Subtract this from camera ts before mapping to a mask "
                        "(account for mask-to-light + write-queue latency).")
    p.add_argument("--jitter-ms-max", type=float, default=8.0,
                   help="Max allowed std of inter-frame hardware-timestamp "
                        "intervals (proof of trigger lock).")
    p.add_argument("--coverage-min", type=float, default=0.99,
                   help="Min fraction of projected masks that must be captured.")
    args = p.parse_args(argv)

    bdir = args.bundle_dir
    if not bdir.is_dir():
        print(f"[verify] ERROR: not a directory: {bdir}", file=sys.stderr)
        return 2

    # Gather rows from both possible CSVs (real run = split; dry-run = one file).
    rows = []
    for name in ("demo_frames.csv", "demo_masklog.csv"):
        rows += _read_rows(bdir / name)

    cam = []      # (ts_ns, hw_ts_ns|None, frame_id)
    sends = []    # (ts_ns, frame_id, name, color, sha, segment)
    metrics = {}  # name -> value
    for r in rows:
        if len(r) < 10:
            continue
        ev = r[EVENT]
        if ev == "camera_meta":
            ts = _int(r[TS]); fid = _int(r[FID]); hw = _int(r[HWTS])
            if ts is not None:
                cam.append((ts, hw, fid))
        elif ev == "projection_send":
            ts = _int(r[TS])
            if ts is not None:
                sends.append((ts, _int(r[FID]), r[MNAME], r[MCOLOR], r[MSHA], r[SEG]))
        elif ev == "metric":
            metrics[r[MNAME]] = r[EXTRA]

    cam.sort()
    sends.sort()
    print(f"[verify] bundle: {bdir}")
    print(f"[verify] camera frames: {len(cam)}   projected masks: {len(sends)}")

    problems = []

    # ── Projection-only (no camera) run ───────────────────────────────────────
    if not cam:
        print("[verify] no camera_meta rows — projection-only run (nothing to sync).")
        if not sends:
            print("[verify] FAIL: no projection_send rows either.")
            return 1
        _print_segment_breakdown(sends)
        print("[verify] (run with the camera to get a full sync/accuracy report)")
        return 0

    if not sends:
        print("[verify] FAIL: camera frames present but no projection_send rows.")
        return 1

    # ── Build the per-frame mask mapping (shared CLOCK_MONOTONIC ts_ns) ────────
    lag_ns = int(args.lag_ms * 1e6)
    send_ts = [s[0] for s in sends]
    mapped_fids = set()
    no_mask = 0
    seg_counts = {}
    synced_rows = []
    for ts, hw, fid in cam:
        # last mask sent at or before (camera_ts - lag)
        idx = bisect.bisect_right(send_ts, ts - lag_ns) - 1
        hws = hw if hw is not None else ""
        if idx < 0:
            no_mask += 1
            synced_rows.append([fid, ts, hws, "", "", "", "", "(pre-first-mask)"])
            continue
        s = sends[idx]
        mapped_fids.add(s[1])
        seg_counts[s[5]] = seg_counts.get(s[5], 0) + 1
        synced_rows.append([fid, ts, hws, s[1], s[2], s[3], s[4], s[5]])

    out_path = bdir / "synced_frames.csv"
    try:
        with open(out_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cam_frame_id", "cam_ts_ns", "hw_ts_ns", "mask_frame_id",
                        "mask_name", "mask_color", "mask_sha256", "segment"])
            w.writerows(synced_rows)
        print(f"[verify] wrote {out_path.name} ({len(synced_rows)} rows)")
    except OSError as e:
        print(f"[verify] (could not write {out_path.name}: {e}; report below still valid)")

    # ── Drops ─────────────────────────────────────────────────────────────────
    write_drops = _int(metrics.get("camera_recorder_write_drops")) or 0
    sdk_lost_raw = metrics.get("camera_recorder_sdk_lost_frames")
    sdk_lost = _int(sdk_lost_raw)
    n_tiff = len(list((bdir / "tiff_frames").glob("*.tif"))) if (bdir / "tiff_frames").is_dir() else None
    print(f"[verify] write_drops={write_drops}  sdk_lost={sdk_lost if sdk_lost is not None else 'unknown'}"
          + (f"  tiff_frames={n_tiff}" if n_tiff is not None else ""))
    if write_drops > 0:
        problems.append(f"{write_drops} write-queue drops")
    if sdk_lost:
        problems.append(f"{sdk_lost} SDK-lost frames")
    if n_tiff is not None and abs(n_tiff - len(cam)) > 1:
        problems.append(f"tiff_frames ({n_tiff}) != camera frames ({len(cam)})")

    # ── FPS + trigger-interval jitter (from hardware timestamps) ──────────────
    hw_list = [hw for (_, hw, _) in cam if hw is not None]
    fps_hw = None
    if len(hw_list) >= 3:
        hw_list.sort()
        intervals_ms = [(b - a) / 1e6 for a, b in zip(hw_list, hw_list[1:]) if b > a]
        if intervals_ms:
            mean_ms = statistics.mean(intervals_ms)
            std_ms = statistics.pstdev(intervals_ms)
            fps_hw = 1000.0 / mean_ms if mean_ms else 0.0
            print(f"[verify] hw-trigger interval: mean={mean_ms:.2f} ms  "
                  f"std={std_ms:.2f} ms  min={min(intervals_ms):.2f}  max={max(intervals_ms):.2f}")
            print(f"[verify] fps (hardware-timestamp): {fps_hw:.2f}")
            if std_ms > args.jitter_ms_max:
                problems.append(f"trigger jitter std={std_ms:.2f} ms > {args.jitter_ms_max} ms "
                                f"(camera may not be cleanly locked to the DMD trigger)")
    else:
        # fall back to host ts span
        span_s = (cam[-1][0] - cam[0][0]) / 1e9
        if span_s > 0:
            fps_hw = (len(cam) - 1) / span_s
        print(f"[verify] (no hardware timestamps; fps from host clock: "
              f"{fps_hw:.2f})" if fps_hw else "[verify] WARN: cannot compute fps")
        problems.append("no IDS hardware timestamps — cannot prove trigger lock")

    if fps_hw is not None and abs(fps_hw - args.fps) > args.fps_tol:
        problems.append(f"fps {fps_hw:.1f} outside {args.fps}±{args.fps_tol}")

    # ── Coverage: was every projected mask actually captured? ─────────────────
    all_fids = {s[1] for s in sends if s[1] is not None}
    captured = mapped_fids & all_fids
    cov = (len(captured) / len(all_fids)) if all_fids else 1.0
    unmapped = sorted(all_fids - captured)
    print(f"[verify] mask coverage: {len(captured)}/{len(all_fids)} = {cov*100:.1f}% captured"
          + (f"  ({no_mask} pre-first-mask frames)" if no_mask else ""))
    if unmapped:
        print(f"[verify]   masks never captured (frame_ids): {unmapped[:20]}"
              + (" …" if len(unmapped) > 20 else ""))
    if cov < args.coverage_min:
        problems.append(f"coverage {cov*100:.1f}% < {args.coverage_min*100:.0f}%")

    _print_segment_breakdown(sends, seg_counts)

    # ── Optional cross-check vs projector.log ─────────────────────────────────
    plog = bdir / "projector.log"
    if plog.exists():
        try:
            txt = plog.read_text(errors="ignore")
            cam_lines = len(re.findall(r"\[CAM ?\].*visible_id=", txt))
            vis_ids = set(re.findall(r"visible_id=(\d+)", txt))
            print(f"[verify] projector.log cross-check: {cam_lines} [CAM] mappings, "
                  f"{len(vis_ids)} distinct visible_ids")
        except Exception as e:
            print(f"[verify] projector.log cross-check skipped: {e}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("─" * 64)
    if problems:
        print("[verify] RESULT: ❌ FAIL")
        for pr in problems:
            print(f"           - {pr}")
        return 1
    print("[verify] RESULT: ✅ PASS — recording is synced and accurate")
    print(f"           {len(cam)} frames @ ~{fps_hw:.1f} fps, 0 drops, "
          f"{cov*100:.0f}% mask coverage, locked to the DMD trigger")
    return 0


def _print_segment_breakdown(sends, seg_counts=None) -> None:
    segs = {}
    for s in sends:
        segs.setdefault(s[5], 0)
        segs[s[5]] += 1
    print("[verify] segments (masks sent" + (" / camera frames" if seg_counts else "") + "):")
    for seg in dict.fromkeys(s[5] for s in sends):
        line = f"           {seg}: {segs.get(seg,0)} masks"
        if seg_counts is not None:
            line += f" / {seg_counts.get(seg,0)} frames"
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
