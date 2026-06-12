"""Mask generation for the base-platform DMD demo recording — inference-free.

Every generator here is pure geometry — no inference-module dependency,
no Cellpose `rois.npz`, no homography dependency. Three of the segments
(neuron_rois / speed_ramp / multi_target_temporal) use DETERMINISTIC
SYNTHETIC ROIs (a fixed-seed blob field) so the "many independent targets"
capability is demonstrated without coupling to any inference / segmentation
output.

Each generator returns a list of ``DemoMask``:
  - name:   e.g. "spatial_r0_c0_red"
  - led:    "R" | "B" — which LED to gate when this mask plays
  - intent: human-readable label
  - img:    (PROJ_H, PROJ_W) uint8 grayscale
  - sha256: hex digest of the raw bytes (determinism check)

Determinism: identical inputs -> identical mask bytes (and identical sha256)
every run. All RNG is seeded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import cv2

PROJ_W = 1920
PROJ_H = 1080


@dataclass
class DemoMask:
    name: str
    led: str            # "R" | "B"
    intent: str
    img: np.ndarray     # (H, W) uint8 grayscale
    sha256: str


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _ensure_shape(mask_2d: np.ndarray) -> np.ndarray:
    if mask_2d.shape != (PROJ_H, PROJ_W):
        mask_2d = cv2.resize(
            mask_2d.astype(np.uint8), (PROJ_W, PROJ_H), interpolation=cv2.INTER_NEAREST
        )
    return mask_2d.astype(np.uint8, copy=False)


def _wrap(name: str, led: str, intent: str, mask_2d: np.ndarray) -> DemoMask:
    img = _ensure_shape(mask_2d)
    return DemoMask(name=name, led=led, intent=intent, img=img, sha256=_sha256(img))


# ─────────────────────────────────────────────────────────────────────────────
# Spatial coverage sweep
# ─────────────────────────────────────────────────────────────────────────────


def spatial_sweep(target_size_px: int = 60, grid: Tuple[int, int] = (5, 5)) -> List[DemoMask]:
    n_cols, n_rows = grid
    margin_x = PROJ_W // (n_cols + 1)
    margin_y = PROJ_H // (n_rows + 1)
    half = target_size_px // 2

    masks: List[DemoMask] = []
    n_pos = n_cols * n_rows
    for r in range(n_rows):
        for c in range(n_cols):
            cx = (c + 1) * margin_x
            cy = (r + 1) * margin_y
            base = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
            base[cy - half : cy + half, cx - half : cx + half] = 255
            pos_idx = r * n_cols + c + 1
            masks.append(_wrap(f"spatial_r{r}_c{c}_red", "R",
                               f"Spatial sweep pos {pos_idx}/{n_pos} — RED LED", base))
            masks.append(_wrap(f"spatial_r{r}_c{c}_blue", "B",
                               f"Spatial sweep pos {pos_idx}/{n_pos} — BLUE LED", base))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Arbitrary shapes
# ─────────────────────────────────────────────────────────────────────────────


def arbitrary_shapes() -> List[DemoMask]:
    cx, cy = PROJ_W // 2, PROJ_H // 2
    radius = 220

    def _circle():
        m = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8); cv2.circle(m, (cx, cy), radius, 255, -1); return m
    def _square():
        m = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8); cv2.rectangle(m, (cx-radius, cy-radius), (cx+radius, cy+radius), 255, -1); return m
    def _triangle():
        m = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
        pts = np.array([[cx, cy-radius], [cx-radius, cy+radius], [cx+radius, cy+radius]], np.int32)
        cv2.fillPoly(m, [pts], 255); return m
    def _hexagon():
        m = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
        pts = np.array([[cx + int(radius*np.cos(t)), cy + int(radius*np.sin(t))]
                        for t in np.linspace(0, 2*np.pi, 7)[:-1]], np.int32)
        cv2.fillPoly(m, [pts], 255); return m
    def _star():
        m = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8); pts = []
        for i in range(10):
            r = radius if i % 2 == 0 else radius // 2
            t = i * np.pi / 5 - np.pi / 2
            pts.append([cx + int(r*np.cos(t)), cy + int(r*np.sin(t))])
        cv2.fillPoly(m, [np.array(pts, np.int32)], 255); return m
    def _irregular():
        m = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
        pts = np.array([
            [cx-250, cy-150], [cx-100, cy-250], [cx+50, cy-100],
            [cx+250, cy-200], [cx+200, cy+50], [cx+300, cy+250],
            [cx+50, cy+200], [cx-150, cy+250], [cx-250, cy+50],
        ], np.int32)
        cv2.fillPoly(m, [pts], 255); return m

    shapes = [
        ("circle", _circle()), ("square", _square()), ("triangle", _triangle()),
        ("hexagon", _hexagon()), ("star", _star()), ("irregular", _irregular()),
    ]
    masks: List[DemoMask] = []
    for name, m in shapes:
        masks.append(_wrap(f"shape_{name}_red", "R", f"Shape: {name} — RED LED", m))
        masks.append(_wrap(f"shape_{name}_blue", "B", f"Shape: {name} — BLUE LED", m))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic SYNTHETIC ROIs (CS-free stand-in for a Cellpose rois.npz)
# ─────────────────────────────────────────────────────────────────────────────


def synthetic_roi_labels(n: int = 24, seed: int = 1234,
                         min_size: int = 18, max_size: int = 45) -> np.ndarray:
    """Deterministic synthetic 'neuron' ROI label field on the projector canvas.

    A CS-free stand-in for a Cellpose ``rois.npz`` so the multi-target segments
    run on base-platform with NO CS dependency and NO hard-exit. Same args ->
    identical layout (and identical mask bytes downstream). Returns an int32
    (PROJ_H, PROJ_W) label image with blobs labelled 1..n (0 = background).

    ``n`` controls how many ROIs; ``min_size``/``max_size`` control the
    half-axis range so a single field can span small-to-large ROIs across the
    full FOV (use e.g. n=40, min_size=12, max_size=70 for a granular field).
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros((PROJ_H, PROJ_W), dtype=np.int32)
    margin = max(60, max_size + 20)
    for nid in range(1, n + 1):
        cx = int(rng.integers(margin, PROJ_W - margin))
        cy = int(rng.integers(margin, PROJ_H - margin))
        ax = int(rng.integers(min_size, max_size + 1))
        ay = int(rng.integers(min_size, max_size + 1))
        ang = int(rng.integers(0, 180))
        cv2.ellipse(labels, (cx, cy), (ax, ay), ang, 0, 360, int(nid), thickness=-1)
    return labels


def synthetic_neuron_rois(max_neurons: int = 20, seed: int = 1234) -> List[DemoMask]:
    """Light each synthetic ROI individually, RED then BLUE — the
    'address many independent targets' capability (CS-free)."""
    labels = synthetic_roi_labels(seed=seed)
    unique_ids = sorted({int(i) for i in np.unique(labels) if i > 0})[:max_neurons]
    masks: List[DemoMask] = []
    for nid in unique_ids:
        m = (labels == nid).astype(np.uint8) * 255
        masks.append(_wrap(f"synth_roi_{nid:02d}_red", "R", f"Synthetic ROI {nid} — RED (stim)", m))
        masks.append(_wrap(f"synth_roi_{nid:02d}_blue", "B", f"Synthetic ROI {nid} — BLUE (observe)", m))
    return masks


def synthetic_speed_ramp(seed: int = 1234) -> DemoMask:
    """All synthetic ROIs in one mask — alternated R<->B at varying rates by the
    runner to show the LED-switch envelope."""
    labels = synthetic_roi_labels(seed=seed)
    allm = (labels > 0).astype(np.uint8) * 255
    return _wrap("synth_speed_ramp_all", "R",
                 "All synthetic ROIs — alternating R<->B at varying rates", allm)


def synthetic_multi_target_temporal(seed: int = 1234) -> List[DemoMask]:
    """Disjoint R-mask + B-mask (stim vs observe subsets) alternating — the
    temporal-multiplex capability that future CS work relies on (CS-free)."""
    labels = synthetic_roi_labels(seed=seed)
    unique_ids = sorted({int(i) for i in np.unique(labels) if i > 0})
    if len(unique_ids) < 4:
        return []
    masks: List[DemoMask] = []
    splits = [(0.25, "stim_minority"), (0.50, "stim_half"), (0.75, "stim_majority")]
    for stim_frac, label in splits:
        n_stim = max(1, int(stim_frac * len(unique_ids)))
        stim_ids = set(unique_ids[:n_stim])
        observe_ids = set(unique_ids[n_stim:])
        red = np.zeros_like(labels, dtype=np.uint8)
        blue = np.zeros_like(labels, dtype=np.uint8)
        for nid in stim_ids:
            red[labels == nid] = 255
        for nid in observe_ids:
            blue[labels == nid] = 255
        masks.append(_wrap(f"synth_multiplex_{label}_R", "R",
                           f"Multiplex: {n_stim} stim ROIs — RED", red))
        masks.append(_wrap(f"synth_multiplex_{label}_B", "B",
                           f"Multiplex: {len(observe_ids)} obs ROIs — BLUE", blue))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Dot/ROI FIELD with per-ROI labels — for the density & scale-ramp sequence
# ─────────────────────────────────────────────────────────────────────────────


def dot_field_labels(dot_size_px: int = 2, spacing_px: int = 24,
                     arrangement: str = "grid", shape: str = "square",
                     seed: int = 0, max_dots: int | None = None) -> np.ndarray:
    """Dense field of small ROIs ("dots"), each UNIQUELY labelled (1..N), so a
    caller can colorize per-ROI (R/B/mix). Deterministic for a given (args).

    dot_size_px  half-extent grows the ROI from pixel-level (1-2) up to groups.
    spacing_px   grid pitch / scatter min-gap (controls density).
    arrangement  'grid' (regular lattice) | 'scatter' (seeded random).
    shape        'square' | 'circle'.
    max_dots     cap the ROI count (None = fill the FOV).

    Returns an int32 (PROJ_H, PROJ_W) label image (0 = background).
    """
    labels = np.zeros((PROJ_H, PROJ_W), dtype=np.int32)
    half = max(0, dot_size_px // 2)
    nid = 0

    def _stamp(x: int, y: int, _id: int) -> None:
        if shape == "circle" and dot_size_px >= 3:
            cv2.circle(labels, (x, y), max(1, dot_size_px // 2), _id, thickness=-1)
        else:
            labels[max(0, y - half): y + half + 1, max(0, x - half): x + half + 1] = _id

    if arrangement == "scatter":
        rng = np.random.default_rng(seed)
        n = max_dots if max_dots else 400
        m = spacing_px
        for _ in range(n):
            x = int(rng.integers(m, PROJ_W - m))
            y = int(rng.integers(m, PROJ_H - m))
            nid += 1
            _stamp(x, y, nid)
    else:  # grid
        for y in range(spacing_px, PROJ_H - spacing_px, spacing_px):
            for x in range(spacing_px, PROJ_W - spacing_px, spacing_px):
                nid += 1
                _stamp(x, y, nid)
                if max_dots and nid >= max_dots:
                    return labels
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-level addressability (dense dot grid)
# ─────────────────────────────────────────────────────────────────────────────


def pixel_grid_dense(dot_size_px: int = 3, spacing_px: int = 30) -> List[DemoMask]:
    """Dense grid of tiny dots — proves pixel-level control (~2300 individually
    addressable targets in the FOV). All-at-once in red, then blue."""
    mask = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
    half = dot_size_px // 2
    n_dots = 0
    for y in range(spacing_px, PROJ_H - spacing_px, spacing_px):
        for x in range(spacing_px, PROJ_W - spacing_px, spacing_px):
            mask[y - half : y + half + 1, x - half : x + half + 1] = 255
            n_dots += 1
    return [
        _wrap(f"pixel_grid_dense_{n_dots}dots_red", "R",
              f"Pixel-level addressability: {n_dots} dots ({dot_size_px}x{dot_size_px} px) — RED", mask),
        _wrap(f"pixel_grid_dense_{n_dots}dots_blue", "B",
              f"Pixel-level addressability: {n_dots} dots ({dot_size_px}x{dot_size_px} px) — BLUE", mask),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-target simultaneous (random scattered targets)
# ─────────────────────────────────────────────────────────────────────────────


def random_scattered_targets(n_targets: int = 20, target_radius: int = 25, seed: int = 42) -> List[DemoMask]:
    """N randomly-positioned ROIs lit simultaneously — many disjoint points at
    once. 3 seeded variants for visual interest (deterministic)."""
    masks: List[DemoMask] = []
    for variant, s in enumerate([seed, seed + 7, seed + 13]):
        rng = np.random.default_rng(s)
        mask = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
        for _ in range(n_targets):
            cx = int(rng.uniform(target_radius * 2, PROJ_W - target_radius * 2))
            cy = int(rng.uniform(target_radius * 2, PROJ_H - target_radius * 2))
            cv2.circle(mask, (cx, cy), target_radius, 255, thickness=-1)
        masks.append(_wrap(f"scatter_v{variant}_red", "R",
                           f"Scattered targets v{variant + 1}: {n_targets} simultaneous — RED", mask))
        masks.append(_wrap(f"scatter_v{variant}_blue", "B",
                           f"Scattered targets v{variant + 1}: {n_targets} simultaneous — BLUE", mask))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Spiral sweep (single target tracing a spiral)
# ─────────────────────────────────────────────────────────────────────────────


def spiral_sweep(n_steps: int = 40, target_radius: int = 30) -> List[DemoMask]:
    """Single target moving along a spiral, alternating R/B per step."""
    cx, cy = PROJ_W // 2, PROJ_H // 2
    masks: List[DemoMask] = []
    max_r = min(PROJ_W, PROJ_H) // 2 - target_radius - 20
    for i in range(n_steps):
        t = i / float(n_steps)
        r = max_r * t
        theta = 4 * np.pi * t  # 2 full revolutions
        x = int(cx + r * np.cos(theta))
        y = int(cy + r * np.sin(theta))
        mask = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
        cv2.circle(mask, (x, y), target_radius, 255, thickness=-1)
        led = "R" if i % 2 == 0 else "B"
        masks.append(_wrap(f"spiral_step_{i:02d}_{led}", led,
                           f"Spiral step {i + 1}/{n_steps} — {led} LED", mask))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Concentric rings (curve precision + multi-target-per-frame)
# ─────────────────────────────────────────────────────────────────────────────


def concentric_rings(n_steps: int = 16) -> List[DemoMask]:
    cx, cy = PROJ_W // 2, PROJ_H // 2
    max_r = min(PROJ_W, PROJ_H) // 2 - 20
    masks: List[DemoMask] = []
    for i in range(n_steps):
        mask = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
        n_rings = 3 + (i % 3)
        for j in range(n_rings):
            r = int(((j + 1) / (n_rings + 1)) * max_r * (0.4 + 0.6 * ((i + 1) / n_steps)))
            cv2.circle(mask, (cx, cy), r, 255, thickness=8)
        led = "R" if i % 2 == 0 else "B"
        masks.append(_wrap(f"rings_step_{i:02d}_{led}", led,
                           f"Concentric rings step {i + 1}/{n_steps} — {led} LED", mask))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Multi-shape composition (many ROIs of different shapes in one frame)
# ─────────────────────────────────────────────────────────────────────────────


def multi_shape_composition() -> List[DemoMask]:
    masks: List[DemoMask] = []
    # Composition 1: 5 circles in a row
    m1 = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
    positions = [(384, 540), (768, 540), (1152, 540), (1536, 540), (192, 540)]
    sizes = [120, 100, 110, 95, 80]
    for (cx, cy), r in zip(positions, sizes):
        cv2.circle(m1, (cx, cy), r, 255, -1)
    masks.append(_wrap("comp_5circles_red", "R", "5 circles, row layout — RED", m1))
    masks.append(_wrap("comp_5circles_blue", "B", "5 circles, row layout — BLUE", m1))

    # Composition 2: center + 8 satellites
    m2 = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
    cv2.circle(m2, (PROJ_W // 2, PROJ_H // 2), 150, 255, -1)
    for dx, dy in [(-400, -300), (0, -300), (400, -300),
                   (-400, 0),                (400, 0),
                   (-400, 300),  (0, 300), (400, 300)]:
        cv2.circle(m2, (PROJ_W // 2 + dx, PROJ_H // 2 + dy), 50, 255, -1)
    masks.append(_wrap("comp_satellite_red", "R", "Center + 8 satellites — RED", m2))
    masks.append(_wrap("comp_satellite_blue", "B", "Center + 8 satellites — BLUE", m2))

    # Composition 3: 4 different shapes in 4 quadrants
    m3 = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
    qw, qh = PROJ_W // 4, PROJ_H // 4
    pts = np.array([[qw, qh - 80], [qw - 80, qh + 80], [qw + 80, qh + 80]], np.int32)
    cv2.fillPoly(m3, [pts], 255)
    cx, cy = 3 * qw, qh
    pts = np.array([[cx + int(80 * np.cos(t)), cy + int(80 * np.sin(t))]
                    for t in np.linspace(0, 2 * np.pi, 7)[:-1]], np.int32)
    cv2.fillPoly(m3, [pts], 255)
    cv2.rectangle(m3, (qw - 80, 3 * qh - 80), (qw + 80, 3 * qh + 80), 255, -1)
    cx, cy = 3 * qw, 3 * qh
    pts = []
    for i in range(10):
        r = 80 if i % 2 == 0 else 40
        t = i * np.pi / 5 - np.pi / 2
        pts.append([cx + int(r * np.cos(t)), cy + int(r * np.sin(t))])
    cv2.fillPoly(m3, [np.array(pts, np.int32)], 255)
    masks.append(_wrap("comp_4shapes_red", "R", "4 shapes in 4 quadrants — RED", m3))
    masks.append(_wrap("comp_4shapes_blue", "B", "4 shapes in 4 quadrants — BLUE", m3))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-density tiers
# ─────────────────────────────────────────────────────────────────────────────


def pixel_density_tiers() -> List[DemoMask]:
    masks: List[DemoMask] = []
    tiers = [(1, 50), (2, 30), (3, 20), (5, 15)]
    for dot_size, spacing in tiers:
        mask = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
        half = max(1, dot_size // 2)
        n_dots = 0
        for y in range(spacing, PROJ_H - spacing, spacing):
            for x in range(spacing, PROJ_W - spacing, spacing):
                mask[max(0, y - half) : min(PROJ_H, y + half + 1),
                     max(0, x - half) : min(PROJ_W, x + half + 1)] = 255
                n_dots += 1
        masks.append(_wrap(f"density_{dot_size}px_{n_dots}dots_red", "R",
                           f"Density tier: {n_dots} dots ({dot_size}x{dot_size} px) — RED", mask))
        masks.append(_wrap(f"density_{dot_size}px_{n_dots}dots_blue", "B",
                           f"Density tier: {n_dots} dots ({dot_size}x{dot_size} px) — BLUE", mask))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Lissajous path (single tiny target tracing a complex curve)
# ─────────────────────────────────────────────────────────────────────────────


def lissajous_path(n_steps: int = 80, target_size: int = 12,
                   freq_x: int = 3, freq_y: int = 4) -> List[DemoMask]:
    cx, cy = PROJ_W // 2, PROJ_H // 2
    ax, ay = PROJ_W // 3, PROJ_H // 3
    masks: List[DemoMask] = []
    half = target_size // 2
    for i in range(n_steps):
        t = (i / n_steps) * 2 * np.pi
        x = int(cx + ax * np.sin(freq_x * t))
        y = int(cy + ay * np.sin(freq_y * t))
        mask = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
        mask[max(0, y - half) : min(PROJ_H, y + half),
             max(0, x - half) : min(PROJ_W, x + half)] = 255
        led = "R" if (i // 4) % 2 == 0 else "B"
        masks.append(_wrap(f"lissajous_step_{i:02d}_{led}", led,
                           f"Lissajous {freq_x}:{freq_y} step {i + 1}/{n_steps} — {led}", mask))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Multi-target choreography (many coordinated moving targets)
# ─────────────────────────────────────────────────────────────────────────────


def multi_target_choreography(n_steps: int = 40, n_targets: int = 8,
                              target_size: int = 18) -> List[DemoMask]:
    cx, cy = PROJ_W // 2, PROJ_H // 2
    radius = min(PROJ_W, PROJ_H) // 3
    half = target_size // 2
    masks: List[DemoMask] = []
    for i in range(n_steps):
        t = (i / n_steps) * 2 * np.pi
        mask = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
        for k in range(n_targets):
            theta = t + (k * 2 * np.pi / n_targets)
            x = int(cx + radius * np.cos(theta))
            y = int(cy + radius * np.sin(theta))
            mask[max(0, y - half) : min(PROJ_H, y + half),
                 max(0, x - half) : min(PROJ_W, x + half)] = 255
        led = "R" if (i // 3) % 2 == 0 else "B"
        masks.append(_wrap(f"choreo_step_{i:02d}_{led}", led,
                           f"{n_targets} targets rotating, step {i + 1}/{n_steps} — {led}", mask))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic wave (sinusoidal band traversing the FOV)
# ─────────────────────────────────────────────────────────────────────────────


def dynamic_wave(n_steps: int = 50) -> List[DemoMask]:
    masks: List[DemoMask] = []
    for i in range(n_steps):
        mask = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
        phase = (i / n_steps) * 4 * np.pi
        for x in range(0, PROJ_W, 4):
            y_center = int(PROJ_H // 2 + (PROJ_H // 4) * np.sin((x / 80.0) + phase))
            cv2.line(mask, (x, y_center - 8), (x, y_center + 8), 255, thickness=4)
        led = "R" if (i // 5) % 2 == 0 else "B"
        masks.append(_wrap(f"wave_step_{i:02d}_{led}", led,
                           f"Dynamic wave step {i + 1}/{n_steps} — {led}", mask))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Random scatter animated (fresh random config each frame)
# ─────────────────────────────────────────────────────────────────────────────


def random_scatter_animated(n_steps: int = 30, n_targets: int = 100,
                            target_radius: int = 8, base_seed: int = 100) -> List[DemoMask]:
    masks: List[DemoMask] = []
    for i in range(n_steps):
        rng = np.random.default_rng(base_seed + i)
        mask = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
        xs = rng.integers(target_radius * 2, PROJ_W - target_radius * 2, size=n_targets)
        ys = rng.integers(target_radius * 2, PROJ_H - target_radius * 2, size=n_targets)
        for x, y in zip(xs, ys):
            cv2.circle(mask, (int(x), int(y)), target_radius, 255, -1)
        led = "R" if (i // 3) % 2 == 0 else "B"
        masks.append(_wrap(f"scatter_anim_step_{i:02d}_{led}", led,
                           f"Animated scatter step {i + 1}/{n_steps}: {n_targets} fresh targets — {led}", mask))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Persistence — write grayscale PNGs + a sha256 manifest
# ─────────────────────────────────────────────────────────────────────────────


def write_library(masks: Iterator[DemoMask], out_dir: Path) -> List[Tuple[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Tuple[str, str]] = []
    for m in masks:
        cv2.imwrite(str(out_dir / f"{m.name}.png"), m.img)
        manifest.append((m.name, m.sha256))
    return manifest


__all__ = [
    "PROJ_W", "PROJ_H", "DemoMask",
    "spatial_sweep", "arbitrary_shapes",
    # CS-free synthetic ROI segments (replace the old Cellpose-coupled ones):
    "synthetic_roi_labels", "synthetic_neuron_rois",
    "synthetic_speed_ramp", "synthetic_multi_target_temporal",
    "dot_field_labels",
    # Pure-geometry rich-visualization segments:
    "pixel_grid_dense", "random_scattered_targets",
    "spiral_sweep", "concentric_rings", "multi_shape_composition",
    "pixel_density_tiers", "lissajous_path",
    "multi_target_choreography", "dynamic_wave", "random_scatter_animated",
    "write_library",
]
