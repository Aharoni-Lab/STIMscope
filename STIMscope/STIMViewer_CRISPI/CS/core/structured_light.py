"""Structured-light calibration subsystem.

Extracted from ``STIMViewer_CRISPI/calibration.py`` during the L3 audit. Calibration.py focuses on ArUco-marker
homography; this module owns the orthogonal Gray-code + phase-shift
+ inverse-LUT pipeline used for high-coverage projector↔camera
calibration when ArUco is insufficient (e.g. wide-FOV bring-up).

Public surface — used by ``qt_interface.py`` and ``gpu_ui.py``:

  generate_gray_code_patterns        — Gray code pattern bank
  generate_phase_shift_patterns      — sinusoidal phase patterns
  save_structured_light_patterns     — write bank to disk
  decode_gray_code_from_files        — captures → forward LUT (cam→proj)
  decode_phase_shift_from_files      — phase captures → subpixel LUT
  invert_cam_to_proj_lut             — forward LUT → inverse LUT (proj→cam)
  prewarp_with_inverse_lut           — apply inverse LUT to a mask
  visualize_lut_quality              — coverage diagnostic image
  SL_PATTERN_DIR                     — legacy disk path constant

``calibration.py`` re-exports these symbols verbatim so existing
``from calibration import generate_gray_code_patterns`` callers keep
working. New callers should import directly from this module.

No behavior change vs the original location — the logger handle is
the only swap (each function now logs through ``core.logging_config``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from core.logging_config import get_logger

logger = get_logger(__name__)


# Legacy disk location for saved patterns. Kept here (not in core.paths)
# because qt_interface.py + calibration.py both reference the same
# ``Assets/Generated/sl_patterns/`` tree and the broader migration to
# core.paths is rolling per module.
_CRISPI_ROOT = Path(__file__).resolve().parents[4]  # …/STIMViewer_CRISPI/
SL_PATTERN_DIR = _CRISPI_ROOT / "Assets" / "Generated" / "sl_patterns"


def generate_gray_code_patterns(
    proj_w: int, proj_h: int,
) -> list:
    """Generate standard Gray code patterns for structured-light calibration.

    Returns a list of dicts, each with keys:
      - 'image': (proj_h, proj_w, 3) uint8 BGR image
      - 'bit': int bit index
      - 'axis': 'x' or 'y'
      - 'inverted': bool
    """
    patterns = []
    n_bits_x = int(np.ceil(np.log2(max(proj_w, 2))))
    n_bits_y = int(np.ceil(np.log2(max(proj_h, 2))))

    white = np.full((proj_h, proj_w), 255, dtype=np.uint8)
    black = np.zeros((proj_h, proj_w), dtype=np.uint8)
    patterns.append({'image': cv2.cvtColor(white, cv2.COLOR_GRAY2BGR),
                     'bit': -1, 'axis': 'threshold', 'inverted': False})
    patterns.append({'image': cv2.cvtColor(black, cv2.COLOR_GRAY2BGR),
                     'bit': -2, 'axis': 'threshold', 'inverted': True})

    def _binary_to_gray(n):
        return n ^ (n >> 1)

    for bit in range(n_bits_x):
        img = np.zeros((proj_h, proj_w), dtype=np.uint8)
        for x in range(proj_w):
            gray_val = _binary_to_gray(x)
            if (gray_val >> (n_bits_x - 1 - bit)) & 1:
                img[:, x] = 255
        img_inv = 255 - img
        patterns.append({'image': cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                         'bit': bit, 'axis': 'x', 'inverted': False})
        patterns.append({'image': cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR),
                         'bit': bit, 'axis': 'x', 'inverted': True})

    for bit in range(n_bits_y):
        img = np.zeros((proj_h, proj_w), dtype=np.uint8)
        for y in range(proj_h):
            gray_val = _binary_to_gray(y)
            if (gray_val >> (n_bits_y - 1 - bit)) & 1:
                img[y, :] = 255
        img_inv = 255 - img
        patterns.append({'image': cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                         'bit': bit, 'axis': 'y', 'inverted': False})
        patterns.append({'image': cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR),
                         'bit': bit, 'axis': 'y', 'inverted': True})

    logger.info("Generated %d Gray code patterns (%d X-bits + %d Y-bits + 2 threshold)",
                len(patterns), n_bits_x, n_bits_y)
    return patterns


def generate_phase_shift_patterns(
    proj_w: int, proj_h: int,
    num_phases: int = 3,
    cycles_x: int = 1,
    cycles_y: int = 1,
    gamma: float = 1.0,
) -> list:
    """Generate sinusoidal phase-shift patterns for subpixel refinement.

    Returns a list of dicts with keys:
      - 'image': (proj_h, proj_w, 3) uint8 BGR
      - 'type': 'phase'
      - 'phase_idx': int
      - 'axis': 'x' or 'y'
      - 'shift_rad': float
    """
    patterns = []
    xs = np.arange(proj_w, dtype=np.float64)
    ys = np.arange(proj_h, dtype=np.float64)

    for axis, coords, n_cycles, length in [
        ('x', xs, cycles_x, proj_w),
        ('y', ys, cycles_y, proj_h),
    ]:
        for phase_idx in range(num_phases):
            shift = 2.0 * np.pi * phase_idx / num_phases
            freq = 2.0 * np.pi * n_cycles / length
            if axis == 'x':
                vals = 0.5 + 0.5 * np.cos(freq * xs + shift)
                img = np.tile(vals, (proj_h, 1))
            else:
                vals = 0.5 + 0.5 * np.cos(freq * ys + shift)
                img = np.tile(vals.reshape(-1, 1), (1, proj_w))
            if gamma != 1.0:
                img = np.power(img, gamma)
            img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
            patterns.append({
                'image': cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR),
                'type': 'phase',
                'phase_idx': phase_idx,
                'axis': axis,
                'shift_rad': shift,
            })

    logger.info("Generated %d phase-shift patterns (%d phases x 2 axes)",
                len(patterns), num_phases)
    return patterns


def save_structured_light_patterns(patterns: list) -> list:
    """Save pattern images to disk.

    Returns list of file paths (same order as input patterns).
    """
    SL_PATTERN_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, pat in enumerate(patterns):
        img = pat.get('image')
        if img is None:
            paths.append('')
            continue
        fname = SL_PATTERN_DIR / f"sl_pattern_{i:03d}.png"
        cv2.imwrite(str(fname), img)
        paths.append(str(fname))
    logger.info("Saved %d structured-light patterns to %s", len(paths), SL_PATTERN_DIR)
    return paths


def decode_gray_code_from_files(
    capture_paths: list,
    meta_list: list,
    cam_h: int, cam_w: int,
    proj_w: int, proj_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode captured Gray code images to per-pixel projector coordinates.

    Returns (proj_x_of_cam, proj_y_of_cam) — both (cam_h, cam_w) float32.
    Pixels where decoding failed are set to -1.
    """
    thresh_imgs = {}
    x_pairs = {}
    y_pairs = {}

    for path, meta in zip(capture_paths, meta_list):
        if not path or not isinstance(meta, dict):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (cam_h, cam_w):
            img = cv2.resize(img, (cam_w, cam_h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)

        bit = meta.get('bit', -99)
        axis = meta.get('axis', '')
        inverted = meta.get('inverted', False)

        if axis == 'threshold':
            thresh_imgs['white' if not inverted else 'black'] = img
            continue

        store = x_pairs if axis == 'x' else y_pairs
        if bit not in store:
            store[bit] = [None, None]
        store[bit][1 if inverted else 0] = img

    white = thresh_imgs.get('white')
    black = thresh_imgs.get('black')
    if white is not None and black is not None:
        shadow_mask = (white - black) < 10.0
    else:
        shadow_mask = np.zeros((cam_h, cam_w), dtype=bool)

    def _decode_axis(pairs, n_proj):
        n_bits = int(np.ceil(np.log2(max(n_proj, 2))))
        decoded = np.zeros((cam_h, cam_w), dtype=np.int32)
        for bit in range(n_bits):
            if bit not in pairs or pairs[bit][0] is None or pairs[bit][1] is None:
                continue
            normal, inverted = pairs[bit]
            bit_val = ((normal - inverted) > 0).astype(np.int32)
            decoded |= (bit_val << (n_bits - 1 - bit))
        result = decoded.copy()
        shift = 1
        while shift < n_bits:
            result ^= (result >> shift)
            shift <<= 1
        return result.astype(np.float32)

    proj_x = _decode_axis(x_pairs, proj_w)
    proj_y = _decode_axis(y_pairs, proj_h)

    proj_x[shadow_mask] = -1.0
    proj_y[shadow_mask] = -1.0
    proj_x[(proj_x < 0) | (proj_x >= proj_w)] = -1.0
    proj_y[(proj_y < 0) | (proj_y >= proj_h)] = -1.0

    valid = (proj_x >= 0) & (proj_y >= 0)
    logger.info("Gray code decoded: %d/%d valid pixels (%.1f%%)",
                int(valid.sum()), cam_h * cam_w,
                100.0 * valid.sum() / (cam_h * cam_w))
    return proj_x, proj_y


def decode_phase_shift_from_files(
    capture_paths: list,
    meta_list: list,
    cam_h: int, cam_w: int,
    proj_w: int, proj_h: int,
    num_phases: int = 3,
    amp_thresh: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode phase-shift captures to subpixel projector coordinates.

    Returns (px_phase, py_phase, amp_x, amp_y) — all (cam_h, cam_w) float32.
    px_phase/py_phase contain projector pixel coordinates (-1 where invalid).
    amp_x/amp_y contain modulation amplitude (for quality gating).
    """
    x_imgs = []
    y_imgs = []

    for path, meta in zip(capture_paths, meta_list):
        if not path or not isinstance(meta, dict):
            continue
        if meta.get('type') != 'phase':
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (cam_h, cam_w):
            img = cv2.resize(img, (cam_w, cam_h), interpolation=cv2.INTER_AREA)
        axis = meta.get('axis', 'x')
        shift = meta.get('shift_rad', 0.0)
        store = x_imgs if axis == 'x' else y_imgs
        store.append((shift, img.astype(np.float64)))

    def _decode_phase_axis(imgs, n_proj, n_cycles):
        if len(imgs) < 2:
            return (np.full((cam_h, cam_w), -1, dtype=np.float32),
                    np.zeros((cam_h, cam_w), dtype=np.float32))
        sin_sum = np.zeros((cam_h, cam_w), dtype=np.float64)
        cos_sum = np.zeros((cam_h, cam_w), dtype=np.float64)
        for shift, img in imgs:
            sin_sum += img * np.sin(shift)
            cos_sum += img * np.cos(shift)
        phase = np.arctan2(-sin_sum, cos_sum)
        phase = (phase + np.pi) / (2.0 * np.pi)
        px = phase * (n_proj / max(n_cycles, 1))
        amp = 2.0 * np.sqrt(sin_sum**2 + cos_sum**2) / len(imgs)
        return px.astype(np.float32), amp.astype(np.float32)

    px_x, amp_x = _decode_phase_axis(x_imgs, proj_w, 1)
    px_y, amp_y = _decode_phase_axis(y_imgs, proj_h, 1)

    px_x[amp_x < amp_thresh] = -1.0
    px_y[amp_y < amp_thresh] = -1.0

    return px_x, px_y, amp_x, amp_y


def invert_cam_to_proj_lut(
    proj_x_of_cam: np.ndarray,
    proj_y_of_cam: np.ndarray,
    proj_w: int, proj_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert forward LUT (cam→proj) to inverse LUT (proj→cam).

    Forward: proj_x_of_cam[cam_y, cam_x] = proj_x
    Inverse: cam_from_proj_x[proj_y, proj_x] = cam_x

    Returns (inv_x, inv_y) — both (proj_h, proj_w) float32, -1 where unmapped.
    """
    cam_h, cam_w = proj_x_of_cam.shape
    inv_x = np.full((proj_h, proj_w), -1.0, dtype=np.float32)
    inv_y = np.full((proj_h, proj_w), -1.0, dtype=np.float32)

    valid = (proj_x_of_cam >= 0) & (proj_y_of_cam >= 0)
    cam_ys, cam_xs = np.where(valid)
    px = proj_x_of_cam[valid].astype(np.int32)
    py = proj_y_of_cam[valid].astype(np.int32)

    mask = (px >= 0) & (px < proj_w) & (py >= 0) & (py < proj_h)
    px, py = px[mask], py[mask]
    cx, cy = cam_xs[mask].astype(np.float32), cam_ys[mask].astype(np.float32)

    inv_x[py, px] = cx
    inv_y[py, px] = cy

    unmapped = (inv_x < 0)
    if unmapped.sum() > 0 and unmapped.sum() < proj_h * proj_w:
        from scipy.ndimage import distance_transform_edt
        _, nearest = distance_transform_edt(unmapped, return_distances=True,
                                             return_indices=True)
        fill_mask = unmapped & ((_ < 5))
        inv_x[fill_mask] = inv_x[nearest[0][fill_mask], nearest[1][fill_mask]]
        inv_y[fill_mask] = inv_y[nearest[0][fill_mask], nearest[1][fill_mask]]

    mapped = (inv_x >= 0).sum()
    logger.info("LUT inverted: %d/%d projector pixels mapped (%.1f%%)",
                mapped, proj_h * proj_w,
                100.0 * mapped / (proj_h * proj_w))
    return inv_x, inv_y


def prewarp_with_inverse_lut(
    image_bgr: np.ndarray,
    inv_x: np.ndarray,
    inv_y: np.ndarray,
    proj_w: int, proj_h: int,
) -> np.ndarray:
    """Warp a camera-space image to projector-space using inverse LUT.

    inv_x[proj_y, proj_x] = cam_x (where to sample from camera image)
    inv_y[proj_y, proj_x] = cam_y

    Returns (proj_h, proj_w, 3) uint8 BGR image ready for projection.
    """
    map_x = inv_x.astype(np.float32)
    map_y = inv_y.astype(np.float32)
    invalid = (map_x < 0) | (map_y < 0)
    map_x[invalid] = -1
    map_y[invalid] = -1
    warped = cv2.remap(image_bgr, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0))
    return warped


def visualize_lut_quality(
    inv_x: np.ndarray,
    inv_y: np.ndarray,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Generate a diagnostic visualization of LUT quality.

    Shows mapped pixels in green, unmapped in red, with a grid overlay.
    Returns (H, W, 3) uint8 BGR image.
    """
    h, w = inv_x.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    valid = (inv_x >= 0) & (inv_y >= 0)
    vis[valid] = (0, 180, 0)
    vis[~valid] = (0, 0, 120)

    for y in range(0, h, 64):
        vis[y, :] = np.where(vis[y, :] > 0, vis[y, :] // 2, vis[y, :])
    for x in range(0, w, 64):
        vis[:, x] = np.where(vis[:, x] > 0, vis[:, x] // 2, vis[:, x])

    pct = 100.0 * valid.sum() / max(valid.size, 1)
    cv2.putText(vis, f"Coverage: {pct:.1f}% ({int(valid.sum())}/{valid.size})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if output_path:
        cv2.imwrite(str(output_path), vis)
        logger.info("LUT diagnostic saved: %s", output_path)
    return vis
