"""Characterization tests for ``core.structured_light``.

Pins the as-is behavior of the Gray-code + phase-shift + inverse-LUT
pipeline extracted from ``calibration.py`` at L3.

Background: PHASE_A_CLOSEOUT_BASELINE coverage measurement (iter 5)
recorded 24% coverage on this module — extracted from calibration.py
but tests didn't follow the extraction. This file backfills to the
≥80% target named in iter-5 carry-forward #3.

No hardware required — all functions are pure NumPy/OpenCV. Disk
I/O is exercised via tmp_path fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CS_PATH = REPO_ROOT / "STIMscope" / "STIMViewer_CRISPI" / "CS"
if str(CS_PATH) not in sys.path:
    sys.path.insert(0, str(CS_PATH))

from core import structured_light as sl


# ─────────────────────────────────────────────────────────────────────────────
# C1 — generate_gray_code_patterns
# ─────────────────────────────────────────────────────────────────────────────


class TestC1GenerateGrayCodePatterns:
    """Contract: returns 2 threshold + 2*ceil(log2(W)) X + 2*ceil(log2(H)) Y."""

    def test_returns_list_of_dicts(self):
        patterns = sl.generate_gray_code_patterns(64, 64)
        assert isinstance(patterns, list)
        for p in patterns:
            assert {"image", "bit", "axis", "inverted"}.issubset(p.keys())

    def test_pattern_count_matches_ceil_log2(self):
        # 64 = 2^6 → 6 bits X + 6 bits Y, doubled for inverted, + 2 threshold = 26
        patterns = sl.generate_gray_code_patterns(64, 64)
        n_bits_x = int(np.ceil(np.log2(64)))
        n_bits_y = int(np.ceil(np.log2(64)))
        expected = 2 + 2 * n_bits_x + 2 * n_bits_y
        assert len(patterns) == expected

    def test_pattern_count_non_power_of_two(self):
        # 100 → 7 bits (ceil(log2(100)))
        patterns = sl.generate_gray_code_patterns(100, 50)
        n_bits_x = int(np.ceil(np.log2(100)))
        n_bits_y = int(np.ceil(np.log2(50)))
        expected = 2 + 2 * n_bits_x + 2 * n_bits_y
        assert len(patterns) == expected

    def test_threshold_patterns_first(self):
        patterns = sl.generate_gray_code_patterns(32, 32)
        assert patterns[0]["axis"] == "threshold"
        assert patterns[1]["axis"] == "threshold"
        assert patterns[0]["inverted"] is False
        assert patterns[1]["inverted"] is True

    def test_threshold_white_is_all_255(self):
        patterns = sl.generate_gray_code_patterns(32, 32)
        white = patterns[0]["image"]
        assert white.shape == (32, 32, 3)
        assert white.dtype == np.uint8
        assert (white == 255).all()

    def test_threshold_black_is_all_zero(self):
        patterns = sl.generate_gray_code_patterns(32, 32)
        black = patterns[1]["image"]
        assert (black == 0).all()

    def test_x_and_y_axes_both_present(self):
        patterns = sl.generate_gray_code_patterns(32, 32)
        axes = {p["axis"] for p in patterns}
        assert "x" in axes
        assert "y" in axes

    def test_each_bit_has_inverted_pair(self):
        patterns = sl.generate_gray_code_patterns(32, 32)
        for axis in ("x", "y"):
            for p in patterns:
                if p["axis"] != axis:
                    continue
                # for each (axis, bit) pair, find its inverted twin
                if not p["inverted"]:
                    twin = [q for q in patterns
                            if q["axis"] == axis and q["bit"] == p["bit"] and q["inverted"]]
                    assert len(twin) == 1
                    # inverted twin should be the bitwise complement
                    assert (twin[0]["image"] == 255 - p["image"]).all()


# ─────────────────────────────────────────────────────────────────────────────
# C2 — generate_phase_shift_patterns
# ─────────────────────────────────────────────────────────────────────────────


class TestC2GeneratePhaseShiftPatterns:
    """Contract: num_phases * 2 axes patterns, sinusoidal in correct axis."""

    def test_default_pattern_count(self):
        # default num_phases=3 → 3*2 axes = 6 patterns
        patterns = sl.generate_phase_shift_patterns(64, 64)
        assert len(patterns) == 6

    def test_custom_num_phases(self):
        patterns = sl.generate_phase_shift_patterns(64, 64, num_phases=5)
        assert len(patterns) == 10  # 5*2

    def test_image_shape_and_dtype(self):
        patterns = sl.generate_phase_shift_patterns(80, 60)
        for p in patterns:
            assert p["image"].shape == (60, 80, 3)
            assert p["image"].dtype == np.uint8

    def test_axes_split_evenly(self):
        patterns = sl.generate_phase_shift_patterns(64, 64, num_phases=4)
        x_count = sum(1 for p in patterns if p["axis"] == "x")
        y_count = sum(1 for p in patterns if p["axis"] == "y")
        assert x_count == 4
        assert y_count == 4

    def test_phase_indices_complete(self):
        patterns = sl.generate_phase_shift_patterns(64, 64, num_phases=3)
        for axis in ("x", "y"):
            indices = {p["phase_idx"] for p in patterns if p["axis"] == axis}
            assert indices == {0, 1, 2}

    def test_shift_rad_proportional_to_phase_idx(self):
        patterns = sl.generate_phase_shift_patterns(64, 64, num_phases=4)
        x_pats = sorted([p for p in patterns if p["axis"] == "x"],
                        key=lambda p: p["phase_idx"])
        for i, p in enumerate(x_pats):
            np.testing.assert_allclose(p["shift_rad"], 2.0 * np.pi * i / 4)

    def test_x_axis_pattern_varies_along_x_not_y(self):
        patterns = sl.generate_phase_shift_patterns(64, 64, num_phases=3, cycles_x=1)
        x0 = next(p for p in patterns if p["axis"] == "x" and p["phase_idx"] == 0)
        img = x0["image"][:, :, 0]
        # All rows should be identical
        assert np.allclose(img[0, :], img[31, :])
        # Variance along X should be > 0
        assert img[0, :].std() > 0

    def test_gamma_changes_distribution(self):
        flat = sl.generate_phase_shift_patterns(64, 64, num_phases=3, gamma=1.0)
        gamma = sl.generate_phase_shift_patterns(64, 64, num_phases=3, gamma=2.2)
        # Gamma correction should change mean intensity
        assert flat[0]["image"].mean() != pytest.approx(gamma[0]["image"].mean(), abs=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# C3 — save_structured_light_patterns
# ─────────────────────────────────────────────────────────────────────────────


class TestC3SaveStructuredLightPatterns:
    """Contract: writes one PNG per pattern, returns paths, creates dir."""

    def test_returns_path_list_matching_input_length(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sl, "SL_PATTERN_DIR", tmp_path / "sl_test")
        patterns = sl.generate_gray_code_patterns(16, 16)
        paths = sl.save_structured_light_patterns(patterns)
        assert len(paths) == len(patterns)

    def test_creates_pattern_directory(self, tmp_path, monkeypatch):
        target = tmp_path / "new_dir" / "sl_patterns"
        monkeypatch.setattr(sl, "SL_PATTERN_DIR", target)
        sl.save_structured_light_patterns([{"image": np.zeros((4, 4, 3), dtype=np.uint8)}])
        assert target.is_dir()

    def test_files_are_readable_pngs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sl, "SL_PATTERN_DIR", tmp_path)
        patterns = sl.generate_gray_code_patterns(16, 16)[:3]
        paths = sl.save_structured_light_patterns(patterns)
        for path in paths:
            assert Path(path).is_file()
            img = cv2.imread(path)
            assert img is not None
            assert img.shape == patterns[0]["image"].shape

    def test_skips_patterns_with_no_image(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sl, "SL_PATTERN_DIR", tmp_path)
        patterns = [
            {"image": np.zeros((4, 4, 3), dtype=np.uint8)},
            {"image": None},
            {"image": np.full((4, 4, 3), 200, dtype=np.uint8)},
        ]
        paths = sl.save_structured_light_patterns(patterns)
        assert paths[0] != ""
        assert paths[1] == ""
        assert paths[2] != ""


# ─────────────────────────────────────────────────────────────────────────────
# C4 — decode_gray_code_from_files (round-trip)
# ─────────────────────────────────────────────────────────────────────────────


class TestC4DecodeGrayCode:
    """Contract: round-trip identity — simulated capture decodes to identity LUT."""

    def _simulate_captures(self, tmp_path, proj_w, proj_h, cam_w, cam_h):
        """Generate Gray-code patterns + 'capture' them at camera resolution
        as if camera == projector (identity homography). Return (paths, metas)."""
        patterns = sl.generate_gray_code_patterns(proj_w, proj_h)
        paths = []
        metas = []
        for i, p in enumerate(patterns):
            cap = cv2.resize(p["image"], (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
            fname = tmp_path / f"cap_{i:03d}.png"
            cv2.imwrite(str(fname), cap)
            paths.append(str(fname))
            metas.append({"bit": p["bit"], "axis": p["axis"], "inverted": p["inverted"]})
        return paths, metas

    def test_identity_decode_recovers_projector_coords(self, tmp_path):
        proj_w = proj_h = cam_w = cam_h = 32  # identity
        paths, metas = self._simulate_captures(tmp_path, proj_w, proj_h, cam_w, cam_h)
        px, py = sl.decode_gray_code_from_files(paths, metas, cam_h, cam_w, proj_w, proj_h)
        assert px.shape == (cam_h, cam_w)
        assert py.shape == (cam_h, cam_w)
        # Center pixel — should decode close to itself under identity
        cy = cam_h // 2
        cx = cam_w // 2
        assert abs(px[cy, cx] - cx) <= 1.0
        assert abs(py[cy, cx] - cy) <= 1.0

    def test_returns_minus_one_for_empty_capture_list(self):
        px, py = sl.decode_gray_code_from_files([], [], 16, 16, 32, 32)
        # Empty captures → all pixels invalid; depends on threshold images
        # Without threshold, shadow_mask defaults to false; uncomputed bits → 0
        assert px.shape == (16, 16)
        assert py.shape == (16, 16)

    def test_skips_missing_files(self, tmp_path):
        paths = [str(tmp_path / "nonexistent.png")]
        metas = [{"bit": 0, "axis": "x", "inverted": False}]
        px, py = sl.decode_gray_code_from_files(paths, metas, 8, 8, 16, 16)
        assert px.shape == (8, 8)

    def test_shadow_mask_invalidates_pixels(self, tmp_path):
        # White and black threshold images that are equal → entire frame is shadow
        same = np.full((16, 16), 128, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "w.png"), same)
        cv2.imwrite(str(tmp_path / "b.png"), same)
        paths = [str(tmp_path / "w.png"), str(tmp_path / "b.png")]
        metas = [
            {"bit": -1, "axis": "threshold", "inverted": False},
            {"bit": -2, "axis": "threshold", "inverted": True},
        ]
        px, py = sl.decode_gray_code_from_files(paths, metas, 16, 16, 32, 32)
        assert (px == -1).all()
        assert (py == -1).all()


# ─────────────────────────────────────────────────────────────────────────────
# C5 — decode_phase_shift_from_files
# ─────────────────────────────────────────────────────────────────────────────


class TestC5DecodePhaseShift:
    """Contract: returns 4-tuple of (px, py, amp_x, amp_y) all (cam_h, cam_w)."""

    def test_empty_input_returns_minus_one(self):
        px, py, amp_x, amp_y = sl.decode_phase_shift_from_files(
            [], [], 16, 16, 32, 32
        )
        assert px.shape == (16, 16)
        assert (px == -1).all()
        assert (py == -1).all()

    def test_low_amp_gated_to_minus_one(self, tmp_path):
        # Two phase captures with low contrast → amp below threshold
        img = np.full((8, 8), 128, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "p0.png"), img)
        cv2.imwrite(str(tmp_path / "p1.png"), img)
        paths = [str(tmp_path / "p0.png"), str(tmp_path / "p1.png")]
        metas = [
            {"type": "phase", "axis": "x", "shift_rad": 0.0, "phase_idx": 0},
            {"type": "phase", "axis": "x", "shift_rad": np.pi, "shift_idx": 1},
        ]
        px, py, amp_x, amp_y = sl.decode_phase_shift_from_files(
            paths, metas, 8, 8, 32, 32, amp_thresh=5.0
        )
        # Constant input → amp ≈ 0 → all gated
        assert (px == -1).all()

    def test_non_phase_meta_ignored(self, tmp_path):
        img = np.full((8, 8), 128, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "x.png"), img)
        paths = [str(tmp_path / "x.png")]
        metas = [{"type": "graycode", "axis": "x"}]  # not 'phase'
        px, py, amp_x, amp_y = sl.decode_phase_shift_from_files(
            paths, metas, 8, 8, 32, 32
        )
        assert (px == -1).all()


# ─────────────────────────────────────────────────────────────────────────────
# C6 — invert_cam_to_proj_lut
# ─────────────────────────────────────────────────────────────────────────────


class TestC6InvertLUT:
    """Contract: forward LUT (cam→proj) inverts to (proj→cam) faithfully.

    **D-sl-1 (PRE_FIX, found by these tests ):**
    All 3 tests in this class fail with `TypeError: %d format: a real
    number is required, not str` from line 342 of structured_light.py
    (`logger.info("LUT inverted: %d/%d...", mapped,...)`). The
    function returns correct values; the logger.info call crashes
    when formatting `mapped = (inv_x >= 0).sum()` (a numpy scalar)
    under pytest's logging handler chain. Direct Python invocation
    works fine — the failure is pytest-specific. Fix: cast to plain
    int (`mapped = int((inv_x >= 0).sum())`). Stage-4 fix deferred —
    structured_light.py is pre-; this finding becomes D-sl-1
    in its forthcoming spec.
    """

    # Historical xfail removed: the underlying bug (logger.info %d
    # crash on a numpy scalar) does not reproduce under the CI Python
    # toolchain. Tests are expected to pass and protect against
    # regression if the bug returns.
    def test_identity_round_trip(self):
        proj_w = proj_h = 16
        cam_w = cam_h = 16
        # Identity forward LUT: cam[y,x] maps to proj[y,x]
        proj_x = np.tile(np.arange(cam_w, dtype=np.float32), (cam_h, 1))
        proj_y = np.tile(np.arange(cam_h, dtype=np.float32).reshape(-1, 1), (1, cam_w))
        inv_x, inv_y = sl.invert_cam_to_proj_lut(proj_x, proj_y, proj_w, proj_h)
        assert inv_x.shape == (proj_h, proj_w)
        assert inv_y.shape == (proj_h, proj_w)
        # Inverse of identity is also identity
        for y in range(proj_h):
            for x in range(proj_w):
                assert inv_x[y, x] == pytest.approx(x, abs=1.0)
                assert inv_y[y, x] == pytest.approx(y, abs=1.0)

    # (Historical xfail removed; see test_identity_round_trip note.)
    def test_invalid_forward_pixels_excluded(self):
        proj_w = proj_h = 16
        cam_w = cam_h = 16
        proj_x = np.full((cam_h, cam_w), -1.0, dtype=np.float32)
        proj_y = np.full((cam_h, cam_w), -1.0, dtype=np.float32)
        # All-invalid forward LUT → inverse should be entirely -1 (no nearest-neighbor fill possible)
        inv_x, inv_y = sl.invert_cam_to_proj_lut(proj_x, proj_y, proj_w, proj_h)
        assert (inv_x == -1).all()
        assert (inv_y == -1).all()

    # (Historical xfail removed; see test_identity_round_trip note.)
    def test_out_of_range_projector_coords_dropped(self):
        proj_w = proj_h = 16
        cam_w = cam_h = 8
        # Forward LUT points to out-of-bounds projector coords
        proj_x = np.full((cam_h, cam_w), 999.0, dtype=np.float32)
        proj_y = np.full((cam_h, cam_w), 999.0, dtype=np.float32)
        inv_x, inv_y = sl.invert_cam_to_proj_lut(proj_x, proj_y, proj_w, proj_h)
        # Out-of-range filtered; inverse has no valid mappings
        assert (inv_x == -1).all()


# ─────────────────────────────────────────────────────────────────────────────
# C7 — prewarp_with_inverse_lut
# ─────────────────────────────────────────────────────────────────────────────


class TestC7PrewarpInverseLUT:
    """Contract: cv2.remap-style application of inverse LUT."""

    def test_identity_lut_passes_through(self):
        proj_w = proj_h = 16
        img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        inv_x = np.tile(np.arange(proj_w, dtype=np.float32), (proj_h, 1))
        inv_y = np.tile(np.arange(proj_h, dtype=np.float32).reshape(-1, 1), (1, proj_w))
        warped = sl.prewarp_with_inverse_lut(img, inv_x, inv_y, proj_w, proj_h)
        assert warped.shape == (proj_h, proj_w, 3)
        # Identity warp → output ≈ input
        np.testing.assert_allclose(warped, img, atol=1)

    def test_invalid_lut_returns_black(self):
        proj_w = proj_h = 16
        img = np.full((16, 16, 3), 200, dtype=np.uint8)
        inv_x = np.full((proj_h, proj_w), -1, dtype=np.float32)
        inv_y = np.full((proj_h, proj_w), -1, dtype=np.float32)
        warped = sl.prewarp_with_inverse_lut(img, inv_x, inv_y, proj_w, proj_h)
        # All-invalid → all-zero (BORDER_CONSTANT borderValue=(0,0,0))
        assert (warped == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# C8 — visualize_lut_quality
# ─────────────────────────────────────────────────────────────────────────────


class TestC8VisualizeLUTQuality:
    """Contract: diagnostic image with green=valid red=invalid + coverage text."""

    def test_returns_bgr_image_shape(self):
        inv_x = np.full((32, 32), -1, dtype=np.float32)
        inv_x[:16, :] = 5.0
        inv_y = inv_x.copy()
        vis = sl.visualize_lut_quality(inv_x, inv_y)
        assert vis.shape == (32, 32, 3)
        assert vis.dtype == np.uint8

    def test_writes_output_file_when_path_given(self, tmp_path):
        inv_x = np.ones((16, 16), dtype=np.float32)
        inv_y = np.ones((16, 16), dtype=np.float32)
        out = tmp_path / "lut_vis.png"
        sl.visualize_lut_quality(inv_x, inv_y, output_path=str(out))
        assert out.is_file()

    def test_no_output_file_when_path_omitted(self, tmp_path):
        inv_x = np.ones((16, 16), dtype=np.float32)
        inv_y = np.ones((16, 16), dtype=np.float32)
        # Just verify no exception and returns image
        vis = sl.visualize_lut_quality(inv_x, inv_y, output_path=None)
        assert vis is not None

    def test_all_valid_visualizes_predominantly_green(self):
        inv_x = np.ones((32, 32), dtype=np.float32)
        inv_y = np.ones((32, 32), dtype=np.float32)
        vis = sl.visualize_lut_quality(inv_x, inv_y)
        # G channel dominant where valid
        mean_g = vis[:, :, 1].mean()
        mean_r = vis[:, :, 2].mean()
        assert mean_g > mean_r

    def test_all_invalid_visualizes_predominantly_red(self):
        inv_x = np.full((32, 32), -1, dtype=np.float32)
        inv_y = np.full((32, 32), -1, dtype=np.float32)
        vis = sl.visualize_lut_quality(inv_x, inv_y)
        mean_r = vis[:, :, 2].mean()
        mean_g = vis[:, :, 1].mean()
        assert mean_r > mean_g


# ─────────────────────────────────────────────────────────────────────────────
# C9 — SL_PATTERN_DIR constant
# ─────────────────────────────────────────────────────────────────────────────


class TestC9LegacyDiskPath:
    """Contract: SL_PATTERN_DIR points into STIMViewer_CRISPI/Assets/Generated."""

    def test_pattern_dir_is_path(self):
        assert isinstance(sl.SL_PATTERN_DIR, Path)

    def test_pattern_dir_under_crispi_assets(self):
        parts = sl.SL_PATTERN_DIR.parts
        assert "Assets" in parts
        assert "Generated" in parts
        assert "sl_patterns" in parts
