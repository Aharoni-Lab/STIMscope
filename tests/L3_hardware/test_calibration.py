"""Stage-2 characterization tests for `STIMViewer_CRISPI/calibration.py`.

Pins the as-is behavior described in
`docs/specs/L3_hardware/calibration.md` §1 (contract) and §12 (divergence
ledger). Stage 4 will mutate the D-cal-9..15 PRE-FIX tests to assert the
CalibrationResult dataclass contract.

Tests are NUMBERED by the contract clause they pin (C1..C6) and by the
divergence they pre-stage (D-cal-N). Uses synthetic ArUco fixtures
generated at test time — no operator-supplied calibration board needed,
suite runs anywhere with cv2.aruco installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def cs_path():
    return (
        Path(__file__).resolve().parent.parent.parent
        / "STIMscope"
        / "STIMViewer_CRISPI"
    )


@pytest.fixture
def calibration_module(monkeypatch, cs_path):
    """Import calibration with the STIMViewer_CRISPI path on sys.path."""
    monkeypatch.syspath_prepend(str(cs_path))
    sys.modules.pop("calibration", None)
    import calibration as mod
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic ArUco board generator — used by C3 + D-cal-9..15 PRE-FIX tests.
# ─────────────────────────────────────────────────────────────────────────────


def _make_aruco_board_png(
    out_path: Path,
    n_markers: int = 12,
    img_w: int = 1200,
    img_h: int = 900,
    marker_size_px: int = 80,
    margin: int = 80,
) -> Path:
    """Render an N-marker ArUco board with DICT_5X5_50 to a PNG file.

    Markers laid out on a 4×3 grid (default) with white background.
    Returns the path. Deterministic.
    """
    import cv2

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    img = np.full((img_h, img_w), 255, dtype=np.uint8)

    cols = 4
    rows = (n_markers + cols - 1) // cols
    cell_w = (img_w - 2 * margin) // cols
    cell_h = (img_h - 2 * margin) // rows

    for mid in range(n_markers):
        r, c = mid // cols, mid % cols
        cx = margin + c * cell_w + cell_w // 2
        cy = margin + r * cell_h + cell_h // 2
        marker = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_size_px)
        y0 = cy - marker_size_px // 2
        x0 = cx - marker_size_px // 2
        img[y0:y0 + marker_size_px, x0:x0 + marker_size_px] = marker

    cv2.imwrite(str(out_path), img)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# C2 — decompose_homography (pure math)
# ─────────────────────────────────────────────────────────────────────────────


class TestC2DecomposeHomography:
    """C2: returns (tx, ty, sx, sy, angle_deg). Pure math; deterministic."""

    def test_identity_decomposes_to_zeros_and_unity(self, calibration_module):
        tx, ty, sx, sy, angle = calibration_module.decompose_homography(np.eye(3))
        assert tx == pytest.approx(0.0)
        assert ty == pytest.approx(0.0)
        assert sx == pytest.approx(1.0)
        assert sy == pytest.approx(1.0)
        assert angle == pytest.approx(0.0)

    def test_pure_translation(self, calibration_module):
        H = np.array([[1, 0, 100], [0, 1, 50], [0, 0, 1]], dtype=np.float64)
        tx, ty, sx, sy, angle = calibration_module.decompose_homography(H)
        assert tx == pytest.approx(100.0)
        assert ty == pytest.approx(50.0)
        assert sx == pytest.approx(1.0)
        assert sy == pytest.approx(1.0)
        assert angle == pytest.approx(0.0)

    def test_pure_scale(self, calibration_module):
        H = np.array([[2.0, 0, 0], [0, 3.0, 0], [0, 0, 1]], dtype=np.float64)
        tx, ty, sx, sy, angle = calibration_module.decompose_homography(H)
        assert sx == pytest.approx(2.0)
        assert sy == pytest.approx(3.0)
        assert angle == pytest.approx(0.0)

    def test_pure_rotation_90deg(self, calibration_module):
        # 90° rotation
        H = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        tx, ty, sx, sy, angle = calibration_module.decompose_homography(H)
        assert sx == pytest.approx(1.0, rel=1e-6)
        assert sy == pytest.approx(1.0, rel=1e-6)
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_invalid_shape_raises_value_error(self, calibration_module):
        with pytest.raises(ValueError, match="3x3"):
            calibration_module.decompose_homography(np.eye(4))

    def test_h22_near_zero_does_not_normalize(self, calibration_module):
        # Documented behavior: prints warning, skips normalize
        H = np.eye(3, dtype=np.float64)
        H[2, 2] = 1e-13
        # Should not raise. Result is unspecified math but call MUST succeed.
        result = calibration_module.decompose_homography(H)
        assert len(result) == 5


# ─────────────────────────────────────────────────────────────────────────────
# C3 — find_homography_aruco happy path (synthetic markers in BOTH images)
# ─────────────────────────────────────────────────────────────────────────────


class TestC3FindHomographyArucoHappy:
    """C3: when both images have the same markers at the same locations,
    the returned H should be approximately the identity (within rtol).
    """

    def test_self_pair_yields_near_identity(self, calibration_module, tmp_path):
        # Same image serves as both reference and "capture" — H must be ~I.
        ref = tmp_path / "ref.png"
        cap = tmp_path / "cap.png"
        _make_aruco_board_png(ref, n_markers=12)
        # Cap is byte-identical
        import shutil
        shutil.copy(str(ref), str(cap))

        # Post-: find_homography_aruco returns CalibrationResult.
        result = calibration_module.find_homography_aruco(
            registration_path=ref, capture_path=cap, save_outputs=False
        )
        assert result.valid, f"happy-path returned invalid: {result.message}"
        assert result.H.shape == (3, 3)
        assert result.H.dtype == np.float64
        # H should be near-identity (markers at same locations)
        assert np.allclose(result.H, np.eye(3), atol=1e-3), (
            f"expected ~identity for self-pair, got H=\n{result.H}"
        )

    def test_returns_calibration_result_on_success(self, calibration_module, tmp_path):
        ref = tmp_path / "ref.png"
        cap = tmp_path / "cap.png"
        _make_aruco_board_png(ref, n_markers=12)
        import shutil
        shutil.copy(str(ref), str(cap))
        result = calibration_module.find_homography_aruco(
            registration_path=ref, capture_path=cap, save_outputs=False
        )
        # Post-: typed return contract
        assert isinstance(result, calibration_module.CalibrationResult)
        assert result.valid is True
        assert result.H.dtype == np.float64
        assert result.H.shape == (3, 3)
        # On success the message carries summary stats
        assert "computed h from" in result.message.lower()
        assert "inliers" in result.message.lower()
        # Inlier ratio populated on success
        assert 0.0 < result.inlier_ratio <= 1.0
        # MSE is finite on success
        assert result.mse != float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# C4 — D-cal-9..15: silent-success PRE-FIX pins
#
# Currently every failure mode in `find_homography_aruco` returns np.eye(3).
# These tests pin the buggy behavior;will mutate them to assert
# the post-fix CalibrationResult contract.
# ─────────────────────────────────────────────────────────────────────────────


class TestC4DCal9PostFixRefMissing:
    """D-cal-9 POST-FIX: registration image missing → CalibrationResult(valid=False)."""

    def test_ref_missing_returns_invalid_result(self, calibration_module, tmp_path):
        ref = tmp_path / "does_not_exist.png"
        cap = tmp_path / "cap.png"
        _make_aruco_board_png(cap, n_markers=12)

        result = calibration_module.find_homography_aruco(
            registration_path=ref, capture_path=cap, save_outputs=False
        )
        # POST-FIX (): typed result, not np.eye(3) sentinel
        assert isinstance(result, calibration_module.CalibrationResult)
        assert result.valid is False
        assert "not found" in result.message.lower()
        # H is still a 3x3 identity placeholder but caller MUST NOT use it
        # without checking.valid first
        assert result.H.shape == (3, 3)


class TestC4DCal10PostFixCapMissing:
    """D-cal-10 POST-FIX: capture image missing → CalibrationResult(valid=False)."""

    def test_cap_missing_returns_invalid_result(self, calibration_module, tmp_path):
        ref = tmp_path / "ref.png"
        cap = tmp_path / "does_not_exist.png"
        _make_aruco_board_png(ref, n_markers=12)

        result = calibration_module.find_homography_aruco(
            registration_path=ref, capture_path=cap, save_outputs=False
        )
        assert isinstance(result, calibration_module.CalibrationResult)
        assert result.valid is False
        assert "not found" in result.message.lower()


class TestC4DCal12PostFixTooFewMarkers:
    """D-cal-12 POST-FIX: **THE USER-REPORTED BUG IS FIXED.**

    Previously a blank capture (zero ArUco markers detected) silently
    returned np.eye(3) and the caller's "✅ Success!" popup fired
    regardless. Now: CalibrationResult(valid=False, message="too few
    markers …"), caller in camera.py:1033 prints
    "❌ Calibration failed: too few markers …" instead.
    """

    def test_blank_capture_returns_invalid_result_with_marker_count(
        self, calibration_module, tmp_path
    ):
        ref = tmp_path / "ref.png"
        cap = tmp_path / "blank.png"
        _make_aruco_board_png(ref, n_markers=12)
        # Blank capture: all-white image, no ArUco markers
        import cv2
        cv2.imwrite(str(cap), np.full((900, 1200), 255, dtype=np.uint8))

        result = calibration_module.find_homography_aruco(
            registration_path=ref, capture_path=cap, save_outputs=False
        )
        # The user-painful bug is fixed: explicit failure signal.
        assert isinstance(result, calibration_module.CalibrationResult)
        assert result.valid is False
        # Message should mention the actual counts so the operator can act
        assert "too few markers" in result.message.lower()
        assert "captured=0" in result.message  # blank capture detected 0
        # Inlier ratio defaults to 0 on failure
        assert result.inlier_ratio == 0.0


class TestC4DCal13PostFixTooFewMatched:
    """D-cal-13 POST-FIX: disjoint marker IDs → CalibrationResult(valid=False)."""

    def test_disjoint_marker_ids_returns_invalid_result(
        self, calibration_module, tmp_path
    ):
        # ref has markers 0..11, cap has markers 20..31 → zero common IDs
        ref = tmp_path / "ref.png"
        cap = tmp_path / "cap.png"
        _make_aruco_board_png(ref, n_markers=12)
        # Cap: same layout but high IDs 20..31
        import cv2
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        img = np.full((900, 1200), 255, dtype=np.uint8)
        for r in range(3):
            for c in range(4):
                mid = 20 + r * 4 + c
                marker = cv2.aruco.generateImageMarker(aruco_dict, mid, 80)
                cy = 80 + r * ((900 - 160) // 3) + ((900 - 160) // 3) // 2
                cx = 80 + c * ((1200 - 160) // 4) + ((1200 - 160) // 4) // 2
                img[cy - 40:cy + 40, cx - 40:cx + 40] = marker
        cv2.imwrite(str(cap), img)

        result = calibration_module.find_homography_aruco(
            registration_path=ref, capture_path=cap, save_outputs=False
        )
        assert isinstance(result, calibration_module.CalibrationResult)
        assert result.valid is False
        assert "too few matched" in result.message.lower()


# Note: D-cal-11 (image-load failure with file present but corrupted) and
# D-cal-14/15 (RANSAC null / identity-sanity-check fail) are harder to
# trigger from outside without elaborate fixtures; covered indirectly by
# the failure-path enumeration. Stage 4's CalibrationResult conversion
# touches all 15 sites uniformly.


# ─────────────────────────────────────────────────────────────────────────────
# C5 — Reproducibility (deterministic-given-input)
# ─────────────────────────────────────────────────────────────────────────────


class TestC5Reproducibility:
    """C5: same input images → bit-identical H across two calls."""

    def test_two_runs_same_input_same_h(self, calibration_module, tmp_path):
        ref = tmp_path / "ref.png"
        cap = tmp_path / "cap.png"
        _make_aruco_board_png(ref, n_markers=12)
        import shutil
        shutil.copy(str(ref), str(cap))

        result1 = calibration_module.find_homography_aruco(
            registration_path=ref, capture_path=cap, save_outputs=False
        )
        result2 = calibration_module.find_homography_aruco(
            registration_path=ref, capture_path=cap, save_outputs=False
        )
        assert result1.valid and result2.valid
        assert np.array_equal(result1.H, result2.H), (
            "ArUco detection is non-deterministic"
        )
        # Decomposed components also deterministic
        assert result1.inlier_ratio == result2.inlier_ratio
        assert result1.mse == result2.mse


# ─────────────────────────────────────────────────────────────────────────────
# C6 — Structured-light subsystem smoke tests
#
# These functions move to core/structured_light.py in. The tests
# here pin the as-is behavior so theextraction can be verified
# as a pure move (same outputs).
# ─────────────────────────────────────────────────────────────────────────────


class TestC6StructuredLight:
    """C6: SL subsystem produces sensible outputs for known inputs."""

    def test_gray_code_patterns_count_and_shapes(self, calibration_module):
        patterns = calibration_module.generate_gray_code_patterns(640, 480)
        assert isinstance(patterns, list)
        assert len(patterns) >= 4  # at minimum threshold-white + threshold-black + 1 bit each axis
        for p in patterns:
            assert {'image', 'bit', 'axis', 'inverted'} <= set(p.keys())
            assert p['image'].shape == (480, 640, 3)
            assert p['image'].dtype == np.uint8

    def test_gray_code_patterns_include_threshold_pair(self, calibration_module):
        patterns = calibration_module.generate_gray_code_patterns(320, 240)
        # threshold pair: one all-white + one all-black
        axes = [p['axis'] for p in patterns]
        assert 'threshold' in axes

    def test_prewarp_with_inverse_lut_returns_proj_sized_image(
        self, calibration_module
    ):
        # Synthetic camera image + identity LUT → prewarp should produce
        # a proj-sized image with the same content (modulo border).
        cam_h, cam_w = 480, 640
        proj_h, proj_w = 480, 640
        cam_img = np.random.randint(0, 256, (cam_h, cam_w, 3), dtype=np.uint8)
        # Identity LUT: each projector pixel samples the same camera pixel
        inv_x, inv_y = np.meshgrid(
            np.arange(proj_w, dtype=np.float32),
            np.arange(proj_h, dtype=np.float32),
        )
        warped = calibration_module.prewarp_with_inverse_lut(
            cam_img, inv_x, inv_y, proj_w, proj_h
        )
        assert warped.shape == (proj_h, proj_w, 3)
        assert warped.dtype == np.uint8
        # Identity LUT should produce exact passthrough (mod numerical
        # precision of cv2.remap)
        assert np.array_equal(warped, cam_img) or np.allclose(warped, cam_img, atol=1)

    def test_prewarp_with_invalid_lut_pixels_produces_black(
        self, calibration_module
    ):
        proj_h, proj_w = 100, 100
        cam_img = np.full((100, 100, 3), 200, dtype=np.uint8)
        # LUT with all -1 (invalid) entries → output should be all black
        inv_x = np.full((proj_h, proj_w), -1.0, dtype=np.float32)
        inv_y = np.full((proj_h, proj_w), -1.0, dtype=np.float32)
        warped = calibration_module.prewarp_with_inverse_lut(
            cam_img, inv_x, inv_y, proj_w, proj_h
        )
        assert warped.shape == (proj_h, proj_w, 3)
        # All-invalid LUT → all-black output (cv2.remap borderValue=(0,0,0))
        assert warped.max() == 0
