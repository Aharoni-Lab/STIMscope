
from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

# Logger seam: prefer the project's structured logger from
# core.logging_config (timestamps + level + module). When this module
# is imported in a context without the CS path on sys.path (some unit
# tests, ad-hoc scripts), fall back to a stdlib basicConfig logger so
# the import doesn't fail. The CS directory is added to sys.path here
# defensively — `core/` lives there in the live GUI runtime.
_CS_DIR = Path(__file__).resolve().parent / "CS"
if _CS_DIR.is_dir() and str(_CS_DIR) not in sys.path:
    sys.path.insert(0, str(_CS_DIR))
try:
    from core.logging_config import get_logger  # type: ignore
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)


# Paths — `Assets/Generated/` is the legacy GUI-coupled location. The
# broader migration to `core.paths` is rolling per
# module; calibration.py keeps these legacy paths because the running GUI
# (qt_interface.py + main.py) still resolves homography_cam2proj.npy from
# Assets/Generated. Migrating these constants requires a coordinated
# write-once-read-many change is NOT done
# in this audit pass.
ASSETS = (Path(__file__).resolve().parent / "Assets").resolve()
GEN_DIR = (ASSETS / "Generated").resolve()
GEN_DIR.mkdir(parents=True, exist_ok=True)

CALIB_CAPTURE_IMG = GEN_DIR / "calibration_capture_image.png"
CALIB_OUTPUT_IMG = GEN_DIR / "CalibOutput.jpg"
HOMOGRAPHY_NPY = GEN_DIR / "homography_cam2proj.npy"

# ArUco dictionary matching the board (DICT_5X5_50, 48 markers detected)
ARUCO_DICT_ID = cv2.aruco.DICT_5X5_50

# ─── ArUco detector tuning ──────────────────────────────────────────────
# Tuned for microscope optics (blur, distortion, low contrast). Values
# chosen empirically against the lab's DLPC3479-projected ChArUco board.
# Do NOT change without re-running tests/L3_hardware/test_calibration.py
# and at least one live hardware capture.
_ARUCO_ADAPTIVE_THRESH_WIN_MIN = 3
_ARUCO_ADAPTIVE_THRESH_WIN_MAX = 53
_ARUCO_ADAPTIVE_THRESH_WIN_STEP = 4
_ARUCO_ADAPTIVE_THRESH_CONSTANT = 7
_ARUCO_MIN_MARKER_PERIMETER_RATE = 0.01
_ARUCO_MAX_MARKER_PERIMETER_RATE = 4.0
_ARUCO_POLYGONAL_APPROX_ACCURACY = 0.05
_ARUCO_MIN_CORNER_DISTANCE_RATE = 0.01
_ARUCO_MIN_DISTANCE_TO_BORDER = 1

# Minimum markers required (4 markers x 4 corners = 16 pts; well above
# the 4-point minimum for findHomography, but allows some outlier
# rejection by RANSAC). One ArUco corner is unreliable in isolation.
_MIN_MARKERS_REQUIRED = 4

# RANSAC reprojection threshold — relaxed because microscope optics
# cause significant non-affine distortion that tight thresholds reject
# as outliers (despite being real corner matches).
_RANSAC_REPROJ_THRESHOLD_PX = 10.0
_RANSAC_CONFIDENCE = 0.999

# Degenerate-homography guard: |det(H[:2, :2])| must exceed this. A
# determinant near zero means the projective mapping collapses to a
# line (rank-deficient) — useless for warping.
_HOMOGRAPHY_MIN_DET_ABS = 0.001

# Alignment-quality MSE thresholds (pixel intensity, ref vs warped capture).
# Conflates geometric error with lighting/contrast differences, so these
# are advisory only — the inlier ratio is the authoritative geometric
# measure. Calibrated against real hardware captures where LED/exposure
# differs between reference and capture.
_MSE_EXCELLENT = 5000
_MSE_GOOD = 15000
_MSE_FAIR = 40000


def _resolve_charuco_board() -> Path:
    """Resolve the ChArUco calibration board image.

    Order (first existing wins):
      1. operator override at ``$STIM_DATA_ROOT/config/calibration_board.png``
         (per ``core.paths``) — lets a site swap in its own board.
      2. the board bundled with the platform at ``Assets/calibration_board.png``
         (committed to the repo, ships in the Docker image) — used by default.

    If neither exists, returns the bundled path so the caller can generate
    one on demand via :func:`generate_registration_board`.

    Resolved lazily at module load — restart Python to pick up a new board
    after copying it into place.
    """
    bundled = Path(__file__).resolve().parent / "Assets" / "calibration_board.png"
    try:
        from core.paths import config_dir  # type: ignore
        override = config_dir() / "calibration_board.png"
        if override.exists():
            return override
    except Exception:
        pass
    return bundled


def generate_registration_board(out_path: Path, width: int, height: int,
                                squares_x: int = 8, squares_y: int = 6) -> bool:
    """Generate a ChArUco board (``ARUCO_DICT_ID``) sized to the projector and
    write it to ``out_path``.

    This is the projected registration pattern used by calibration:
    :func:`find_homography_aruco` detects the board's ArUco markers in both
    the projected reference and the camera capture, matches them by ID, and
    solves for the camera→projector homography. Generating it here makes
    calibration self-contained — no operator-supplied physical board needed.

    Square/marker lengths are arbitrary units; only their ratio and the
    output pixel size matter for a flat projected pattern. Returns True on
    success.
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        try:
            # OpenCV >= 4.7 API
            board = cv2.aruco.CharucoBoard((squares_x, squares_y), 0.04, 0.02, aruco_dict)
            img = board.generateImage((int(width), int(height)))
        except AttributeError:
            # OpenCV < 4.7 legacy API
            board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, 0.04, 0.02, aruco_dict)
            img = board.draw((int(width), int(height)))
        cv2.imwrite(str(out_path), img)
        return out_path.exists()
    except Exception as e:
        logger.error(f"failed to generate registration board: {e}")
        return False


# User-provided ChArUco calibration board (resolved at import; restart
# the GUI / Python session after moving the file).
CHARUCO_BOARD_IMG = _resolve_charuco_board()


# ─────────────────────────────────────────────────────────────────────────────
# CalibrationResult — typed contract for calibration return values
# ─────────────────────────────────────────────────────────────────────────────
#
# Pre-audit, find_homography_aruco returned `np.ndarray` with `np.eye(3)`
# on EVERY failure path (15 sites in this file, 7 of which live in
# find_homography_aruco). Caller in camera.py:1033 could not distinguish
# real H from silent-success identity — popup showed "✅ Homography
# Computed Successfully!" regardless. Operator-painful bug.
#
# Post-audit: find_homography_aruco returns a CalibrationResult.
# - On success: valid=True, H=computed matrix, message=summary,
#   inlier_ratio + decomposed components filled.
# - On failure: valid=False, H=np.eye(3) (placeholder — NOT a valid
#   calibration), message=diagnostic.
# - Caller MUST check result.valid before using result.H.
#
# Structure mirrors `core.calibration_service.CalibrationResult`
# (the Stack B equivalent) — uniform contract across both calibration
# stacks in the codebase.


@dataclass
class CalibrationResult:
    """Result of a homography-calibration attempt.

    Attributes
    ----------
    H : (3, 3) float64 ndarray
        On success: the camera→projector homography. On failure: identity
        placeholder; do NOT use without first checking ``valid``.
    valid : bool
        True iff the homography is a real computed result. False if any
        failure mode hit (file missing, too few markers, RANSAC null,
        degenerate determinant, etc.).
    message : str
        Diagnostic — on success, summary stats. On failure, the reason
        (suitable for operator-facing popup display).
    inlier_ratio : float
        Fraction of RANSAC inliers among the matched point pairs. 0.0 on
        failure.
    mse : float
        Reprojection MSE on inliers. ``float('inf')`` if not computed.
    tx, ty : float
        Translation components from `decompose_homography(H)`. Zero on failure.
    sx, sy : float
        Scale components. 1.0 on failure (identity placeholder).
    angle_deg : float
        Rotation in degrees. 0.0 on failure.
    ref_image, cap_image : Optional[ndarray]
        Reference + captured grayscale images (kept for debugging /
        overlay generation). Not serialized in ``__repr__``.
    """

    H: np.ndarray
    valid: bool = False
    message: str = ''
    inlier_ratio: float = 0.0
    mse: float = float('inf')
    tx: float = 0.0
    ty: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    angle_deg: float = 0.0
    ref_image: Optional[np.ndarray] = field(default=None, repr=False)
    cap_image: Optional[np.ndarray] = field(default=None, repr=False)





def decompose_homography(H: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Decompose 3x3 homography into translation (tx, ty), scale (sx, sy), rotation (deg).
    Returns (tx, ty, sx, sy, angle_deg).
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError("Homography must be 3x3.")

    if abs(H[2, 2]) < 1e-12:
        logger.warning("Homography H[2,2] ~ 0; normalizing skipped.")
    else:
        H = H / H[2, 2]

    tx = float(H[0, 2])
    ty = float(H[1, 2])

    A = H[:2, :2]

    sx = float(np.linalg.norm(A[:, 0]))
    sy = float(np.linalg.norm(A[:, 1])) if np.linalg.norm(A[:, 1]) > 1e-12 else 1.0


    R = np.zeros_like(A)
    if sx > 1e-12:
        R[:, 0] = A[:, 0] / sx
    if sy > 1e-12:
        R[:, 1] = A[:, 1] / sy



    angle = math.degrees(math.atan2(R[1, 0], R[0, 0]))

    return tx, ty, sx, sy, angle


def find_homography_aruco(
    registration_path: Path = CHARUCO_BOARD_IMG,
    capture_path: Path = CALIB_CAPTURE_IMG,
    save_outputs: bool = True,
) -> CalibrationResult:
    """Compute homography using ArUco marker detection.

    Detects ArUco markers in both the reference (projected) and captured
    (camera) images, matches them by marker ID, and computes a homography
    from the matched corner points. Much more robust than SIFT/ORB through
    microscope optics because ArUco detection is designed for this.

    Returns
    -------
    CalibrationResult
        On success: ``valid=True``, ``H`` = computed camera→projector
        homography, ``inlier_ratio`` + decomposed components filled,
        ``message`` = summary stats.

        On failure: ``valid=False``, ``H = np.eye(3)`` placeholder,
        ``message`` = operator-facing diagnostic. **Caller MUST check
        ``result.valid`` before using ``result.H``.**

    Notes
    -----
    Replaces 7 prior silent-success
    ``return np.eye(3)`` sites replaced with explicit
    ``CalibrationResult(valid=False, message=…)`` returns. Pre-fix, the
    caller in ``camera.py:1033`` could not distinguish real H from
    failure → popup showed "✅ Success!" on every operator action.
    """
    reg_p = Path(registration_path)
    cap_p = Path(capture_path)

    def _fail(msg: str) -> CalibrationResult:
        """Build a CalibrationResult for the failure path (D-cal-9..15 fix).

        Identity placeholder for ``H`` — kept so legacy callers reading
        ``result.H`` directly (without checking ``valid``) won't crash on
        type errors. Caller MUST gate on ``result.valid``.
        """
        logger.error(msg)
        return CalibrationResult(
            H=np.eye(3, dtype=np.float64), valid=False, message=msg
        )

    if not reg_p.exists():
        return _fail(f"reference board image not found: {reg_p}")        # D-cal-9
    if not cap_p.exists():
        return _fail(f"calibration capture image not found: {cap_p}")    # D-cal-10

    img_ref = cv2.imread(str(reg_p), cv2.IMREAD_GRAYSCALE)
    img_cap = cv2.imread(str(cap_p), cv2.IMREAD_GRAYSCALE)
    if img_ref is None or img_cap is None:
        return _fail("failed to load images for ArUco calibration")       # D-cal-11

    # Detect ArUco markers in both images with tuned parameters for
    # microscope optics (blur, distortion, low contrast)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = _ARUCO_ADAPTIVE_THRESH_WIN_MIN
    params.adaptiveThreshWinSizeMax = _ARUCO_ADAPTIVE_THRESH_WIN_MAX
    params.adaptiveThreshWinSizeStep = _ARUCO_ADAPTIVE_THRESH_WIN_STEP
    params.adaptiveThreshConstant = _ARUCO_ADAPTIVE_THRESH_CONSTANT
    params.minMarkerPerimeterRate = _ARUCO_MIN_MARKER_PERIMETER_RATE
    params.maxMarkerPerimeterRate = _ARUCO_MAX_MARKER_PERIMETER_RATE
    params.polygonalApproxAccuracyRate = _ARUCO_POLYGONAL_APPROX_ACCURACY
    params.minCornerDistanceRate = _ARUCO_MIN_CORNER_DISTANCE_RATE
    params.minDistanceToBorder = _ARUCO_MIN_DISTANCE_TO_BORDER
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    ref_corners, ref_ids, _ = detector.detectMarkers(img_ref)
    cap_corners, cap_ids, _ = detector.detectMarkers(img_cap)

    n_ref = len(ref_ids) if ref_ids is not None else 0
    n_cap = len(cap_ids) if cap_ids is not None else 0
    logger.info("ArUco markers: reference=%d, captured=%d", n_ref, n_cap)

    if n_ref < _MIN_MARKERS_REQUIRED or n_cap < _MIN_MARKERS_REQUIRED:    # D-cal-12
        return _fail(
            f"too few markers detected: reference={n_ref}, captured={n_cap} "
            f"(need ≥{_MIN_MARKERS_REQUIRED} each)"
        )

    # Build lookup: marker_id -> 4 corners for each image
    ref_map = {int(ref_ids[i][0]): ref_corners[i][0] for i in range(n_ref)}
    cap_map = {int(cap_ids[i][0]): cap_corners[i][0] for i in range(n_cap)}

    # Match by ID — each marker contributes 4 corner points
    common_ids = sorted(set(ref_map.keys()) & set(cap_map.keys()))
    logger.info("Matched markers: %d", len(common_ids))

    if len(common_ids) < _MIN_MARKERS_REQUIRED:                           # D-cal-13
        return _fail(
            f"too few matched markers: only {len(common_ids)} common IDs "
            f"(need ≥{_MIN_MARKERS_REQUIRED})"
        )

    pts_ref = np.vstack([ref_map[mid] for mid in common_ids]).astype(np.float32)
    pts_cap = np.vstack([cap_map[mid] for mid in common_ids]).astype(np.float32)

    logger.debug("Point correspondences: %d (from %d markers x 4 corners)",
                 len(pts_ref), len(common_ids))

    # Compute homography: maps capture → reference (camera → projector)
    # Use relaxed reproj threshold — microscope optics cause significant
    # distortion that tight thresholds would reject as outliers.
    H, inliers = cv2.findHomography(
        pts_cap, pts_ref, cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJ_THRESHOLD_PX,
        confidence=_RANSAC_CONFIDENCE,
    )
    if H is None:                                                         # D-cal-14
        return _fail("findHomography returned None (RANSAC failed)")

    inlier_count = int(inliers.sum()) if inliers is not None else 0
    total = len(pts_ref)
    inlier_ratio = (inlier_count / total) if total > 0 else 0.0
    logger.info("Homography: %d/%d inliers (%.1f%%)",
                inlier_count, total, 100 * inlier_ratio)

    # Validate — only reject truly degenerate results
    try:
        tx, ty, sx, sy, ang = decompose_homography(H)
        logger.debug("H decomposition: tx=%.1f, ty=%.1f, sx=%.3f, sy=%.3f, angle=%.1f",
                     tx, ty, sx, sy, ang)
        det = np.linalg.det(H[:2, :2])
        if abs(det) < _HOMOGRAPHY_MIN_DET_ABS:                            # D-cal-15
            return _fail(
                f"degenerate homography: det(H[:2,:2])={det:.6f} "
                f"(|det| < {_HOMOGRAPHY_MIN_DET_ABS})"
            )
        # With ArUco markers, even a few matched markers give reliable H.
        # Don't reject based on inlier ratio — the markers are trustworthy.
    except Exception as e:
        # Decomposition failure isn't fatal — still surface partial result
        # but mark valid=False so caller sees the issue.
        return _fail(f"H validation error: {e}")

    if save_outputs:
        h, w = img_ref.shape[:2]
        warped = cv2.warpPerspective(img_cap, H, (w, h))
        try:
            cv2.imwrite(str(CALIB_OUTPUT_IMG), warped)
            np.save(str(HOMOGRAPHY_NPY), H.astype(np.float64))
            logger.info("Saved warped preview: %s", CALIB_OUTPUT_IMG)
            logger.info("Saved homography: %s", HOMOGRAPHY_NPY)
            _generate_alignment_verification(
                cv2.imread(str(reg_p), cv2.IMREAD_COLOR),
                cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR) if warped.ndim == 2 else warped,
                H,
            )
        except Exception as e:
            logger.error("Output save failed: %s", e)

    # Compute reprojection MSE on inliers (audit-grade quality metric)
    mse = float('inf')
    try:
        if inliers is not None and inlier_count > 0:
            inlier_mask = inliers.ravel().astype(bool)
            src_in = pts_cap[inlier_mask]
            dst_in = pts_ref[inlier_mask]
            src_h = np.hstack([src_in, np.ones((len(src_in), 1), dtype=np.float32)])
            proj = (H @ src_h.T).T
            proj = proj[:, :2] / proj[:, 2:3]
            mse = float(np.mean(np.sum((proj - dst_in) ** 2, axis=1)))
    except Exception as e:
        logger.warning("MSE compute failed (non-fatal): %s", e)

    logger.info("ArUco calibration completed successfully.")
    return CalibrationResult(
        H=H.astype(np.float64),
        valid=True,
        message=(
            f"computed H from {len(common_ids)} ArUco markers, "
            f"{inlier_count}/{total} RANSAC inliers ({100 * inlier_ratio:.1f}%), "
            f"MSE={mse:.2f}px²"
        ),
        inlier_ratio=inlier_ratio,
        mse=mse,
        tx=tx, ty=ty, sx=sx, sy=sy, angle_deg=ang,
        ref_image=img_ref,
        cap_image=img_cap,
    )


# ---------------------------------------------------------------------------
#  Structured-Light Calibration — moved to core/structured_light.py
#  (audit). Re-exported here so existing callers in
#  qt_interface.py and gpu_ui.py that import these symbols from
#  ``calibration`` keep working without touching the GUI.
# ---------------------------------------------------------------------------

from core.structured_light import (  # noqa: E402, F401
    SL_PATTERN_DIR,
    generate_gray_code_patterns,
    generate_phase_shift_patterns,
    save_structured_light_patterns,
    decode_gray_code_from_files,
    decode_phase_shift_from_files,
    invert_cam_to_proj_lut,
    prewarp_with_inverse_lut,
    visualize_lut_quality,
)


def _generate_alignment_verification(reference, warped, _homography):
    # `_homography` (leading underscore) marks intentionally-unused — the
    # function generates a pixel-intensity comparison image from reference
    # and warped only. H is kept in the signature for caller-side
    # readability (`_generate_alignment_verification(ref, warped, H)`).
    try:

        h, w = reference.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        

        if len(reference.shape) == 3:
            comparison[:, :w] = reference
        else:
            comparison[:, :w] = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
        

        if len(warped.shape) == 3:
            comparison[:, w:] = warped
        else:
            comparison[:, w:] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        

        cv2.line(comparison, (w, 0), (w, h), (0, 255, 0), 2)
        

        cv2.putText(comparison, "REFERENCE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "ALIGNED CAPTURE", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

        verification_path = CALIB_OUTPUT_IMG.parent / "calibration_verification.png"
        cv2.imwrite(str(verification_path), comparison)
        logger.info("Alignment verification saved: %s", verification_path)
        

        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference
            
        if len(warped.shape) == 3:
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            warped_gray = warped
        

        mse = np.mean((ref_gray.astype(float) - warped_gray.astype(float)) ** 2)
        # MSE compares pixel intensities of reference vs captured-then-warped
        # image — it conflates geometric error with lighting/contrast differences.
        # The inlier ratio reported above (e.g. "Homography: N/M inliers (X%)")
        # is the authoritative geometric measure. These thresholds are tuned to
        # only flag truly poor alignments; expect MSE in the 5k–20k range even
        # for excellent geometric fits because LED/exposure differs.
        logger.info("Alignment quality MSE: %.2f (geometric inliers above are authoritative)", mse)

        if mse < _MSE_EXCELLENT:
            logger.info("Excellent alignment quality.")
        elif mse < _MSE_GOOD:
            logger.info("Good alignment quality.")
        elif mse < _MSE_FAIR:
            logger.warning("Fair alignment — geometry may still be fine, check inlier ratio above.")
        else:
            logger.warning("Poor alignment quality — recalibration recommended (also check inlier ratio).")

    except Exception as e:
        logger.warning("Verification image generation failed: %s", e)





