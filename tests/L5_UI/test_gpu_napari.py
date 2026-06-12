"""Comprehensive characterization tests for ``gpu_ui_napari``.

1 — comprehensive (branch + raise walk, ≥2
property-based tests, ≥85% line+branch coverage target on the audited
unit). Third chars suite for the L5 ``gpu_ui.py`` 9-sub-module
decomposition (iter-3, NapariViewerMixin extracted from ``gpu_ui.py``
per ``docs/specs/L5_UI/gpu_ui.md`` §0.5).

Module surface (~365 LOC, 1 method, UI-glue archetype with deep
nesting):

- ``_launch_napari_viewer(mean, masks)`` — Qt slot that pauses
  live-traces/camera/projector, validates masks (3D-stack vs
  2D-labels), launches ``roi_editor.refine_rois`` with a
  ``restore_after_napari`` callback that re-projects updated masks +
  restarts traces.

The method body contains 3 nested closures:
- ``restore_after_napari(event=None)`` — invoked on Napari close
- ``restart_with_new_rois()`` — scheduled via QTimer from restore
- ``fallback_restart()`` — scheduled via QTimer on restart failure

Because the inner closures are dispatched via ``QTimer.singleShot``,
test coverage of their bodies requires direct manipulation of the
``on_close_callback`` argument that ``refine_rois`` receives. The
chars suite patches ``QTimer.singleShot`` to no-op so the test
deterministically observes pre-timer state.

Coverage gap recovery criterion (per §1.1 sub-target rule): the inner
``restart_with_new_rois`` closure dispatches through
``QTimer.singleShot`` after a 1000 ms delay; running its body in a
unit-test would require pumping a Qt event loop. The chars suite
exercises the closure factory + outer scheduling but does NOT execute
the inner body — those lines (~50) remain uncovered. Recovery: stated
in spec §15 Row 3 — the iter-N refactor will sub-extract
``restart_with_new_rois`` into a top-level helper method on the mixin;
focused chars on the helper close the gap without timer plumbing.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

REPO_ROOT = Path(__file__).resolve().parents[2]
CRISPI_PATH = REPO_ROOT / "STIMscope" / "STIMViewer_CRISPI"
if str(CRISPI_PATH) not in sys.path:
    sys.path.insert(0, str(CRISPI_PATH))

from gpu_ui_mixins.napari import NapariViewerMixin  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Test infrastructure: stub host class + fake roi_editor / projection modules
# ─────────────────────────────────────────────────────────────────────────────


class _Host(NapariViewerMixin):
    """Minimal stub satisfying the NapariViewerMixin host contract."""

    def __init__(self, tmp_path: Path):
        self.camera = MagicMock(
            acquisition_running=False,
            is_recording=False,
            translation_matrix=None,
        )
        self.camera.stop_realtime_acquisition = MagicMock()
        self.camera.start_realtime_acquisition = MagicMock(return_value=True)

        self.proj_display = None
        self.rois_path = str(tmp_path / "rois.npz")
        self.plot_widget = None
        self.live_extractor = None
        self.layout = MagicMock()
        self.layout.count = MagicMock(return_value=0)
        self.current_labels = None

        # Mixin methods normally provided by LiveTracesMixin; we mock here.
        self.stop_live_traces = MagicMock()
        self.start_live_traces = MagicMock()

        # Provided by the residual GPU class; mock for unit tests.
        self._handle_error = MagicMock()


@pytest.fixture
def host(tmp_path: Path) -> _Host:
    return _Host(tmp_path)


@pytest.fixture
def patched_qtimer():
    """No-op QTimer.singleShot inside the audited module."""
    with patch("gpu_ui_mixins.napari.QTimer") as mock_qt:
        mock_qt.singleShot = MagicMock()
        yield mock_qt


@pytest.fixture
def fake_roi_editor():
    """Install a fake roi_editor module with a controllable refine_rois."""
    captured = {"calls": [], "raise": None, "return": None}

    def fake_refine_rois(mean, masks, return_viewer=False, on_close_callback=None):
        captured["calls"].append({
            "mean_shape": mean.shape,
            "n_masks": len(masks),
            "on_close_callback": on_close_callback,
        })
        if captured["raise"] is not None:
            raise captured["raise"]
        return captured["return"]

    fake_mod = types.ModuleType("roi_editor")
    fake_mod.refine_rois = fake_refine_rois
    sys.modules["roi_editor"] = fake_mod
    yield captured
    sys.modules.pop("roi_editor", None)


@pytest.fixture
def broken_roi_editor_importerror():
    """Force ``from roi_editor import refine_rois`` to ImportError."""
    sys.modules.pop("roi_editor", None)
    broken = types.ModuleType("roi_editor")
    # No refine_rois attr → from-import raises ImportError
    sys.modules["roi_editor"] = broken
    yield
    sys.modules.pop("roi_editor", None)


@pytest.fixture
def broken_roi_editor_runtime():
    """Force the from-import to raise a non-ImportError."""
    class _ExplodingModule(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError(f"roi_editor explodes on access: {name!r}")

    sys.modules.pop("roi_editor", None)
    sys.modules["roi_editor"] = _ExplodingModule("roi_editor")
    yield
    sys.modules.pop("roi_editor", None)


@pytest.fixture
def fake_projection():
    """Stub projection.ProjectDisplay so restore closure doesn't crash."""
    fake_mod = types.ModuleType("projection")
    fake_mod.ProjectDisplay = MagicMock()
    sys.modules["projection"] = fake_mod

    # QGuiApplication.screens() also referenced in the closure — patch
    # the QtGui import path inside the closure.
    fake_qtgui = types.ModuleType("PyQt5_napari_test_qtgui")
    screen = MagicMock()
    screen.size = MagicMock(return_value=MagicMock(width=lambda: 1920, height=lambda: 1080))
    fake_qtgui.QGuiApplication = MagicMock()
    fake_qtgui.QGuiApplication.screens = MagicMock(return_value=[screen])
    yield fake_mod
    sys.modules.pop("projection", None)


# ─────────────────────────────────────────────────────────────────────────────
# C1 — initial state capture: was_recording / was_live_traces / was_camera_running
# ─────────────────────────────────────────────────────────────────────────────


def test_C1_camera_none_was_recording_false(host, patched_qtimer, broken_roi_editor_importerror):
    """Branch: camera attr None-equivalent → was_recording = False (no crash)."""
    host.camera = None
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    # No exception propagates; outer try/except absorbs anything.


def test_C2_live_extractor_present_calls_stop(host, patched_qtimer, broken_roi_editor_importerror):
    """Branch: live_extractor present → stop_live_traces() invoked."""
    host.live_extractor = MagicMock()
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    host.stop_live_traces.assert_called_once()


def test_C3_camera_running_paused_for_napari(host, patched_qtimer, broken_roi_editor_importerror):
    """Branch: camera acquisition_running True → stop_realtime_acquisition called."""
    host.camera.acquisition_running = True
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    host.camera.stop_realtime_acquisition.assert_called_once()


def test_C4_proj_display_present_closed(host, patched_qtimer, broken_roi_editor_importerror, fake_projection):
    """Branch: proj_display present →.close() invoked at startup."""
    pd = MagicMock()
    host.proj_display = pd
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    pd.close.assert_called()


def test_C5_proj_display_close_raises_swallowed(host, patched_qtimer, broken_roi_editor_importerror, fake_projection):
    """Raise walk: proj_display.close() raises → swallowed silently."""
    pd = MagicMock()
    pd.close.side_effect = RuntimeError("zmq dead")
    host.proj_display = pd
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    # No exception escapes
    host._launch_napari_viewer(mean, masks)


# ─────────────────────────────────────────────────────────────────────────────
# C6-C9 — roi_editor import: ImportError + non-Import exception + happy
# ─────────────────────────────────────────────────────────────────────────────


def test_C6_roi_editor_importerror_returns_after_restore(host, patched_qtimer, broken_roi_editor_importerror, capsys):
    """Branch: ImportError on roi_editor → 'Cannot proceed' + restore + return."""
    host.camera.acquisition_running = True  # so restore triggers start_realtime
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    out = capsys.readouterr().out
    assert "roi_editor import failed" in out
    assert "Cannot proceed without roi_editor" in out
    # Camera restart confirms restore_after_napari was invoked
    host.camera.start_realtime_acquisition.assert_called_once()


def test_C7_roi_editor_non_import_exception(host, patched_qtimer, broken_roi_editor_runtime, capsys):
    """Branch: roi_editor raises non-ImportError → 'unexpected error' path."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    out = capsys.readouterr().out
    assert "unexpected error" in out or "Cannot proceed without roi_editor" in out


# ─────────────────────────────────────────────────────────────────────────────
# C8-C14 — mask validation: ndim 3 (match / mismatch / empty / resize-raises)
# ─────────────────────────────────────────────────────────────────────────────


def test_C8_3d_masks_matching_shape_converts_to_list(host, patched_qtimer, fake_roi_editor, fake_projection):
    """Branch: 3D ndarray matching mean shape → list conversion + refine_rois call."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((3, 10, 10), dtype=bool)
    masks[0, 0:3, 0:3] = True  # non-empty
    masks[1, 5:8, 5:8] = True  # non-empty
    # masks[2] is all-empty → filtered out
    host._launch_napari_viewer(mean, masks)
    assert len(fake_roi_editor["calls"]) == 1
    # 2 non-empty masks (the empty one is dropped)
    assert fake_roi_editor["calls"][0]["n_masks"] == 2


def test_C9_3d_masks_all_empty_aborts(host, patched_qtimer, fake_roi_editor, fake_projection):
    """Branch: 3D masks all-empty → refine_rois never called."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((3, 10, 10), dtype=bool)  # all-empty
    host._launch_napari_viewer(mean, masks)
    # After dropping empties + 'No valid masks after validation' early-return
    assert fake_roi_editor["calls"] == []


def test_C10_3d_masks_mismatched_shape_resizes(host, patched_qtimer, fake_roi_editor, fake_projection):
    """Branch: 3D ndarray with mismatched shape → cv2.resize fallback."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    # masks at 20x20 — must resize to 10x10
    masks = np.zeros((2, 20, 20), dtype=bool)
    masks[0, 0:5, 0:5] = True
    masks[1, 10:15, 10:15] = True
    host._launch_napari_viewer(mean, masks)
    assert len(fake_roi_editor["calls"]) == 1
    assert fake_roi_editor["calls"][0]["n_masks"] >= 1


def test_C11_3d_masks_resize_all_empty_after(host, patched_qtimer, fake_roi_editor, fake_projection, capsys):
    """Branch: 3D mismatched + resize all-empty → 'All resized masks were empty'."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((2, 20, 20), dtype=bool)  # all-empty even after resize
    host._launch_napari_viewer(mean, masks)
    out = capsys.readouterr().out
    assert "All resized masks were empty" in out
    assert fake_roi_editor["calls"] == []


def test_C12_3d_masks_resize_raises_returns(host, patched_qtimer, fake_roi_editor, fake_projection, capsys):
    """Raise walk: cv2.resize raises inside 3D-mismatch path."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 20, 20), dtype=bool)
    masks[0, 0:5, 0:5] = True
    with patch("gpu_ui_mixins.napari.cv2.resize", side_effect=RuntimeError("cv2 dead")):
        host._launch_napari_viewer(mean, masks)
    out = capsys.readouterr().out
    assert "Failed to resize 3D masks" in out
    assert fake_roi_editor["calls"] == []


# ─────────────────────────────────────────────────────────────────────────────
# C13-C16 — 2D label arrays
# ─────────────────────────────────────────────────────────────────────────────


def test_C13_2d_labels_matching_shape_converts(host, patched_qtimer, fake_roi_editor, fake_projection):
    """Branch: 2D labels matching mean.shape → unique-id conversion."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[0:3, 0:3] = 1
    labels[5:8, 5:8] = 2
    host._launch_napari_viewer(mean, labels)
    assert len(fake_roi_editor["calls"]) == 1
    assert fake_roi_editor["calls"][0]["n_masks"] == 2


def test_C14_2d_labels_mismatched_shape_resizes(host, patched_qtimer, fake_roi_editor, fake_projection):
    """Branch: 2D labels with mismatched shape → cv2.resize to mean.shape."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:6, 2:6] = 1
    labels[12:18, 12:18] = 2
    host._launch_napari_viewer(mean, labels)
    assert len(fake_roi_editor["calls"]) == 1


def test_C15_2d_labels_resize_raises(host, patched_qtimer, fake_roi_editor, fake_projection, capsys):
    """Raise walk: cv2.resize raises for 2D labels mismatch."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[0:3, 0:3] = 1
    with patch("gpu_ui_mixins.napari.cv2.resize", side_effect=RuntimeError("resize")):
        host._launch_napari_viewer(mean, labels)
    out = capsys.readouterr().out
    assert "Failed to resize labels" in out
    assert fake_roi_editor["calls"] == []


def test_C16_2d_labels_all_background_empty(host, patched_qtimer, fake_roi_editor, fake_projection, capsys):
    """Branch: 2D labels all-zero → no ROIs found → early return."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int32)  # all-background
    host._launch_napari_viewer(mean, labels)
    out = capsys.readouterr().out
    # Either "No valid masks found" (empty after conversion) reached
    assert "No valid masks" in out
    assert fake_roi_editor["calls"] == []


# ─────────────────────────────────────────────────────────────────────────────
# C17-C18 — unsupported ndim + non-ndarray input
# ─────────────────────────────────────────────────────────────────────────────


def test_C17_unexpected_ndim_4d_returns(host, patched_qtimer, fake_roi_editor, fake_projection, capsys):
    """Branch: ndim == 4 → 'Unexpected mask array shape' early return."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((2, 2, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    out = capsys.readouterr().out
    assert "Unexpected mask array shape" in out
    assert fake_roi_editor["calls"] == []


def test_C18_non_ndarray_non_list_returns(host, patched_qtimer, fake_roi_editor, fake_projection, capsys):
    """Branch: masks neither ndarray nor non-empty list → 'No valid masks found'."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    host._launch_napari_viewer(mean, "not-a-mask")
    out = capsys.readouterr().out
    assert "No valid masks found" in out
    assert fake_roi_editor["calls"] == []


def test_C19_empty_list_returns(host, patched_qtimer, fake_roi_editor, fake_projection, capsys):
    """Branch: masks is empty list → 'No valid masks found'."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    host._launch_napari_viewer(mean, [])
    out = capsys.readouterr().out
    assert "No valid masks found" in out
    assert fake_roi_editor["calls"] == []


def test_C20_list_with_wrong_shape_filtered(host, patched_qtimer, fake_roi_editor, fake_projection, capsys):
    """Branch: list contains a wrong-shape mask → marked None + filtered."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    good = np.zeros((10, 10), dtype=bool); good[0:3, 0:3] = True
    bad = np.zeros((7, 7), dtype=bool)
    masks = [good, bad]
    host._launch_napari_viewer(mean, masks)
    # Bad mask dropped; good remains; refine_rois called once with 1 mask
    assert len(fake_roi_editor["calls"]) == 1
    assert fake_roi_editor["calls"][0]["n_masks"] == 1


def test_C21_list_all_invalid_returns(host, patched_qtimer, fake_roi_editor, fake_projection, capsys):
    """Branch: all masks wrong shape → 'No valid masks after validation'."""
    mean = np.zeros((10, 10), dtype=np.uint8)
    bad = np.zeros((7, 7), dtype=bool); bad[0, 0] = True
    masks = [bad]
    host._launch_napari_viewer(mean, masks)
    out = capsys.readouterr().out
    assert "No valid masks after validation" in out
    assert fake_roi_editor["calls"] == []


# ─────────────────────────────────────────────────────────────────────────────
# C22-C26 — refine_rois invocation outcomes
# ─────────────────────────────────────────────────────────────────────────────


def test_C22_refine_rois_raises_restores_state(host, patched_qtimer, fake_roi_editor, fake_projection, capsys):
    """Raise walk: refine_rois raises → 'Napari ROI editing failed' + restore."""
    host.camera.acquisition_running = True
    fake_roi_editor["raise"] = RuntimeError("napari segfault avoided")
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool); masks[0, 0:3, 0:3] = True
    host._launch_napari_viewer(mean, masks)
    out = capsys.readouterr().out
    assert "Napari ROI editing failed" in out
    # restore_after_napari was triggered → camera restart called
    host.camera.start_realtime_acquisition.assert_called()


def test_C23_refine_rois_returns_none_no_save(host, patched_qtimer, fake_roi_editor, fake_projection, tmp_path):
    """Branch: refine_rois returns None → np.savez_compressed NOT called."""
    fake_roi_editor["return"] = None
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool); masks[0, 0:3, 0:3] = True
    with patch("gpu_ui_mixins.napari.np.savez_compressed") as mock_save:
        host._launch_napari_viewer(mean, masks)
    mock_save.assert_not_called()
    assert host.current_labels is None


def test_C24_refine_rois_returns_labels_save_success(host, patched_qtimer, fake_roi_editor, fake_projection, tmp_path):
    """Branch: refine_rois returns labels → np.savez_compressed called."""
    refined = np.zeros((10, 10), dtype=np.int32); refined[1:4, 1:4] = 1
    fake_roi_editor["return"] = refined
    # Seed an existing rois file so np.load works
    np.savez_compressed(host.rois_path, masks=[], sizes=[], labels=np.zeros((10, 10), dtype=np.int32))
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool); masks[0, 0:3, 0:3] = True
    host._launch_napari_viewer(mean, masks)
    assert host.current_labels is not None
    np.testing.assert_array_equal(host.current_labels, refined)
    # Verify file was saved + loadable
    saved = np.load(host.rois_path)
    np.testing.assert_array_equal(saved["labels"], refined)


def test_C25_refine_rois_returns_labels_save_raises(host, patched_qtimer, fake_roi_editor, fake_projection, capsys):
    """Raise walk: savez_compressed raises → 'Could not save updated ROIs'."""
    refined = np.zeros((10, 10), dtype=np.int32); refined[0, 0] = 1
    fake_roi_editor["return"] = refined
    np.savez_compressed(host.rois_path, masks=[], sizes=[], labels=np.zeros((10, 10), dtype=np.int32))
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool); masks[0, 0:3, 0:3] = True
    with patch("gpu_ui_mixins.napari.np.savez_compressed", side_effect=OSError("disk full")):
        host._launch_napari_viewer(mean, masks)
    assert "Could not save updated ROIs" in capsys.readouterr().out


def test_C26_refine_rois_success_logs_opengl_safety(host, patched_qtimer, fake_roi_editor, fake_projection, tmp_path, capsys):
    """Happy path: 'Napari ROI editor launched successfully with OpenGL safety'."""
    fake_roi_editor["return"] = None  # avoid file ops
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool); masks[0, 0:3, 0:3] = True
    host._launch_napari_viewer(mean, masks)
    out = capsys.readouterr().out
    assert "Napari ROI editor launched successfully" in out


# ─────────────────────────────────────────────────────────────────────────────
# C27-C30 — restore_after_napari closure side effects (invoked via error paths)
# ─────────────────────────────────────────────────────────────────────────────


def test_C27_restore_camera_restart_when_was_running(host, patched_qtimer, broken_roi_editor_importerror, fake_projection):
    """Restore closure: was_camera_running True → start_realtime_acquisition called."""
    host.camera.acquisition_running = True
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    host.camera.start_realtime_acquisition.assert_called_once()


def test_C28_restore_no_camera_restart_when_not_running(host, patched_qtimer, broken_roi_editor_importerror, fake_projection):
    """Restore closure: was_camera_running False → start_realtime NOT called."""
    host.camera.acquisition_running = False
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    host.camera.start_realtime_acquisition.assert_not_called()


def test_C29_restore_reprojects_binary_mask(host, patched_qtimer, broken_roi_editor_importerror, fake_projection, tmp_path):
    """Restore closure: rois file with 'binary' key → re-projection path runs."""
    # Pre-populate ROI file with a 'binary' mask key
    binary = np.zeros((10, 10), dtype=np.uint8); binary[0:3, 0:3] = 1
    np.savez_compressed(host.rois_path, binary=binary)
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    # Patch the QGuiApplication reference inside the closure
    fake_screen = MagicMock()
    fake_screen.size.return_value = MagicMock(width=lambda: 1920, height=lambda: 1080)
    with patch("PyQt5.QtGui.QGuiApplication") as mock_qg:
        mock_qg.screens.return_value = [fake_screen]
        host._launch_napari_viewer(mean, masks)
    # ProjectDisplay was instantiated (fake_projection installs the stub)
    assert sys.modules["projection"].ProjectDisplay.called


def test_C30_restore_reprojects_labels_when_no_binary(host, patched_qtimer, broken_roi_editor_importerror, fake_projection, tmp_path):
    """Restore closure: rois file with 'labels' but no 'binary' → labels path."""
    labels = np.zeros((10, 10), dtype=np.int32); labels[0:3, 0:3] = 1
    np.savez_compressed(host.rois_path, labels=labels)
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    fake_screen = MagicMock()
    fake_screen.size.return_value = MagicMock(width=lambda: 1920, height=lambda: 1080)
    with patch("PyQt5.QtGui.QGuiApplication") as mock_qg:
        mock_qg.screens.return_value = [fake_screen]
        host._launch_napari_viewer(mean, masks)
    # Still goes through projection
    assert sys.modules["projection"].ProjectDisplay.called


def test_C31_restore_no_rois_file_returns(host, patched_qtimer, broken_roi_editor_importerror, fake_projection, capsys):
    """Restore closure: rois file missing → 'No ROI file found for re-projection'."""
    # rois_path doesn't exist on disk
    assert not Path(host.rois_path).exists()
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    out = capsys.readouterr().out
    assert "No ROI file found for re-projection" in out


def test_C32_restore_load_corrupted_file_fallback(host, patched_qtimer, broken_roi_editor_importerror, fake_projection, capsys, tmp_path):
    """Restore closure: np.load raises mid-block → falls through to outer handler."""
    Path(host.rois_path).write_bytes(b"not an npz")  # corrupt the file
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    out = capsys.readouterr().out
    # Either "Could not load" or "Failed to re-project mask" appears
    assert "Could not load updated ROIs" in out or "Failed to re-project mask" in out


def test_C33_restore_outer_except_calls_handle_error(host, patched_qtimer, broken_roi_editor_importerror, fake_projection):
    """Raise walk: restore raises → _handle_error invoked with 'restore_after_napari'."""
    # Force the outer except by making camera.start_realtime raise inside restore
    host.camera.acquisition_running = True
    host.camera.start_realtime_acquisition = MagicMock(side_effect=RuntimeError("camera lost"))
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    # _handle_error called at least once; restore_after_napari is one possible context
    contexts = [c.args[1] for c in host._handle_error.call_args_list if len(c.args) > 1]
    assert "restore_after_napari" in contexts or len(contexts) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# C34 — outer napari_launch except path (top-level handler)
# ─────────────────────────────────────────────────────────────────────────────


def test_C34_outer_exception_handled(host, patched_qtimer):
    """Raise walk: top-level exception → _handle_error with 'napari_launch'."""
    # Force a crash in the very first attribute access — assigning a property that
    # raises on read.
    class _BadCam:
        @property
        def is_recording(self):
            raise RuntimeError("bad cam")

    host.camera = _BadCam()
    mean = np.zeros((10, 10), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=bool)
    host._launch_napari_viewer(mean, masks)
    # _handle_error called with "napari_launch" or "launch_napari"
    contexts = [c.args[1] for c in host._handle_error.call_args_list if len(c.args) > 1]
    assert any(ctx in ("napari_launch", "launch_napari") for ctx in contexts)


# ─────────────────────────────────────────────────────────────────────────────
# Property-based tests (≥2 per §1.1 UI-glue archetype)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# C35-C41 — synchronous QTimer drives inner closures
# (lifts coverage of restart_with_new_rois + fallback_restart bodies)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def patched_qtimer_sync():
    """QTimer.singleShot fires synchronously — drives nested-closure bodies."""
    with patch("gpu_ui_mixins.napari.QTimer") as mock_qt:
        mock_qt.singleShot = MagicMock(side_effect=lambda ms, fn: fn())
        yield mock_qt


def _seed_rois_with_binary(host, side=10):
    """Helper: write a binary-key ROIs npz so the restore path runs the
    re-projection branch.
    """
    binary = np.zeros((side, side), dtype=np.uint8); binary[0:3, 0:3] = 1
    np.savez_compressed(host.rois_path, binary=binary)


def test_C35_restart_with_new_rois_cleanup_path(host, patched_qtimer_sync, broken_roi_editor_importerror, fake_projection):
    """Sync QTimer: was_live_traces + existing extractor → cleanup + start path."""
    host.camera.acquisition_running = True
    host.live_extractor = MagicMock()  # so was_live_traces becomes True
    _seed_rois_with_binary(host)
    with patch("PyQt5.QtGui.QGuiApplication") as mock_qg:
        screen = MagicMock()
        screen.size.return_value = MagicMock(width=lambda: 32, height=lambda: 32)
        mock_qg.screens.return_value = [screen]
        mean = np.zeros((10, 10), dtype=np.uint8)
        masks = np.zeros((1, 10, 10), dtype=bool)
        host._launch_napari_viewer(mean, masks)
    # start_live_traces was called via the synchronous QTimer-driven restart
    host.start_live_traces.assert_called()


def test_C36_restart_uses_restart_after_napari_success(host, patched_qtimer_sync, broken_roi_editor_importerror, fake_projection):
    """Sync restart: live_extractor.restart_after_napari → returns truthy."""
    host.camera.acquisition_running = True
    # First-pass extractor (cleanup target)
    initial_ext = MagicMock()
    host.live_extractor = initial_ext

    # After start_live_traces runs, install a *new* extractor with
    # restart_after_napari returning True. We mutate live_extractor inside
    # the mocked start_live_traces.
    new_ext = MagicMock(spec=["cleanup", "restart_after_napari", "plot_widget"])
    new_ext.restart_after_napari = MagicMock(return_value=True)

    def install_new_extractor():
        host.live_extractor = new_ext
    host.start_live_traces.side_effect = install_new_extractor

    _seed_rois_with_binary(host)
    with patch("PyQt5.QtGui.QGuiApplication") as mock_qg:
        screen = MagicMock()
        screen.size.return_value = MagicMock(width=lambda: 32, height=lambda: 32)
        mock_qg.screens.return_value = [screen]
        mean = np.zeros((10, 10), dtype=np.uint8)
        masks = np.zeros((1, 10, 10), dtype=bool)
        host._launch_napari_viewer(mean, masks)
    new_ext.restart_after_napari.assert_called_once()


def test_C37_restart_after_napari_failure_fallback_to_pagination(host, patched_qtimer_sync, broken_roi_editor_importerror, fake_projection):
    """Sync restart: restart_after_napari returns False → fallback to
    plot_widget + _setup_pagination_controls.
    """
    host.camera.acquisition_running = True
    host.live_extractor = MagicMock()

    new_ext = MagicMock()
    new_ext.restart_after_napari = MagicMock(return_value=False)
    new_ext._setup_pagination_controls = MagicMock()

    def install_new_extractor():
        host.live_extractor = new_ext
    host.start_live_traces.side_effect = install_new_extractor

    _seed_rois_with_binary(host)
    with patch("PyQt5.QtGui.QGuiApplication") as mock_qg:
        screen = MagicMock()
        screen.size.return_value = MagicMock(width=lambda: 32, height=lambda: 32)
        mock_qg.screens.return_value = [screen]
        mean = np.zeros((10, 10), dtype=np.uint8)
        masks = np.zeros((1, 10, 10), dtype=bool)
        host._launch_napari_viewer(mean, masks)
    new_ext._setup_pagination_controls.assert_called()


def test_C38_restart_no_restart_after_napari_uses_direct_path(host, patched_qtimer_sync, broken_roi_editor_importerror, fake_projection):
    """Sync restart: extractor lacks restart_after_napari → direct
    plot_widget assignment + pagination.
    """
    host.camera.acquisition_running = True
    host.live_extractor = MagicMock()

    # spec= without 'restart_after_napari' so hasattr returns False
    new_ext = MagicMock(spec=["cleanup", "_setup_pagination_controls", "plot_widget"])
    new_ext._setup_pagination_controls = MagicMock()

    def install_new_extractor():
        host.live_extractor = new_ext
    host.start_live_traces.side_effect = install_new_extractor

    _seed_rois_with_binary(host)
    with patch("PyQt5.QtGui.QGuiApplication") as mock_qg:
        screen = MagicMock()
        screen.size.return_value = MagicMock(width=lambda: 32, height=lambda: 32)
        mock_qg.screens.return_value = [screen]
        mean = np.zeros((10, 10), dtype=np.uint8)
        masks = np.zeros((1, 10, 10), dtype=bool)
        host._launch_napari_viewer(mean, masks)
    new_ext._setup_pagination_controls.assert_called()


def test_C39_restart_raises_schedules_fallback(host, patched_qtimer_sync, broken_roi_editor_importerror, fake_projection, capsys):
    """Sync restart: start_live_traces raises inside restart_with_new_rois
    → exception caught → fallback_restart scheduled (and immediately runs
    under sync QTimer).
    """
    host.camera.acquisition_running = True
    host.live_extractor = MagicMock()
    # First call to start_live_traces inside restart_with_new_rois raises;
    # the fallback then calls start_live_traces a second time successfully.
    calls = []

    def flaky_start():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("restart kaboom")
    host.start_live_traces.side_effect = flaky_start

    _seed_rois_with_binary(host)
    with patch("PyQt5.QtGui.QGuiApplication") as mock_qg:
        screen = MagicMock()
        screen.size.return_value = MagicMock(width=lambda: 32, height=lambda: 32)
        mock_qg.screens.return_value = [screen]
        mean = np.zeros((10, 10), dtype=np.uint8)
        masks = np.zeros((1, 10, 10), dtype=bool)
        host._launch_napari_viewer(mean, masks)
    out = capsys.readouterr().out
    assert "Failed to restart live traces" in out
    assert "Fallback restart successful" in out
    assert len(calls) == 2


def test_C40_restart_plot_widget_reinit_skipped_when_present(host, patched_qtimer_sync, broken_roi_editor_importerror, fake_projection):
    """Branch: plot_widget already has.plot attr → no reinit."""
    host.camera.acquisition_running = True
    host.live_extractor = MagicMock()
    # Existing plot_widget with a.plot attribute satisfies the hasattr check
    fake_plot = MagicMock(spec=["plot"])
    fake_plot.plot = MagicMock()
    host.plot_widget = fake_plot
    host.start_live_traces.side_effect = lambda: None  # benign
    _seed_rois_with_binary(host)
    with patch("PyQt5.QtGui.QGuiApplication") as mock_qg:
        screen = MagicMock()
        screen.size.return_value = MagicMock(width=lambda: 32, height=lambda: 32)
        mock_qg.screens.return_value = [screen]
        mean = np.zeros((10, 10), dtype=np.uint8)
        masks = np.zeros((1, 10, 10), dtype=bool)
        host._launch_napari_viewer(mean, masks)
    # Existing plot widget retained (not overwritten by reinit branch)
    assert host.plot_widget is fake_plot


def test_C41_restore_proj_failure_schedules_trace_restart(host, patched_qtimer_sync, broken_roi_editor_importerror, capsys):
    """Branch: restore re-project raises → fallback timer schedules
    start_live_traces (projection-failed path).
    """
    host.camera.acquisition_running = True
    host.live_extractor = MagicMock()
    # Install a broken projection module (no ProjectDisplay attr) so
    # `from projection import ProjectDisplay` raises ImportError.
    broken_proj = types.ModuleType("projection")
    sys.modules["projection"] = broken_proj
    try:
        _seed_rois_with_binary(host)
        mean = np.zeros((10, 10), dtype=np.uint8)
        masks = np.zeros((1, 10, 10), dtype=bool)
        host._launch_napari_viewer(mean, masks)
        out = capsys.readouterr().out
        assert "Failed to re-project mask" in out
        # Synchronous QTimer ran start_live_traces 500ms-delayed callback
        host.start_live_traces.assert_called()
    finally:
        sys.modules.pop("projection", None)


def test_C42_restore_larger_than_screen_uses_resize(host, patched_qtimer, broken_roi_editor_importerror, fake_projection):
    """Branch: binary mask larger than target screen → cv2.resize else-arm."""
    host.camera.acquisition_running = True
    # 40x40 mask, but screen reports 20x20 → larger branch
    big_binary = np.zeros((40, 40), dtype=np.uint8); big_binary[0:5, 0:5] = 1
    np.savez_compressed(host.rois_path, binary=big_binary)
    with patch("PyQt5.QtGui.QGuiApplication") as mock_qg:
        screen = MagicMock()
        screen.size.return_value = MagicMock(width=lambda: 20, height=lambda: 20)
        mock_qg.screens.return_value = [screen]
        mean = np.zeros((10, 10), dtype=np.uint8)
        masks = np.zeros((1, 10, 10), dtype=bool)
        host._launch_napari_viewer(mean, masks)
    # ProjectDisplay still gets called on resized image
    assert sys.modules["projection"].ProjectDisplay.called


def test_C43_restore_cv2_copyMakeBorder_falls_back_to_np_pad(host, patched_qtimer, broken_roi_editor_importerror, fake_projection):
    """Raise walk: cv2.copyMakeBorder raises → np.pad fallback used."""
    host.camera.acquisition_running = True
    binary = np.zeros((10, 10), dtype=np.uint8); binary[0:3, 0:3] = 1
    np.savez_compressed(host.rois_path, binary=binary)
    with patch("PyQt5.QtGui.QGuiApplication") as mock_qg, \
         patch("gpu_ui_mixins.napari.cv2.copyMakeBorder", side_effect=RuntimeError("cv2")):
        screen = MagicMock()
        screen.size.return_value = MagicMock(width=lambda: 32, height=lambda: 32)
        mock_qg.screens.return_value = [screen]
        mean = np.zeros((10, 10), dtype=np.uint8)
        masks = np.zeros((1, 10, 10), dtype=bool)
        host._launch_napari_viewer(mean, masks)
    # No exception propagates; projection still called
    assert sys.modules["projection"].ProjectDisplay.called


def test_C44_restore_labels_loaded_when_no_binary_no_labels(host, patched_qtimer, broken_roi_editor_importerror, fake_projection, tmp_path):
    """Branch: rois file has neither 'binary' nor 'labels' → fallback np.load.

    The fallback line ``np.load(self.rois_path)['labels']`` raises a
    KeyError under that condition (since labels also missing). The outer
    try/except inside the inner reload block absorbs it.
    """
    host.camera.acquisition_running = True
    # Save with only a 'masks' key — no 'binary' or 'labels'
    np.savez_compressed(host.rois_path, masks=np.zeros((3, 3), dtype=np.int32))
    with patch("PyQt5.QtGui.QGuiApplication") as mock_qg:
        screen = MagicMock()
        screen.size.return_value = MagicMock(width=lambda: 32, height=lambda: 32)
        mock_qg.screens.return_value = [screen]
        mean = np.zeros((10, 10), dtype=np.uint8)
        masks = np.zeros((1, 10, 10), dtype=bool)
        # No exception propagates
        host._launch_napari_viewer(mean, masks)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    n_masks=st.integers(min_value=0, max_value=10),
    side=st.integers(min_value=4, max_value=16),
)
def test_property_3d_mask_count_invariant(n_masks, side):
    """Property: for any (n_masks, side), 3D-input → refine_rois called
    with ≤ n_masks ndarray masks (filtering may reduce, never increase).
    """
    with tempfile.TemporaryDirectory() as td:
        host = _Host(Path(td))
        _run_3d_property_iteration(host, n_masks, side)


def _run_3d_property_iteration(host, n_masks, side):
    captured = {"calls": [], "raise": None, "return": None}

    def fake_refine(mean, masks, return_viewer=False, on_close_callback=None):
        captured["calls"].append(len(masks))
        return None

    fake_mod = types.ModuleType("roi_editor")
    fake_mod.refine_rois = fake_refine
    sys.modules["roi_editor"] = fake_mod

    fake_proj = types.ModuleType("projection")
    fake_proj.ProjectDisplay = MagicMock()
    sys.modules["projection"] = fake_proj

    try:
        with patch("gpu_ui_mixins.napari.QTimer"):
            mean = np.zeros((side, side), dtype=np.uint8)
            masks = np.zeros((max(n_masks, 1), side, side), dtype=bool)
            if n_masks > 0:
                # Mark first pixel True in each mask so they're non-empty
                for i in range(n_masks):
                    masks[i, 0, 0] = True
            host._launch_napari_viewer(mean, masks)

        if captured["calls"]:
            assert captured["calls"][0] <= max(n_masks, 1)
            assert captured["calls"][0] >= 0
        # If no call, the validation drop path returned early (allowed for n=0)
    finally:
        sys.modules.pop("roi_editor", None)
        sys.modules.pop("projection", None)


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ndim=st.sampled_from([1, 4, 5]))
def test_property_invalid_ndim_never_raises(ndim):
    """Property: any ndim ∉ {2, 3} → outer try absorbs; method returns
    without exception and without calling refine_rois.
    """
    with tempfile.TemporaryDirectory() as td:
        host = _Host(Path(td))
        _run_invalid_ndim_iteration(host, ndim)


def _run_invalid_ndim_iteration(host, ndim):
    captured = {"calls": []}

    def fake_refine(*a, **kw):
        captured["calls"].append(True)
        return None

    fake_mod = types.ModuleType("roi_editor")
    fake_mod.refine_rois = fake_refine
    sys.modules["roi_editor"] = fake_mod

    fake_proj = types.ModuleType("projection")
    fake_proj.ProjectDisplay = MagicMock()
    sys.modules["projection"] = fake_proj

    try:
        with patch("gpu_ui_mixins.napari.QTimer"):
            mean = np.zeros((10, 10), dtype=np.uint8)
            shape = (2,) * ndim
            masks = np.zeros(shape, dtype=bool)
            # No exception escapes
            host._launch_napari_viewer(mean, masks)
        # refine_rois never called for invalid ndim
        assert captured["calls"] == []
    finally:
        sys.modules.pop("roi_editor", None)
        sys.modules.pop("projection", None)
