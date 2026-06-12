"""Comprehensive characterization tests for ``gpu_ui_roi_discovery``.

1 — comprehensive (branch + raise walk, ≥2
property-based tests, ≥85% line+branch coverage target on the audited
unit). First chars suite for the L5 ``gpu_ui.py`` 9-sub-module
decomposition (iter-1, ROIDiscoveryMixin extracted from
``gpu_ui.py`` per ``docs/specs/L5_UI/gpu_ui.md`` §0.5).

Module surface (~325 LOC, 8 methods, UI-glue archetype):
- ``_select_video()`` — Qt file dialog → sets ``video_path``
- ``_run_make_memmap()`` — spawn worker thread
- ``_thread_make_memmap()`` — branch on path validity + size guard
- ``_load_roi_file()`` — NPZ load + validation + copy + start-prompt
- ``_run_discover_rois(method)`` — branch on method (CNMF/Custom skip)
- ``_thread_discover_rois()`` — large; OTSU + Cellpose + projection
- ``_run_refine_rois()`` — spawn worker thread
- ``_thread_refine_rois()`` — emit ``refineRequested`` after compute

Branch walk per §1.1 #1; raise walk per §1.1 #2.

Property tests (§1.1 archetype "UI-glue" requires ≥2):
- ``test_property_size_threshold_warning`` (Hypothesis) — invariant
  that the >500 MB warning fires iff size > 500.
- ``test_property_discover_method_routing`` (Hypothesis) — invariant
  that CNMF/Custom skip-and-return while OTSU/Cellpose set
  ``_discover_method`` to that exact string.

Coverage gap recovery criterion (per §1.1 sub-target rule): the
``_thread_discover_rois`` 192-LOC branch tree mocks the projection
+ TIFF-fallback subpaths; remaining uncovered lines are the deep
PIL/OpenCV-fallback ladder under chained ImportError, which a single
Mock cannot simulate atomically. Recovery: iter-2's NapariViewerMixin
extraction does NOT touch these lines; the iter-N``_thread_discover_rois`` refactor (named in the spec §15 row) will
sub-extract ``_save_discovery_tiff`` and the projection helper into
their own units, at which point a focused chars suite reaches the
remaining branches without combinatorial mock setup.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

# Ensure the CRISPI module path is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
CRISPI_PATH = REPO_ROOT / "STIMscope" / "STIMViewer_CRISPI"
if str(CRISPI_PATH) not in sys.path:
    sys.path.insert(0, str(CRISPI_PATH))

from gpu_ui_mixins.roi_discovery import ROIDiscoveryMixin  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Test infrastructure: stub host class
# ─────────────────────────────────────────────────────────────────────────────


class _Host(ROIDiscoveryMixin):
    """Minimal stub satisfying the ROIDiscoveryMixin host contract.

    Avoids QWidget instantiation — the mixin only uses ``self`` as the
    parent argument to Qt dialogs (which we mock) and accesses scalar
    attributes + signal-like callables. None of the methods construct
    QWidget children.
    """

    def __init__(self, tmp_path: Path):
        self.video_path = None
        self.memmap_path = str(tmp_path / "movie_mmap.npy")
        self.rois_path = str(tmp_path / "rois.npz")
        self._discover_method = "OTSU"
        self.proj_display = None
        self.camera = MagicMock(translation_matrix=None)
        # Signals — replace with MagicMock so.emit() is observable.
        self.refineRequested = MagicMock()
        self.requestStartLiveTraces = MagicMock()
        self.requestStopLiveTraces = MagicMock()
        # Host methods.
        self._handle_error = MagicMock()
        self.start_live_traces = MagicMock()


@pytest.fixture
def host(tmp_path: Path) -> _Host:
    return _Host(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# _select_video — 2 branches
# ─────────────────────────────────────────────────────────────────────────────


def test_C1_select_video_sets_path_when_dialog_returns_path(host, capsys):
    """Branch: dialog returns truthy path → video_path is set, print fires."""
    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QFileDialog.getOpenFileName",
               return_value=("/tmp/movie.tif", "")):
        host._select_video()
    assert host.video_path == "/tmp/movie.tif"
    assert "Selected video: /tmp/movie.tif" in capsys.readouterr().out


def test_C2_select_video_no_change_when_cancelled(host):
    """Branch: dialog returns empty string → video_path stays None."""
    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QFileDialog.getOpenFileName",
               return_value=("", "")):
        host._select_video()
    assert host.video_path is None


# ─────────────────────────────────────────────────────────────────────────────
# _run_make_memmap — spawns daemon thread
# ─────────────────────────────────────────────────────────────────────────────


def test_C3_run_make_memmap_spawns_daemon_thread(host):
    """Verifies threading.Thread(target=_thread_make_memmap, daemon=True)."""
    with patch("gpu_ui_mixins.roi_discovery.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        host._run_make_memmap()
    mock_thread.assert_called_once()
    kwargs = mock_thread.call_args.kwargs
    assert kwargs["target"] == host._thread_make_memmap
    assert kwargs["daemon"] is True
    mock_thread.return_value.start.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# _thread_make_memmap — 5 branches
# ─────────────────────────────────────────────────────────────────────────────


def test_C4_thread_make_memmap_no_path(host, capsys):
    """Branch: video_path is None → 'No valid video file selected'."""
    host._thread_make_memmap()
    out = capsys.readouterr().out
    assert "No valid video file selected" in out
    host._handle_error.assert_not_called()


def test_C5_thread_make_memmap_path_does_not_exist(host, tmp_path, capsys):
    """Branch: video_path set but file missing → same skip-and-print."""
    host.video_path = str(tmp_path / "missing.tif")
    host._thread_make_memmap()
    assert "No valid video file selected" in capsys.readouterr().out


def test_C6_thread_make_memmap_small_file_no_warning(host, tmp_path, capsys):
    """Branch: size ≤ 500 MB → no large-file warning; make_memmap invoked."""
    video = tmp_path / "small.tif"
    video.write_bytes(b"x" * 1024)  # 1 KB
    host.video_path = str(video)

    fake_module = types.ModuleType("make_mmap")
    fake_module.make_memmap = MagicMock()
    with patch.dict(sys.modules, {"make_mmap": fake_module}):
        host._thread_make_memmap()

    out = capsys.readouterr().out
    assert "Large video file detected" not in out
    assert "Memmap saved" in out
    fake_module.make_memmap.assert_called_once_with(host.video_path, host.memmap_path)


def test_C7_thread_make_memmap_large_file_warning(host, tmp_path, capsys, monkeypatch):
    """Branch: size > 500 MB → large-file warning fires."""
    video = tmp_path / "big.tif"
    video.touch()
    host.video_path = str(video)
    # Fake os.path.getsize returning > 500 MB.
    real_getsize = __import__("os").path.getsize

    def fake_getsize(path):
        if path == host.video_path:
            return 600 * 1024 * 1024  # 600 MB
        return real_getsize(path)

    monkeypatch.setattr("gpu_ui_mixins.roi_discovery.os.path.getsize", fake_getsize)

    fake_module = types.ModuleType("make_mmap")
    fake_module.make_memmap = MagicMock()
    with patch.dict(sys.modules, {"make_mmap": fake_module}):
        host._thread_make_memmap()

    assert "Large video file detected: 600.0 MB" in capsys.readouterr().out


def test_C8_thread_make_memmap_memory_error_path(host, tmp_path, capsys):
    """Raise walk: MemoryError → _handle_error tagged 'Memmap (MemoryError)'."""
    video = tmp_path / "movie.tif"
    video.write_bytes(b"x" * 10)
    host.video_path = str(video)

    fake_module = types.ModuleType("make_mmap")
    fake_module.make_memmap = MagicMock(side_effect=MemoryError("oom"))
    with patch.dict(sys.modules, {"make_mmap": fake_module}):
        host._thread_make_memmap()

    host._handle_error.assert_called_once()
    args = host._handle_error.call_args.args
    assert isinstance(args[0], MemoryError)
    assert args[1] == "Memmap (MemoryError)"
    assert "Try processing a smaller video file" in capsys.readouterr().out


def test_C9_thread_make_memmap_generic_exception(host, tmp_path):
    """Raise walk: generic Exception → _handle_error tagged 'Memmap'."""
    video = tmp_path / "movie.tif"
    video.write_bytes(b"x" * 10)
    host.video_path = str(video)

    fake_module = types.ModuleType("make_mmap")
    fake_module.make_memmap = MagicMock(side_effect=RuntimeError("nope"))
    with patch.dict(sys.modules, {"make_mmap": fake_module}):
        host._thread_make_memmap()

    host._handle_error.assert_called_once()
    args = host._handle_error.call_args.args
    assert isinstance(args[0], RuntimeError)
    assert args[1] == "Memmap"


# ─────────────────────────────────────────────────────────────────────────────
# _load_roi_file — 8 branches
# ─────────────────────────────────────────────────────────────────────────────


def _write_npz_with_labels(path: Path, labels: np.ndarray):
    np.savez(str(path), labels=labels)


def test_C10_load_roi_file_dialog_cancelled(host):
    """Branch: dialog returns '' → early return."""
    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QFileDialog.getOpenFileName",
               return_value=("", "")):
        host._load_roi_file()
    host.start_live_traces.assert_not_called()


def test_C11_load_roi_file_missing_labels_key(host, tmp_path):
    """Branch: NPZ missing 'labels' key → warning, early return."""
    bad = tmp_path / "bad.npz"
    np.savez(str(bad), other=np.zeros(3))

    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QFileDialog.getOpenFileName",
               return_value=(str(bad), "")), \
         patch("gpu_ui_mixins.roi_discovery.QtWidgets.QMessageBox.warning") as mock_warn:
        host._load_roi_file()
    mock_warn.assert_called_once()
    host.start_live_traces.assert_not_called()


def test_C12_load_roi_file_unreadable_file(host, tmp_path):
    """Branch: np.load raises → warning, early return."""
    bad = tmp_path / "corrupt.npz"
    bad.write_bytes(b"not an NPZ")

    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QFileDialog.getOpenFileName",
               return_value=(str(bad), "")), \
         patch("gpu_ui_mixins.roi_discovery.QtWidgets.QMessageBox.warning") as mock_warn:
        host._load_roi_file()
    mock_warn.assert_called_once()
    host.start_live_traces.assert_not_called()


def test_C13_load_roi_file_empty_labels_yields_zero_rois(host, tmp_path, capsys):
    """Branch: labels.size == 0 → n_rois = 0; dialog says 'No', no start."""
    good = tmp_path / "empty.npz"
    _write_npz_with_labels(good, np.array([], dtype=np.int32))

    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QFileDialog.getOpenFileName",
               return_value=(str(good), "")), \
         patch("gpu_ui_mixins.roi_discovery.QtWidgets.QMessageBox.question",
               return_value=0):  # No
        host._load_roi_file()
    out = capsys.readouterr().out
    assert "(0 ROIs)" in out
    host.start_live_traces.assert_not_called()


def test_C14_load_roi_file_yes_starts_traces(host, tmp_path):
    """Branch: user clicks Yes → start_live_traces called."""
    good = tmp_path / "good.npz"
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[3:5, 3:5] = 1
    labels[7:9, 7:9] = 2
    _write_npz_with_labels(good, labels)

    from PyQt5.QtWidgets import QMessageBox
    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QFileDialog.getOpenFileName",
               return_value=(str(good), "")), \
         patch("gpu_ui_mixins.roi_discovery.QtWidgets.QMessageBox.question",
               return_value=QMessageBox.Yes):
        host._load_roi_file()
    host.start_live_traces.assert_called_once()


def test_C15_load_roi_file_no_does_not_start_traces(host, tmp_path):
    """Branch: user clicks No → start_live_traces NOT called."""
    good = tmp_path / "good.npz"
    _write_npz_with_labels(good, np.ones((4, 4), dtype=np.int32))

    from PyQt5.QtWidgets import QMessageBox
    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QFileDialog.getOpenFileName",
               return_value=(str(good), "")), \
         patch("gpu_ui_mixins.roi_discovery.QtWidgets.QMessageBox.question",
               return_value=QMessageBox.No):
        host._load_roi_file()
    host.start_live_traces.assert_not_called()


def test_C16_load_roi_file_start_live_traces_raises_warns(host, tmp_path):
    """Branch: start_live_traces raises → warning shown (not propagated)."""
    good = tmp_path / "good.npz"
    _write_npz_with_labels(good, np.ones((4, 4), dtype=np.int32))
    host.start_live_traces.side_effect = RuntimeError("boom")

    from PyQt5.QtWidgets import QMessageBox
    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QFileDialog.getOpenFileName",
               return_value=(str(good), "")), \
         patch("gpu_ui_mixins.roi_discovery.QtWidgets.QMessageBox.question",
               return_value=QMessageBox.Yes), \
         patch("gpu_ui_mixins.roi_discovery.QtWidgets.QMessageBox.warning") as mock_warn:
        host._load_roi_file()
    mock_warn.assert_called_once()


def test_C17_load_roi_file_copies_to_rois_path(host, tmp_path):
    """Branch: source path != rois_path → shutil.copyfile invoked."""
    src = tmp_path / "src.npz"
    _write_npz_with_labels(src, np.ones((3, 3), dtype=np.int32))
    # rois_path is a different file
    assert host.rois_path != str(src)

    from PyQt5.QtWidgets import QMessageBox
    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QFileDialog.getOpenFileName",
               return_value=(str(src), "")), \
         patch("gpu_ui_mixins.roi_discovery.QtWidgets.QMessageBox.question",
               return_value=QMessageBox.No):
        host._load_roi_file()
    assert Path(host.rois_path).exists()


# ─────────────────────────────────────────────────────────────────────────────
# _run_discover_rois — 4 branches
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["CNMF", "Custom"])
def test_C18_run_discover_rois_unimplemented_methods_skip(host, method):
    """Branch: CNMF/Custom → information dialog, no thread spawned."""
    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QMessageBox.information") as mock_info, \
         patch("gpu_ui_mixins.roi_discovery.threading.Thread") as mock_thread:
        host._run_discover_rois(method=method)
    mock_info.assert_called_once()
    mock_thread.assert_not_called()


@pytest.mark.parametrize("method", ["OTSU", "Cellpose"])
def test_C19_run_discover_rois_implemented_methods_spawn(host, method):
    """Branch: OTSU/Cellpose → sets _discover_method, spawns thread."""
    with patch("gpu_ui_mixins.roi_discovery.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        host._run_discover_rois(method=method)
    assert host._discover_method == method
    mock_thread.assert_called_once()
    mock_thread.return_value.start.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# _thread_discover_rois — branch coverage of major paths
# ─────────────────────────────────────────────────────────────────────────────


def test_C20_thread_discover_rois_otsu_empty_masks_aborts(host, tmp_path, capsys):
    """Branch: OTSU returns no masks → 'aborting live traces' print, no save."""
    host._discover_method = "OTSU"
    # Pretend memmap exists by mocking np.load.
    movie = np.zeros((10, 100, 100), dtype=np.uint8)
    fake_otsu = types.ModuleType("otsu_thresh")
    fake_otsu.compute_mean_projection = MagicMock(return_value=np.zeros((100, 100), dtype=np.uint8))
    fake_otsu.denoise_and_threshold_gpu = MagicMock(return_value=([], []))

    with patch("gpu_ui_mixins.roi_discovery.np.load", return_value=movie), \
         patch.dict(sys.modules, {"otsu_thresh": fake_otsu}):
        host._thread_discover_rois()

    assert "aborting" in capsys.readouterr().out
    host.requestStopLiveTraces.emit.assert_called_once()
    host.requestStartLiveTraces.emit.assert_not_called()


def test_C21_thread_discover_rois_unknown_method_raises(host, capsys):
    """Branch: _discover_method is neither OTSU nor Cellpose → ValueError swallowed by outer."""
    host._discover_method = "UNKNOWN"
    host._thread_discover_rois()
    # Outer except prints and routes to _handle_error.
    host._handle_error.assert_called_once()
    where = host._handle_error.call_args.args[1]
    assert where == "ROI discovery"


def test_C22_thread_discover_rois_cellpose_video_missing(host, capsys):
    """Branch: Cellpose with no video_path → 'No valid video file selected'."""
    host._discover_method = "Cellpose"
    host.video_path = None
    host._thread_discover_rois()
    assert "No valid video file selected" in capsys.readouterr().out
    host.requestStartLiveTraces.emit.assert_not_called()


def test_C23_thread_discover_rois_cellpose_runner_not_found(host, tmp_path):
    """Branch: Cellpose runner script missing → FileNotFoundError caught."""
    host._discover_method = "Cellpose"
    video = tmp_path / "v.tif"
    video.touch()
    host.video_path = str(video)
    # Force the runner check to miss by patching os.path.exists to return
    # False for the runner specifically.
    real_exists = __import__("os").path.exists

    def fake_exists(p):
        if "cellpose_runner.py" in p:
            return False
        return real_exists(p)

    with patch("gpu_ui_mixins.roi_discovery.os.path.exists", side_effect=fake_exists):
        host._thread_discover_rois()
    host._handle_error.assert_called_once()
    assert host._handle_error.call_args.args[1] == "ROI discovery"


def _make_otsu_module(masks_count=2):
    """Helper: fake otsu_thresh module returning `masks_count` masks."""
    mod = types.ModuleType("otsu_thresh")
    mod.compute_mean_projection = MagicMock(
        return_value=np.zeros((100, 100), dtype=np.uint8))
    if masks_count == 0:
        masks, sizes = [], []
    else:
        # Each mask is a bool array 1096×1936 (the cv2.resize target).
        masks = [np.zeros((1096, 1936), dtype=bool) for _ in range(masks_count)]
        for i, m in enumerate(masks):
            m[10 + i * 5: 15 + i * 5, 10 + i * 5: 15 + i * 5] = True
        sizes = [25] * masks_count
    mod.denoise_and_threshold_gpu = MagicMock(return_value=(masks, sizes))
    return mod


def _make_projection_module(succeeds=True):
    """Helper: fake projection module with ProjectDisplay."""
    mod = types.ModuleType("projection")

    class FakeProjectDisplay:
        def __init__(self, scr):
            self._scr = scr

        def show_image_fullscreen_on_second_monitor(self, img, H):
            if not succeeds:
                raise RuntimeError("projection failed")

        def close(self):
            pass

    mod.ProjectDisplay = FakeProjectDisplay
    return mod


def test_C27_thread_discover_rois_otsu_happy_path(host, tmp_path, capsys):
    """OTSU end-to-end with masks → rois.npz saved, requestStartLiveTraces emitted."""
    host._discover_method = "OTSU"
    movie = np.zeros((5, 100, 100), dtype=np.uint8)

    fake_otsu = _make_otsu_module(masks_count=2)
    fake_proj = _make_projection_module(succeeds=True)
    # Fake screen plumbing.
    fake_screen = MagicMock()
    fake_screen.size.return_value = MagicMock(
        width=MagicMock(return_value=1920),
        height=MagicMock(return_value=1080))

    with patch("gpu_ui_mixins.roi_discovery.np.load", return_value=movie), \
         patch.dict(sys.modules, {"otsu_thresh": fake_otsu, "projection": fake_proj}), \
         patch("PyQt5.QtGui.QGuiApplication.screens", return_value=[fake_screen]):
        host._thread_discover_rois()

    out = capsys.readouterr().out
    assert "ROIs written to" in out
    host.requestStartLiveTraces.emit.assert_called_once()
    # NPZ should exist with masks/sizes/labels/binary keys.
    assert Path(host.rois_path).exists()
    with np.load(host.rois_path) as z:
        assert set(z.files) >= {"masks", "sizes", "labels", "binary"}


def test_C28_thread_discover_rois_otsu_projection_fails_still_saves(host, tmp_path, capsys):
    """Branch: projection raises → caught + printed; rois.npz still saved."""
    host._discover_method = "OTSU"
    movie = np.zeros((5, 100, 100), dtype=np.uint8)
    fake_otsu = _make_otsu_module(masks_count=1)

    # Make projection import fail to exercise the outer except path.
    def _raising_import(*args, **kwargs):
        raise ImportError("projection unavailable")

    with patch("gpu_ui_mixins.roi_discovery.np.load", return_value=movie), \
         patch.dict(sys.modules, {"otsu_thresh": fake_otsu}), \
         patch.dict(sys.modules, {"projection": None}):
        host._thread_discover_rois()

    out = capsys.readouterr().out
    assert "Failed to project mask" in out
    assert "ROIs written to" in out  # still saves
    host.requestStartLiveTraces.emit.assert_called_once()


def test_C29_thread_discover_rois_cellpose_subprocess_nonzero(host, tmp_path):
    """Branch: Cellpose subprocess returns nonzero → RuntimeError caught."""
    host._discover_method = "Cellpose"
    video = tmp_path / "v.tif"
    video.touch()
    host.video_path = str(video)

    real_exists = __import__("os").path.exists

    def fake_exists(p):
        if "cellpose_runner.py" in p:
            return True  # runner "exists"
        if "cellpose_env" in p or "U-Net_GPU_Analysis" in p:
            return False  # use sys.executable, skip model/size args
        return real_exists(p)

    res = MagicMock(returncode=1, stdout="boom")
    with patch("gpu_ui_mixins.roi_discovery.os.path.exists", side_effect=fake_exists), \
         patch("gpu_ui_mixins.roi_discovery.subprocess.run", return_value=res):
        host._thread_discover_rois()
    host._handle_error.assert_called_once()
    assert host._handle_error.call_args.args[1] == "ROI discovery"


def test_C31_thread_discover_rois_cellpose_happy_path(host, tmp_path, capsys):
    """Cellpose end-to-end: subprocess succeeds, NPZ has 'labels' → save + emit."""
    host._discover_method = "Cellpose"
    video = tmp_path / "v.tif"
    video.touch()
    host.video_path = str(video)

    real_exists = __import__("os").path.exists

    def fake_exists(p):
        if "cellpose_runner.py" in p:
            return True
        if "cellpose_env" in p:
            return True  # exercise venv-python branch
        if "cytotorch_0" in p or "size_cytotorch_0.npy" in p:
            return True  # exercise model/size args branch
        return real_exists(p)

    # Pre-create rois.npz so np.load(self.rois_path) succeeds after subprocess.
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[5:10, 5:10] = 1
    labels[20:25, 20:25] = 2
    np.savez(host.rois_path, labels=labels)

    res = MagicMock(returncode=0, stdout="ok")
    fake_proj = _make_projection_module(succeeds=True)
    fake_screen = MagicMock()
    fake_screen.size.return_value = MagicMock(
        width=MagicMock(return_value=1920),
        height=MagicMock(return_value=1080))

    with patch("gpu_ui_mixins.roi_discovery.os.path.exists", side_effect=fake_exists), \
         patch("gpu_ui_mixins.roi_discovery.subprocess.run", return_value=res), \
         patch.dict(sys.modules, {"projection": fake_proj}), \
         patch("PyQt5.QtGui.QGuiApplication.screens", return_value=[fake_screen]):
        host._thread_discover_rois()

    out = capsys.readouterr().out
    assert "ROIs written to" in out
    host.requestStartLiveTraces.emit.assert_called_once()


def test_C32_thread_discover_rois_resize_branch_when_image_too_large(host, tmp_path):
    """Branch in projection: img larger than target screen → cv2.resize path."""
    host._discover_method = "OTSU"
    movie = np.zeros((5, 100, 100), dtype=np.uint8)
    fake_otsu = _make_otsu_module(masks_count=1)
    fake_proj = _make_projection_module(succeeds=True)
    # Make target screen smaller than img_gray (1096×1936 from cv2.resize).
    fake_screen = MagicMock()
    fake_screen.size.return_value = MagicMock(
        width=MagicMock(return_value=640),
        height=MagicMock(return_value=480))

    with patch("gpu_ui_mixins.roi_discovery.np.load", return_value=movie), \
         patch.dict(sys.modules, {"otsu_thresh": fake_otsu, "projection": fake_proj}), \
         patch("PyQt5.QtGui.QGuiApplication.screens", return_value=[fake_screen]):
        host._thread_discover_rois()

    host.requestStartLiveTraces.emit.assert_called_once()


def test_C33_thread_discover_rois_otsu_with_existing_proj_display(host, tmp_path):
    """Branch: existing proj_display present → its.close() called before new."""
    host._discover_method = "OTSU"
    existing = MagicMock()
    host.proj_display = existing
    movie = np.zeros((5, 100, 100), dtype=np.uint8)
    fake_otsu = _make_otsu_module(masks_count=1)
    fake_proj = _make_projection_module(succeeds=True)
    fake_screen = MagicMock()
    fake_screen.size.return_value = MagicMock(
        width=MagicMock(return_value=1920),
        height=MagicMock(return_value=1080))

    with patch("gpu_ui_mixins.roi_discovery.np.load", return_value=movie), \
         patch.dict(sys.modules, {"otsu_thresh": fake_otsu, "projection": fake_proj}), \
         patch("PyQt5.QtGui.QGuiApplication.screens", return_value=[fake_screen]):
        host._thread_discover_rois()

    existing.close.assert_called_once()


def test_C30_thread_discover_rois_load_roi_copy_failure_fallback(host, tmp_path, capsys):
    """Branch in _load_roi_file: shutil.copyfile fails → fallback to rois_path = path."""
    good = tmp_path / "src.npz"
    _write_npz_with_labels(good, np.ones((4, 4), dtype=np.int32))

    from PyQt5.QtWidgets import QMessageBox
    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QFileDialog.getOpenFileName",
               return_value=(str(good), "")), \
         patch("gpu_ui_mixins.roi_discovery.QtWidgets.QMessageBox.question",
               return_value=QMessageBox.No), \
         patch("shutil.copyfile", side_effect=OSError("disk full")):
        host._load_roi_file()
    assert host.rois_path == str(good)
    assert "copyfile failed" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────────
# _run_refine_rois + _thread_refine_rois
# ─────────────────────────────────────────────────────────────────────────────


def test_C24_run_refine_rois_spawns_thread(host):
    with patch("gpu_ui_mixins.roi_discovery.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        host._run_refine_rois()
    mock_thread.assert_called_once()
    assert mock_thread.call_args.kwargs["target"] == host._thread_refine_rois
    assert mock_thread.call_args.kwargs["daemon"] is True


def test_C25_thread_refine_rois_emits_refine_request(host, tmp_path):
    """Happy path: emit refineRequested with (mean, masks)."""
    video = tmp_path / "v.tif"
    video.touch()
    host.video_path = str(video)
    # Pre-populate rois.npz with a 'masks' key.
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    np.savez(host.rois_path, masks=masks)

    fake_otsu = types.ModuleType("otsu_thresh")
    fake_otsu.compute_mean_projection = MagicMock(
        return_value=np.zeros((8, 8), dtype=np.float32))
    fake_otsu.load_movie = MagicMock(return_value=np.zeros((4, 8, 8), dtype=np.uint8))

    with patch.dict(sys.modules, {"otsu_thresh": fake_otsu}):
        host._thread_refine_rois()

    host.requestStopLiveTraces.emit.assert_called_once()
    host.refineRequested.emit.assert_called_once()


def test_C26_thread_refine_rois_handle_error_on_exception(host):
    """Raise walk: rois_path doesn't exist → _handle_error 'ROI refinement'."""
    host.video_path = "/nonexistent.tif"
    host._thread_refine_rois()
    host._handle_error.assert_called_once()
    assert host._handle_error.call_args.args[1] == "ROI refinement"


# ─────────────────────────────────────────────────────────────────────────────
# Property tests (Hypothesis) — §1.1 archetype "UI-glue" requires ≥2
# ─────────────────────────────────────────────────────────────────────────────


@settings(deadline=None, max_examples=25)
@given(size_mb=st.floats(min_value=0.001, max_value=2000.0,
                          allow_nan=False, allow_infinity=False))
def test_property_size_threshold_warning(tmp_path_factory, size_mb):
    """Invariant: the 'Large video file detected' message fires iff size_mb > 500.

    The branch is at module line ~76; this property pins the threshold
    so a future change to the constant cannot silently slip through.
    """
    tmp_path = tmp_path_factory.mktemp("size_thresh")
    host = _Host(tmp_path)
    video = tmp_path / f"v_{int(size_mb*1000)}.tif"
    video.touch()
    host.video_path = str(video)

    bytes_for_size = int(size_mb * 1024 * 1024)

    def fake_getsize(path):
        if path == host.video_path:
            return bytes_for_size
        return 0

    fake_module = types.ModuleType("make_mmap")
    fake_module.make_memmap = MagicMock()

    import io
    import contextlib
    buf = io.StringIO()
    with patch("gpu_ui_mixins.roi_discovery.os.path.getsize", side_effect=fake_getsize), \
         patch.dict(sys.modules, {"make_mmap": fake_module}), \
         contextlib.redirect_stdout(buf):
        host._thread_make_memmap()

    fired = "Large video file detected" in buf.getvalue()
    expected = size_mb > 500
    assert fired == expected, (
        f"size_mb={size_mb}: fired={fired}, expected={expected}")


@settings(deadline=None, max_examples=20)
@given(method=st.sampled_from(["OTSU", "Cellpose", "CNMF", "Custom",
                                "Random", "", "otsu", "cellpose"]))
def test_property_discover_method_routing(tmp_path_factory, method):
    """Invariant: method in {'CNMF','Custom'} short-circuits; everything
    else falls through to thread spawn AND sets _discover_method to the
    exact method string.

    Pins the case-sensitivity of the method-name dispatch.
    """
    tmp_path = tmp_path_factory.mktemp("method_routing")
    host = _Host(tmp_path)
    host._discover_method = "PREVIOUS"
    with patch("gpu_ui_mixins.roi_discovery.QtWidgets.QMessageBox.information"), \
         patch("gpu_ui_mixins.roi_discovery.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        host._run_discover_rois(method=method)

    if method in ("CNMF", "Custom"):
        mock_thread.assert_not_called()
        assert host._discover_method == "PREVIOUS"  # untouched
    else:
        mock_thread.assert_called_once()
        assert host._discover_method == method
