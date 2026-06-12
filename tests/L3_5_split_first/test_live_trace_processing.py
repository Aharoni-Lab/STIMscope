"""Comprehensive characterization tests for ``live_trace_processing``.

target ~85% path coverage on the LiveTraceProcessingMixin (extracted
iter 35 commit 70560b6).

Note on coverage ceiling: the GPU branch of `_on_frame_processed` and
half of `_initialize_processing_structures` use `cp.*` calls that
require a working CUDA runtime. The L3.5 test host's CUDA driver is
broken (12 L1 GPU failures pre-existing). For deterministic CI, this
suite patches `CUDA_USABLE = False` for the CPU branch and uses
`patch.object(lt_proc, "cp", FakeCupy)` for GPU-branch wire-format
tests. Some lines inside the GPU branch (e.g. `cp.bincount` argument
positions) inherit the same untestable status as L1 algorithms.

Module surface (~430 LOC, 9 methods):
- `_on_frame_processed(processed_data)` — main frame slot
- `_on_processing_error(msg)` — @pyqtSlot(str) error relay
- `_build_rois_for_shape(H, W)` — runtime ROI builder
- `_compute_dff(rid_key, raw_val)` — rolling-percentile baseline dF/F
- `_cleanup_existing_rois()` — GPU + CPU teardown
- `_initialize_empty_state()` — safe-empty fallback
- `_initialize_buffers_safely()` — per-ROI deque allocation
- `_initialize_processing_structures(resized)` — GPU/CPU label init
- `_initialize_cpu_fallback(flat)` — CPU-only init

Branches exercised per method are listed in each test docstring.
QApp + offscreen + sys.path are handled by conftest.py (session autouse).
"""

from __future__ import annotations

import threading
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from PyQt5.QtCore import QObject, pyqtSignal

import live_trace.processing as lt_proc
from live_trace.processing import LiveTraceProcessingMixin


# ─────────────────────────────────────────────────────────────────────────────
# Test infrastructure: stub host class
# ─────────────────────────────────────────────────────────────────────────────


class _Host(QObject, LiveTraceProcessingMixin):
    """Stub satisfying the 25-attribute mixin contract."""

    error_occurred = pyqtSignal(str)

    def __init__(self, *, max_rois_cfg=10, max_points_cfg=100, neuropil_r=0.0,
                 process_every_n=1, oasis_enabled=False):
        QObject.__init__(self)
        # Config snapshots
        self._max_rois_cfg = max_rois_cfg
        self._max_points_cfg = max_points_cfg
        self._neuropil_r = neuropil_r
        self._neuropil_inner_gap = 2
        self._neuropil_ring_width = 10
        self._baseline_window_s = 30.0
        self._baseline_percentile = 10.0
        self._oasis_enabled = oasis_enabled
        self._oasis_gamma = 0.95
        self._oasis_lambda = 0.1
        self._oasis_prev_c = {}
        # Frame decimation
        self._proc_gate = -1
        self._process_every_n = process_every_n
        # Threading
        self._gpu_lock = threading.Lock()
        # ROI state (filled by methods under test)
        self._labels_orig = None
        self.ids = np.array([], dtype=np.int32)
        self.buffers = {}
        self._dff_buffers = {}
        self._spike_buffers = {}
        self._labels_gpu = None
        self._ids_gpu = None
        self._roi_sizes_gpu = None
        self._f_gpu = None
        self._flat_labels_cpu = None
        self._roi_sizes_cpu = None
        self._npil_labels_gpu = None
        self._npil_sizes_gpu = None
        self._npil_labels_flat_cpu = None
        self._npil_sizes_cpu = None
        self._max_label = 0
        self._roi_ready = False
        # Stats + counters
        self.stats = {
            "frames_processed": 0,
            "frames_failed": 0,
            "last_frame_time": 0.0,
        }
        self._global_frame_index = 0
        self._last_fps_est = 30.0
        # Plot state
        self.plot_widget = None
        self._plot_curves = {}
        # Pygame flag for downstream sanity (not used here)
        self.use_pygame_plot = False


# ─────────────────────────────────────────────────────────────────────────────
# C1 — _compute_dff (pure)
# ─────────────────────────────────────────────────────────────────────────────


class TestC1ComputeDff:
    """Contract: rolling-percentile baseline dF/F.

    Branches:
    - buffer missing → 0.0
    - buffer < 3 entries → 0.0
    - small window after fps/baseline_window_s math → 0.0
    - happy path: returns (raw - f0) / f0
    - f0 ~ 0 → divide-by-zero clamp uses 1.0
    """

    def test_missing_buffer_returns_zero(self):
        host = _Host()
        assert host._compute_dff(rid_key=42, raw_val=100.0) == 0.0

    def test_buffer_too_short_returns_zero(self):
        host = _Host()
        host.buffers[1] = deque([10.0, 20.0], maxlen=100)
        assert host._compute_dff(rid_key=1, raw_val=30.0) == 0.0

    def test_happy_path_returns_dff(self):
        host = _Host()
        # Fill buffer enough to satisfy fps * baseline_window_s clamp
        host._last_fps_est = 10.0
        host._baseline_window_s = 1.0  # win = 10 points
        host._baseline_percentile = 10.0
        host.buffers[1] = deque([100.0] * 10, maxlen=100)
        # f0 = percentile(recent, 10) = 100 → dff = (200-100)/100 = 1.0
        result = host._compute_dff(rid_key=1, raw_val=200.0)
        assert result == pytest.approx(1.0)

    def test_f0_near_zero_clamp(self):
        host = _Host()
        host._last_fps_est = 10.0
        host._baseline_window_s = 1.0
        host._baseline_percentile = 10.0
        host.buffers[1] = deque([0.0] * 10, maxlen=100)
        # f0 = 0 → clamped to 1.0 → dff = (5.0 - 1.0) / 1.0 = 4.0
        # (The clamp REPLACES f0 with 1.0 BEFORE the subtraction, so
        # the numerator uses the clamped value.)
        result = host._compute_dff(rid_key=1, raw_val=5.0)
        assert result == pytest.approx(4.0)

    def test_negative_dff_allowed(self):
        host = _Host()
        host._last_fps_est = 10.0
        host._baseline_window_s = 1.0
        host._baseline_percentile = 10.0
        host.buffers[1] = deque([100.0] * 10, maxlen=100)
        # Raw below baseline → negative dff
        result = host._compute_dff(rid_key=1, raw_val=50.0)
        assert result == pytest.approx(-0.5)

    def test_window_truncated_by_baseline_window_s(self):
        """Buffer has 100 entries but baseline_window_s caps the window."""
        host = _Host()
        host._last_fps_est = 10.0
        host._baseline_window_s = 1.0  # win = 10 entries
        host._baseline_percentile = 10.0
        # First 90 values are 0, last 10 are 100 — window should use last 10
        host.buffers[1] = deque([0.0] * 90 + [100.0] * 10, maxlen=200)
        # f0 = percentile(last 10, 10%) = 100 → dff = (200-100)/100 = 1.0
        result = host._compute_dff(rid_key=1, raw_val=200.0)
        assert result == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# C2 — _initialize_empty_state (pure)
# ─────────────────────────────────────────────────────────────────────────────


class TestC2InitializeEmptyState:
    """Contract: reset to safe-empty state."""

    def test_resets_all_attrs(self):
        host = _Host()
        host.ids = np.array([1, 2, 3], dtype=np.int32)
        host.buffers = {1: deque(), 2: deque()}
        host._dff_buffers = {1: deque()}
        host._roi_ready = True
        host._labels_gpu = "junk"
        host._flat_labels_cpu = np.array([1])

        host._initialize_empty_state()

        assert host.ids.size == 0
        assert host.ids.dtype == np.int32
        assert host.buffers == {}
        assert host._dff_buffers == {}
        assert host._roi_ready is False
        assert host._labels_gpu is None
        assert host._ids_gpu is None
        assert host._roi_sizes_gpu is None
        assert host._f_gpu is None
        assert host._flat_labels_cpu is None
        assert host._roi_sizes_cpu is None


# ─────────────────────────────────────────────────────────────────────────────
# C3 — _initialize_buffers_safely
# ─────────────────────────────────────────────────────────────────────────────


class TestC3InitializeBuffersSafely:
    """Contract: per-ROI deque allocation; verify count + retry missing."""

    def test_buffers_allocated_per_id(self):
        host = _Host(max_points_cfg=50)
        host.ids = np.array([5, 10, 15], dtype=np.int32)
        host._initialize_buffers_safely()
        assert set(host.buffers.keys()) == {5, 10, 15}
        assert set(host._dff_buffers.keys()) == {5, 10, 15}
        assert set(host._spike_buffers.keys()) == {5, 10, 15}

    def test_deque_maxlen_from_config(self):
        host = _Host(max_points_cfg=42)
        host.ids = np.array([1], dtype=np.int32)
        host._initialize_buffers_safely()
        assert host.buffers[1].maxlen == 42
        assert host._dff_buffers[1].maxlen == 42
        assert host._spike_buffers[1].maxlen == 42

    def test_empty_ids_yields_empty_buffers(self):
        host = _Host()
        host.ids = np.array([], dtype=np.int32)
        host._initialize_buffers_safely()
        assert host.buffers == {}
        assert host._dff_buffers == {}
        assert host._spike_buffers == {}

    def test_ids_with_duplicate_int_cast_collapses_to_single_key(self):
        """int(np.int32(7)) == 7 so duplicates collapse — verifies behavior."""
        host = _Host()
        host.ids = np.array([7, 7, 7], dtype=np.int32)
        host._initialize_buffers_safely()
        assert set(host.buffers.keys()) == {7}


# ─────────────────────────────────────────────────────────────────────────────
# C4 — _cleanup_existing_rois
# ─────────────────────────────────────────────────────────────────────────────


class TestC4CleanupExistingRois:
    """Contract: best-effort teardown of GPU + CPU + plot-curve state."""

    def test_clears_buffers(self):
        host = _Host()
        host.buffers = {1: deque([1, 2]), 2: deque([3, 4])}
        host._dff_buffers = {1: deque([0.1])}
        host._cleanup_existing_rois()
        assert host.buffers == {}
        assert host._dff_buffers == {}

    def test_nulls_cpu_labels(self):
        host = _Host()
        host._flat_labels_cpu = np.array([1, 2, 3])
        host._roi_sizes_cpu = np.array([10, 20], dtype=np.float32)
        host._cleanup_existing_rois()
        assert host._flat_labels_cpu is None
        assert host._roi_sizes_cpu is None

    def test_clears_plot_curves(self):
        host = _Host()
        host._plot_curves = {1: MagicMock(), 2: MagicMock()}
        host._cleanup_existing_rois()
        assert host._plot_curves == {}

    def test_exception_swallowed(self, capsys):
        host = _Host()
        # Force the AttributeError-protected path to error
        host.buffers = MagicMock()
        host.buffers.clear.side_effect = RuntimeError("clear broken")
        host._cleanup_existing_rois()  # must not raise
        captured = capsys.readouterr()
        assert "Error during ROI cleanup" in captured.out

    def test_gpu_deletion_when_cuda_available(self):
        """When CUDA_AVAILABLE is True, GPU attrs are deleted via `del`."""
        host = _Host()
        host._labels_gpu = MagicMock()
        host._ids_gpu = MagicMock()
        host._roi_sizes_gpu = MagicMock()
        host._f_gpu = MagicMock()
        with patch.object(lt_proc, "CUDA_AVAILABLE", True):
            host._cleanup_existing_rois()
        # After del, attribute access raises AttributeError
        with pytest.raises(AttributeError):
            _ = host._labels_gpu

    def test_no_gpu_deletion_when_cuda_unavailable(self):
        host = _Host()
        host._labels_gpu = "kept"
        with patch.object(lt_proc, "CUDA_AVAILABLE", False):
            host._cleanup_existing_rois()
        # del branch skipped — attribute still there
        assert host._labels_gpu == "kept"


# ─────────────────────────────────────────────────────────────────────────────
# C5 — _initialize_cpu_fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestC5InitializeCpuFallback:
    """Contract: bincount-based ROI sizes + null out GPU attrs."""

    def test_happy_path_computes_sizes(self):
        host = _Host()
        host.ids = np.array([1, 2], dtype=np.int32)
        host._max_label = 2
        flat = np.array([0, 1, 1, 2, 2, 2], dtype=np.int32)
        host._initialize_cpu_fallback(flat)
        # ROI 1 has 2 pixels, ROI 2 has 3 pixels
        assert host._roi_sizes_cpu[0] == pytest.approx(2.0)
        assert host._roi_sizes_cpu[1] == pytest.approx(3.0)
        assert host._roi_sizes_cpu.dtype == np.float32

    def test_nulls_gpu_attrs(self):
        host = _Host()
        host.ids = np.array([1], dtype=np.int32)
        host._max_label = 1
        host._labels_gpu = "junk"
        host._ids_gpu = "junk"
        host._roi_sizes_gpu = "junk"
        host._f_gpu = "junk"
        flat = np.array([0, 1, 1], dtype=np.int32)
        host._initialize_cpu_fallback(flat)
        assert host._labels_gpu is None
        assert host._ids_gpu is None
        assert host._roi_sizes_gpu is None
        assert host._f_gpu is None

    def test_exception_triggers_empty_state(self, capsys):
        host = _Host()
        # Provide bad ids → indexing into counts will fail
        host.ids = np.array([99], dtype=np.int32)  # out-of-range
        host._max_label = 1
        flat = np.array([0, 1], dtype=np.int32)  # bincount → [1, 1]
        host._initialize_cpu_fallback(flat)
        # IndexError caught → empty state
        captured = capsys.readouterr()
        assert "CPU initialization also failed" in captured.out
        assert host._roi_ready is False
        assert host.ids.size == 0


# ─────────────────────────────────────────────────────────────────────────────
# C6 — _initialize_processing_structures
# ─────────────────────────────────────────────────────────────────────────────


class TestC6InitializeProcessingStructures:
    """Contract: build CPU label arrays; maybe build GPU + neuropil."""

    def test_cpu_path_when_cuda_disabled(self):
        host = _Host()
        host.ids = np.array([1, 2], dtype=np.int32)
        host._max_label = 0  # forced reset
        resized = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._initialize_processing_structures(resized)
        assert host._flat_labels_cpu is not None
        assert host._max_label == 2  # max of resized
        assert host._roi_sizes_cpu is not None

    def test_neuropil_zero_skips_npil_build(self):
        host = _Host(neuropil_r=0.0)
        host.ids = np.array([1], dtype=np.int32)
        resized = np.array([[1, 0], [0, 0]], dtype=np.int32)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._initialize_processing_structures(resized)
        assert host._npil_labels_flat_cpu is None
        assert host._npil_sizes_cpu is None

    def test_neuropil_build_failure_caught_and_zeros_r(self, capsys):
        """When `build_neuropil_labels` import or call fails, exception is
        caught + `_neuropil_r` zeroed (graceful degradation)."""
        host = _Host(neuropil_r=0.5)
        host.ids = np.array([1], dtype=np.int32)
        resized = np.array([[1, 0]], dtype=np.int32)
        # Patch the lazy import target to raise
        import sys
        fake_te = type(sys)("trace_extractor_fake")
        fake_te.build_neuropil_labels = MagicMock(side_effect=RuntimeError("npil broken"))
        with patch.dict(sys.modules, {"trace_extractor": fake_te}):
            with patch.object(lt_proc, "CUDA_USABLE", False):
                host._initialize_processing_structures(resized)
        assert host._neuropil_r == 0.0  # zeroed after failure
        captured = capsys.readouterr()
        assert "Neuropil ring build failed" in captured.out

    def test_plot_curves_built_when_widget_and_pyqtgraph_available(self):
        """When plot_widget set + PYQTPGRAPH_AVAILABLE True, allocate curves."""
        host = _Host()
        host.ids = np.array([5, 7], dtype=np.int32)
        host.plot_widget = MagicMock()
        host.plot_widget.plot.return_value = MagicMock()
        resized = np.array([[5, 0], [7, 0]], dtype=np.int32)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            with patch.object(lt_proc, "PYQTPGRAPH_AVAILABLE", True):
                host._initialize_processing_structures(resized)
        assert set(host._plot_curves.keys()) == {5, 7}

    def test_plot_curves_skipped_when_pyqtgraph_unavailable(self):
        host = _Host()
        host.ids = np.array([5], dtype=np.int32)
        host.plot_widget = MagicMock()
        resized = np.array([[5]], dtype=np.int32)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            with patch.object(lt_proc, "PYQTPGRAPH_AVAILABLE", False):
                host._initialize_processing_structures(resized)
        assert host._plot_curves == {}


# ─────────────────────────────────────────────────────────────────────────────
# C7 — _build_rois_for_shape
# ─────────────────────────────────────────────────────────────────────────────


class TestC7BuildRoisForShape:
    """Contract: orchestrates cleanup → resize → init."""

    def test_happy_path_sets_ready(self):
        host = _Host(max_rois_cfg=10)
        host._labels_orig = np.array([[1, 1], [2, 2]], dtype=np.int32)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._build_rois_for_shape(2, 2)
        assert host._roi_ready is True
        assert host._H == 2 and host._W == 2
        assert set(host.ids.tolist()) == {1, 2}

    def test_shape_mismatch_triggers_resize(self):
        host = _Host(max_rois_cfg=10)
        # 2x2 labels but frame is 4x4 → cv2.resize NEAREST
        host._labels_orig = np.array([[1, 1], [2, 2]], dtype=np.int32)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._build_rois_for_shape(4, 4)
        # After NEAREST resize to 4x4, ids should still be {1, 2}
        assert host._roi_ready is True
        assert set(host.ids.tolist()) == {1, 2}

    def test_no_positive_labels_yields_empty_state(self, capsys):
        host = _Host()
        host._labels_orig = np.zeros((4, 4), dtype=np.int32)
        host._build_rois_for_shape(4, 4)
        assert host._roi_ready is False
        assert host.ids.size == 0
        captured = capsys.readouterr()
        assert "No positive ROI labels found" in captured.out

    def test_max_rois_cfg_truncates_ids(self):
        host = _Host(max_rois_cfg=2)
        host._labels_orig = np.array([[1, 2], [3, 4]], dtype=np.int32)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._build_rois_for_shape(2, 2)
        assert len(host.ids) == 2

    def test_exception_falls_back_to_empty(self, capsys):
        host = _Host()
        # _labels_orig is None → AttributeError on.shape access
        host._labels_orig = None
        host._build_rois_for_shape(4, 4)
        captured = capsys.readouterr()
        assert "Error building ROIs" in captured.out
        assert host._roi_ready is False
        assert host.ids.size == 0


# ─────────────────────────────────────────────────────────────────────────────
# C8 — _on_processing_error
# ─────────────────────────────────────────────────────────────────────────────


class TestC8OnProcessingError:
    """Contract: print + emit error_occurred."""

    def test_emits_signal(self, capsys):
        host = _Host()
        emitted = []
        host.error_occurred.connect(lambda msg: emitted.append(msg))
        host._on_processing_error("boom")
        assert emitted == ["boom"]
        captured = capsys.readouterr()
        assert "Processing error: boom" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C9 — _on_frame_processed (CPU branch)
# ─────────────────────────────────────────────────────────────────────────────


class TestC9OnFrameProcessedCpuBranch:
    """Contract: dispatcher + CPU path. CUDA_USABLE forced False.

    Branches exercised in this class:
    - Invalid input (not dict) → skip
    - 'frame' key missing → skip
    - frame is None → skip
    - frame has no shape → skip
    - frame dimensions unreasonable → skip
    - _roi_ready False + no labels → skip
    - _roi_ready False + labels → build_rois called
    - proc_gate decimation → skip
    - happy CPU path → buffers populated + stats updated
    - missing CPU labels → skip
    - missing CPU roi_sizes → lazy init
    """

    def _ready_host(self):
        """Host with ROI structures pre-initialised for CPU path."""
        host = _Host(max_rois_cfg=10, max_points_cfg=100, process_every_n=1)
        host._labels_orig = np.array([[1, 1], [2, 2]], dtype=np.int32)
        host.ids = np.array([1, 2], dtype=np.int32)
        host._max_label = 2
        host._flat_labels_cpu = host._labels_orig.ravel().astype(np.int32)
        host._roi_sizes_cpu = np.array([2.0, 2.0], dtype=np.float32)
        for rid in [1, 2]:
            host.buffers[rid] = deque(maxlen=100)
            host._dff_buffers[rid] = deque(maxlen=100)
            host._spike_buffers[rid] = deque(maxlen=100)
        host._roi_ready = True
        return host

    def test_non_dict_input_skipped(self, capsys):
        host = _Host()
        host._on_frame_processed("not a dict")
        captured = capsys.readouterr()
        assert "Invalid frame data" in captured.out

    def test_missing_frame_key_skipped(self, capsys):
        host = _Host()
        host._on_frame_processed({"other": 1})
        captured = capsys.readouterr()
        assert "Invalid frame data" in captured.out

    def test_none_frame_skipped(self, capsys):
        host = _Host()
        host._on_frame_processed({"frame": None})
        captured = capsys.readouterr()
        assert "Received None frame" in captured.out

    def test_no_shape_frame_skipped(self, capsys):
        host = _Host()
        host._on_frame_processed({"frame": "no shape attr"})
        captured = capsys.readouterr()
        assert "Invalid frame shape" in captured.out

    def test_1d_frame_skipped(self, capsys):
        host = _Host()
        gray = np.zeros(10, dtype=np.uint8)  # 1D
        host._on_frame_processed({"frame": gray})
        captured = capsys.readouterr()
        assert "Invalid frame shape" in captured.out

    def test_unreasonable_dims_skipped(self, capsys):
        host = _Host()
        gray = MagicMock()
        gray.shape = (20000, 20000)
        host._on_frame_processed({"frame": gray})
        captured = capsys.readouterr()
        assert "Unreasonable frame dimensions" in captured.out

    def test_no_labels_skipped(self, capsys):
        host = _Host()
        host._labels_orig = None
        gray = np.zeros((4, 4), dtype=np.uint8)
        host._on_frame_processed({"frame": gray})
        captured = capsys.readouterr()
        assert "No ROI labels loaded" in captured.out

    def test_first_frame_triggers_build_rois(self):
        host = _Host(max_rois_cfg=10, max_points_cfg=100)
        host._labels_orig = np.array([[1, 1], [2, 2]], dtype=np.int32)
        host._roi_ready = False
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._on_frame_processed({"frame": gray})
        # After build: _roi_ready True, ids populated
        assert host._roi_ready is True
        assert host.ids.size > 0

    def test_proc_gate_decimation_skips(self):
        host = self._ready_host()
        host._process_every_n = 2  # skip every-other frame
        host._proc_gate = -1
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        # First call: gate becomes 0 → not skipped → processed
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._on_frame_processed({"frame": gray})
        first_processed = host.stats['frames_processed']
        # Second call: gate becomes 1 → truthy → skipped (only last_frame_time updates)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._on_frame_processed({"frame": gray})
        assert host.stats['frames_processed'] == first_processed

    def test_happy_cpu_path_populates_buffers(self):
        host = self._ready_host()
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._on_frame_processed({"frame": gray})
        # Both ROIs should have one entry
        assert len(host.buffers[1]) == 1
        assert len(host.buffers[2]) == 1
        # Stats incremented
        assert host.stats['frames_processed'] == 1
        assert host._global_frame_index == 1

    def test_missing_cpu_labels_skipped(self, capsys):
        host = self._ready_host()
        host._flat_labels_cpu = None
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._on_frame_processed({"frame": gray})
        captured = capsys.readouterr()
        assert "CPU labels not initialized" in captured.out

    def test_missing_cpu_sizes_lazy_init(self, capsys):
        host = self._ready_host()
        host._roi_sizes_cpu = None
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._on_frame_processed({"frame": gray})
        captured = capsys.readouterr()
        assert "CPU ROI sizes not initialized" in captured.out
        assert host._roi_sizes_cpu is not None  # got lazily initialised

    def test_oasis_enabled_writes_spike(self):
        host = self._ready_host()
        host._oasis_enabled = True
        # Fill enough buffer to compute dF/F
        for v in [100.0] * 10:
            host.buffers[1].append(v)
            host.buffers[2].append(v)
        gray = np.array([[200, 200], [200, 200]], dtype=np.uint8)
        host._last_fps_est = 10.0
        host._baseline_window_s = 1.0
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._on_frame_processed({"frame": gray})
        # spike buffers should now have an entry
        assert len(host._spike_buffers[1]) == 1
        assert len(host._spike_buffers[2]) == 1

    def test_unexpected_exception_increments_failed(self, capsys):
        """Force an internal exception via a frame whose `.ravel()` raises."""
        host = self._ready_host()
        gray = MagicMock()
        gray.shape = (2, 2)
        gray.ravel.side_effect = RuntimeError("ravel broken")
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._on_frame_processed({"frame": gray})
        assert host.stats['frames_failed'] >= 1
        captured = capsys.readouterr()
        assert "Frame processing error" in captured.out

    def test_index_error_triggers_roi_reinit(self, capsys):
        """An IndexError msg with 'index' triggers reinit attempt."""
        host = self._ready_host()
        # Force ids to be out-of-range → bincount-index error
        host.ids = np.array([99, 100], dtype=np.int32)
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._on_frame_processed({"frame": gray})
        captured = capsys.readouterr()
        assert "Attempting ROI reinitialization" in captured.out

    def test_cpu_neuropil_subtraction_path(self):
        """When neuropil_r > 0 + npil arrays present, mean subtraction
        kicks in."""
        host = self._ready_host()
        host._neuropil_r = 0.5
        # Mirror flat labels (simple: same labels also for neuropil)
        host._npil_labels_flat_cpu = host._flat_labels_cpu.copy()
        host._npil_sizes_cpu = np.array([2.0, 2.0], dtype=np.float32)
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._on_frame_processed({"frame": gray})
        assert host.stats['frames_processed'] == 1

    def test_keyerror_reinit_branch(self):
        """When a ROI id is missing from buffers mid-loop, the recovery
        path reinitialises all missing buffers from self.ids."""
        host = self._ready_host()
        # Drop ROI 2's buffer to force the reinit branch
        del host.buffers[2]
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", False):
            host._on_frame_processed({"frame": gray})
        # After the loop, buffer 2 should be recreated AND populated
        assert 2 in host.buffers
        assert len(host.buffers[2]) >= 1


class TestC10OnFrameProcessedGpuBranchExtended:
    """Extended GPU-branch coverage via _FakeCp shim."""

    def _gpu_ready_host(self, *, oasis=False, neuropil=0.0):
        host = _Host(max_rois_cfg=10, max_points_cfg=100, process_every_n=1,
                     oasis_enabled=oasis, neuropil_r=neuropil)
        host._labels_orig = np.array([[1, 1], [2, 2]], dtype=np.int32)
        host.ids = np.array([1, 2], dtype=np.int32)
        host._max_label = 2
        flat = host._labels_orig.ravel().astype(np.int32)
        host._labels_gpu = _FakeCpArr(flat)
        host._ids_gpu = _FakeCpArr(host.ids)
        host._roi_sizes_gpu = _FakeCpArr(np.array([2.0, 2.0], dtype=np.float32))
        host._f_gpu = _FakeCpArr(np.zeros(4, dtype=np.float32))
        host._flat_labels_cpu = flat
        host._roi_sizes_cpu = np.array([2.0, 2.0], dtype=np.float32)
        if neuropil > 0:
            host._npil_labels_gpu = _FakeCpArr(flat)
            host._npil_sizes_gpu = _FakeCpArr(np.array([2.0, 2.0], dtype=np.float32))
        for rid in [1, 2]:
            host.buffers[rid] = deque(maxlen=100)
            host._dff_buffers[rid] = deque(maxlen=100)
            host._spike_buffers[rid] = deque(maxlen=100)
        host._roi_ready = True
        return host

    def test_gpu_neuropil_subtraction(self):
        host = self._gpu_ready_host(neuropil=0.4)
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", True):
            with patch.object(lt_proc, "cp", _FakeCp):
                host._on_frame_processed({"frame": gray})
        assert host.stats['frames_processed'] == 1

    def test_gpu_oasis_enabled_writes_spike(self):
        host = self._gpu_ready_host(oasis=True)
        # Pre-fill buffer to make dF/F nontrivial
        for v in [100.0] * 10:
            host.buffers[1].append(v)
            host.buffers[2].append(v)
        host._last_fps_est = 10.0
        host._baseline_window_s = 1.0
        gray = np.array([[200, 200], [200, 200]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", True):
            with patch.object(lt_proc, "cp", _FakeCp):
                host._on_frame_processed({"frame": gray})
        assert len(host._spike_buffers[1]) == 1
        assert len(host._spike_buffers[2]) == 1

    def test_gpu_missing_buffer_lazy_create(self):
        """GPU path also has the 'ROI not in buffers, creating' branch."""
        host = self._gpu_ready_host()
        del host.buffers[2]
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", True):
            with patch.object(lt_proc, "cp", _FakeCp):
                host._on_frame_processed({"frame": gray})
        assert 2 in host.buffers
        assert len(host.buffers[2]) >= 1

    def test_gpu_diagnostic_prints_after_5s(self, capsys):
        """The 5-second diagnostic block prints frame stats."""
        host = self._gpu_ready_host()
        # Make last_extract_log_t very old so >5s gate passes
        host._last_extract_log_t = 0.0
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", True):
            with patch.object(lt_proc, "cp", _FakeCp):
                host._on_frame_processed({"frame": gray})
        captured = capsys.readouterr()
        assert "[Extractor]" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C10 — _on_frame_processed (GPU branch — wire-format via cp monkey-patch)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeCp:
    """Minimal cupy-like shim. Just enough to exercise the GPU branch's
    wire-format code path (call sequence + named arguments) without
    needing a real CUDA runtime. Returns numpy-backed objects that
    quack like cupy arrays for the call chain in _on_frame_processed.
    """

    @staticmethod
    def bincount(labels, weights=None, minlength=0):
        # Return a _FakeCpArr from numpy bincount
        return _FakeCpArr(np.bincount(_unwrap(labels), weights=_unwrap(weights),
                                      minlength=minlength))

    @staticmethod
    def maximum(a, b):
        return _FakeCpArr(np.maximum(_unwrap(a), b))

    @staticmethod
    def asarray(a, *args, **kwargs):
        return _FakeCpArr(np.asarray(a))

    @staticmethod
    def empty(n, dtype=None):
        return _FakeCpArr(np.empty(n, dtype=dtype))


def _unwrap(x):
    if isinstance(x, _FakeCpArr):
        return x._a
    return x


class _FakeCpArr:
    def __init__(self, a):
        self._a = np.asarray(a)

    def __getitem__(self, idx):
        return _FakeCpArr(self._a[_unwrap(idx)])

    def __truediv__(self, other):
        return _FakeCpArr(self._a / _unwrap(other))

    def __sub__(self, other):
        return _FakeCpArr(self._a - _unwrap(other))

    def __rsub__(self, other):
        return _FakeCpArr(other - self._a)

    def __mul__(self, other):
        return _FakeCpArr(self._a * _unwrap(other))

    def __rmul__(self, other):
        return _FakeCpArr(self._a * _unwrap(other))

    def set(self, src):
        self._a = np.asarray(src).copy()

    def get(self):
        return self._a

    def astype(self, dtype):
        return _FakeCpArr(self._a.astype(dtype))

    def max(self):
        return _FakeCpArr(np.array(self._a.max()))

    def __len__(self):
        return len(self._a)

    @property
    def shape(self):
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype


class TestC10OnFrameProcessedGpuBranch:
    """Wire-format test for the GPU branch using a fake cupy module.

    Only validates call sequence + state mutation. Does NOT validate
    pixel-perfect numerical equivalence with a real CUDA run — that
    is L1 algorithm territory.

    Branches:
    - GPU path with _roi_sizes_gpu absent → CPU fallback message
    - GPU path happy → buffers + stats updated
    """

    def _gpu_ready_host(self):
        host = _Host(max_rois_cfg=10, max_points_cfg=100, process_every_n=1)
        host._labels_orig = np.array([[1, 1], [2, 2]], dtype=np.int32)
        host.ids = np.array([1, 2], dtype=np.int32)
        host._max_label = 2
        flat = host._labels_orig.ravel().astype(np.int32)
        host._labels_gpu = _FakeCpArr(flat)
        host._ids_gpu = _FakeCpArr(host.ids)
        host._roi_sizes_gpu = _FakeCpArr(np.array([2.0, 2.0], dtype=np.float32))
        host._f_gpu = _FakeCpArr(np.zeros(4, dtype=np.float32))
        # CPU buffers (always written, regardless of GPU path)
        host._flat_labels_cpu = flat
        host._roi_sizes_cpu = np.array([2.0, 2.0], dtype=np.float32)
        for rid in [1, 2]:
            host.buffers[rid] = deque(maxlen=100)
            host._dff_buffers[rid] = deque(maxlen=100)
            host._spike_buffers[rid] = deque(maxlen=100)
        host._roi_ready = True
        return host

    def test_gpu_path_missing_sizes_falls_to_cpu(self, capsys):
        host = self._gpu_ready_host()
        host._roi_sizes_gpu = None
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", True):
            with patch.object(lt_proc, "cp", _FakeCp):
                host._on_frame_processed({"frame": gray})
        captured = capsys.readouterr()
        assert "GPU ROI sizes not initialized" in captured.out

    def test_gpu_happy_path_populates_buffers(self):
        host = self._gpu_ready_host()
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        with patch.object(lt_proc, "CUDA_USABLE", True):
            with patch.object(lt_proc, "cp", _FakeCp):
                host._on_frame_processed({"frame": gray})
        # Buffers populated via GPU path
        assert len(host.buffers[1]) == 1
        assert len(host.buffers[2]) == 1
        assert host.stats['frames_processed'] == 1


# ─────────────────────────────────────────────────────────────────────────────
# C11 — Mixin integration
# ─────────────────────────────────────────────────────────────────────────────


class TestC11MixinIntegration:
    """Contract: 9 methods accessible on subclass; mixin has no __init__."""

    METHODS = (
        "_on_frame_processed",
        "_on_processing_error",
        "_build_rois_for_shape",
        "_compute_dff",
        "_cleanup_existing_rois",
        "_initialize_empty_state",
        "_initialize_buffers_safely",
        "_initialize_processing_structures",
        "_initialize_cpu_fallback",
    )

    def test_all_9_methods_on_subclass(self):
        host = _Host()
        for name in self.METHODS:
            method = getattr(host, name, None)
            assert callable(method), f"Missing or non-callable: {name}"

    def test_methods_defined_on_mixin(self):
        for name in self.METHODS:
            assert name in LiveTraceProcessingMixin.__dict__, (
                f"{name} not defined on LiveTraceProcessingMixin"
            )

    def test_mixin_has_no_init(self):
        assert "__init__" not in LiveTraceProcessingMixin.__dict__

    def test_module_flags_present(self):
        assert isinstance(lt_proc.CUDA_AVAILABLE, bool)
        assert isinstance(lt_proc.CUDA_USABLE, bool)
        assert isinstance(lt_proc.PYQTPGRAPH_AVAILABLE, bool)


# ─────────────────────────────────────────────────────────────────────────────
# §1.1 L3.5 matrix backfill — Property + Snapshot + Concurrency (iter-57)
#
# §1.1 L3.5 row requires:
#   - Property ≥2 per sub-module (universal floor)
#   - Snapshot required for trace outputs (_compute_dff IS the trace
#     output transform: raw fluorescence → dF/F is the per-frame trace
#     value; _initialize_empty_state defines the post-cleanup contract)
#   - Concurrency ≥1 if mixin touches threads (_gpu_lock guards the
#     GPU branch of _on_frame_processed; _compute_dff must be safe
#     across per-ROI parallel calls)
#
# Closes part of the OPEN BLOCK on iter-42 L3.5 PROMOTION per
# audit_findings.log lines 1655-2235 + docs/PHASE_A5_DEFERRAL.md.
# Fourth L3.5 sub-mixin backfill (live_trace_processing), 4 of 8.
# ─────────────────────────────────────────────────────────────────────────────

import hashlib  # noqa: E402

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


class TestPropertyComputeDff:
    """§1.1 universal floor: ≥2 property tests for `_compute_dff`.

    `_compute_dff` is the dF/F transform: raw fluorescence → fractional
    change above baseline (percentile of a rolling window). Invariants
    that must hold for any input:
    - len(buf) < 3 OR win < 3 → returns 0.0 exactly
    - On constant-fill buffer: percentile == fill → dF/F = (raw - fill)/fill
    """

    @given(
        raw_val=st.floats(min_value=-1e6, max_value=1e6,
                          allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=60, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_short_buffer_always_zero(self, raw_val):
        """For any raw_val, if buf is None / len < 3, _compute_dff
        returns exactly 0.0. Pins the short-buffer-no-signal contract;
        a regression that returned raw_val instead would corrupt every
        trace at startup."""
        host = _Host()
        # No buffer at all
        host.buffers = {}
        assert host._compute_dff(rid_key=1, raw_val=raw_val) == 0.0
        # Empty buffer
        host.buffers = {1: deque(maxlen=100)}
        assert host._compute_dff(rid_key=1, raw_val=raw_val) == 0.0
        # 2-element buffer (< 3 → short path)
        buf = deque(maxlen=100)
        buf.append(10.0)
        buf.append(20.0)
        host.buffers = {1: buf}
        assert host._compute_dff(rid_key=1, raw_val=raw_val) == 0.0

    @given(
        fill=st.floats(min_value=1.0, max_value=1e4,
                       allow_nan=False, allow_infinity=False),
        raw_val=st.floats(min_value=-1e4, max_value=1e4,
                          allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=40, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_constant_fill_dff_identity(self, fill, raw_val):
        """On a constant-fill buffer (any percentile of constant ==
        constant), f0 == fill so _compute_dff returns (raw - fill)/fill
        exactly. Pins the dF/F arithmetic identity — any change to the
        formula (e.g. swapping numerator/denominator, missing the
        baseline subtraction) breaks this for every test case."""
        host = _Host()
        host._last_fps_est = 30.0
        host._baseline_window_s = 1.0  # win = min(N, 30) — easy ladder
        host._baseline_percentile = 50.0  # median of constant is constant
        buf = deque(maxlen=200)
        for _ in range(60):
            buf.append(fill)
        host.buffers = {7: buf}

        out = host._compute_dff(rid_key=7, raw_val=raw_val)
        expected = (raw_val - fill) / fill  # f0 == fill, non-zero
        assert out == pytest.approx(expected, rel=1e-5, abs=1e-7), (
            f"dF/F identity failed: got {out}, expected {expected} for "
            f"fill={fill}, raw_val={raw_val}"
        )


class TestSnapshotProcessingContract:
    """§1.1 L3.5 row: snapshot required for trace outputs.

    Two contract snapshots:
    - `_initialize_empty_state` post-state field set — the canonical
      no-labels fallback that every downstream method reads from
    - `_initialize_buffers_safely` produces deterministic per-ROI
      deque sizing for canonical ids — the buffer-shape contract
    """

    def test_initialize_empty_state_post_state_snapshot(self):
        """Pin the sha256 of the post-_initialize_empty_state field
        snapshot: ids dtype/shape + buffers/_dff_buffers identity
        (empty dicts) + GPU/CPU buffer null state + _roi_ready flag.
        Any field rename, dtype change, or non-empty default breaks
        this — downstream code reads these fields as the "no-labels"
        contract."""
        host = _Host()
        # Pre-dirty all state to be cleared
        host.ids = np.array([7, 8, 9], dtype=np.int32)
        host.buffers = {7: deque([1.0])}
        host._dff_buffers = {7: deque([0.5])}
        host._roi_ready = True
        host._labels_gpu = "not-none"
        host._ids_gpu = "not-none"
        host._roi_sizes_gpu = "not-none"
        host._f_gpu = "not-none"
        host._flat_labels_cpu = "not-none"
        host._roi_sizes_cpu = "not-none"

        host._initialize_empty_state()

        payload = b"|".join([
            b"ids_dtype:" + str(host.ids.dtype).encode(),
            b"ids_shape:" + repr(host.ids.shape).encode(),
            b"ids_len:" + str(len(host.ids)).encode(),
            b"buffers_is_empty_dict:"
            + str(host.buffers == {} and isinstance(host.buffers, dict)).encode(),
            b"dff_buffers_is_empty_dict:"
            + str(host._dff_buffers == {} and isinstance(host._dff_buffers, dict)).encode(),
            b"roi_ready:" + str(host._roi_ready).encode(),
            b"labels_gpu_is_none:" + str(host._labels_gpu is None).encode(),
            b"ids_gpu_is_none:" + str(host._ids_gpu is None).encode(),
            b"roi_sizes_gpu_is_none:" + str(host._roi_sizes_gpu is None).encode(),
            b"f_gpu_is_none:" + str(host._f_gpu is None).encode(),
            b"flat_labels_cpu_is_none:" + str(host._flat_labels_cpu is None).encode(),
            b"roi_sizes_cpu_is_none:" + str(host._roi_sizes_cpu is None).encode(),
        ])
        h = hashlib.sha256(payload).hexdigest()

        expected_payload = b"|".join([
            b"ids_dtype:int32",
            b"ids_shape:(0,)",
            b"ids_len:0",
            b"buffers_is_empty_dict:True",
            b"dff_buffers_is_empty_dict:True",
            b"roi_ready:False",
            b"labels_gpu_is_none:True",
            b"ids_gpu_is_none:True",
            b"roi_sizes_gpu_is_none:True",
            b"f_gpu_is_none:True",
            b"flat_labels_cpu_is_none:True",
            b"roi_sizes_cpu_is_none:True",
        ])
        expected = hashlib.sha256(expected_payload).hexdigest()
        assert h == expected, (
            f"_initialize_empty_state contract regression. Got {h}, "
            f"expected {expected}. Downstream no-labels-fallback callers "
            f"may now see different field shapes/values."
        )

    def test_initialize_buffers_safely_canonical_snapshot(self):
        """Snapshot the buffer-shape contract for canonical ids.
        For ids=[1, 2, 3] and max_points_cfg=50, expect three deques
        per buffer dict (buffers, _dff_buffers, _spike_buffers), each
        with maxlen=50 and empty initial length. Pins the per-ROI
        buffer surface; any change to maxlen, key dtype, or buffer
        identity set would shift downstream trace storage."""
        host = _Host(max_points_cfg=50)
        host.ids = np.array([1, 2, 3], dtype=np.int32)
        host._initialize_buffers_safely()

        def _describe(d):
            keys = sorted(d.keys())
            return ",".join(
                f"{k}:maxlen={d[k].maxlen}:len={len(d[k])}" for k in keys
            )

        payload = b"|".join([
            b"buffers:" + _describe(host.buffers).encode(),
            b"dff_buffers:" + _describe(host._dff_buffers).encode(),
            b"spike_buffers:" + _describe(host._spike_buffers).encode(),
        ])
        h = hashlib.sha256(payload).hexdigest()
        expected_payload = (
            b"buffers:1:maxlen=50:len=0,2:maxlen=50:len=0,3:maxlen=50:len=0|"
            b"dff_buffers:1:maxlen=50:len=0,2:maxlen=50:len=0,3:maxlen=50:len=0|"
            b"spike_buffers:1:maxlen=50:len=0,2:maxlen=50:len=0,3:maxlen=50:len=0"
        )
        expected = hashlib.sha256(expected_payload).hexdigest()
        assert h == expected, (
            f"_initialize_buffers_safely shape regression. Got {h}, "
            f"expected {expected}. Buffer maxlen, key dtype, or surface "
            f"identity has changed."
        )


class TestConcurrencyProcessing:
    """§1.1 L3.5 row: concurrency ≥1 if mixin touches threads.

    The processing mixin touches `self._gpu_lock` in
    `_on_frame_processed` GPU branch. Per §1.2 concurrency playbook:
    pin state-machine invariants without sleep-as-control.

    - _compute_dff is per-ROI pure: concurrent calls on different
      rid_keys must not corrupt each other's results.
    - _initialize_empty_state is idempotent: concurrent calls must
      converge to the canonical empty state.
    """

    def test_compute_dff_per_roi_isolation(self):
        """Many threads compute dF/F concurrently against independent
        ROIs. The result for each rid_key must match the same call
        made serially (no shared-state leak between ROIs).

        Each thread owns a distinct rid_key + buffer; if the mixin
        cached intermediate state on `self` (e.g. last-baseline), the
        results would scramble under contention. This test pins the
        no-shared-state contract."""
        host = _Host()
        host._last_fps_est = 30.0
        host._baseline_window_s = 1.0
        host._baseline_percentile = 50.0

        N_ROIS = 16
        # Each ROI has a distinct constant-fill baseline
        for rid in range(N_ROIS):
            buf = deque(maxlen=200)
            fill = float(rid + 1)
            for _ in range(60):
                buf.append(fill)
            host.buffers[rid] = buf

        # Expected: for each rid, raw_val = 2*(rid+1), so dF/F = 1.0
        expected = {rid: 1.0 for rid in range(N_ROIS)}
        actual = {}
        actual_lock = threading.Lock()

        def _worker(rid):
            raw = 2.0 * (rid + 1)
            val = host._compute_dff(rid_key=rid, raw_val=raw)
            with actual_lock:
                actual[rid] = val

        threads = [
            threading.Thread(target=_worker, args=(rid,), daemon=True)
            for rid in range(N_ROIS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)
            assert not t.is_alive(), "compute_dff worker hung"

        # All results match serial expectation
        for rid in range(N_ROIS):
            assert actual[rid] == pytest.approx(expected[rid], rel=1e-5), (
                f"per-ROI isolation broken: rid={rid} got {actual[rid]}, "
                f"expected {expected[rid]}"
            )

    def test_initialize_empty_state_idempotent_under_contention(self):
        """Concurrent calls to _initialize_empty_state from N threads
        must converge to the canonical empty state. The method only
        assigns fresh containers — no read-modify-write — so the
        final state is deterministic regardless of interleaving.
        Pin this so a future refactor that introduced merge semantics
        (e.g. preserving prior buffers) fails immediately."""
        host = _Host()
        # Pre-dirty state so a no-op would fail the post-condition
        host.ids = np.array([10, 11, 12], dtype=np.int32)
        host.buffers = {10: deque([1.0, 2.0, 3.0])}
        host._dff_buffers = {10: deque([0.1])}
        host._labels_gpu = "stale"
        host._ids_gpu = "stale"
        host._roi_ready = True

        N_THREADS = 8
        barrier = threading.Barrier(N_THREADS)

        def _worker():
            barrier.wait(timeout=2.0)
            host._initialize_empty_state()

        threads = [
            threading.Thread(target=_worker, daemon=True)
            for _ in range(N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)
            assert not t.is_alive(), "init-empty-state worker hung"

        # Canonical empty state regardless of interleaving
        assert host.ids.dtype == np.int32
        assert len(host.ids) == 0
        assert host.buffers == {}
        assert host._dff_buffers == {}
        assert host._roi_ready is False
        assert host._labels_gpu is None
        assert host._ids_gpu is None
        assert host._roi_sizes_gpu is None
        assert host._f_gpu is None
        assert host._flat_labels_cpu is None
        assert host._roi_sizes_cpu is None
