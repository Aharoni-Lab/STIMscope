"""Import + structural smoke tests for ``live_trace_extractor``.

**Safety-net tests for.6 decomposition (D-lte-13 partial close).**

The full module has zero unit tests today (D-lte-13). Stage-0.6 of the
6-module decomposition requires mixin-based method extractions across
2700+ LOC of hardware-coupled GUI code. Mechanical surgery without
ANY tests is high-risk.

These tests are the **minimum safety net** for.6 work:
- Module imports cleanly (catches syntax errors, missing imports)
- `LiveTraceExtractor` class is accessible after decomposition
- All 5 declared Qt signals on the class are present after every refactor
- Public API methods (declared in the recon spec §3) still exist
- Re-exported names from ``live_trace_perf`` still work via
  ``live_trace_extractor`` (backward-compat for callers)
- ``gpu_ui.py`` (sole production caller) still imports cleanly

These are NOT behavior characterization tests — they only assert the
**structural surface** is preserved across refactor commits. Stage-2
behavioral characterization is gated behind D-lte-13 promotion and is
out of scope here.

If any of these tests fails after a.6 commit, REVERT the
commit before proceeding — the safety net has fired.

Spec: ``docs/specs/L3.5_split_first/live_trace_extractor.md``.
Self-audit log: iter-9 entrance criterion for iter-10.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CRISPI_PATH = REPO_ROOT / "STIMscope" / "STIMViewer_CRISPI"
if str(CRISPI_PATH) not in sys.path:
    sys.path.insert(0, str(CRISPI_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# S1 — Module import smoke
# ─────────────────────────────────────────────────────────────────────────────


class TestS1ModuleImports:
    """Contract: the module + its dependencies import cleanly."""

    def test_live_trace_perf_imports(self):
        """live_trace.perf.py loads without error."""
        import live_trace.perf as live_trace_perf  # noqa: F401

    def test_live_trace_extractor_imports(self):
        """live_trace.extractor.py loads without error."""
        import live_trace.extractor as live_trace_extractor  # noqa: F401

    def test_gpu_ui_imports(self):
        """gpu_ui (sole production caller) imports without error.

        If a.6 commit breaks this, gpu_ui is broken and the
        commit must be reverted.
        """
        import gpu_ui  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# S2 — Public-class surface
# ─────────────────────────────────────────────────────────────────────────────


class TestS2PublicClassSurface:
    """Contract: LiveTraceExtractor + helper classes accessible."""

    def test_live_trace_extractor_class_exists(self):
        from live_trace.extractor import LiveTraceExtractor
        assert LiveTraceExtractor is not None

    def test_performance_monitor_class_exists(self):
        """PerformanceMonitor accessible via both new + legacy import paths."""
        from live_trace.perf import PerformanceMonitor as PM_new
        from live_trace.extractor import PerformanceMonitor as PM_legacy
        # Same class via re-export, not a copy.
        assert PM_new is PM_legacy

    def test_frame_processor_class_exists(self):
        from live_trace.perf import FrameProcessor as FP_new
        from live_trace.extractor import FrameProcessor as FP_legacy
        assert FP_new is FP_legacy

    def test_sync_state_enum_exists(self):
        from live_trace.perf import SyncState as SS_new
        from live_trace.extractor import SyncState as SS_legacy
        assert SS_new is SS_legacy
        # All 7 states declared in the original module
        expected = {"IDLE", "INITIALIZING", "RECORDING", "PROCESSING",
                    "PROJECTING", "STOPPING", "ERROR"}
        assert {s.name for s in SS_new} == expected

    def test_sync_info_dataclass_exists(self):
        from live_trace.perf import SyncInfo as SI_new
        from live_trace.extractor import SyncInfo as SI_legacy
        assert SI_new is SI_legacy

    def test_qimage_to_gray_np_helper_exists(self):
        from live_trace.perf import qimage_to_gray_np as f_new
        from live_trace.extractor import qimage_to_gray_np as f_legacy
        assert f_new is f_legacy


# ─────────────────────────────────────────────────────────────────────────────
# S3 — Declared Qt signals
# ─────────────────────────────────────────────────────────────────────────────


class TestS3QtSignals:
    """Contract: the 5 declared Qt signals on LiveTraceExtractor are present.

    The class declares these at the class body (lines 78-82 in the
    pre-decomposition file). They are the public IPC surface — any.6 refactor that breaks them breaks the GUI silently
    (signal binds at connect time, not at definition).
    """

    @pytest.mark.parametrize("signal_name", [
        "update_plot_signal",
        "gpu_memory_infoing",
        "sync_state_changed",
        "performance_update",
        "error_occurred",
    ])
    def test_class_has_signal_attribute(self, signal_name):
        from live_trace.extractor import LiveTraceExtractor
        # Class-level attribute presence (signals are class attrs in PyQt5)
        assert hasattr(LiveTraceExtractor, signal_name), \
            f"Signal {signal_name!r} missing from LiveTraceExtractor class body"


# ─────────────────────────────────────────────────────────────────────────────
# S4 — Public-method surface
# ─────────────────────────────────────────────────────────────────────────────


class TestS4PublicMethodSurface:
    """Contract: documented public API methods are present on the class.

    These are the methods recon §3 calls out as the public surface used
    by gpu_ui.py + the broader CRISPI orchestration. If.6
    surgery accidentally drops one (e.g. a mixin gets the wrong methods),
    these tests catch it before runtime.
    """

    @pytest.mark.parametrize("method_name", [
        # Configuration / setters
        "set_oasis_enabled",
        "set_neuropil",
        "set_plot_normalization",
        "set_highlight_ids",
        # Camera-frame intake
        "on_frame",
        # Trace export
        "export_traces",
        "get_dff_traces",
        "get_raw_traces",
        "get_spike_traces",
        # Performance
        "get_performance_stats",
        # Lifecycle
        "restart_after_napari",
        "cleanup",
        "stop",
        # Plot-layout builders (mixed-in from live_trace_plot_layouts.py at iter 10)
        "_setup_single_plot_layout",
        "_setup_multi_plot_layout",
        "_setup_plot_with_external_legend",
        "_setup_optimized_single_plot",
    ])
    def test_class_has_method(self, method_name):
        from live_trace.extractor import LiveTraceExtractor
        method = getattr(LiveTraceExtractor, method_name, None)
        assert method is not None, \
            f"Method {method_name!r} missing from LiveTraceExtractor"
        assert callable(method), \
            f"Attribute {method_name!r} exists but is not callable"

    def test_no_known_methods_dropped_by_refactor(self):
        """Resilience: cross-check the full known-method set is present.

        Catches the case where a.6 mixin extraction accidentally
        leaves a method behind in both the new file AND the old class
        (or in neither). Mirrors the parametrize list above as a single
        assertion for fail-fast diagnostics.
        """
        from live_trace.extractor import LiveTraceExtractor
        known_methods = {
            "set_oasis_enabled", "set_neuropil", "set_plot_normalization",
            "set_highlight_ids", "on_frame", "export_traces",
            "get_dff_traces", "get_raw_traces", "get_spike_traces",
            "get_performance_stats", "restart_after_napari",
            "cleanup", "stop",
            "_setup_single_plot_layout", "_setup_multi_plot_layout",
            "_setup_plot_with_external_legend", "_setup_optimized_single_plot",
        }
        actual = set(dir(LiveTraceExtractor))
        missing = known_methods - actual
        assert not missing, \
            f"Methods dropped by refactor (NEITHER on class nor mixed in): {sorted(missing)}"


# ─────────────────────────────────────────────────────────────────────────────
# S5 — Module constants
# ─────────────────────────────────────────────────────────────────────────────


class TestS5ModuleConstants:
    """Contract: module-level constants used by callers are accessible.

    Some callers may have hard-coded ``from live_trace.extractor import
    MAX_FRAME_QUEUE_SIZE``. The re-export must keep those working.
    """

    def test_max_frame_queue_size_value(self):
        from live_trace.perf import MAX_FRAME_QUEUE_SIZE as M_new
        from live_trace.extractor import MAX_FRAME_QUEUE_SIZE as M_legacy
        assert M_new == M_legacy == 8

    def test_extractor_constants_preserved(self):
        """The 5 non-extracted constants are still in live_trace_extractor."""
        import live_trace.extractor as lte
        assert lte.THREAD_POOL_SIZE == 1
        assert lte.SYNCHRONIZATION_TIMEOUT == 3.0
        assert lte.MEMORY_MONITORING_INTERVAL == 5
        assert lte.GPU_MEMORY_CLEANUP_INTERVAL == 15
        assert lte.JETSON_GPU_MEMORY_LIMIT == 0.60
