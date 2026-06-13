"""Comprehensive characterization tests for ``live_trace_init``.

target ~90% path coverage on the LiveTraceInitMixin (extracted at
iter 33 commit 568ab34).

Module surface (~178 LOC, 5 methods):
- ``_init_roi_processing(label_path, max_rois, max_points)`` — load
  labels.npz, initialise ROI buffer state on the host
- ``_limit_cuda_pools()`` — cap cupy default + pinned memory pools at
  256 MB each (best-effort, swallow exceptions)
- ``_init_plotting(plot_widget)`` — wire plot widget + QTimer at
  camera-matched interval
- ``_detect_camera_fps()`` — auto-detect FPS via 5 cascading strategies
- ``_calculate_update_throttle(max_rois)`` — pure throttle ladder

Mixin contract — subclass provides:
- ``self.camera`` (any of: get_actual_fps / node_map / fps-attrs /
  get_fps)
- ``self.use_pygame_plot`` (bool — skip plotting when True)
- ``self.ids`` (list[int], writable)
- ``self.update_plot_signal`` (pyqtSignal())
- ``self._setup_single_plot_layout`` / ``self._setup_multi_plot_layout``
  (from LiveTracePlotLayoutsMixin)

QApp + QT_QPA_PLATFORM offscreen + sys.path are handled by
``tests/L3_5_split_first/conftest.py`` (session autouse).

Branches exercised per method are listed in each test docstring.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from PyQt5.QtCore import QObject, pyqtSignal

import live_trace.init as lti_init
from live_trace.init import LiveTraceInitMixin


# ─────────────────────────────────────────────────────────────────────────────
# Test infrastructure: stub host class for the mixin
# ─────────────────────────────────────────────────────────────────────────────


class _Host(QObject, LiveTraceInitMixin):
    """Stub host class satisfying the mixin's `self.X` expectations.

    Inherits QObject so `_init_plotting` can pass `self` as QTimer parent.
    """

    update_plot_signal = pyqtSignal()

    def __init__(self, ids=None, use_pygame_plot=False, camera=None):
        QObject.__init__(self)
        self.ids = ids if ids is not None else [1, 2, 3]
        self.use_pygame_plot = use_pygame_plot
        self.camera = camera if camera is not None else MagicMock()
        # Spy on plot-layout dispatchers
        self._setup_single_plot_layout = MagicMock()
        self._setup_multi_plot_layout = MagicMock()


def _write_labels_npz(tmp_path, labels):
    """Write a labels.npz file matching the loader's expectations."""
    path = tmp_path / "labels.npz"
    np.savez(path, labels=labels)
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# C1 — _init_roi_processing
# ─────────────────────────────────────────────────────────────────────────────


class TestC1InitRoiProcessing:
    """Contract: load labels, snapshot config, zero out GPU buffer state.

    Branches:
    - happy path: 2D labels array → all state initialised
    - labels.ndim != 2 → ValueError raised
    - empty labels (max=0) → max_label snapshotted as 0
    """

    def test_happy_path_assigns_all_state(self, tmp_path):
        labels = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
        path = _write_labels_npz(tmp_path, labels)
        host = _Host()
        host._init_roi_processing(path, max_rois=10, max_points=500)
        assert np.array_equal(host._labels_orig, labels)
        assert host._roi_max == 2
        assert host._max_rois_cfg == 10
        assert host._max_points_cfg == 500
        assert host._roi_ready is False
        assert host._ids_gpu is None
        assert host._roi_sizes_gpu is None
        assert host._f_gpu is None
        assert host._roi_sizes_cpu is None
        assert host._flat_labels_cpu is None
        assert host._max_label == 0  # explicit zero on init regardless of labels
        assert host.ids == []

    def test_labels_ndim_not_2_raises(self, tmp_path):
        labels = np.array([1, 2, 3], dtype=np.int32)  # 1D
        path = _write_labels_npz(tmp_path, labels)
        host = _Host()
        with pytest.raises(ValueError, match="labels must be 2D"):
            host._init_roi_processing(path, max_rois=10, max_points=500)

    def test_labels_3d_also_raises(self, tmp_path):
        labels = np.zeros((2, 2, 2), dtype=np.int32)  # 3D
        path = _write_labels_npz(tmp_path, labels)
        host = _Host()
        with pytest.raises(ValueError, match="labels must be 2D"):
            host._init_roi_processing(path, max_rois=10, max_points=500)

    def test_empty_labels_roi_max_is_zero(self, tmp_path):
        labels = np.zeros((4, 4), dtype=np.int32)
        path = _write_labels_npz(tmp_path, labels)
        host = _Host()
        host._init_roi_processing(path, max_rois=5, max_points=100)
        assert host._roi_max == 0

    def test_labels_cast_to_int32(self, tmp_path):
        labels = np.array([[0, 1], [2, 0]], dtype=np.int64)
        path = _write_labels_npz(tmp_path, labels)
        host = _Host()
        host._init_roi_processing(path, max_rois=5, max_points=100)
        assert host._labels_orig.dtype == np.int32


# ─────────────────────────────────────────────────────────────────────────────
# C2 — _limit_cuda_pools
# ─────────────────────────────────────────────────────────────────────────────


class TestC2LimitCudaPools:
    """Contract: best-effort cap default + pinned cupy pools at 256 MB.

    Branches:
    - happy path: both pools have set_limit → both called with 256 MB
    - mempool lacks set_limit → skipped silently
    - pinned mempool lacks set_limit → skipped silently
    - any exception → swallowed with diagnostic print
    - cp is None (no cuda) → swallowed via exception (AttributeError)
    """

    def test_happy_path_sets_both_limits(self, capsys):
        host = _Host()
        mempool = MagicMock()
        pinned = MagicMock()
        fake_cp = MagicMock()
        fake_cp.get_default_memory_pool.return_value = mempool
        fake_cp.get_default_pinned_memory_pool.return_value = pinned
        with patch.object(lti_init, "cp", fake_cp):
            host._limit_cuda_pools()
        mempool.set_limit.assert_called_once_with(size=2**28)
        pinned.set_limit.assert_called_once_with(size=2**28)
        captured = capsys.readouterr()
        assert "256MB" in captured.out

    def test_mempool_without_set_limit_skipped(self, capsys):
        host = _Host()
        # Force hasattr to be False on set_limit by using a spec without it
        mempool = MagicMock(spec=[])
        pinned = MagicMock()
        fake_cp = MagicMock()
        fake_cp.get_default_memory_pool.return_value = mempool
        fake_cp.get_default_pinned_memory_pool.return_value = pinned
        with patch.object(lti_init, "cp", fake_cp):
            host._limit_cuda_pools()
        pinned.set_limit.assert_called_once_with(size=2**28)
        captured = capsys.readouterr()
        # Only pinned pool message should appear
        assert captured.out.count("256MB") == 1

    def test_pinned_pool_without_set_limit_skipped(self, capsys):
        host = _Host()
        mempool = MagicMock()
        pinned = MagicMock(spec=[])
        fake_cp = MagicMock()
        fake_cp.get_default_memory_pool.return_value = mempool
        fake_cp.get_default_pinned_memory_pool.return_value = pinned
        with patch.object(lti_init, "cp", fake_cp):
            host._limit_cuda_pools()
        mempool.set_limit.assert_called_once_with(size=2**28)
        captured = capsys.readouterr()
        assert captured.out.count("256MB") == 1

    def test_exception_swallowed_with_diagnostic(self, capsys):
        host = _Host()
        fake_cp = MagicMock()
        fake_cp.get_default_memory_pool.side_effect = RuntimeError("cuda blew up")
        with patch.object(lti_init, "cp", fake_cp):
            host._limit_cuda_pools()  # must not raise
        captured = capsys.readouterr()
        assert "Could not set CUDA pool limits" in captured.out
        assert "cuda blew up" in captured.out

    def test_cp_is_none_no_crash(self, capsys):
        """When cupy import failed at module load, cp is None — should
        be caught by the try/except wrapper without propagating."""
        host = _Host()
        with patch.object(lti_init, "cp", None):
            host._limit_cuda_pools()  # must not raise
        captured = capsys.readouterr()
        assert "Could not set CUDA pool limits" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C3 — _init_plotting
# ─────────────────────────────────────────────────────────────────────────────


class TestC3InitPlotting:
    """Contract: skip if pygame mode; else build layout + QTimer at
    camera-matched interval.

    Branches:
    - use_pygame_plot=True → early return; _legend=None; no timer
    - plot_widget=None → skip layout setup, still build timer
    - roi_count <= 20 → _setup_single_plot_layout called
    - roi_count > 20 → _setup_multi_plot_layout called
    - PYQTPGRAPH_AVAILABLE=False → skip layout setup, still build timer
    """

    def test_pygame_mode_early_return(self):
        host = _Host(use_pygame_plot=True)
        host._init_plotting(plot_widget=MagicMock())
        assert host._legend is None
        host._setup_single_plot_layout.assert_not_called()
        host._setup_multi_plot_layout.assert_not_called()
        assert not hasattr(host, "_plot_timer")

    def test_plot_widget_none_skips_layout_but_builds_timer(self):
        host = _Host(ids=[1, 2])
        host.camera.get_actual_fps = MagicMock(return_value=30.0)
        with patch.object(lti_init, "PYQTPGRAPH_AVAILABLE", True):
            host._init_plotting(plot_widget=None)
        host._setup_single_plot_layout.assert_not_called()
        host._setup_multi_plot_layout.assert_not_called()
        assert hasattr(host, "_plot_timer")
        assert host._plot_timer.isActive()
        # Plot timer is capped at ~15Hz (67ms) regardless of camera fps
        # to avoid saturating the Qt event loop during many-ROI plotting.
        assert host._plot_timer.interval() == max(int(1000 / 30.0), 67)
        host._plot_timer.stop()

    def test_roi_count_le_20_uses_single_layout(self):
        host = _Host(ids=list(range(20)))  # exactly 20
        host.camera.get_actual_fps = MagicMock(return_value=30.0)
        pw = MagicMock()
        with patch.object(lti_init, "PYQTPGRAPH_AVAILABLE", True):
            host._init_plotting(plot_widget=pw)
        host._setup_single_plot_layout.assert_called_once_with(pw, 20)
        host._setup_multi_plot_layout.assert_not_called()
        host._plot_timer.stop()

    def test_roi_count_gt_20_uses_multi_layout(self):
        host = _Host(ids=list(range(21)))  # 21
        host.camera.get_actual_fps = MagicMock(return_value=30.0)
        pw = MagicMock()
        with patch.object(lti_init, "PYQTPGRAPH_AVAILABLE", True):
            host._init_plotting(plot_widget=pw)
        host._setup_multi_plot_layout.assert_called_once_with(pw, 21)
        host._setup_single_plot_layout.assert_not_called()
        host._plot_timer.stop()

    def test_pyqtgraph_unavailable_skips_layout(self):
        host = _Host(ids=[1, 2])
        host.camera.get_actual_fps = MagicMock(return_value=30.0)
        pw = MagicMock()
        with patch.object(lti_init, "PYQTPGRAPH_AVAILABLE", False):
            host._init_plotting(plot_widget=pw)
        host._setup_single_plot_layout.assert_not_called()
        host._setup_multi_plot_layout.assert_not_called()
        assert hasattr(host, "_plot_timer")
        host._plot_timer.stop()

    def test_timer_interval_capped_at_15hz(self):
        host = _Host(ids=[1])
        host.camera.get_actual_fps = MagicMock(return_value=60.0)
        with patch.object(lti_init, "PYQTPGRAPH_AVAILABLE", True):
            host._init_plotting(plot_widget=MagicMock())
        # 60fps would give 16ms but production caps at ~15Hz (67ms minimum)
        # to keep the Qt main thread from saturating during many-ROI plotting.
        assert host._plot_timer.interval() == max(int(1000 / 60.0), 67)
        host._plot_timer.stop()

    def test_last_fps_est_recorded(self):
        host = _Host(ids=[1])
        host.camera.get_actual_fps = MagicMock(return_value=45.0)
        with patch.object(lti_init, "PYQTPGRAPH_AVAILABLE", True):
            host._init_plotting(plot_widget=MagicMock())
        assert host._last_fps_est == 45.0
        host._plot_timer.stop()


# ─────────────────────────────────────────────────────────────────────────────
# C4 — _detect_camera_fps (5 cascading strategies)
# ─────────────────────────────────────────────────────────────────────────────


class TestC4DetectCameraFps:
    """Contract: 5 cascading strategies, default 30.0 on full miss.

    Strategy cascade:
    1. camera.get_actual_fps() → if truthy + >0, return float(fps)
    2. camera.node_map.FindNode("AcquisitionFrameRate") → if Readable + >0
    3. camera.{fps, framerate, frame_rate, acquisition_fps} → first truthy
    4. camera.get_fps() → if truthy + >0
    5. default 30.0

    Plus: outer try/except → default 30.0 on unexpected crash.
    """

    def _bare_camera(self):
        """A real bare object with NO methods/attrs — hasattr returns False
        for every probe (MagicMock auto-supplies attributes which defeats
        hasattr-based strategies)."""

        class _Bare:
            pass

        return _Bare()

    # ── Strategy 1: get_actual_fps ──────────────────────────────────────

    def test_strategy1_get_actual_fps_returns_float(self, capsys):
        cam = self._bare_camera()
        cam.get_actual_fps = lambda: 42.5
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == pytest.approx(42.5)
        captured = capsys.readouterr()
        assert "get_actual_fps" in captured.out

    def test_strategy1_get_actual_fps_zero_falls_through(self):
        cam = self._bare_camera()
        cam.get_actual_fps = lambda: 0.0  # falls through
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 30.0  # default

    def test_strategy1_get_actual_fps_none_falls_through(self):
        cam = self._bare_camera()
        cam.get_actual_fps = lambda: None
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 30.0

    # ── Strategy 2: node_map ────────────────────────────────────────────

    def test_strategy2_node_map_returns_fps(self, capsys):
        cam = self._bare_camera()
        node = MagicMock()
        node.IsReadable.return_value = True
        node.Value.return_value = 25.0
        node_map = MagicMock()
        node_map.FindNode.return_value = node
        cam.node_map = node_map
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 25.0
        captured = capsys.readouterr()
        assert "node map" in captured.out

    def test_strategy2_node_map_not_readable_falls_through(self):
        cam = self._bare_camera()
        node = MagicMock()
        node.IsReadable.return_value = False
        node_map = MagicMock()
        node_map.FindNode.return_value = node
        cam.node_map = node_map
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 30.0

    def test_strategy2_node_map_zero_falls_through(self):
        cam = self._bare_camera()
        node = MagicMock()
        node.IsReadable.return_value = True
        node.Value.return_value = 0.0  # falsy in the > 0 check
        node_map = MagicMock()
        node_map.FindNode.return_value = node
        cam.node_map = node_map
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 30.0

    def test_strategy2_node_map_falsy_skipped(self):
        """node_map is set but falsy (e.g. None) — should skip the block."""
        cam = self._bare_camera()
        cam.node_map = None
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 30.0

    def test_strategy2_node_map_exception_logged_and_skipped(self, capsys):
        cam = self._bare_camera()
        node_map = MagicMock()
        node_map.FindNode.side_effect = RuntimeError("node map crashed")
        cam.node_map = node_map
        host = _Host(camera=cam)
        result = host._detect_camera_fps()
        assert result == 30.0  # fell through to default
        captured = capsys.readouterr()
        assert "Node map FPS detection failed" in captured.out

    # ── Strategy 3: fps / framerate / frame_rate / acquisition_fps ──────

    def test_strategy3_fps_attr_returned(self, capsys):
        cam = self._bare_camera()
        cam.fps = 24.0
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 24.0
        captured = capsys.readouterr()
        assert "via fps" in captured.out

    def test_strategy3_framerate_attr_returned(self):
        cam = self._bare_camera()
        cam.framerate = 50.0
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 50.0

    def test_strategy3_frame_rate_attr_returned(self):
        cam = self._bare_camera()
        cam.frame_rate = 18.0
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 18.0

    def test_strategy3_acquisition_fps_attr_returned(self):
        cam = self._bare_camera()
        cam.acquisition_fps = 33.0
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 33.0

    def test_strategy3_zero_value_falls_through(self):
        cam = self._bare_camera()
        cam.fps = 0
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 30.0

    def test_strategy3_exception_during_getattr_swallowed(self):
        """Accessing the attribute raises — outer try/except catches it.

        The property raises on every access, so `hasattr` itself re-raises
        the RuntimeError (Python 3 hasattr only swallows AttributeError),
        which bubbles to the outermost try/except → default 30.0.
        """

        class _ExplodeCam:
            @property
            def fps(self):
                raise RuntimeError("property exploded")

        host = _Host(camera=_ExplodeCam())
        assert host._detect_camera_fps() == 30.0  # outer except → default

    def test_strategy3_inner_except_swallows_comparison_failure(self):
        """Targets the inner `except Exception: pass` (lines 145-146).

        hasattr-probe succeeds (property returns a non-numeric sentinel),
        but the subsequent `if fps > 0` comparison raises TypeError. The
        inner except swallows it and the loop proceeds to the next
        candidate attribute. With no other fps-shaped attrs, returns 30.0.
        """

        class _SentinelCam:
            @property
            def fps(self):
                # Truthy object — hasattr succeeds; `if fps` is True; but
                # `fps > 0` raises TypeError on object().
                return object()

        host = _Host(camera=_SentinelCam())
        # Inner except swallows TypeError; loop falls through to default
        assert host._detect_camera_fps() == 30.0

    # ── Strategy 4: get_fps ─────────────────────────────────────────────

    def test_strategy4_get_fps_returns(self, capsys):
        cam = self._bare_camera()
        cam.get_fps = lambda: 19.0
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 19.0
        captured = capsys.readouterr()
        assert "get_fps" in captured.out

    def test_strategy4_get_fps_zero_falls_through(self):
        cam = self._bare_camera()
        cam.get_fps = lambda: 0.0
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 30.0

    def test_strategy4_get_fps_exception_swallowed(self):
        cam = self._bare_camera()

        def _boom():
            raise RuntimeError("get_fps boom")

        cam.get_fps = _boom
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 30.0

    # ── Default + outer exception ───────────────────────────────────────

    def test_no_camera_methods_returns_default(self, capsys):
        cam = self._bare_camera()
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 30.0
        captured = capsys.readouterr()
        assert "30 fps default" in captured.out

    def test_outer_exception_returns_default(self, capsys):
        """If a `hasattr` probe itself raises (e.g. __getattr__ blows up),
        the outer try/except catches it and returns the default."""

        class _NastyCam:
            def __getattribute__(self, name):
                # __init__ etc still work, but any user-attribute probe raises
                if name.startswith("_") or name in ("__class__",):
                    return object.__getattribute__(self, name)
                raise RuntimeError(f"{name} probe exploded")

        host = _Host(camera=_NastyCam())
        assert host._detect_camera_fps() == 30.0
        captured = capsys.readouterr()
        assert "Camera FPS detection error" in captured.out

    # ── Strategy ordering invariant ─────────────────────────────────────

    def test_strategy_ordering_first_match_wins(self):
        """When multiple strategies would return distinct values, the
        earliest one in the cascade wins."""
        cam = self._bare_camera()
        cam.get_actual_fps = lambda: 100.0  # strategy 1
        cam.fps = 200.0  # strategy 3
        cam.get_fps = lambda: 300.0  # strategy 4
        host = _Host(camera=cam)
        assert host._detect_camera_fps() == 100.0  # strategy 1 wins


# ─────────────────────────────────────────────────────────────────────────────
# C5 — _calculate_update_throttle (pure ladder)
# ─────────────────────────────────────────────────────────────────────────────


class TestC5CalculateUpdateThrottle:
    """Contract: pure 4-step ladder on max_rois.

    Ladder:
    - max_rois <= 10 → 2
    - 10 < max_rois <= 25 → 3
    - 25 < max_rois <= 50 → 5
    - max_rois > 50 → 8
    """

    @pytest.mark.parametrize(
        "max_rois,expected",
        [
            (0, 2),    # edge: 0
            (1, 2),
            (10, 2),   # boundary low
            (11, 3),   # boundary +1
            (25, 3),   # boundary
            (26, 5),
            (50, 5),   # boundary
            (51, 8),
            (1000, 8),
        ],
    )
    def test_ladder_boundaries(self, max_rois, expected):
        host = _Host()
        assert host._calculate_update_throttle(max_rois) == expected

    def test_negative_treated_as_low(self):
        """Negative max_rois <= 10 so returns 2 (pure functional behavior)."""
        host = _Host()
        assert host._calculate_update_throttle(-5) == 2


# ─────────────────────────────────────────────────────────────────────────────
# C6 — Mixin integration
# ─────────────────────────────────────────────────────────────────────────────


class TestC6MixinIntegration:
    """Contract: methods accessible on subclass; mixin has no __init__."""

    def test_all_5_methods_on_subclass(self):
        host = _Host()
        for name in (
            "_init_roi_processing",
            "_limit_cuda_pools",
            "_init_plotting",
            "_detect_camera_fps",
            "_calculate_update_throttle",
        ):
            method = getattr(host, name, None)
            assert callable(method), f"Missing or non-callable: {name}"

    def test_methods_defined_on_mixin(self):
        """Confirm the 5 methods come from LiveTraceInitMixin, not from
        _Host or QObject accidentally."""
        for name in (
            "_init_roi_processing",
            "_limit_cuda_pools",
            "_init_plotting",
            "_detect_camera_fps",
            "_calculate_update_throttle",
        ):
            assert name in LiveTraceInitMixin.__dict__

    def test_mixin_has_no_init(self):
        """The mixin relies entirely on subclass-provided state, so it
        must not define its own __init__."""
        assert "__init__" not in LiveTraceInitMixin.__dict__

    def test_pyqtpgraph_available_flag_exists(self):
        """Module-level constant should always exist (True or False)."""
        assert isinstance(lti_init.PYQTPGRAPH_AVAILABLE, bool)

    def test_cuda_available_flag_exists(self):
        assert isinstance(lti_init.CUDA_AVAILABLE, bool)


# ─────────────────────────────────────────────────────────────────────────────
# §1.1 L3.5 matrix backfill — Property + Snapshot + Concurrency (iter-55)
#
# §1.1 L3.5 row requires:
#   - Property ≥2 per sub-module (universal floor)
#   - Snapshot required for trace outputs (init wires the labels→ROI
#     state that all downstream trace extraction reads — snapshot the
#     post-init state for canonical labels)
#   - Concurrency ≥1 if mixin touches threads (`_init_plotting` owns a
#     QTimer that drives the plot-update signal — pin shutdown invariants)
#
# Closes part of the OPEN BLOCK on iter-42 L3.5 PROMOTION per
# audit_findings.log lines 1655-2235 + docs/PHASE_A5_DEFERRAL.md.
# Second L3.5 sub-mixin backfill (live_trace_init), 2 of 8.
# ─────────────────────────────────────────────────────────────────────────────

import hashlib  # noqa: E402
import time  # noqa: E402

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


class TestPropertyCalculateUpdateThrottle:
    """§1.1 universal floor: ≥2 property tests for `_calculate_update_throttle`.

    The throttle ladder is the pure-functional plot-update governor;
    it must satisfy invariants across the entire non-negative range
    of max_rois, not just the hand-picked boundaries in C5.
    """

    @given(max_rois=st.integers(min_value=-100, max_value=10_000))
    @settings(max_examples=80, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_throttle_monotonic_nondecreasing(self, max_rois):
        """For any (a, b) with a <= b in the supported range, the
        throttle output is monotonic non-decreasing. Pins the ladder
        ordering invariant — a regression that inverted any band
        (e.g. swapping the 25 and 50 thresholds) would fail this.
        """
        host = _Host()
        t_a = host._calculate_update_throttle(max_rois)
        t_b = host._calculate_update_throttle(max_rois + 1)
        assert t_a <= t_b, (
            f"Throttle ladder not monotonic: f({max_rois})={t_a} > "
            f"f({max_rois + 1})={t_b}"
        )

    @given(max_rois=st.integers(min_value=-100, max_value=10_000))
    @settings(max_examples=80, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_throttle_codomain_is_fixed_ladder_set(self, max_rois):
        """The throttle output is always one of the four canonical
        ladder values {2, 3, 5, 8}. Pins that the function is a
        total function over int → fixed codomain — a regression
        that introduced a stray default branch (e.g. returning
        ``max_rois // 10``) would fail this."""
        host = _Host()
        assert host._calculate_update_throttle(max_rois) in {2, 3, 5, 8}


class TestSnapshotInitRoiPostState:
    """§1.1 L3.5 row: snapshot required for trace outputs.

    `_init_roi_processing` is the entry point for the labels→ROI
    pipeline that every downstream trace-extraction call reads.
    Pin a sha256 of the post-init state for a canonical labels
    array; any regression in label loading, dtype coercion, or
    ROI state zero-initialisation will fail the hash.

    Per §1.5 snapshot policy: deterministically-derivable
    artifacts → hash assertion in-line (< 100KB).
    """

    def _canonical_labels(self):
        """Reproducible 8×6 label tile with 3 ROIs (ids 1, 2, 3) and
        a background of 0. Same fixture across snapshot tests."""
        return np.array(
            [
                [0, 0, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 0, 2, 2, 0],
                [0, 0, 2, 2, 2, 0],
                [0, 3, 3, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
                [3, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

    def test_canonical_labels_post_init_state_hash(self, tmp_path):
        """Pin the post-init host state for canonical labels. Combines:
        - the labels array bytes (dtype int32, row-major)
        - the (_roi_max, _max_label) tuple
        - the GPU-buffer null-state markers
        - the ids list (must be empty at this stage)
        Into a single sha256 digest. Any change to label loading,
        dtype coercion, or ROI state init breaks this.
        """
        labels = self._canonical_labels()
        path = _write_labels_npz(tmp_path, labels)
        host = _Host()
        host._init_roi_processing(path, max_rois=10, max_points=500)

        # Build the post-init state payload deterministically
        payload = b"".join([
            b"labels:",
            host._labels_orig.tobytes(),
            b"|shape:",
            repr(host._labels_orig.shape).encode(),
            b"|dtype:",
            str(host._labels_orig.dtype).encode(),
            b"|roi_max:",
            str(host._roi_max).encode(),
            b"|max_label:",
            str(host._max_label).encode(),
            b"|ids:",
            repr(host.ids).encode(),
            b"|f_gpu_is_none:",
            str(host._f_gpu is None).encode(),
            b"|ids_gpu_is_none:",
            str(host._ids_gpu is None).encode(),
            b"|roi_sizes_gpu_is_none:",
            str(host._roi_sizes_gpu is None).encode(),
            b"|roi_sizes_cpu_is_none:",
            str(host._roi_sizes_cpu is None).encode(),
            b"|flat_labels_cpu_is_none:",
            str(host._flat_labels_cpu is None).encode(),
            b"|roi_ready:",
            str(host._roi_ready).encode(),
        ])
        h = hashlib.sha256(payload).hexdigest()

        # Recovery: if the contract intentionally evolves, regenerate
        # by printing payload and updating both the hash and the spec.
        expected_payload = b"".join([
            b"labels:",
            labels.tobytes(),
            b"|shape:",
            repr((8, 6)).encode(),
            b"|dtype:",
            b"int32",
            b"|roi_max:",
            b"3",
            b"|max_label:",
            b"0",
            b"|ids:",
            b"[]",
            b"|f_gpu_is_none:True",
            b"|ids_gpu_is_none:True",
            b"|roi_sizes_gpu_is_none:True",
            b"|roi_sizes_cpu_is_none:True",
            b"|flat_labels_cpu_is_none:True",
            b"|roi_ready:False",
        ])
        expected = hashlib.sha256(expected_payload).hexdigest()
        assert h == expected, (
            f"_init_roi_processing post-state regression. Got {h}, "
            f"expected {expected}. Either labels coercion, ROI-state "
            f"init, or the ids-empty invariant changed."
        )

    def test_throttle_ladder_table_snapshot(self):
        """Pin the entire (max_rois → throttle) table for the
        canonical sweep N ∈ [0, 60]. The trace-update cadence is
        ladder-driven; any silent change to a band threshold (e.g.
        moving the 25 boundary to 30) would shift the plot-update
        rate at runtime — fail this hash.
        """
        host = _Host()
        table = b",".join(
            f"{n}:{host._calculate_update_throttle(n)}".encode()
            for n in range(0, 61)
        )
        h = hashlib.sha256(table).hexdigest()
        # Manually derived expected table per the iter-33 spec
        # (<=10 → 2; <=25 → 3; <=50 → 5; else 8)
        expected_table = b",".join(
            f"{n}:{2 if n <= 10 else 3 if n <= 25 else 5 if n <= 50 else 8}".encode()
            for n in range(0, 61)
        )
        expected = hashlib.sha256(expected_table).hexdigest()
        assert h == expected, (
            f"Throttle ladder boundary regression. Got {h}, expected "
            f"{expected}. A band threshold or output value has shifted."
        )


class TestConcurrencyInitPlotting:
    """§1.1 L3.5 row: concurrency ≥1 if mixin touches threads.

    `_init_plotting` owns a QTimer that emits ``update_plot_signal``
    on its interval — the timer is the live ROI-plot pacemaker.
    Per §1.2 concurrency-test playbook: pin shutdown invariants
    (state-machine, no time-based sleeps as control flow).

    Note: the QTimer fires under the QApplication event loop; with
    the offscreen platform and no processEvents(), it will NOT
    actually emit. That is fine for these tests — we pin the
    state-machine surface (isActive, interval, stop idempotency,
    parenting), not the actual emit cadence. This mirrors the
    iter-54 FrameProcessor approach (no.start() call, just
    state invariants).
    """

    def test_plot_timer_stop_idempotent(self):
        """§1.2.3 inspired: timer.stop() must flip isActive() to False
        and be safe to call repeatedly. Any future refactor that puts
        non-idempotent cleanup in stop() (closing a queue, joining a
        worker thread) would fail this — surfacing the regression
        before it crashes on the real ROI-plot shutdown path."""
        host = _Host(ids=[1, 2, 3])
        host.camera.get_actual_fps = lambda: 30.0
        with patch.object(lti_init, "PYQTPGRAPH_AVAILABLE", True):
            host._init_plotting(plot_widget=MagicMock())
        assert host._plot_timer.isActive() is True
        host._plot_timer.stop()
        assert host._plot_timer.isActive() is False
        # Idempotent: stopping a stopped timer must not raise/deadlock
        host._plot_timer.stop()
        assert host._plot_timer.isActive() is False

    def test_plot_timer_parented_for_qt_cleanup(self):
        """§1.2 lifecycle invariant: the QTimer must be parented to
        the host QObject so Qt's parent-owns-child deletion cleans
        it up when the host is destroyed. An un-parented QTimer
        leaks across the trial loop — pin parenting here so a
        regression to ``QTimer()`` (no parent) fails immediately."""
        host = _Host(ids=[1])
        host.camera.get_actual_fps = lambda: 30.0
        with patch.object(lti_init, "PYQTPGRAPH_AVAILABLE", True):
            host._init_plotting(plot_widget=MagicMock())
        try:
            assert host._plot_timer.parent() is host, (
                "QTimer must be parented to host for Qt-owned cleanup."
            )
        finally:
            host._plot_timer.stop()

    def test_plot_timer_creation_completes_within_budget(self):
        """§1.2.3: timer wiring must complete in bounded wall-clock
        time. Even with the fps-detection cascade, plotting init
        should finish well under 1s — a regression that introduced
        a blocking probe (e.g. a synchronous network call to fetch
        config) would fail this budget. No `sleep` is used as a
        control mechanism; we measure elapsed wall-clock around the
        synchronous init call."""
        host = _Host(ids=[1, 2, 3])
        host.camera.get_actual_fps = lambda: 30.0
        t0 = time.monotonic()
        with patch.object(lti_init, "PYQTPGRAPH_AVAILABLE", True):
            host._init_plotting(plot_widget=MagicMock())
        elapsed = time.monotonic() - t0
        try:
            assert elapsed < 1.0, (
                f"_init_plotting took {elapsed:.3f}s — over the 1s budget. "
                f"A blocking probe was likely introduced into the init path."
            )
        finally:
            host._plot_timer.stop()
