"""Comprehensive characterization tests for ``live_trace_plot_aggregation``.

target ~80-85 % path coverage on the LiveTracePlotAggregationMixin
(extracted iter 39 commit 6f04e80).

Note on coverage ceiling: `_expand_all_rois` is ~170 LOC of QDialog
construction. Some early-return branches are easy to test, but
fully exercising the construction body requires either a real QDialog
under offscreen Qt (which conftest already configures) or extensive
patching. Tests use the real Qt widgets under `QT_QPA_PLATFORM=offscreen`
where convenient and skip the heavier paths via early-return
fixtures.

Module surface (~517 LOC, 6 methods):
- ``_expand_all_rois()`` — open QDialog with all-ROI view
- ``_update_expanded_plot()`` — incremental update
- ``_update_statistical_aggregation_mode()`` — population mean + std + pXX
- ``_setup_statistical_plot()`` — build curves
- ``_update_density_heatmap_mode()`` — pyqtgraph ImageItem heatmap
- ``_setup_density_plot()`` — build ImageItem + summary curves

Pre-existing SMELLs surfaced in this iter:
- D-lta-1: duplicate "Selected (top-5)" block in _expand_all_rois
  (lines 184-204 of new mixin; two identical try/except blocks back
  to back). Pin via TestC1ExpandAllRois::test_dlta1_duplicate_selected_block.

Branches exercised per method are listed in each test docstring.
QApp + offscreen + sys.path are handled by conftest.py (session autouse).
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from PyQt5.QtCore import QObject

import live_trace.plot_aggregation as lt_pa
from live_trace.plot_aggregation import LiveTracePlotAggregationMixin


# ─────────────────────────────────────────────────────────────────────────────
# Test infrastructure: stub host class
# ─────────────────────────────────────────────────────────────────────────────


class _FakePg:
    """Minimal pyqtgraph shim — covers pg.mkPen + pg.PlotWidget + pg.ImageItem
    + pg.QtCore.Qt.SolidLine/DashLine/DotLine for the setup methods. Real
    pyqtgraph isn't reliable headless in CI."""

    class QtCore:
        class Qt:
            SolidLine = "SolidLine"
            DashLine = "DashLine"
            DotLine = "DotLine"

    class ViewBox:
        XYAxes = "XYAxes"

    @staticmethod
    def mkPen(*args, **kwargs):
        m = MagicMock()
        m.kwargs = kwargs
        return m

    @staticmethod
    def PlotWidget(*args, **kwargs):
        # Returns a MagicMock that quacks like a PlotWidget
        w = MagicMock()
        w.plot.return_value = MagicMock()
        w.getViewBox.return_value = MagicMock()
        return w

    @staticmethod
    def ImageItem(*args, **kwargs):
        return MagicMock()


_MISSING = object()


class _Host(QObject, LiveTracePlotAggregationMixin):
    """Stub satisfying the mixin contract."""

    def __init__(self, *, plot_widget=_MISSING, buffers=None, highlight_ids=None,
                 global_frame_index=0, max_points_cfg=100):
        QObject.__init__(self)
        # Use sentinel so callers can explicitly pass `None` to test the
        # "no plot widget" early-return path
        self.plot_widget = MagicMock() if plot_widget is _MISSING else plot_widget
        self.buffers = buffers if buffers is not None else {}
        self._highlight_ids = highlight_ids if highlight_ids is not None else set()
        self._global_frame_index = global_frame_index
        self._last_fps_est = 30.0
        self._max_points_cfg = max_points_cfg
        self._plot_curves = {}
        # parent-class methods called via MRO
        self._resolve_trace_y = MagicMock(side_effect=lambda rid: np.array(
            list(self.buffers.get(rid, deque())), dtype=np.float32))
        self._get_unified_roi_color = MagicMock(return_value='#FF6B6B')
        self._setup_pagination_controls = MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
# C1 — _setup_statistical_plot
# ─────────────────────────────────────────────────────────────────────────────


class TestC1SetupStatisticalPlot:
    """Contract: build 8 pyqtgraph curves on plot_widget.

    Branches:
    - happy path: 8 curves added (mean, upper_std, lower_std, p75, p25,
      highlight_0/1/2)
    - clears existing _plot_curves
    - exception swallowed
    """

    def test_happy_path_creates_8_curves(self):
        host = _Host()
        with patch.object(lt_pa, "pg", _FakePg):
            host._setup_statistical_plot()
        assert host.plot_widget.plot.call_count == 8
        assert "mean" in host._stat_curves
        assert "upper_std" in host._stat_curves
        assert "lower_std" in host._stat_curves
        assert "p75" in host._stat_curves
        assert "p25" in host._stat_curves
        for i in range(3):
            assert f"highlight_{i}" in host._stat_curves

    def test_clears_existing_plot_curves(self):
        host = _Host()
        # Pre-populate _plot_curves to verify it gets cleared
        host._plot_curves = {1: MagicMock(), 2: MagicMock()}
        with patch.object(lt_pa, "pg", _FakePg):
            host._setup_statistical_plot()
        assert host._plot_curves == {}
        assert host.plot_widget.removeItem.call_count == 2

    def test_exception_swallowed(self, capsys):
        host = _Host()
        host.plot_widget.plot.side_effect = RuntimeError("plot broken")
        with patch.object(lt_pa, "pg", _FakePg):
            host._setup_statistical_plot()  # must not raise
        captured = capsys.readouterr()
        assert "Statistical plot setup error" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C2 — _update_statistical_aggregation_mode
# ─────────────────────────────────────────────────────────────────────────────


class TestC2UpdateStatisticalAggregationMode:
    """Contract: compute mean/std/percentiles across ROI buffers, update
    pyqtgraph curves.

    Branches:
    - lazy init: _stat_curves missing → calls _setup_statistical_plot
    - max_len=0 → early return
    - empty trace_matrix → early return
    - buffers < target_points → padding
    - buffers > target_points → resampling
    - ≥3 ROIs → pagination init + highlight curves
    - _roi_total_pages mismatch → re-sync
    - exception swallowed
    """

    def _stat_ready_host(self):
        """Host with _stat_curves already initialised."""
        host = _Host()
        host._stat_curves = {
            "mean": MagicMock(),
            "upper_std": MagicMock(),
            "lower_std": MagicMock(),
            "p75": MagicMock(),
            "p25": MagicMock(),
            "highlight_0": MagicMock(),
            "highlight_1": MagicMock(),
            "highlight_2": MagicMock(),
        }
        return host

    def test_lazy_init_when_missing_stat_curves(self):
        host = _Host(buffers={1: deque([1.0, 2.0]), 2: deque([3.0, 4.0])})
        with patch.object(lt_pa, "pg", _FakePg):
            host._update_statistical_aggregation_mode()
        # _setup_statistical_plot was triggered → _stat_curves now exists
        assert hasattr(host, "_stat_curves")
        assert host._stat_curves  # non-empty

    def test_empty_buffers_early_return(self):
        host = self._stat_ready_host()
        host.buffers = {}
        # max() of empty generator would raise — but max_len is computed
        # only when buffers has values, so we test that
        with patch.object(lt_pa, "pg", _FakePg):
            # No buffers → the `max(...)` call raises (no buffers > 0)
            # which is caught by outer try/except → no crash
            host._update_statistical_aggregation_mode()
        # No curve updates
        host._stat_curves["mean"].setData.assert_not_called()

    def test_max_len_zero_early_return(self):
        host = self._stat_ready_host()
        host.buffers = {1: deque([5.0])}  # only one point, len=1
        # In the code: `max_len = max(len(buf) for buf in self.buffers.values() if len(buf) > 0)`
        # len(buf)=1 > 0, so max_len=1. Then trace_matrix loop filters len(buf)<2 → skip.
        # trace_matrix empty → second early return.
        with patch.object(lt_pa, "pg", _FakePg):
            host._update_statistical_aggregation_mode()
        host._stat_curves["mean"].setData.assert_not_called()

    def test_happy_path_updates_curves(self):
        host = self._stat_ready_host()
        host.buffers = {
            1: deque([10.0, 20.0, 30.0]),
            2: deque([5.0, 15.0, 25.0]),
        }
        with patch.object(lt_pa, "pg", _FakePg):
            host._update_statistical_aggregation_mode()
        # mean curve updated
        host._stat_curves["mean"].setData.assert_called_once()
        host._stat_curves["upper_std"].setData.assert_called_once()
        host._stat_curves["lower_std"].setData.assert_called_once()
        host._stat_curves["p75"].setData.assert_called_once()
        host._stat_curves["p25"].setData.assert_called_once()

    def test_pagination_init_at_3_rois(self):
        host = self._stat_ready_host()
        host.buffers = {
            i: deque([float(j) for j in range(5)]) for i in range(1, 4)
        }
        with patch.object(lt_pa, "pg", _FakePg):
            host._update_statistical_aggregation_mode()
        # Pagination should have been initialised
        host._setup_pagination_controls.assert_called_once()
        assert host._roi_page_index == 0
        assert host._roi_total_pages == 3

    def test_resampling_when_buffer_longer_than_target(self):
        """target_points = min(300, max_len). When buffer >300, resample."""
        host = self._stat_ready_host()
        host.buffers = {1: deque([float(i) for i in range(400)]),
                        2: deque([float(i) for i in range(400)])}
        with patch.object(lt_pa, "pg", _FakePg):
            host._update_statistical_aggregation_mode()
        # No crash → resampling path exercised
        host._stat_curves["mean"].setData.assert_called_once()

    def test_padding_when_buffer_shorter_than_target(self):
        """When buffer < target_points, last value padded forward."""
        host = self._stat_ready_host()
        # Two different-length buffers: target = min(300, max_len=5) = 5
        host.buffers = {
            1: deque([10.0, 20.0, 30.0]),  # 3 < 5 → padded
            2: deque([5.0, 15.0, 25.0, 35.0, 45.0]),  # 5 = 5
        }
        with patch.object(lt_pa, "pg", _FakePg):
            host._update_statistical_aggregation_mode()
        host._stat_curves["mean"].setData.assert_called_once()

    def test_exception_swallowed(self, capsys):
        host = self._stat_ready_host()
        host.buffers = {1: deque([1.0, 2.0])}
        # Force exception during mean curve update
        host._stat_curves["mean"].setData.side_effect = RuntimeError("boom")
        with patch.object(lt_pa, "pg", _FakePg):
            host._update_statistical_aggregation_mode()
        captured = capsys.readouterr()
        assert "Statistical aggregation mode error" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C3 — _setup_density_plot
# ─────────────────────────────────────────────────────────────────────────────


class TestC3SetupDensityPlot:
    """Contract: build ImageItem + 3 summary curves on plot_widget."""

    def test_happy_path(self):
        host = _Host()
        with patch.object(lt_pa, "pg", _FakePg):
            host._setup_density_plot()
        host.plot_widget.clear.assert_called_once()
        host.plot_widget.addItem.assert_called_once()
        assert host.plot_widget.plot.call_count == 3
        assert "mean" in host._summary_curves
        assert "upper" in host._summary_curves
        assert "lower" in host._summary_curves
        assert hasattr(host, "_density_image")

    def test_exception_swallowed(self, capsys):
        host = _Host()
        host.plot_widget.clear.side_effect = RuntimeError("clear broken")
        with patch.object(lt_pa, "pg", _FakePg):
            host._setup_density_plot()
        captured = capsys.readouterr()
        assert "Density plot setup error" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C4 — _update_density_heatmap_mode
# ─────────────────────────────────────────────────────────────────────────────


class TestC4UpdateDensityHeatmapMode:
    """Contract: build density matrix + update ImageItem + summary curves."""

    def _density_ready_host(self):
        host = _Host()
        host._density_plot = MagicMock()
        host._density_image = MagicMock()
        host._summary_curves = {
            "mean": MagicMock(),
            "upper": MagicMock(),
            "lower": MagicMock(),
        }
        return host

    def test_lazy_init_when_missing_density_plot(self):
        host = _Host(buffers={1: deque([1.0, 2.0])})
        with patch.object(lt_pa, "pg", _FakePg):
            host._update_density_heatmap_mode()
        assert hasattr(host, "_density_image")

    def test_empty_buffers_exception_swallowed(self, capsys):
        host = self._density_ready_host()
        host.buffers = {}
        with patch.object(lt_pa, "pg", _FakePg):
            host._update_density_heatmap_mode()
        captured = capsys.readouterr()
        # max() of empty generator → ValueError → swallowed
        assert "Density heatmap mode error" in captured.out

    def test_happy_path_updates_image(self):
        host = self._density_ready_host()
        host.buffers = {
            1: deque([10.0, 20.0, 30.0]),
            2: deque([5.0, 15.0, 25.0]),
        }
        with patch.object(lt_pa, "pg", _FakePg):
            host._update_density_heatmap_mode()
        host._density_image.setImage.assert_called_once()
        host._summary_curves["mean"].setData.assert_called_once()

    def test_resampling_when_buffer_longer_than_target(self):
        host = self._density_ready_host()
        host.buffers = {1: deque([float(i) for i in range(300)]),
                        2: deque([float(i) for i in range(300)])}
        with patch.object(lt_pa, "pg", _FakePg):
            host._update_density_heatmap_mode()
        host._density_image.setImage.assert_called_once()

    def test_short_buffer_skipped(self):
        """Buffers with len<2 should be skipped via `if len(buf) < 2: continue`."""
        host = self._density_ready_host()
        host.buffers = {
            1: deque([5.0]),  # length 1 — skipped
            2: deque([10.0, 20.0]),
        }
        with patch.object(lt_pa, "pg", _FakePg):
            host._update_density_heatmap_mode()
        host._density_image.setImage.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# C5 — _update_expanded_plot
# ─────────────────────────────────────────────────────────────────────────────


class TestC5UpdateExpandedPlot:
    """Contract: incremental update of the expanded-dialog curves."""

    def _expanded_ready_host(self):
        host = _Host()
        host._expanded_dialog = MagicMock()
        host._expanded_dialog.isVisible.return_value = True
        host._expanded_curves = {
            1: MagicMock(),
            2: MagicMock(),
        }
        host._expanded_plot = MagicMock()
        return host

    def test_missing_dialog_early_return(self):
        host = _Host()
        # No _expanded_dialog or _expanded_curves attrs → early return
        host._update_expanded_plot()  # must not raise

    def test_dialog_invisible_early_return(self):
        host = self._expanded_ready_host()
        host._expanded_dialog.isVisible.return_value = False
        host._update_expanded_plot()
        # No curve updates
        host._expanded_curves[1].setData.assert_not_called()

    def test_happy_path_updates_curves(self):
        host = self._expanded_ready_host()
        host.buffers = {
            1: deque([10.0, 20.0, 30.0]),
            2: deque([5.0, 15.0, 25.0]),
        }
        host._update_expanded_plot()
        host._expanded_curves[1].setData.assert_called_once()
        host._expanded_curves[2].setData.assert_called_once()

    def test_highlight_pen_width_3(self):
        host = self._expanded_ready_host()
        host._highlight_ids = {1}
        host.buffers = {
            1: deque([10.0, 20.0]),
            2: deque([5.0, 15.0]),
        }
        # Set up pen mocks so the setWidth call can be observed
        pen1 = MagicMock()
        pen2 = MagicMock()
        host._expanded_curves[1].opts.get.return_value = pen1
        host._expanded_curves[2].opts.get.return_value = pen2
        host._update_expanded_plot()
        # Pen for ROI 1 (highlighted) → width 3
        pen1.setWidth.assert_called_with(3)
        # Pen for ROI 2 (not highlighted) → width 1
        pen2.setWidth.assert_called_with(1)

    def test_x_mode_seconds_path(self):
        host = self._expanded_ready_host()
        host._x_mode_seconds = True
        host._last_fps_est = 30.0
        host._global_frame_index = 100
        host.buffers = {1: deque([10.0, 20.0])}
        host._update_expanded_plot()
        host._expanded_curves[1].setData.assert_called_once()

    def test_expand_update_count_initialised(self):
        host = self._expanded_ready_host()
        host.buffers = {1: deque([10.0, 20.0])}
        host._update_expanded_plot()
        assert hasattr(host, "_expand_update_count")
        assert host._expand_update_count == 0

    def test_expand_update_count_incremented_on_second_call(self):
        host = self._expanded_ready_host()
        host._expand_update_count = 5
        host.buffers = {1: deque([10.0, 20.0])}
        host._update_expanded_plot()
        assert host._expand_update_count == 6

    def test_outer_exception_swallowed_silently(self):
        """Outer try/except has `pass` — no diagnostic, just swallow."""
        host = self._expanded_ready_host()
        host._expanded_dialog.isVisible.side_effect = RuntimeError("isVisible broken")
        # Must not raise; no print expected since outer except has bare `pass`
        host._update_expanded_plot()


# ─────────────────────────────────────────────────────────────────────────────
# C6 — _expand_all_rois (QDialog construction)
# ─────────────────────────────────────────────────────────────────────────────


class TestC6ExpandAllRois:
    """Contract: open QDialog with all-ROI view.

    Branches:
    - plot_widget None → early return with warning
    - >10 ROIs → spacing-offset path
    - ≤10 ROIs → direct-plot path
    - exception swallowed with traceback
    """

    def test_no_plot_widget_early_return(self, capsys):
        host = _Host(plot_widget=None)
        host._expand_all_rois()
        captured = capsys.readouterr()
        assert "No plot widget available" in captured.out

    def test_exception_path_swallowed(self, capsys):
        """When pyqtgraph import succeeds but QDialog construction errors."""
        host = _Host()
        host.buffers = {1: deque([10.0, 20.0])}
        # Patch the lazy `import pyqtgraph as pg` inside the method to raise
        import sys
        import importlib
        # Save original then inject bad module
        original = sys.modules.get('pyqtgraph')
        bad_pg = MagicMock()
        bad_pg.PlotWidget.side_effect = RuntimeError("pg broken")
        with patch.dict(sys.modules, {'pyqtgraph': bad_pg}):
            host._expand_all_rois()
        captured = capsys.readouterr()
        assert "Error creating expanded view" in captured.out

    def test_le_10_rois_direct_plot_path(self):
        """≤10 active ROIs goes through the direct-plot branch (no spacing
        offset). Exercises lines 153-163 of the mixin."""
        host = _Host()
        # 5 ROIs, all with >=2 points → ≤10 path
        host.buffers = {
            rid: deque([float(rid * 10), float(rid * 10 + 5)])
            for rid in range(1, 6)
        }
        # Need a real-ish PlotWidget mock — pyqtgraph.PlotWidget() and
        #.plot() return MagicMocks that quack
        import sys
        fake_pg = MagicMock()
        fake_pg.PlotWidget.return_value = MagicMock()
        fake_pg.mkPen.return_value = MagicMock()
        with patch.dict(sys.modules, {'pyqtgraph': fake_pg}):
            host._expand_all_rois()
        # Curves should have been created for the 5 ROIs
        assert len(host._expanded_curves) == 5

    def test_gt_10_rois_spacing_path(self):
        """>10 active ROIs goes through the spacing-offset branch with
        global_min/global_max normalization. Exercises lines 121-149."""
        host = _Host()
        # 11 ROIs, all with >=2 points → >10 path
        host.buffers = {
            rid: deque([float(rid * 10), float(rid * 10 + 5)])
            for rid in range(1, 12)
        }
        import sys
        fake_pg = MagicMock()
        fake_pg.PlotWidget.return_value = MagicMock()
        fake_pg.mkPen.return_value = MagicMock()
        with patch.dict(sys.modules, {'pyqtgraph': fake_pg}):
            host._expand_all_rois()
        # Curves should have been created for the 11 ROIs
        assert len(host._expanded_curves) == 11

    def test_selected_ids_legend_added_when_highlight_ids_set(self):
        """When _highlight_ids is non-empty, the duplicate Selected blocks
        each run (lines 195-199 + 206-210 — pinned by D-lta-1)."""
        host = _Host(highlight_ids={1, 2, 3})
        host.buffers = {
            rid: deque([float(rid), float(rid + 1)]) for rid in range(1, 6)
        }
        import sys
        fake_pg = MagicMock()
        fake_pg.PlotWidget.return_value = MagicMock()
        fake_pg.mkPen.return_value = MagicMock()
        with patch.dict(sys.modules, {'pyqtgraph': fake_pg}):
            host._expand_all_rois()
        # The duplicate "Selected" blocks both ran without crashing
        assert len(host._expanded_curves) == 5

    def test_full_dialog_construction_with_fully_mocked_widgets(self):
        """Patch BOTH pyqtgraph AND PyQt5.QtWidgets in sys.modules so the
        lazy from-imports inside _expand_all_rois resolve to MagicMocks.
        This lets the bulk of the construction body run; downstream
        widget-tree assembly hits a MagicMock-vs-int comparison and
        the outer try/except catches gracefully."""
        host = _Host(highlight_ids={1})
        host.buffers = {
            rid: deque([float(rid), float(rid + 1)]) for rid in range(1, 6)
        }
        import sys
        fake_pg = MagicMock()
        fake_pg.PlotWidget.return_value = MagicMock()
        fake_pg.mkPen.return_value = MagicMock()
        # Build a fake PyQt5.QtWidgets module with the 7 names imported.
        # Set QHBoxLayout's count() to return 0 so the `> 0` branch can
        # evaluate without TypeError.
        fake_qtw = MagicMock()
        fake_hbox_instance = MagicMock()
        fake_hbox_instance.count.return_value = 0
        fake_qtw.QHBoxLayout.return_value = fake_hbox_instance
        with patch.dict(sys.modules, {
            'pyqtgraph': fake_pg,
            'PyQt5.QtWidgets': fake_qtw,
        }):
            host._expand_all_rois()
        # Curves stored → reached deep enough into the construction body
        assert len(host._expanded_curves) == 5
        # _expanded_dialog set up
        assert host._expanded_dialog is not None

    def test_dlta1_duplicate_selected_block_removed(self):
        """D-lta-1fix iter 43: the previously-duplicated
        "Selected (top-5)" block was deduped. The post-fix source has
        exactly ONE occurrence. Regression guard against re-introduction
        of the duplicate via copy-paste during future refactors.
        """
        import inspect
        src = inspect.getsource(lt_pa._expand_all_rois if hasattr(lt_pa, '_expand_all_rois')
                                else LiveTracePlotAggregationMixin._expand_all_rois)
        # POST D-lta-1 fix: exactly 1 occurrence
        count = src.count('Selected (top-5):')
        assert count == 1, (
            f"D-lta-1 regression: expected exactly 1 occurrence of "
            f"'Selected (top-5):' after iter-43dedup, found {count}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# C7 — Mixin integration
# ─────────────────────────────────────────────────────────────────────────────


class TestC7MixinIntegration:
    """Contract: 6 methods accessible on subclass; mixin has no __init__."""

    METHODS = (
        "_expand_all_rois",
        "_update_expanded_plot",
        "_update_statistical_aggregation_mode",
        "_setup_statistical_plot",
        "_update_density_heatmap_mode",
        "_setup_density_plot",
    )

    def test_all_6_methods_on_subclass(self):
        host = _Host()
        for name in self.METHODS:
            method = getattr(host, name, None)
            assert callable(method), f"Missing or non-callable: {name}"

    def test_methods_defined_on_mixin(self):
        for name in self.METHODS:
            assert name in LiveTracePlotAggregationMixin.__dict__, (
                f"{name} not defined on LiveTracePlotAggregationMixin"
            )

    def test_mixin_has_no_init(self):
        assert "__init__" not in LiveTracePlotAggregationMixin.__dict__

    def test_pyqtpgraph_flag_present(self):
        assert isinstance(lt_pa.PYQTPGRAPH_AVAILABLE, bool)


# ─────────────────────────────────────────────────────────────────────────────
# §1.1 L3.5 matrix backfill — Property + Snapshot + Structural (iter-60)
#
# §1.1 L3.5 row requires:
#   - Property ≥2 per sub-module (universal floor)
#   - Snapshot required for trace outputs (statistical-curve keyset +
#     pen-color contract; both pinned)
#   - Concurrency: live_trace_plot_aggregation mixin does NOT touch
#     threads (Qt-main-thread pyqtgraph rendering only). Per §1.1
#     "≥1 IF mixin touches threads" — N/A; pinned structurally.
#
# Closes part of the OPEN BLOCK on iter-42 L3.5 PROMOTION per
# audit_findings.log lines 1655-2235 + docs/PHASE_A5_DEFERRAL.md.
# Seventh L3.5 sub-mixin backfill (live_trace_plot_aggregation), 7 of 8.
# ─────────────────────────────────────────────────────────────────────────────

import hashlib  # noqa: E402

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


class TestPropertyAggregationStats:
    """§1.1 universal floor: ≥2 property tests."""

    @given(
        n_rois=st.integers(min_value=2, max_value=20),
        n_points=st.integers(min_value=2, max_value=50),
        fill=st.floats(min_value=-1e4, max_value=1e4,
                       allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_constant_fill_identity_stats(self, n_rois, n_points, fill):
        """When all ROI buffers contain the same constant `fill`, the
        statistical aggregation must yield: mean == fill, std == 0,
        p25 == p75 == fill. Pins the aggregation arithmetic identity
        — any change in axis (e.g. axis=1 vs axis=0) or to the
        percentile-vs-mean dispatch would break this for many seeds.

        Implementation detail: we exercise the trace_matrix-construction
        + np.mean/std/percentile code path by invoking
        _update_statistical_aggregation_mode with mocked setData curves
        and reading back the captured y arrays."""
        host = _Host()
        host.buffers = {
            rid: deque([fill] * n_points)
            for rid in range(n_rois)
        }
        # Pre-build _stat_curves so the setup branch is skipped
        host._stat_curves = {
            k: MagicMock() for k in (
                "mean", "upper_std", "lower_std", "p75", "p25",
                "highlight_0", "highlight_1", "highlight_2",
            )
        }
        host._roi_page_index = 0
        host._roi_page_size = 3
        host._roi_total_pages = n_rois

        with patch.object(lt_pa, "pg", _FakePg):
            host._update_statistical_aggregation_mode()

        # mean curve setData(x=..., y=mean_trace) — read y from call
        mean_call = host._stat_curves["mean"].setData.call_args
        y_mean = mean_call.kwargs["y"]
        np.testing.assert_allclose(y_mean, fill, rtol=1e-5, atol=1e-5)

        # upper_std: y == mean + std == mean + 0 == mean == fill
        upper_call = host._stat_curves["upper_std"].setData.call_args
        np.testing.assert_allclose(upper_call.kwargs["y"], fill,
                                   rtol=1e-5, atol=1e-5)

        # p75 and p25 of constant: == fill
        p75_call = host._stat_curves["p75"].setData.call_args
        p25_call = host._stat_curves["p25"].setData.call_args
        np.testing.assert_allclose(p75_call.kwargs["y"], fill,
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(p25_call.kwargs["y"], fill,
                                   rtol=1e-5, atol=1e-5)

    @given(
        n_rois=st.integers(min_value=2, max_value=10),
        n_points=st.integers(min_value=2, max_value=500),
    )
    @settings(max_examples=20, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_density_target_points_capped_at_200(self, n_rois, n_points):
        """For ANY (n_rois, n_points), _update_density_heatmap_mode
        produces a density matrix with second-axis size
        min(200, n_points). Pins the target_points ceiling — a
        regression that removed the min(200, max_len) cap would
        cause OOM at high frame counts."""
        host = _Host()
        host.buffers = {
            rid: deque([float(i + rid) for i in range(n_points)])
            for rid in range(n_rois)
        }
        # Pre-build _density_image so the setup branch is skipped
        host._density_image = MagicMock()
        host._summary_curves = {
            k: MagicMock() for k in ("mean", "upper", "lower")
        }
        host._density_plot = True  # mark setup done

        with patch.object(lt_pa, "pg", _FakePg):
            host._update_density_heatmap_mode()

        # The density image setImage receives the density_matrix.
        # Capture the matrix and check shape.
        call_args = host._density_image.setImage.call_args
        density_matrix = call_args.args[0]
        assert density_matrix.shape[0] == n_rois
        assert density_matrix.shape[1] == min(200, n_points), (
            f"target_points cap violated: matrix shape "
            f"{density_matrix.shape}, expected ({n_rois}, "
            f"{min(200, n_points)})"
        )


class TestSnapshotAggregationContract:
    """§1.1 L3.5 row: snapshot required for trace outputs.

    Two operator-visible contract snapshots:
    - Statistical-curve key set (8 curves: mean, ±std, p25/75, 3 highlights)
    - Statistical pen-color contract (the canonical color palette
      operators recognize on the statistical-aggregation page)
    """

    def test_statistical_plot_curve_keyset_snapshot(self):
        """Pin the 8-curve key set produced by _setup_statistical_plot.
        Downstream code in _update_statistical_aggregation_mode looks
        up curves by these exact string keys; any rename or addition
        would crash silently with an `if X in self._stat_curves`
        miss. Snapshot guarantees the contract."""
        host = _Host()
        with patch.object(lt_pa, "pg", _FakePg):
            host._setup_statistical_plot()

        keys = tuple(sorted(host._stat_curves.keys()))
        h = hashlib.sha256(repr(keys).encode()).hexdigest()
        expected_keys = (
            "highlight_0", "highlight_1", "highlight_2",
            "lower_std", "mean", "p25", "p75", "upper_std",
        )
        expected = hashlib.sha256(repr(expected_keys).encode()).hexdigest()
        assert h == expected, (
            f"statistical-curve keyset regression. Got {keys!r}, "
            f"expected {expected_keys!r}. A curve has been renamed, "
            f"added, or removed."
        )

    def test_statistical_pen_color_palette_snapshot(self):
        """Pin the 6 canonical pen colors used by _setup_statistical_plot:
        - mean: #3498db (blue)
        - std (upper/lower): #85c1e8 (light blue)
        - p75/p25: #2ecc71 (green)
        - highlight_0: #e74c3c (red)
        - highlight_1: #f39c12 (orange)
        - highlight_2: #9b59b6 (purple)

        These are the colors operators visually recognize on the
        statistical-aggregation plot; a silent palette shift would
        change the visual contract."""
        host = _Host()
        captured = []

        class _ColorCapturingPg(_FakePg):
            @staticmethod
            def mkPen(*args, **kwargs):
                color = kwargs.get("color", args[0] if args else None)
                captured.append(color)
                m = MagicMock()
                m.kwargs = kwargs
                return m

        with patch.object(lt_pa, "pg", _ColorCapturingPg):
            host._setup_statistical_plot()

        # Filter to hex-string colors (the canonical pens; the dashed
        # std uses same color twice but mkPen is called per curve)
        hex_colors = [c for c in captured if isinstance(c, str) and c.startswith("#")]
        h = hashlib.sha256(",".join(hex_colors).encode()).hexdigest()
        # mkPen call order (one pen reused for std upper/lower and for
        # p75/p25): mean(#3498db), std(#85c1e8), perc(#2ecc71),
        # highlight_0(#e74c3c), highlight_1(#f39c12), highlight_2(#9b59b6).
        expected_colors = [
            "#3498db",
            "#85c1e8",
            "#2ecc71",
            "#e74c3c", "#f39c12", "#9b59b6",
        ]
        expected = hashlib.sha256(",".join(expected_colors).encode()).hexdigest()
        assert h == expected, (
            f"statistical pen-color palette regression. Got "
            f"{hex_colors!r}, expected {expected_colors!r}."
        )


class TestStructuralNoThreadAffordanceAggregation:
    """§1.1 L3.5 row: concurrency cell justification.

    `live_trace_plot_aggregation` is Qt-main-thread-only pyqtgraph
    rendering. Per §1.1 "≥1 IF mixin touches threads" — N/A.
    Pinned structurally.
    """

    def test_module_does_not_import_threading_primitives(self):
        """No threading / Lock / RLock / Semaphore / QThread / Future
        references. If introduced, force §1.1 concurrency tests."""
        import inspect
        src = inspect.getsource(lt_pa)
        forbidden = [
            "import threading",
            "from threading import",
            "Lock(",
            "RLock(",
            "Semaphore(",
            "Event(",
            "QThread",
            "concurrent.futures",
            "Future(",
        ]
        offenders = [tok for tok in forbidden if tok in src]
        assert not offenders, (
            f"live_trace_plot_aggregation introduced threading "
            f"primitives: {offenders}. Per §1.1 L3.5 row, add ≥1 "
            f"concurrency tests before removing this guard."
        )
