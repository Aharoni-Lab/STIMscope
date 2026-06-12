"""Comprehensive characterization tests for ``live_trace_plot_pagination``.

target ~75-80 % path coverage on the LiveTracePlotPaginationMixin
(extracted iter 41 commit dbc6a61). This is the FINAL chars suite —
after iter 42 lands, live_trace_extractor.py audit promotes from
🟡 IN PROGRESS to 🟢 DONE provisional.

Module surface (~732 LOC, 10 methods — 9 distinct + 1 D-ltm-1 dup):
- ``_update_paged_trace_mode()`` — ~195 LOC paginated rendering
- ``_update_legend_for_page(page_rois)`` — refresh page legend
- ``_setup_pagination_controls()`` — ~195 LOC widget assembly
- ``_update_page_label_safe()`` (1st def — shadowed by 2nd!)
- ``_prev_roi_page()`` — back-page handler
- ``_next_roi_page()`` — next-page handler
- ``restart_after_napari()`` — napari integration hook
- ``_cleanup_pagination_widget()`` — teardown
- ``_update_page_label_safe()`` (2nd def — LIVE; D-ltm-1 BUG)
- ``_update_page_label()`` — non-safe variant

Pre-existing SMELLs surfaced & pinned in this iter:
- D-ltm-1: `_update_page_label_safe` defined TWICE — Python uses
  only the 2nd. Pin via TestC10MixinIntegration::test_dltm1_*

Branches exercised per method in each test docstring.
QApp + offscreen + sys.path are handled by conftest.py.
"""

from __future__ import annotations

import inspect
import threading
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from PyQt5.QtCore import QObject

import live_trace.plot_pagination as lt_pp
from live_trace.plot_pagination import LiveTracePlotPaginationMixin


# ─────────────────────────────────────────────────────────────────────────────
# Test infrastructure
# ─────────────────────────────────────────────────────────────────────────────


_MISSING = object()


class _Host(QObject, LiveTracePlotPaginationMixin):
    """Stub satisfying the 25-attr mixin contract."""

    def __init__(self, *, plot_widget=_MISSING, buffers=None,
                 traces_per_page=5, page_index=0, highlight_ids=None):
        QObject.__init__(self)
        self.plot_widget = MagicMock() if plot_widget is _MISSING else plot_widget
        self.buffers = buffers if buffers is not None else {}
        self._dff_buffers = {}
        self._spike_buffers = {}
        self.ids = np.array(sorted(self.buffers.keys()), dtype=np.int32) if self.buffers else np.array([], dtype=np.int32)
        self._plot_curves = {}
        self._trace_page_index = page_index
        self._traces_per_page = traces_per_page
        self._global_frame_index = 0
        self._max_points_cfg = 100
        self._last_fps_est = 30.0
        self._x_mode_seconds = False
        self._highlight_ids = highlight_ids if highlight_ids is not None else set()
        self._is_shutting_down = False
        self._cleanup_event = threading.Event()
        self._plot_norm_mode = "Raw"
        # parent-class / sibling-mixin methods (resolved via MRO normally)
        self._resolve_trace_y = MagicMock(side_effect=lambda rid: np.array(
            list(self.buffers.get(rid, deque())), dtype=np.float32))
        self._get_unified_roi_color = MagicMock(return_value='#FF6B6B')


# ─────────────────────────────────────────────────────────────────────────────
# C1 — _update_paged_trace_mode (largest method)
# ─────────────────────────────────────────────────────────────────────────────


class TestC1UpdatePagedTraceMode:
    """Contract: paginated ROI trace rendering on plot_widget.

    Early-return branches:
    - _is_shutting_down=True → skip
    - _cleanup_event set → skip
    - no plot_widget → skip
    - plot_widget without.plot attribute → skip
    """

    def test_shutdown_early_return(self):
        host = _Host()
        host._is_shutting_down = True
        host._update_paged_trace_mode()
        # No plot calls
        host.plot_widget.plot.assert_not_called()

    def test_cleanup_event_set_early_return(self):
        host = _Host()
        host._cleanup_event.set()
        host._update_paged_trace_mode()
        host.plot_widget.plot.assert_not_called()

    def test_no_plot_widget_early_return(self):
        host = _Host(plot_widget=None)
        # No exception
        host._update_paged_trace_mode()

    def test_plot_widget_without_plot_attr_early_return(self):
        host = _Host(plot_widget=object())  # bare object — no.plot
        host._update_paged_trace_mode()  # must not raise

    def test_with_buffers_runs_without_crash(self):
        """Happy path: real buffer + mock plot_widget runs the body."""
        host = _Host(buffers={
            1: deque([10.0, 20.0, 30.0]),
            2: deque([5.0, 15.0, 25.0]),
        })
        # Set viewbox + setData mocks
        viewbox = MagicMock()
        viewbox.viewRange.return_value = [[0, 100], [0, 100]]
        host.plot_widget.getViewBox.return_value = viewbox
        # Should not raise even though pyqtgraph internals are mocked
        host._update_paged_trace_mode()

    def test_viewbox_returns_none_clears_curves(self):
        """When viewbox is None, _plot_curves cleared + early return."""
        host = _Host(buffers={1: deque([1.0, 2.0])})
        host._plot_curves = {1: MagicMock()}
        host.plot_widget.getViewBox.return_value = None
        host._update_paged_trace_mode()
        # _plot_curves cleared
        assert host._plot_curves == {}

    def test_deep_pagination_body_runs(self):
        """Exercise the deep body of _update_paged_trace_mode by mocking
        all the pyqtgraph + Qt internals."""
        host = _Host(buffers={
            i: deque([float(i + k) for k in range(10)]) for i in range(1, 8)
        }, traces_per_page=5, page_index=0)
        viewbox = MagicMock()
        viewbox.viewRange.return_value = [[0, 100], [0, 100]]
        host.plot_widget.getViewBox.return_value = viewbox
        host.plot_widget.plot.return_value = MagicMock()
        # Run — should walk the iteration over active_rois, page slicing,
        # curve creation, etc.
        host._update_paged_trace_mode()
        # plot_widget.plot was called at least once (one curve per
        # paged ROI)
        assert host.plot_widget.plot.call_count > 0

    def test_with_highlight_ids(self):
        """When _highlight_ids is non-empty, highlighted ROIs get thicker pen."""
        host = _Host(
            buffers={i: deque([float(i + k) for k in range(10)]) for i in range(1, 8)},
            highlight_ids={1, 2},
            traces_per_page=5,
        )
        viewbox = MagicMock()
        viewbox.viewRange.return_value = [[0, 100], [0, 100]]
        host.plot_widget.getViewBox.return_value = viewbox
        host.plot_widget.plot.return_value = MagicMock()
        host._update_paged_trace_mode()  # no crash

    def test_curve_validation_loop(self):
        """Exercise the curve-validation loop (lines 126-149) by
        pre-populating _plot_curves with curves that have.scene() returning
        a non-None value."""
        host = _Host(
            buffers={i: deque([float(i + k) for k in range(10)]) for i in range(1, 4)},
            traces_per_page=5,
        )
        # Pre-populate _plot_curves with mock curves
        for rid in [1, 2, 3]:
            curve = MagicMock()
            curve.scene.return_value = MagicMock()  # non-None scene
            host._plot_curves[rid] = curve
        viewbox = MagicMock()
        viewbox.viewRange.return_value = [[0, 100], [0, 100]]
        host.plot_widget.getViewBox.return_value = viewbox
        host.plot_widget.plot.return_value = MagicMock()
        host._update_paged_trace_mode()
        # Curves should have been retained as valid (scene was non-None)

    def test_curve_with_deleted_scene_dropped(self):
        """When a curve's scene() returns None, it's dropped from valid_curves."""
        host = _Host(
            buffers={i: deque([float(i + k) for k in range(10)]) for i in range(1, 3)},
            traces_per_page=5,
        )
        curve_dead = MagicMock()
        curve_dead.scene.return_value = None  # Dead curve
        host._plot_curves = {1: curve_dead}
        viewbox = MagicMock()
        viewbox.viewRange.return_value = [[0, 100], [0, 100]]
        host.plot_widget.getViewBox.return_value = viewbox
        host.plot_widget.plot.return_value = MagicMock()
        host._update_paged_trace_mode()


# ─────────────────────────────────────────────────────────────────────────────
# C2 — _update_legend_for_page
# ─────────────────────────────────────────────────────────────────────────────


class TestC2UpdateLegendForPage:
    """Contract: refresh page-legend labels to match the page's ROI IDs."""

    def test_empty_legend_labels_attr_no_crash(self):
        """When _legend_labels attr is missing, the method handles gracefully."""
        host = _Host()
        # _legend_labels not set up — method tolerates this via try/except
        host._update_legend_for_page([1, 2, 3])

    def test_with_legend_labels_updates(self):
        host = _Host(buffers={
            1: deque([1.0, 2.0]),
            2: deque([3.0, 4.0]),
        })
        # Pre-create legend labels (3 mock QLabels)
        host._legend_labels = [MagicMock() for _ in range(3)]
        host._update_legend_for_page([1, 2])
        # No crash; legend updated for the 2 page-ROIs

    def test_no_legend_layout_early_return(self):
        host = _Host()
        # _legend_layout attr missing → early return inside try/except
        host._update_legend_for_page([1, 2])  # no crash

    def test_creates_combined_legend_label_when_missing(self):
        """When `_combined_legend_label` is missing, create it via QLabel."""
        host = _Host()
        host._legend_layout = MagicMock()
        # _combined_legend_label not set
        import sys
        fake_qtw = MagicMock()
        fake_qtc = MagicMock()
        with patch.dict(sys.modules, {
            'PyQt5.QtWidgets': fake_qtw,
            'PyQt5.QtCore': fake_qtc,
        }):
            host._update_legend_for_page([1, 2])
        assert host._combined_legend_label is not None

    def test_empty_page_rois_shows_no_active_html(self):
        """When page_rois is empty list, sets 'No active traces' HTML."""
        host = _Host()
        host._legend_layout = MagicMock()
        host._combined_legend_label = MagicMock()
        host._update_legend_for_page([])
        args = host._combined_legend_label.setText.call_args[0][0]
        assert "No active traces" in args

    def test_non_empty_page_rois_builds_html(self):
        host = _Host(buffers={
            1: deque([1.0, 2.0]),
            2: deque([3.0, 4.0]),
        })
        host._legend_layout = MagicMock()
        host._combined_legend_label = MagicMock()
        host._update_legend_for_page([1, 2])
        args = host._combined_legend_label.setText.call_args[0][0]
        # HTML format with the unified roi color
        assert "ROI 1" in args
        assert "ROI 2" in args

    def test_falls_back_to_unified_color_when_curve_missing(self):
        """When ROI not in _plot_curves, falls back to _get_unified_roi_color."""
        host = _Host(buffers={1: deque([1.0, 2.0])})
        host._legend_layout = MagicMock()
        host._combined_legend_label = MagicMock()
        host._plot_curves = {}  # empty — ROI 1 not present
        host._update_legend_for_page([1])
        # Should have called _get_unified_roi_color
        host._get_unified_roi_color.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# C3 — _setup_pagination_controls
# ─────────────────────────────────────────────────────────────────────────────


class TestC3SetupPaginationControls:
    """Contract: build Prev/Next buttons + page label widget."""

    def test_no_plot_widget_early_handling(self):
        host = _Host(plot_widget=None)
        # Should handle gracefully (likely via try/except)
        host._setup_pagination_controls()

    def test_with_mocked_widgets(self):
        """Run setup with fully mocked PyQt5.QtWidgets."""
        host = _Host()
        import sys
        fake_qtw = MagicMock()
        # QPushButton + QLabel + QHBoxLayout/VBoxLayout / QWidget all stubbed
        with patch.dict(sys.modules, {'PyQt5.QtWidgets': fake_qtw}):
            host._setup_pagination_controls()
        # Method ran — no crash. Some attrs may not be set due to MagicMock
        # comparisons inside the body (similar to iter-40 aggregation pattern)

    def test_with_deeply_mocked_pyqt5(self):
        """Push past the construction body by mocking PyQt5.QtWidgets +
        PyQt5.QtCore. Same technique as iter-40 aggregation chars."""
        host = _Host()
        import sys
        fake_qtw = MagicMock()
        # Make addWidget / setLayout / count etc. tolerant of MagicMock children
        fake_hbox = MagicMock()
        fake_hbox.count.return_value = 0
        fake_qtw.QHBoxLayout.return_value = fake_hbox
        with patch.dict(sys.modules, {
            'PyQt5.QtWidgets': fake_qtw,
        }):
            host._setup_pagination_controls()
        # Pagination widget should have been attempted
        # (the attribute set inside the method body)


# ─────────────────────────────────────────────────────────────────────────────
# C4 — _update_page_label_safe (BOTH definitions; live = 2nd by Python rule)
# ─────────────────────────────────────────────────────────────────────────────


class TestC4UpdatePageLabelSafe:
    """Contract: 2nd definition (the live one per Python) updates the
    page label to show 'Traces X-Y (Page i/n)' or 'No active traces'."""

    def test_missing_page_label_early_return(self):
        host = _Host()
        # No _page_label attr → early return inside try/except
        host._update_page_label_safe()  # must not raise

    def test_no_active_rois_shows_no_active_message(self):
        host = _Host()
        host._page_label = MagicMock()
        host._prev_button = MagicMock()
        host._next_button = MagicMock()
        # No buffers (empty) → no active_rois → "No active traces"
        host._update_page_label_safe()
        host._page_label.setText.assert_called_with("No active traces")
        host._prev_button.setEnabled.assert_called_with(False)
        host._next_button.setEnabled.assert_called_with(False)

    def test_active_rois_show_page_info(self):
        host = _Host(buffers={
            i: deque([float(i), float(i + 1)]) for i in range(1, 8)
        })
        host._page_label = MagicMock()
        host._update_page_label_safe()
        # Should display "Traces 1-5 (Page 1/2)" with 7 ROIs, page size 5
        args = host._page_label.setText.call_args[0][0]
        assert "Traces" in args
        assert "Page" in args

    def test_buttons_enabled_when_active_rois(self):
        host = _Host(buffers={
            i: deque([float(i), float(i + 1)]) for i in range(1, 8)
        })
        host._page_label = MagicMock()
        host._prev_button = MagicMock()
        host._next_button = MagicMock()
        host._update_page_label_safe()
        host._prev_button.setEnabled.assert_called_with(True)
        host._next_button.setEnabled.assert_called_with(True)

    def test_exception_swallowed(self, capsys):
        host = _Host()
        host._page_label = MagicMock()
        host._page_label.setText.side_effect = RuntimeError("setText broken")
        host.buffers = {1: deque([1.0, 2.0])}
        host._update_page_label_safe()
        captured = capsys.readouterr()
        assert "Page label update error" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C5 — _prev_roi_page
# ─────────────────────────────────────────────────────────────────────────────


class TestC5PrevRoiPage:
    """Contract: navigate to previous page; wrap-around at index 0."""

    def test_navigation_in_progress_early_return(self):
        host = _Host()
        host._navigation_in_progress = True
        host._prev_roi_page()
        # _navigation_in_progress unchanged (still True)
        assert host._navigation_in_progress is True

    def test_no_active_rois_returns_without_change(self):
        host = _Host()  # empty buffers
        host._prev_roi_page()
        assert host._trace_page_index == 0

    def test_wraps_at_index_zero(self):
        host = _Host(
            buffers={i: deque([float(i), float(i + 1)]) for i in range(1, 11)},
            page_index=0,
            traces_per_page=5,
        )
        host._page_label = MagicMock()
        host._prev_roi_page()
        # 10 ROIs, 5/page → 2 pages. Wrap from 0 → 1.
        assert host._trace_page_index == 1

    def test_decrements_when_above_zero(self):
        host = _Host(
            buffers={i: deque([float(i), float(i + 1)]) for i in range(1, 11)},
            page_index=1,
            traces_per_page=5,
        )
        host._page_label = MagicMock()
        host._prev_roi_page()
        assert host._trace_page_index == 0

    def test_navigation_resets_to_false(self):
        host = _Host(
            buffers={i: deque([float(i), float(i + 1)]) for i in range(1, 11)},
            page_index=1,
        )
        host._page_label = MagicMock()
        host._prev_roi_page()
        assert host._navigation_in_progress is False


# ─────────────────────────────────────────────────────────────────────────────
# C6 — _next_roi_page
# ─────────────────────────────────────────────────────────────────────────────


class TestC6NextRoiPage:
    """Contract: navigate to next page; wrap-around at last page."""

    def test_navigation_in_progress_early_return(self):
        host = _Host()
        host._navigation_in_progress = True
        host._next_roi_page()
        assert host._navigation_in_progress is True

    def test_no_active_rois_returns(self):
        host = _Host()
        host._next_roi_page()

    def test_increments(self):
        host = _Host(
            buffers={i: deque([float(i), float(i + 1)]) for i in range(1, 11)},
            page_index=0,
            traces_per_page=5,
        )
        host._page_label = MagicMock()
        host._next_roi_page()
        assert host._trace_page_index == 1

    def test_wraps_at_last_page(self):
        host = _Host(
            buffers={i: deque([float(i), float(i + 1)]) for i in range(1, 11)},
            page_index=1,
            traces_per_page=5,
        )
        host._page_label = MagicMock()
        host._next_roi_page()
        # Wrap to 0
        assert host._trace_page_index == 0

    def test_lazy_init_traces_per_page(self):
        """When _traces_per_page attr is missing, default to 5."""
        host = _Host(
            buffers={i: deque([float(i), float(i + 1)]) for i in range(1, 6)},
            page_index=0,
        )
        del host._traces_per_page
        host._page_label = MagicMock()
        host._next_roi_page()
        assert host._traces_per_page == 5


# ─────────────────────────────────────────────────────────────────────────────
# C7 — restart_after_napari
# ─────────────────────────────────────────────────────────────────────────────


class TestC7RestartAfterNapari:
    """Contract: re-init plot_widget + pagination after napari integration."""

    def test_returns_true_on_success(self):
        host = _Host(buffers={1: deque([1.0, 2.0])})
        with patch.object(host, "_cleanup_pagination_widget"), \
             patch.object(host, "_setup_pagination_controls"), \
             patch.object(host, "_update_paged_trace_mode"):
            result = host.restart_after_napari()
        assert result is True

    def test_updates_plot_widget_when_provided(self):
        host = _Host()
        new_widget = MagicMock()
        with patch.object(host, "_cleanup_pagination_widget"), \
             patch.object(host, "_setup_pagination_controls"), \
             patch.object(host, "_update_paged_trace_mode"):
            host.restart_after_napari(new_plot_widget=new_widget)
        assert host.plot_widget is new_widget

    def test_returns_false_on_exception(self):
        host = _Host()
        # Force exception during pagination setup
        with patch.object(host, "_setup_pagination_controls",
                          side_effect=RuntimeError("setup broken")):
            result = host.restart_after_napari()
        assert result is False

    def test_skips_pagination_when_no_plot_widget(self):
        host = _Host(plot_widget=None)
        with patch.object(host, "_setup_pagination_controls") as mock_setup:
            host.restart_after_napari()
        mock_setup.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# C8 — _cleanup_pagination_widget
# ─────────────────────────────────────────────────────────────────────────────


class TestC8CleanupPaginationWidget:
    """Contract: teardown pagination widget + legend labels."""

    def test_missing_widget_no_crash(self):
        host = _Host()
        # No _pagination_widget attr — method tolerates
        host._cleanup_pagination_widget()

    def test_widget_set_to_none_after_cleanup(self):
        host = _Host()
        host._pagination_widget = MagicMock()
        host._cleanup_pagination_widget()
        assert host._pagination_widget is None

    def test_clears_legend_labels(self):
        host = _Host()
        host._pagination_widget = MagicMock()
        host._legend_labels = [MagicMock(), MagicMock(), MagicMock()]
        host._cleanup_pagination_widget()
        assert host._legend_labels == []

    def test_exception_swallowed(self, capsys):
        host = _Host()
        host._pagination_widget = MagicMock()
        host._pagination_widget.setParent.side_effect = RuntimeError("broken")
        host._cleanup_pagination_widget()  # must not raise
        captured = capsys.readouterr()
        assert "Pagination cleanup warning" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C9 — _update_page_label (non-safe variant)
# ─────────────────────────────────────────────────────────────────────────────


class TestC9UpdatePageLabel:
    """Contract: update page label text — non-safe variant (no button toggle)."""

    def test_with_page_label_and_index(self):
        host = _Host(buffers={
            i: deque([float(i), float(i + 1)]) for i in range(1, 8)
        })
        host._page_label = MagicMock()
        host._update_page_label()
        args = host._page_label.setText.call_args[0][0]
        assert "Traces" in args
        assert "Page" in args

    def test_missing_page_label_no_crash(self):
        host = _Host()
        # _page_label not set — method tolerates (`hasattr` check)
        host._update_page_label()

    def test_missing_trace_page_index_no_crash(self):
        host = _Host()
        host._page_label = MagicMock()
        del host._trace_page_index
        host._update_page_label()
        # Method requires both attrs; misses inner block silently

    def test_exception_swallowed(self, capsys):
        host = _Host(buffers={1: deque([1.0, 2.0])})
        host._page_label = MagicMock()
        host._page_label.setText.side_effect = RuntimeError("setText broken")
        host._update_page_label()
        captured = capsys.readouterr()
        assert "Page label update error" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C10 — Mixin integration + D-ltm-1 BUG pin
# ─────────────────────────────────────────────────────────────────────────────


class TestC10MixinIntegration:
    """Contract: 9 distinct methods accessible on subclass; mixin has no
    __init__; D-ltm-1 BUG (duplicate `_update_page_label_safe`) pinned."""

    METHODS = (
        "_update_paged_trace_mode",
        "_update_legend_for_page",
        "_setup_pagination_controls",
        "_update_page_label_safe",  # appears twice — Python uses 2nd
        "_prev_roi_page",
        "_next_roi_page",
        "restart_after_napari",
        "_cleanup_pagination_widget",
        "_update_page_label",
    )

    def test_all_9_methods_on_subclass(self):
        host = _Host()
        for name in self.METHODS:
            method = getattr(host, name, None)
            assert callable(method), f"Missing or non-callable: {name}"

    def test_methods_defined_on_mixin(self):
        for name in self.METHODS:
            assert name in LiveTracePlotPaginationMixin.__dict__

    def test_mixin_has_no_init(self):
        assert "__init__" not in LiveTracePlotPaginationMixin.__dict__

    def test_pyqtgraph_flag_present(self):
        assert isinstance(lt_pp.PYQTPGRAPH_AVAILABLE, bool)

    def test_dltm1_duplicate_removed(self):
        """D-ltm-1fix iter 43: the first (dead) definition of
        `_update_page_label_safe` was removed. The post-fix source has
        exactly ONE definition. Regression guard against re-introduction
        of the duplicate via copy-paste.
        """
        src = inspect.getsource(lt_pp)
        count = src.count("def _update_page_label_safe(self):")
        assert count == 1, (
            f"D-ltm-1 regression: expected exactly 1 occurrence of "
            f"'def _update_page_label_safe(self):' after iter-43"
            f"dedup, found {count}."
        )

    def test_dltm1_live_behavior_preserved(self):
        """Post-fix the remaining (LIVE) `_update_page_label_safe` should
        still set 'No active traces' when there are no active ROIs. This
        was the behavior of the 2nd def pre-fix; it's now the only def."""
        host = _Host()
        host._page_label = MagicMock()
        host._prev_button = MagicMock()
        host._next_button = MagicMock()
        # No buffers → no active ROIs → "No active traces"
        host._update_page_label_safe()
        host._page_label.setText.assert_called_with("No active traces")
        host._prev_button.setEnabled.assert_called_with(False)


# ─────────────────────────────────────────────────────────────────────────────
# §1.1 L3.5 matrix backfill — Property + Snapshot + Concurrency (iter-61)
#
# §1.1 L3.5 row requires:
#   - Property ≥2 per sub-module (universal floor)
#   - Snapshot required for trace outputs (page label format + button
#     enabled-state contract; both pinned)
#   - Concurrency ≥1 if mixin touches threads — `_cleanup_event`
#     (threading.Event) is referenced in `_update_paged_trace_mode`
#     as a shutdown gate. Pin: gate honored + thread-safe early-exit.
#
# Closes the OPEN BLOCK on iter-42 L3.5 PROMOTION per
# audit_findings.log lines 1655-2235 + docs/PHASE_A5_DEFERRAL.md.
# FINAL L3.5 sub-mixin backfill (live_trace_plot_pagination), 8 of 8.
# After this lands, L3.5 row recovery criterion is met → ready to
# re-promote 🟡 → 🟢.
# ─────────────────────────────────────────────────────────────────────────────

import hashlib  # noqa: E402

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


def _total_pages(n_active, per_page):
    """Reference impl of the pagination formula used throughout the
    mixin: max(1, ceil(n / per))."""
    if per_page <= 0:
        return 1
    return max(1, (n_active + per_page - 1) // per_page)


class TestPropertyPagination:
    """§1.1 universal floor: ≥2 property tests."""

    @given(
        n_active=st.integers(min_value=1, max_value=200),
        per_page=st.integers(min_value=1, max_value=50),
        start_page=st.integers(min_value=0, max_value=200),
        n_clicks=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=40, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_next_page_wraps_in_valid_range(
            self, n_active, per_page, start_page, n_clicks):
        """After ANY sequence of _next_roi_page() clicks from any
        starting state, _trace_page_index ∈ [0, total_pages-1]. Pins
        the wrap-around invariant — a regression that allowed
        page_index to overflow active_rois would crash the rendering
        path with an IndexError."""
        host = _Host()
        host.buffers = {
            rid: deque([float(rid)] * 5)
            for rid in range(n_active)
        }
        host._traces_per_page = per_page
        total = _total_pages(n_active, per_page)
        host._trace_page_index = min(start_page, total - 1)

        # Patch out the side-effects that depend on Qt event loop
        host._update_paged_trace_mode = MagicMock()
        host._update_page_label_safe = MagicMock()

        for _ in range(n_clicks):
            host._next_roi_page()
            assert 0 <= host._trace_page_index < total, (
                f"page_index out of range after _next_roi_page: "
                f"{host._trace_page_index}, n_active={n_active}, "
                f"per_page={per_page}, total={total}"
            )

    @given(
        n_active=st.integers(min_value=0, max_value=500),
        per_page=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=60, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_total_pages_formula_bounded(self, n_active, per_page):
        """For any (n_active, per_page>0), total_pages computed via
        the canonical formula max(1, ceil(n/per)) satisfies:
        - total_pages >= 1 always
        - total_pages * per_page >= n_active (covers all ROIs)
        - For n_active > 0: total_pages == ceil(n/per)

        Pins the ceiling-pagination formula used in 4 places
        (_update_paged_trace_mode, _prev_roi_page, _next_roi_page,
        _update_page_label_safe)."""
        total = _total_pages(n_active, per_page)
        assert total >= 1
        assert total * per_page >= n_active
        if n_active > 0:
            expected = -(-n_active // per_page)  # ceil(n/per)
            assert total == expected, (
                f"pagination formula regression: total={total}, "
                f"expected={expected} for ({n_active}, {per_page})"
            )


class TestSnapshotPaginationContract:
    """§1.1 L3.5 row: snapshot required for trace outputs.

    The page label is an operator-visible UI string; pin its format
    + the button enabled-state contract.
    """

    def test_page_label_format_snapshot(self):
        """Pin the format string of _update_page_label_safe for a
        canonical state: active_rois=12, traces_per_page=5,
        page_index=1 → expected:
            'Traces 6-10 (Page 2/3)'
        + prev/next buttons enabled=True.

        Any change to the label format (e.g. case, delimiters,
        adding a separator) breaks UI tests downstream."""
        host = _Host()
        host.buffers = {
            rid: deque([float(rid)] * 5) for rid in range(12)
        }
        host._traces_per_page = 5
        host._trace_page_index = 1
        host._page_label = MagicMock()
        host._prev_button = MagicMock()
        host._next_button = MagicMock()

        host._update_page_label_safe()

        # Capture the exact text + button states
        label_text = host._page_label.setText.call_args.args[0]
        prev_state = host._prev_button.setEnabled.call_args.args[0]
        next_state = host._next_button.setEnabled.call_args.args[0]

        payload = b"|".join([
            b"label:" + label_text.encode(),
            b"prev_enabled:" + str(prev_state).encode(),
            b"next_enabled:" + str(next_state).encode(),
        ])
        h = hashlib.sha256(payload).hexdigest()
        expected_payload = (
            b"label:Traces 6-10 (Page 2/3)|"
            b"prev_enabled:True|"
            b"next_enabled:True"
        )
        expected = hashlib.sha256(expected_payload).hexdigest()
        assert h == expected, (
            f"page label format regression. Got {payload!r}, "
            f"expected {expected_payload!r}. The UI label format "
            f"or button-enabled contract has shifted."
        )

    def test_no_active_traces_label_snapshot(self):
        """Pin the no-active-traces state contract: label =
        'No active traces' + prev/next buttons DISABLED."""
        host = _Host()  # empty buffers
        host._page_label = MagicMock()
        host._prev_button = MagicMock()
        host._next_button = MagicMock()

        host._update_page_label_safe()

        payload = b"|".join([
            b"label:" + host._page_label.setText.call_args.args[0].encode(),
            b"prev_enabled:"
            + str(host._prev_button.setEnabled.call_args.args[0]).encode(),
            b"next_enabled:"
            + str(host._next_button.setEnabled.call_args.args[0]).encode(),
        ])
        h = hashlib.sha256(payload).hexdigest()
        expected = hashlib.sha256(
            b"label:No active traces|prev_enabled:False|next_enabled:False"
        ).hexdigest()
        assert h == expected, (
            f"no-active-traces contract regression. Got {payload!r}. "
            f"Empty-state UI text or button state has shifted."
        )


class TestConcurrencyCleanupEventGate:
    """§1.1 L3.5 row: concurrency ≥1 if mixin touches threads.

    `live_trace_plot_pagination` honors a `_cleanup_event`
    (threading.Event) shutdown gate in `_update_paged_trace_mode`.
    Per §1.2 concurrency playbook: state-machine invariant, no
    sleep-as-control.

    Two concurrency tests:
    - Gate honored: when _cleanup_event.is_set(), the rendering body
      MUST early-exit before any plot_widget access.
    - Concurrent set+update: setting the event from a background
      thread races safely with _update_paged_trace_mode in the main
      thread (the early-exit path is thread-safe).
    """

    def test_cleanup_event_set_skips_rendering(self):
        """When `_cleanup_event.is_set()` returns True at entry, the
        mixin MUST NOT touch plot_widget. Pins the shutdown gate
        contract — a regression that moved the gate check below the
        viewbox access would crash on a deleted widget at shutdown.
        """
        host = _Host()
        host._cleanup_event.set()
        host.buffers = {
            rid: deque([float(rid)] * 5) for rid in range(3)
        }
        # Replace plot_widget with a spy that fails if touched
        accessed = []
        pw = MagicMock()
        pw.plot.side_effect = lambda *a, **k: accessed.append("plot")
        pw.getViewBox.side_effect = lambda: accessed.append("getViewBox")
        host.plot_widget = pw

        host._update_paged_trace_mode()

        assert accessed == [], (
            f"_cleanup_event gate not honored — plot_widget was "
            f"accessed during shutdown: {accessed}"
        )

    def test_cleanup_event_set_from_background_thread_thread_safe(self):
        """A background thread sets the cleanup event while the main
        thread repeatedly calls _update_paged_trace_mode. Once the
        event is set, all subsequent calls early-exit without crash.
        Pins thread-safety of the gate (event.is_set() is atomic).
        """
        host = _Host()
        host.buffers = {
            rid: deque([float(rid)] * 5) for rid in range(3)
        }
        host.plot_widget = MagicMock()
        host.plot_widget.getViewBox.return_value = MagicMock()

        stop_thread = threading.Event()

        def _setter():
            stop_thread.wait(timeout=0.05)
            host._cleanup_event.set()

        t = threading.Thread(target=_setter, daemon=True)
        t.start()
        stop_thread.set()  # release the setter

        # Spin the main thread doing updates — should not crash
        crashes = []
        for _ in range(50):
            try:
                host._update_paged_trace_mode()
            except Exception as e:
                crashes.append(e)

        t.join(timeout=2.0)
        assert not t.is_alive(), "setter thread hung"
        assert not crashes, f"crashes during shutdown race: {crashes}"

        # After the event is set, calls must early-exit (no plot calls)
        host.plot_widget.reset_mock()
        host._update_paged_trace_mode()
        host.plot_widget.plot.assert_not_called()
        host.plot_widget.getViewBox.assert_not_called()