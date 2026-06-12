"""Comprehensive characterization tests for ``live_trace_plot_layouts``.

target ~90% path coverage. Tests pin the AS-IS behavior of the 4
plot-layout builder methods on ``LiveTracePlotLayoutsMixin``.

Module surface (~205 LOC, 4 methods):
- ``_setup_single_plot_layout(plot_widget, roi_count)`` — single-plot + legend
- ``_setup_multi_plot_layout(plot_widget, roi_count)`` — dispatcher
- ``_setup_plot_with_external_legend(plot_widget, parent_widget, roi_count)``
  — sidecar legend in parent layout
- ``_setup_optimized_single_plot(plot_widget, roi_count)`` — no-legend
  fallback for high ROI counts

The mixin expects subclass state:
- ``self.ids`` — List[int]
- ``self._plot_curves`` — Dict[int, curve] (populated by methods)
- ``self._legend`` — set in _setup_single_plot_layout
- ``self.plot_widget`` — set in all 4 methods
- ``self._get_unified_roi_color(rid)`` — returns hex color string

Tests use a stub host class that satisfies the mixin contract +
MagicMock for the plot_widget so no real pyqtgraph rendering happens.

Branches exercised per method are listed in each test docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Note: QApplication setup + sys.path + QT_QPA_PLATFORM offscreen are
# handled by tests/L3_5_split_first/conftest.py (session autouse).
from live_trace.plot_layouts import LiveTracePlotLayoutsMixin


# ─────────────────────────────────────────────────────────────────────────────
# Test infrastructure: stub host class for the mixin
# ─────────────────────────────────────────────────────────────────────────────


class _Host(LiveTracePlotLayoutsMixin):
    """Stub subclass satisfying the mixin's `self.X` expectations."""

    def __init__(self, ids=None):
        self.ids = ids if ids is not None else [1, 2, 3]
        self._plot_curves = {}
        self._legend = None
        self.plot_widget = None

    def _get_unified_roi_color(self, rid: int) -> str:
        # Return a stable test color per ROI
        return "#FF8040"


def _make_plot_widget_mock():
    """MagicMock that satisfies the pyqtgraph PlotWidget interface used
    by the mixin (setBackground, setDownsampling, setClipToView,
    showGrid, setMouseEnabled, setLabel, addLegend, plot, parent)."""
    pw = MagicMock()
    pw.parent.return_value = None  # default: no parent
    # plot() returns a curve mock
    pw.plot.return_value = MagicMock()
    pw.addLegend.return_value = MagicMock()
    return pw


# ─────────────────────────────────────────────────────────────────────────────
# C1 — _setup_single_plot_layout
# ─────────────────────────────────────────────────────────────────────────────


class TestC1SingleLayout:
    """Contract: configures plot widget + legend + adds one curve per ROI.

    Branches:
    - normal: plot configured, legend added, curve per ID stored in _plot_curves
    - exception in try block: caught + logged, no crash
    """

    def test_assigns_plot_widget_to_self(self):
        host = _Host(ids=[1, 2, 3])
        pw = _make_plot_widget_mock()
        host._setup_single_plot_layout(pw, roi_count=3)
        assert host.plot_widget is pw

    def test_configures_widget(self):
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        host._setup_single_plot_layout(pw, roi_count=1)
        pw.setBackground.assert_called_with('k')
        pw.setDownsampling.assert_called_with(auto=True, mode='peak')
        pw.setClipToView.assert_called_with(True)
        pw.showGrid.assert_called_with(x=True, y=True, alpha=0.25)
        pw.setMouseEnabled.assert_called_with(x=True, y=True)

    def test_labels_axes(self):
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        host._setup_single_plot_layout(pw, roi_count=1)
        label_calls = pw.setLabel.call_args_list
        # Should be called for 'left' and 'bottom'
        positions = {c.args[0] for c in label_calls}
        assert 'left' in positions
        assert 'bottom' in positions

    def test_adds_legend(self):
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        host._setup_single_plot_layout(pw, roi_count=1)
        pw.addLegend.assert_called_once_with(offset=(10, 10))
        assert host._legend is not None

    def test_one_curve_per_id(self):
        host = _Host(ids=[10, 20, 30])
        pw = _make_plot_widget_mock()
        host._setup_single_plot_layout(pw, roi_count=3)
        assert set(host._plot_curves.keys()) == {10, 20, 30}
        assert pw.plot.call_count == 3

    def test_uses_unified_color_per_roi(self):
        host = _Host(ids=[1, 2])
        host._get_unified_roi_color = MagicMock(return_value="#FF0000")
        pw = _make_plot_widget_mock()
        host._setup_single_plot_layout(pw, roi_count=2)
        # _get_unified_roi_color called once per ID
        assert host._get_unified_roi_color.call_count == 2

    def test_swallows_exception_no_crash(self, capsys):
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        pw.addLegend.side_effect = RuntimeError("legend broken")
        # Should not raise
        host._setup_single_plot_layout(pw, roi_count=1)
        captured = capsys.readouterr()
        assert "Single plot setup failed" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C2 — _setup_multi_plot_layout (dispatcher)
# ─────────────────────────────────────────────────────────────────────────────


class TestC2MultiLayout:
    """Contract: dispatches to external_legend or optimized_single based on
    parent widget's layout attribute. Exception falls back to optimized.

    Branches:
    - parent has 'layout' attr → external_legend path
    - parent has 'setLayout' (but not layout?) → external_legend path
    - parent has neither → optimized_single path
    - exception during dispatch → optimized_single fallback
    """

    def test_parent_with_layout_dispatches_to_external_legend(self):
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        # Parent has both layout AND setLayout
        parent = MagicMock()
        parent.layout.return_value = MagicMock()
        pw.parent.return_value = parent
        with patch.object(host, "_setup_plot_with_external_legend") as mock_ext:
            host._setup_multi_plot_layout(pw, roi_count=5)
            mock_ext.assert_called_once_with(pw, parent, 5)

    def test_no_parent_uses_plot_widget_as_parent_then_external_legend(self):
        """When plot_widget.parent() returns None, the function uses
        plot_widget itself. MagicMock plot_widget has both layout +
        setLayout (auto-stubbed), so external_legend is called."""
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        pw.parent.return_value = None
        with patch.object(host, "_setup_plot_with_external_legend") as mock_ext:
            host._setup_multi_plot_layout(pw, roi_count=5)
            mock_ext.assert_called_once()

    def test_parent_without_layout_or_setlayout_goes_to_optimized(self):
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        # Parent is a strict object that lacks layout AND setLayout
        class _BareParent:
            pass
        parent = _BareParent()
        pw.parent.return_value = parent
        with patch.object(host, "_setup_optimized_single_plot") as mock_opt:
            host._setup_multi_plot_layout(pw, roi_count=5)
            mock_opt.assert_called_once_with(pw, 5)

    def test_exception_falls_back_to_optimized(self, capsys):
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        pw.parent.side_effect = RuntimeError("parent failed")
        with patch.object(host, "_setup_optimized_single_plot") as mock_opt:
            host._setup_multi_plot_layout(pw, roi_count=5)
            mock_opt.assert_called_once_with(pw, 5)
        captured = capsys.readouterr()
        assert "Multi-plot setup failed" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C3 — _setup_plot_with_external_legend
# ─────────────────────────────────────────────────────────────────────────────


class TestC3ExternalLegend:
    """Contract: builds sidecar legend widget + adds curves with downsampling
    for high ROI counts.

    Branches:
    - parent.layout() truthy → add to parent layout, complete normally
    - parent.layout() falsy → fall back to optimized_single + return
    - roi_count > 30 → curve.setDownsampling enabled
    - roi_count <= 30 → no downsampling
    - exception → optimized_single fallback
    """

    def test_normal_path_stores_curves(self):
        """Verify curves are stored even with mock plot_widget.

        NOTE: full external-legend path (incl. parent_layout.addLayout)
        requires a real QWidget as plot_widget — main_layout.addWidget()
        rejects MagicMock. We verify the pre-addWidget portion of the
        path here. The lines 152-153 + 159 (post-addWidget calls) are
        the 3% uncovered statements — testing them requires real
        pyqtgraph PlotWidget which is overkill for unit-level tests.
        """
        host = _Host(ids=[1, 2, 3])
        pw = _make_plot_widget_mock()
        parent = MagicMock()
        parent.layout.return_value = MagicMock()  # truthy
        host._setup_plot_with_external_legend(pw, parent, roi_count=3)
        # Curves stored (happens inside the for loop, before the failing
        # main_layout.addWidget call)
        assert set(host._plot_curves.keys()) == {1, 2, 3}

    def test_high_roi_count_enables_downsampling_on_curves(self):
        host = _Host(ids=list(range(40)))  # > 30
        pw = _make_plot_widget_mock()
        # Each curve is a separate MagicMock
        curves = [MagicMock() for _ in range(40)]
        pw.plot.side_effect = curves
        parent = MagicMock()
        parent.layout.return_value = MagicMock()
        host._setup_plot_with_external_legend(pw, parent, roi_count=40)
        # All 40 curves should have setDownsampling called
        for c in curves:
            c.setDownsampling.assert_called_with(factor=2, auto=True, method='peak')

    def test_low_roi_count_no_downsampling(self):
        host = _Host(ids=[1, 2, 3])  # <= 30
        pw = _make_plot_widget_mock()
        curves = [MagicMock() for _ in range(3)]
        pw.plot.side_effect = curves
        parent = MagicMock()
        parent.layout.return_value = MagicMock()
        host._setup_plot_with_external_legend(pw, parent, roi_count=3)
        for c in curves:
            c.setDownsampling.assert_not_called()

    def test_parent_without_layout_falls_back_to_optimized(self, capsys):
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        parent = MagicMock()
        parent.layout.return_value = None  # falsy
        # The code path: hasattr(parent_widget, 'layout') is True (it's a
        # MagicMock so hasattr is True), but parent_widget.layout() is None
        # → goes to else branch → optimized fallback
        with patch.object(host, "_setup_optimized_single_plot") as mock_opt:
            host._setup_plot_with_external_legend(pw, parent, roi_count=1)
            mock_opt.assert_called_once_with(pw, 1)
        captured = capsys.readouterr()
        assert "Could not create external legend" in captured.out

    def test_exception_during_setup_falls_back_to_optimized(self, capsys):
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        pw.setBackground.side_effect = RuntimeError("widget broken")
        parent = MagicMock()
        with patch.object(host, "_setup_optimized_single_plot") as mock_opt:
            host._setup_plot_with_external_legend(pw, parent, roi_count=1)
            mock_opt.assert_called_once_with(pw, 1)
        captured = capsys.readouterr()
        assert "External legend setup failed" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C4 — _setup_optimized_single_plot
# ─────────────────────────────────────────────────────────────────────────────


class TestC4OptimizedSinglePlot:
    """Contract: no-legend single plot + auto-color via pg.intColor.

    Branches:
    - normal path → all curves stored
    - roi_count > 25 → curve.setDownsampling enabled
    - roi_count <= 25 → no downsampling
    - exception → caught + logged
    """

    def test_normal_path(self):
        host = _Host(ids=[1, 2, 3])
        pw = _make_plot_widget_mock()
        host._setup_optimized_single_plot(pw, roi_count=3)
        assert set(host._plot_curves.keys()) == {1, 2, 3}

    def test_configures_widget(self):
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        host._setup_optimized_single_plot(pw, roi_count=1)
        pw.setBackground.assert_called_with('k')
        pw.setDownsampling.assert_called_with(auto=True, mode='peak')

    def test_assigns_widget_to_self(self):
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        host._setup_optimized_single_plot(pw, roi_count=1)
        assert host.plot_widget is pw

    def test_high_roi_count_enables_downsampling(self):
        host = _Host(ids=list(range(30)))  # > 25
        pw = _make_plot_widget_mock()
        curves = [MagicMock() for _ in range(30)]
        pw.plot.side_effect = curves
        host._setup_optimized_single_plot(pw, roi_count=30)
        for c in curves:
            c.setDownsampling.assert_called_with(factor=3, auto=True, method='peak')

    def test_low_roi_count_no_downsampling(self):
        host = _Host(ids=[1, 2])  # <= 25
        pw = _make_plot_widget_mock()
        curves = [MagicMock() for _ in range(2)]
        pw.plot.side_effect = curves
        host._setup_optimized_single_plot(pw, roi_count=2)
        for c in curves:
            c.setDownsampling.assert_not_called()

    def test_exception_caught(self, capsys):
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        pw.setBackground.side_effect = RuntimeError("widget exploded")
        # Should not raise
        host._setup_optimized_single_plot(pw, roi_count=1)
        captured = capsys.readouterr()
        assert "Optimized plot setup failed" in captured.out

    def test_uses_pg_intcolor_with_hue_count(self):
        """Hue count is min(15, max(8, roi_count)). Verify pg.intColor
        receives the hues kwarg."""
        host = _Host(ids=[1, 2, 3, 4, 5])  # roi_count=5 → hues=max(8,5)=8
        pw = _make_plot_widget_mock()
        import live_trace.plot_layouts as ltpl
        with patch.object(ltpl.pg, "intColor", wraps=ltpl.pg.intColor) as spy:
            host._setup_optimized_single_plot(pw, roi_count=5)
            # All 5 ROIs should have called intColor with hues=8
            for call in spy.call_args_list:
                assert call.kwargs.get("hues") == 8

    def test_pg_intcolor_high_roi_count_caps_at_15_hues(self):
        host = _Host(ids=list(range(20)))  # > 15
        pw = _make_plot_widget_mock()
        import live_trace.plot_layouts as ltpl
        with patch.object(ltpl.pg, "intColor", wraps=ltpl.pg.intColor) as spy:
            host._setup_optimized_single_plot(pw, roi_count=20)
            for call in spy.call_args_list:
                assert call.kwargs.get("hues") == 15  # capped at 15


# ─────────────────────────────────────────────────────────────────────────────
# C5 — Mixin integration: methods accessible as instance methods
# ─────────────────────────────────────────────────────────────────────────────


class TestC5MixinIntegration:
    """Contract: when host class inherits the mixin, methods are accessible
    via self.method()."""

    def test_all_4_methods_on_subclass(self):
        host = _Host()
        for name in ("_setup_single_plot_layout", "_setup_multi_plot_layout",
                     "_setup_plot_with_external_legend",
                     "_setup_optimized_single_plot"):
            method = getattr(host, name, None)
            assert callable(method), f"Missing or non-callable: {name}"

    def test_methods_not_inherited_from_object(self):
        """Confirm the 4 methods come from LiveTracePlotLayoutsMixin, not
        accidentally defined on _Host or object."""
        host = _Host()
        for name in ("_setup_single_plot_layout", "_setup_multi_plot_layout",
                     "_setup_plot_with_external_legend",
                     "_setup_optimized_single_plot"):
            # The unbound function should be defined on the mixin class
            assert name in LiveTracePlotLayoutsMixin.__dict__

    def test_mixin_has_no_init_state(self):
        """The mixin should not require its own __init__ — relies entirely
        on subclass state."""
        # LiveTracePlotLayoutsMixin should not define __init__
        assert "__init__" not in LiveTracePlotLayoutsMixin.__dict__


# ─────────────────────────────────────────────────────────────────────────────
# §1.1 L3.5 matrix backfill — Property + Snapshot + Concurrency (iter-58)
#
# §1.1 L3.5 row requires:
#   - Property ≥2 per sub-module (universal floor)
#   - Snapshot required for trace outputs (plot-widget configuration is
#     a downstream-visible contract; pin the call set + downsampling
#     ladder thresholds)
#   - Concurrency: live_trace_plot_layouts mixin does NOT touch threads
#     (no threading imports, no Lock/RLock acquisition, no QThread).
#     Per §1.1 "≥1 IF mixin touches threads" — N/A. We document this
#     and add a structural test pinning the no-thread-affordance
#     contract so a future refactor that introduced threading would
#     fail this test and force §1.1 concurrency tests to be added.
#
# Closes part of the OPEN BLOCK on iter-42 L3.5 PROMOTION per
# audit_findings.log lines 1655-2235 + docs/PHASE_A5_DEFERRAL.md.
# Fifth L3.5 sub-mixin backfill (live_trace_plot_layouts), 5 of 8.
# ─────────────────────────────────────────────────────────────────────────────

import hashlib  # noqa: E402

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import live_trace.plot_layouts as ltp_layouts  # noqa: E402


class TestPropertyPlotCurvesPopulation:
    """§1.1 universal floor: ≥2 property tests for plot-curve population.

    All 4 layout methods produce `self._plot_curves` keyed by int(rid).
    Invariants that must hold across any (ids list, roi_count):
    - Exactly len(unique ids) keys after setup
    - All keys are int, regardless of input ROI ID dtype
    """

    @given(
        ids=st.lists(
            st.integers(min_value=0, max_value=10_000),
            min_size=1, max_size=30, unique=True,
        ),
    )
    @settings(max_examples=30, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_single_layout_curve_count_matches_unique_ids(self, ids):
        """For any unique-ids list, _setup_single_plot_layout produces
        exactly len(ids) entries in _plot_curves. Pins the per-ROI
        1:1 curve-creation contract; a regression that dropped or
        duplicated curves would fail this for many seeds."""
        host = _Host(ids=ids)
        pw = _make_plot_widget_mock()
        host._setup_single_plot_layout(pw, roi_count=len(ids))
        assert len(host._plot_curves) == len(ids)
        # All keys are int (cast at insertion)
        for k in host._plot_curves.keys():
            assert isinstance(k, int)

    @given(
        ids=st.lists(
            st.integers(min_value=0, max_value=10_000),
            min_size=1, max_size=30, unique=True,
        ),
    )
    @settings(max_examples=30, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_optimized_layout_no_legend_attribute_set(self, ids):
        """`_setup_optimized_single_plot` MUST NOT set `_legend` (the
        no-legend fallback contract). Pins that a regression that
        added an addLegend call to the optimized path would break
        the "optimized=no-legend" contract that the dispatcher
        (_setup_multi_plot_layout) relies on."""
        host = _Host(ids=ids)
        host._legend = "sentinel"  # if optimized writes, this changes
        pw = _make_plot_widget_mock()
        host._setup_optimized_single_plot(pw, roi_count=len(ids))
        # _legend not touched; addLegend not called
        assert host._legend == "sentinel"
        pw.addLegend.assert_not_called()
        # Still populated curves
        assert len(host._plot_curves) == len(ids)


class TestSnapshotPlotConfig:
    """§1.1 L3.5 row: snapshot required for trace outputs.

    The plot-widget configuration set (background, downsampling,
    clip-to-view, grid alpha, mouse-enable, axis labels) is a
    UI-visible contract that operators rely on. Pin the canonical
    call set + downsampling ladder thresholds.
    """

    def test_plot_widget_config_call_set_snapshot(self):
        """Pin the sha256 of the canonical plot-widget config call set
        applied by `_setup_single_plot_layout`. Any change to a
        constant (e.g. grid alpha from 0.25 to 0.5, background from
        'k' to 'w') would silently shift the visual contract.

        Note: setLabel is called twice (left + bottom); the snapshot
        captures both call-arg tuples in deterministic order."""
        host = _Host(ids=[1])
        pw = _make_plot_widget_mock()
        host._setup_single_plot_layout(pw, roi_count=1)

        # Collect the deterministic config-call surface
        payload = b"|".join([
            b"setBackground:" + repr(pw.setBackground.call_args).encode(),
            b"setDownsampling:" + repr(pw.setDownsampling.call_args).encode(),
            b"setClipToView:" + repr(pw.setClipToView.call_args).encode(),
            b"showGrid:" + repr(pw.showGrid.call_args).encode(),
            b"setMouseEnabled:" + repr(pw.setMouseEnabled.call_args).encode(),
            b"addLegend:" + repr(pw.addLegend.call_args).encode(),
            b"setLabel_left:" + repr(
                [c for c in pw.setLabel.call_args_list if c.args and c.args[0] == 'left']
            ).encode(),
            b"setLabel_bottom:" + repr(
                [c for c in pw.setLabel.call_args_list if c.args and c.args[0] == 'bottom']
            ).encode(),
        ])
        h = hashlib.sha256(payload).hexdigest()

        expected_payload = b"|".join([
            b"setBackground:call('k')",
            b"setDownsampling:call(auto=True, mode='peak')",
            b"setClipToView:call(True)",
            b"showGrid:call(x=True, y=True, alpha=0.25)",
            b"setMouseEnabled:call(x=True, y=True)",
            b"addLegend:call(offset=(10, 10))",
            b"setLabel_left:[call('left', 'Intensity', units='AU')]",
            b"setLabel_bottom:[call('bottom', 'Time Points', units='frames')]",
        ])
        expected = hashlib.sha256(expected_payload).hexdigest()
        assert h == expected, (
            f"plot-widget config call set regression. Got {h}, expected "
            f"{expected}. A configuration constant (background, grid "
            f"alpha, axis label) has silently changed."
        )

    def test_downsampling_ladder_threshold_snapshot(self):
        """Pin the downsampling threshold ladder used by the multi-plot
        and optimized paths:
        - external_legend path: roi_count > 30 → curve.setDownsampling(
          factor=2, auto=True, method='peak')
        - optimized path: roi_count > 25 → curve.setDownsampling(
          factor=3, auto=True, method='peak')

        These thresholds are runtime perf-vs-fidelity tradeoffs; a
        silent shift (e.g. 30→50) would change which trial counts
        get downsampled. Pin via a probe across the boundary."""
        # Optimized-path: probe roi_count=25 (no downsample) vs 26 (downsample)
        downsample_calls_at_25 = []
        downsample_calls_at_26 = []

        def _capture_curve(downsample_list):
            def _plot(*args, **kwargs):
                curve = MagicMock()

                def _track(*a, **k):
                    downsample_list.append((a, k))

                curve.setDownsampling = MagicMock(side_effect=_track)
                return curve

            return _plot

        host_25 = _Host(ids=list(range(25)))
        pw_25 = _make_plot_widget_mock()
        pw_25.plot.side_effect = _capture_curve(downsample_calls_at_25)
        host_25._setup_optimized_single_plot(pw_25, roi_count=25)

        host_26 = _Host(ids=list(range(26)))
        pw_26 = _make_plot_widget_mock()
        pw_26.plot.side_effect = _capture_curve(downsample_calls_at_26)
        host_26._setup_optimized_single_plot(pw_26, roi_count=26)

        # At 25 ROIs: NO curve.setDownsampling calls
        # At 26 ROIs: 26 curve.setDownsampling calls with factor=3
        payload = b"|".join([
            b"at_25:" + repr(downsample_calls_at_25).encode(),
            b"at_26_count:" + str(len(downsample_calls_at_26)).encode(),
            b"at_26_first_call:" + (
                repr(downsample_calls_at_26[0]).encode()
                if downsample_calls_at_26 else b"NONE"
            ),
        ])
        h = hashlib.sha256(payload).hexdigest()

        expected_payload = b"|".join([
            b"at_25:[]",
            b"at_26_count:26",
            b"at_26_first_call:((), {'factor': 3, 'auto': True, 'method': 'peak'})",
        ])
        expected = hashlib.sha256(expected_payload).hexdigest()
        assert h == expected, (
            f"downsampling ladder regression. Got {h}, expected {expected}. "
            f"The roi_count > 25 → factor=3 threshold has shifted, or the "
            f"setDownsampling args changed."
        )


class TestStructuralNoThreadAffordance:
    """§1.1 L3.5 row: concurrency cell justification.

    The plot-layouts mixin does NOT touch threads (Qt main-thread only,
    pyqtgraph widget construction). Per §1.1 "Concurrency ≥1 if mixin
    touches threads" — N/A for this mixin. We pin the no-thread-
    affordance contract structurally: any future refactor that
    introduced threading primitives into this module MUST also add
    §1.1 concurrency tests, and this guard fails first to remind.
    """

    def test_module_does_not_import_threading_primitives(self):
        """No threading / Lock / RLock / Semaphore / QThread / Future
        references in the module source. If a refactor introduces any,
        this test fails — forcing the developer to ALSO add §1.1
        concurrency tests (per the L3.5 row matrix)."""
        import inspect
        src = inspect.getsource(ltp_layouts)
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
            f"live_trace_plot_layouts introduced threading primitives: "
            f"{offenders}. Per §1.1 L3.5 row, this mixin must now also "
            f"have ≥1 concurrency tests added before this guard can be "
            f"updated.1 + §1.2 playbook."
        )
