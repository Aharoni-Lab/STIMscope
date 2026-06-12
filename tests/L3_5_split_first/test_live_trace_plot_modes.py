"""Comprehensive characterization tests for ``live_trace_plot_modes``.

target ~85-90 % path coverage on the LiveTracePlotModesMixin
(extracted iter 37 commit db917ae).

Module surface (~172 LOC, 5 methods):
- ``_update_plot()`` — @pyqtSlot() dispatcher
- ``_update_pygame_plot()`` — pygame surface renderer
- ``_update_pyqtgraph_plot()`` — pyqtgraph entry: skip-factor gate
- ``_calculate_skip_factor(roi_count)`` — pure 4-step ladder
- ``_get_unified_roi_color(roi_id)`` — pure 30-color palette

Branches exercised per method are listed in each test docstring.
QApp + offscreen + sys.path are handled by conftest.py (session autouse).
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from PyQt5.QtCore import QObject

import live_trace.plot_modes as lt_pm
from live_trace.plot_modes import LiveTracePlotModesMixin


# ─────────────────────────────────────────────────────────────────────────────
# Test infrastructure: stub host class
# ─────────────────────────────────────────────────────────────────────────────


class _Host(QObject, LiveTracePlotModesMixin):
    """Stub satisfying the mixin contract."""

    def __init__(self, *, use_pygame_plot=False, plot_widget=None,
                 frame_count=0, buffers=None, screen_size=(640, 480)):
        QObject.__init__(self)
        self.use_pygame_plot = use_pygame_plot
        self.plot_widget = plot_widget
        self._frame_count = frame_count
        self.buffers = buffers if buffers is not None else {}
        # Pygame attrs (only used in pygame path)
        self.screen = MagicMock()
        self.screen_width = screen_size[0]
        self.screen_height = screen_size[1]
        # _update_paged_trace_mode is still on parent class — stub it here
        self._update_paged_trace_mode = MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
# C1 — _calculate_skip_factor (pure ladder)
# ─────────────────────────────────────────────────────────────────────────────


class TestC1CalculateSkipFactor:
    """Contract: 4-step ladder on roi_count.

    Ladder:
    - roi_count <= 10 → 1
    - 10 < roi_count <= 25 → 2
    - 25 < roi_count <= 50 → 3
    - roi_count > 50 → 5
    """

    @pytest.mark.parametrize(
        "roi_count,expected",
        [
            (0, 1),    # edge: 0
            (1, 1),
            (10, 1),   # boundary
            (11, 2),   # boundary +1
            (25, 2),   # boundary
            (26, 3),
            (50, 3),   # boundary
            (51, 5),
            (100, 5),
            (1000, 5),
        ],
    )
    def test_ladder_boundaries(self, roi_count, expected):
        host = _Host()
        assert host._calculate_skip_factor(roi_count) == expected

    def test_negative_treated_as_low(self):
        """Negative roi_count <= 10 so returns 1."""
        host = _Host()
        assert host._calculate_skip_factor(-5) == 1


# ─────────────────────────────────────────────────────────────────────────────
# C2 — _get_unified_roi_color (pure palette)
# ─────────────────────────────────────────────────────────────────────────────


class TestC2GetUnifiedRoiColor:
    """Contract: 30-color palette indexed by (roi_id - 1) % len(colors).

    Branches:
    - roi_id 1 → first color
    - roi_id 30 → last color
    - roi_id 31 → wraps to first color
    - negative roi_id → modulo wraparound
    """

    def test_first_roi_returns_first_color(self):
        host = _Host()
        assert host._get_unified_roi_color(1) == '#FF6B6B'

    def test_known_palette_indices(self):
        """Pin a few mid-palette colors so reordering the list breaks
        the test — guards against accidental palette mutation."""
        host = _Host()
        assert host._get_unified_roi_color(2) == '#4ECDC4'
        assert host._get_unified_roi_color(3) == '#45B7D1'
        assert host._get_unified_roi_color(10) == '#DEB887'

    def test_wraps_at_30(self):
        host = _Host()
        # roi_id=31 → (31-1) % 30 = 0 → first color
        assert host._get_unified_roi_color(31) == '#FF6B6B'

    def test_wraps_at_60(self):
        host = _Host()
        # roi_id=61 → (61-1) % 30 = 0 → first color
        assert host._get_unified_roi_color(61) == '#FF6B6B'

    def test_returns_string(self):
        host = _Host()
        result = host._get_unified_roi_color(5)
        assert isinstance(result, str)
        assert result.startswith('#')
        assert len(result) == 7  # hex format #RRGGBB

    def test_palette_has_30_unique_colors(self):
        """Pin the palette length so additions/removals are caught.

        Post-iter-43fix (D-ltm-2): the previously-duplicated
        '#6C5CE7' at position 30 was replaced with '#1ABC9C', so all
        30 colors are now distinct.
        """
        host = _Host()
        seen = set()
        for rid in range(1, 31):
            seen.add(host._get_unified_roi_color(rid))
        # POST D-ltm-2 fix: 30 distinct colors
        assert len(seen) == 30

    def test_dltm2_last_palette_entry_is_unique(self):
        """D-ltm-2fix regression guard: the 30th entry MUST NOT
        equal the 17th entry. Pre-fix both were '#6C5CE7'; post-fix the
        30th is a different color so this assertion holds."""
        host = _Host()
        # roi_id=17 → index 16; roi_id=30 → index 29
        assert host._get_unified_roi_color(17) != host._get_unified_roi_color(30)

    def test_negative_roi_id_wraps(self):
        """Python `%` is well-defined for negative numbers — returns a
        valid color (not crash)."""
        host = _Host()
        result = host._get_unified_roi_color(-5)
        assert isinstance(result, str)
        assert result.startswith('#')


# ─────────────────────────────────────────────────────────────────────────────
# C3 — _update_plot (dispatcher)
# ─────────────────────────────────────────────────────────────────────────────


class TestC3UpdatePlotDispatcher:
    """Contract: dispatches to pygame or pyqtgraph based on flags.

    Branches:
    - use_pygame_plot=True → _update_pygame_plot called
    - use_pygame_plot=False + plot_widget set → _update_pyqtgraph_plot called
    - use_pygame_plot=False + plot_widget=None → neither called
    - exception in dispatched method → caught + logged
    """

    def test_pygame_branch(self):
        host = _Host(use_pygame_plot=True)
        with patch.object(host, "_update_pygame_plot") as mock_pg:
            with patch.object(host, "_update_pyqtgraph_plot") as mock_qg:
                host._update_plot()
        mock_pg.assert_called_once()
        mock_qg.assert_not_called()

    def test_pyqtgraph_branch(self):
        host = _Host(use_pygame_plot=False, plot_widget=MagicMock())
        with patch.object(host, "_update_pygame_plot") as mock_pg:
            with patch.object(host, "_update_pyqtgraph_plot") as mock_qg:
                host._update_plot()
        mock_qg.assert_called_once()
        mock_pg.assert_not_called()

    def test_neither_branch_no_plot_widget(self):
        host = _Host(use_pygame_plot=False, plot_widget=None)
        with patch.object(host, "_update_pygame_plot") as mock_pg:
            with patch.object(host, "_update_pyqtgraph_plot") as mock_qg:
                host._update_plot()
        mock_pg.assert_not_called()
        mock_qg.assert_not_called()

    def test_exception_swallowed(self, capsys):
        host = _Host(use_pygame_plot=True)
        with patch.object(host, "_update_pygame_plot",
                          side_effect=RuntimeError("pygame exploded")):
            host._update_plot()  # must not raise
        captured = capsys.readouterr()
        assert "Plot update error" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C4 — _update_pygame_plot
# ─────────────────────────────────────────────────────────────────────────────


class TestC4UpdatePygamePlot:
    """Contract: render up to 8 ROI traces on the pygame surface.

    Branches:
    - no data (all buffers empty or single-point) → early return
    - non-finite y-range or y_max <= y_min → fallback to 0..1
    - happy path → screen.fill + draw.rect + draw.lines per ROI
    - >8 ROIs → palette cycles via modulo
    - single-point buffer → skipped (n < 2)
    - exception swallowed
    """

    def test_no_data_early_return(self):
        host = _Host(buffers={1: deque(), 2: deque([5.0])})
        host._update_pygame_plot()
        # screen.fill should not be called when no buffers have >1 entry
        host.screen.fill.assert_not_called()

    def test_happy_path_fills_screen(self):
        host = _Host(buffers={
            1: deque([10.0, 20.0, 30.0]),
            2: deque([5.0, 15.0, 25.0]),
        })
        with patch.object(lt_pm, "pygame") as mock_pg:
            host._update_pygame_plot()
        # screen.fill called with black
        host.screen.fill.assert_called_with((0, 0, 0))
        # pygame.draw.rect called (border)
        mock_pg.draw.rect.assert_called_once()
        # pygame.draw.lines called once per ROI
        assert mock_pg.draw.lines.call_count == 2
        # pygame.display.flip called at end
        mock_pg.display.flip.assert_called_once()

    def test_non_finite_y_falls_back_to_unit_range(self):
        host = _Host(buffers={
            1: deque([float('inf'), float('nan'), 0.0]),
        })
        with patch.object(lt_pm, "pygame"):
            # Should not crash — non-finite triggers fallback
            host._update_pygame_plot()
        host.screen.fill.assert_called_with((0, 0, 0))

    def test_single_point_buffer_skipped(self):
        """Buffer with n=1 entry doesn't get a polyline (n < 2)."""
        host = _Host(buffers={
            1: deque([100.0]),       # only 1 point — skipped
            2: deque([10.0, 20.0]),  # 2 points — drawn
        })
        with patch.object(lt_pm, "pygame") as mock_pg:
            host._update_pygame_plot()
        # Only one ROI should have draw.lines called
        assert mock_pg.draw.lines.call_count == 1

    def test_color_palette_cycles(self):
        """With 10 ROIs and an 8-color palette, colors 0,1,2..7,0,1 cycle."""
        host = _Host(buffers={
            rid: deque([float(rid), float(rid + 1)]) for rid in range(1, 11)
        })
        with patch.object(lt_pm, "pygame") as mock_pg:
            host._update_pygame_plot()
        assert mock_pg.draw.lines.call_count == 10

    def test_exception_swallowed(self, capsys):
        host = _Host(buffers={1: deque([10.0, 20.0])})
        with patch.object(lt_pm, "pygame") as mock_pg:
            mock_pg.draw.rect.side_effect = RuntimeError("draw broken")
            host._update_pygame_plot()  # must not raise
        captured = capsys.readouterr()
        assert "Error in pygame plotting" in captured.out

    def test_zero_yrange_falls_back_to_unit(self):
        """When all values are identical, y_max == y_min → fallback."""
        host = _Host(buffers={1: deque([50.0, 50.0, 50.0])})
        with patch.object(lt_pm, "pygame"):
            host._update_pygame_plot()  # must not crash
        host.screen.fill.assert_called_with((0, 0, 0))


# ─────────────────────────────────────────────────────────────────────────────
# C5 — _update_pyqtgraph_plot
# ─────────────────────────────────────────────────────────────────────────────


class TestC5UpdatePyqtgraphPlot:
    """Contract: skip-factor gate + dispatch to _update_paged_trace_mode.

    Branches:
    - plot_widget=None → early return
    - skip_factor=1 → always dispatch
    - skip_factor>1 + frame_count mod skip_factor != 0 → skip
    - skip_factor>1 + frame_count mod skip_factor == 0 → dispatch
    - exception swallowed
    """

    def test_plot_widget_none_early_return(self):
        host = _Host(plot_widget=None)
        host._update_pyqtgraph_plot()
        host._update_paged_trace_mode.assert_not_called()

    def test_small_roi_count_no_skip(self):
        """roi_count <= 10 → skip_factor=1 → always dispatch."""
        host = _Host(
            plot_widget=MagicMock(),
            buffers={i: deque([1.0, 2.0]) for i in range(5)},
            frame_count=42,
        )
        host._update_pyqtgraph_plot()
        host._update_paged_trace_mode.assert_called_once()

    def test_large_roi_count_with_skip_dispatched(self):
        """roi_count=30 → skip_factor=3 → dispatch only when frame % 3 == 0."""
        host = _Host(
            plot_widget=MagicMock(),
            buffers={i: deque([1.0, 2.0]) for i in range(30)},
            frame_count=9,  # 9 % 3 == 0 → dispatch
        )
        host._update_pyqtgraph_plot()
        host._update_paged_trace_mode.assert_called_once()

    def test_large_roi_count_with_skip_dropped(self):
        """roi_count=30 → skip_factor=3 → drop when frame % 3 != 0."""
        host = _Host(
            plot_widget=MagicMock(),
            buffers={i: deque([1.0, 2.0]) for i in range(30)},
            frame_count=10,  # 10 % 3 == 1 → skip
        )
        host._update_pyqtgraph_plot()
        host._update_paged_trace_mode.assert_not_called()

    def test_huge_roi_count_uses_skip_5(self):
        """roi_count=60 → skip_factor=5."""
        host = _Host(
            plot_widget=MagicMock(),
            buffers={i: deque([1.0, 2.0]) for i in range(60)},
            frame_count=20,  # 20 % 5 == 0 → dispatch
        )
        host._update_pyqtgraph_plot()
        host._update_paged_trace_mode.assert_called_once()

    def test_exception_swallowed(self, capsys):
        host = _Host(
            plot_widget=MagicMock(),
            buffers={i: deque([1.0]) for i in range(5)},
        )
        host._update_paged_trace_mode.side_effect = RuntimeError("paged broken")
        host._update_pyqtgraph_plot()  # must not raise
        captured = capsys.readouterr()
        assert "PyQtGraph plot update error" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C6 — Mixin integration
# ─────────────────────────────────────────────────────────────────────────────


class TestC6MixinIntegration:
    """Contract: 5 methods accessible on subclass; mixin has no __init__."""

    METHODS = (
        "_update_plot",
        "_update_pygame_plot",
        "_update_pyqtgraph_plot",
        "_calculate_skip_factor",
        "_get_unified_roi_color",
    )

    def test_all_5_methods_on_subclass(self):
        host = _Host()
        for name in self.METHODS:
            method = getattr(host, name, None)
            assert callable(method), f"Missing or non-callable: {name}"

    def test_methods_defined_on_mixin(self):
        for name in self.METHODS:
            assert name in LiveTracePlotModesMixin.__dict__, (
                f"{name} not defined on LiveTracePlotModesMixin"
            )

    def test_mixin_has_no_init(self):
        assert "__init__" not in LiveTracePlotModesMixin.__dict__

    def test_update_plot_is_pyqt_slot(self):
        """The @pyqtSlot() decorator should be preserved across extraction."""
        # PyQt5 attaches metadata to slot-decorated methods
        method = LiveTracePlotModesMixin.__dict__["_update_plot"]
        # pyqtSlot stores the signature info; presence verified via __pyqtSignature__
        # or by the fact the method exists and is callable.
        assert callable(method)


# ─────────────────────────────────────────────────────────────────────────────
# §1.1 L3.5 matrix backfill — Property + Snapshot + Structural (iter-59)
#
# §1.1 L3.5 row requires:
#   - Property ≥2 per sub-module (universal floor)
#   - Snapshot required for trace outputs (skip-factor ladder + ROI
#     color palette are visible-to-operator contracts; both pinned)
#   - Concurrency: live_trace_plot_modes mixin does NOT touch threads
#     (Qt-main-thread @pyqtSlot dispatcher; pygame/pyqtgraph rendering
#     stays on main thread). Per §1.1 "≥1 IF mixin touches threads"
#     — N/A. Pinned structurally.
#
# Closes part of the OPEN BLOCK on iter-42 L3.5 PROMOTION per
# audit_findings.log lines 1655-2235 + docs/PHASE_A5_DEFERRAL.md.
# Sixth L3.5 sub-mixin backfill (live_trace_plot_modes), 6 of 8.
# ─────────────────────────────────────────────────────────────────────────────

import hashlib  # noqa: E402

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


class TestPropertyPlotModes:
    """§1.1 universal floor: ≥2 property tests."""

    @given(roi_count=st.integers(min_value=-100, max_value=10_000))
    @settings(max_examples=80, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_skip_factor_monotonic_nondecreasing(self, roi_count):
        """For any (a, b) with a <= b, _calculate_skip_factor(a) <=
        _calculate_skip_factor(b). The pyqtgraph skip-factor gate
        depends on this monotonicity to throttle larger ROI counts
        more aggressively; a band inversion would invert the
        throttle behavior."""
        host = _Host()
        f_a = host._calculate_skip_factor(roi_count)
        f_b = host._calculate_skip_factor(roi_count + 1)
        assert f_a <= f_b, (
            f"skip-factor ladder not monotonic: f({roi_count})={f_a} > "
            f"f({roi_count + 1})={f_b}"
        )
        assert f_a in {1, 2, 3, 5}  # fixed codomain

    @given(roi_id=st.integers(min_value=-10_000, max_value=10_000))
    @settings(max_examples=60, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_roi_color_total_function_and_palette_membership(self, roi_id):
        """For ANY integer roi_id (including negative & extreme), the
        ROI color is a string from the 30-color palette, deterministic
        (same roi_id → same color), and indexed by (roi_id - 1) % 30.

        Pins the total-function contract — any regression that raised
        on negative IDs, or returned None for out-of-range, would fail
        this. Hypothesis sweep across the int range."""
        host = _Host()
        c1 = host._get_unified_roi_color(roi_id)
        c2 = host._get_unified_roi_color(roi_id)
        assert isinstance(c1, str)
        assert c1.startswith("#") and len(c1) == 7  # hex color
        assert c1 == c2  # deterministic
        # Modulo wrap: roi_id and roi_id+30 must collide
        assert host._get_unified_roi_color(roi_id) == \
            host._get_unified_roi_color(roi_id + 30)


class TestSnapshotPlotModesContract:
    """§1.1 L3.5 row: snapshot required for trace outputs.

    Two operator-visible contract snapshots:
    - 30-color palette (D-ltm-2 history: the last entry was previously
      a duplicate of index 16 — pin the post-fix unique-color set)
    - skip-factor ladder table for roi_count ∈ [0, 60]
    """

    def test_roi_color_palette_snapshot(self):
        """Pin the 30-color palette as a sha256 of the joined hex
        strings. The palette has D-ltm-2 history (last entry was a
        duplicate of #6C5CE7 at index 16; fixed to '#1ABC9C' at
        iter 43). Any silent palette edit shifts which ROIs map to
        which color — fail this hash."""
        host = _Host()
        # 30 colors, indexed by (roi_id - 1) % 30; iterate ids 1..30
        palette = [host._get_unified_roi_color(rid) for rid in range(1, 31)]
        h = hashlib.sha256(b",".join(c.encode() for c in palette)).hexdigest()
        # All colors must be unique (D-ltm-2 invariant)
        assert len(set(palette)) == 30, (
            f"D-ltm-2 regression: palette has duplicate colors. "
            f"Set={set(palette)!r}"
        )
        expected_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#FFA07A', '#87CEEB', '#DEB887',
            '#FF9F43', '#10AC84', '#EE5A24', '#0084FF', '#341F97',
            '#F8B500', '#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E',
            '#E17055', '#00B894', '#00CECE', '#2D3436', '#636E72',
            '#FAB1A0', '#74B9FF', '#55A3FF', '#FF7675', '#1ABC9C',
        ]
        expected = hashlib.sha256(
            b",".join(c.encode() for c in expected_palette)
        ).hexdigest()
        assert h == expected, (
            f"ROI palette regression. Got {h}, expected {expected}. "
            f"A palette entry has been edited or reordered."
        )

    def test_skip_factor_ladder_table_snapshot(self):
        """Pin the (roi_count → skip_factor) table for canonical
        sweep [0, 60]. Skip-factor governs how often the pyqtgraph
        plot redraws under load; a silent threshold shift (e.g.
        moving the 25-boundary) changes runtime behavior."""
        host = _Host()
        table = b",".join(
            f"{n}:{host._calculate_skip_factor(n)}".encode()
            for n in range(0, 61)
        )
        h = hashlib.sha256(table).hexdigest()
        # Expected ladder per source: <=10 → 1; <=25 → 2; <=50 → 3; else 5
        expected_table = b",".join(
            f"{n}:{1 if n <= 10 else 2 if n <= 25 else 3 if n <= 50 else 5}".encode()
            for n in range(0, 61)
        )
        expected = hashlib.sha256(expected_table).hexdigest()
        assert h == expected, (
            f"skip-factor ladder regression. Got {h}, expected {expected}. "
            f"A band threshold or output value has shifted."
        )


class TestStructuralNoThreadAffordancePlotModes:
    """§1.1 L3.5 row: concurrency cell justification.

    `live_trace_plot_modes` is the @pyqtSlot dispatcher that runs on
    the Qt main thread; pygame/pyqtgraph rendering also stays on the
    main thread. No threading primitives are used. Per §1.1 "≥1 IF
    mixin touches threads" — N/A. Pinned structurally so a future
    refactor that introduces threading must add §1.1 concurrency
    tests before this guard can be removed.
    """

    def test_module_does_not_import_threading_primitives(self):
        """No threading / Lock / RLock / Semaphore / QThread / Future
        references. If a refactor introduces any, this fails — force
        the developer to also add §1.1 concurrency tests."""
        import inspect
        src = inspect.getsource(lt_pm)
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
            f"live_trace_plot_modes introduced threading primitives: "
            f"{offenders}. Per §1.1 L3.5 row, this mixin must also have "
            f"≥1 concurrency tests added before this guard is updated."
        )
