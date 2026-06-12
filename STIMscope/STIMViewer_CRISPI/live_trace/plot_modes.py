"""Plot-mode dispatcher + base rendering helpers extracted from
``live_trace_extractor``.

Stage-0.6 of the 6-module decomposition (sub-module 6 of 6, FIRST of
3 sub-mixins covering the plot-modes surface). Extracted
from ``live_trace_extractor.py`` (iter 37).

Per the iter-36 carry-forward, the full plot-modes surface (~1100 LOC
across 22 methods) is being sub-split into three mixins to honor the
user's ≤500 LOC granularity verdict (§0.5):

- **iter 37 (this file)**: `live_trace_plot_modes.py` (~100 LOC) —
  dispatcher + pygame renderer + pyqtgraph entry + skip-factor +
  unified ROI color
- **iter 39 (planned)**: `live_trace_plot_aggregation.py` (~459 LOC)
  — expanded/statistical/density-heatmap modes
- **iter 41 (planned)**: `live_trace_plot_pagination.py` (~638 LOC,
  may sub-split) — paged-trace mode + page navigation + pagination
  controls + page-label-safe (two definitions, BUG: D-ltm-1 to flag)

The 5 helpers in THIS mixin:
- ``_update_plot()`` — top-level @pyqtSlot() dispatcher: pygame vs
  pyqtgraph based on `use_pygame_plot` + `plot_widget` presence
- ``_update_pygame_plot()`` — pygame surface renderer; y-range
  auto-scaled with 5 % padding; cycling 8-color palette
- ``_update_pyqtgraph_plot()`` — pyqtgraph entry: skip-factor gate
  + dispatch to `_update_paged_trace_mode` (still on parent class
  until iter 41)
- ``_calculate_skip_factor(roi_count)`` — pure 4-step ladder
- ``_get_unified_roi_color(roi_id)`` — pure 30-color ROI palette

The mixin expects the subclass (LiveTraceExtractor) to provide:
- ``self.use_pygame_plot`` (bool)
- ``self.plot_widget`` (pyqtgraph widget or None)
- ``self.buffers`` (Dict[int, deque[float]])
- ``self.screen`` (pygame surface) + ``self.screen_width``/``screen_height``
  (only required if `use_pygame_plot` is True)
- ``self._frame_count`` (counter, written by parent's frame loop)
- ``self._update_paged_trace_mode()`` (method, still on parent class)

No behavior change vs the original location. Pygame import goes
through the same warnings-suppressed dance to avoid the pkg_resources
DeprecationWarning during pygame's namespace setup.

Safety: smoke + sibling chars tests in ``tests/L3_5_split_first/``
must remain green.
"""

from __future__ import annotations

import warnings

import numpy as np

from PyQt5.QtCore import pyqtSlot

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "pkg_resources is deprecated", DeprecationWarning)
    import pygame


class LiveTracePlotModesMixin:
    """Dispatcher + base plot renderers for ``LiveTraceExtractor``."""

    @pyqtSlot()
    def _update_plot(self):
        try:
            if self.use_pygame_plot:
                self._update_pygame_plot()
            elif self.plot_widget is not None:
                self._update_pyqtgraph_plot()
        except Exception as e:
            print(f"Plot update error: {e}")

    def _update_pygame_plot(self):
        try:
            any_data = any(len(buf) > 1 for buf in self.buffers.values())
            if not any_data:
                return


            y_min = min(min(buf) for buf in self.buffers.values() if len(buf) > 0)
            y_max = max(max(buf) for buf in self.buffers.values() if len(buf) > 0)
            if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                y_min, y_max = 0.0, 1.0

            yr = y_max - y_min
            y_min -= 0.05 * yr
            y_max += 0.05 * yr

            self.screen.fill((0, 0, 0))
            margin = 50
            w = self.screen_width
            h = self.screen_height
            plot_w = w - 2 * margin
            plot_h = h - 2 * margin

            axis_color = (160, 160, 160)
            pygame.draw.rect(self.screen, axis_color, (margin-1, margin-1, plot_w+2, plot_h+2), 1)


            def to_xy(j, val, npoints):
                x = margin + int(j * (plot_w / max(1, npoints-1)))

                t = (val - y_min) / max(1e-6, (y_max - y_min))
                y = margin + (plot_h - int(t * plot_h))
                return x, y

            colors = [(255, 64, 64), (64, 255, 64), (64, 64, 255),
                    (255, 255, 64), (255, 64, 255), (64, 255, 255),
                    (200, 200, 200), (255, 128, 0)]

            for i, (rid, buf) in enumerate(self.buffers.items()):
                n = len(buf)
                if n < 2:
                    continue
                color = colors[i % len(colors)]

                pts = [to_xy(j, buf[j], n) for j in range(n)]
                pygame.draw.lines(self.screen, color, False, pts, 1)

            pygame.display.flip()
        except Exception as e:
            print(f"Error in pygame plotting: {e}")

    def _update_pyqtgraph_plot(self):

        if self.plot_widget is None:
            return
        try:
            roi_count = len(self.buffers)


            skip_factor = self._calculate_skip_factor(roi_count)
            if skip_factor > 1 and self._frame_count % skip_factor != 0:
                return

            self._update_paged_trace_mode()

        except Exception as e:
            print(f"❌ PyQtGraph plot update error: {e}")

    def _calculate_skip_factor(self, roi_count):

        if roi_count <= 10:
            return 1
        elif roi_count <= 25:
            return 2
        elif roi_count <= 50:
            return 3
        else:
            return 5

    def _get_unified_roi_color(self, roi_id):


        # 30-color palette indexed by (roi_id - 1) % 30. Each color
        # MUST be unique — D-ltm-2 (fix iter 43): the last
        # entry was previously '#6C5CE7' duplicating index 16.
        # Replaced with '#1ABC9C' (mid-teal) so the palette has 30
        # distinct colors.
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#FFA07A', '#87CEEB', '#DEB887',
            '#FF9F43', '#10AC84', '#EE5A24', '#0084FF', '#341F97',
            '#F8B500', '#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E',
            '#E17055', '#00B894', '#00CECE', '#2D3436', '#636E72',
            '#FAB1A0', '#74B9FF', '#55A3FF', '#FF7675', '#1ABC9C',
        ]


        color_index = (roi_id - 1) % len(colors)
        return colors[color_index]


__all__ = ["LiveTracePlotModesMixin"]
