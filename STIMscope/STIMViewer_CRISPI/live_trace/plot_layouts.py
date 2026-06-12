"""Plot-layout builders extracted from ``live_trace_extractor``.

Stage-0.6 of the 6-module decomposition (sub-module 3 of 6).
Extracted from ``live_trace_extractor.py``.

Contains the 4 plot-layout setup methods as a mixin class:
- ``_setup_single_plot_layout`` — single legend on plot widget
- ``_setup_multi_plot_layout`` — dispatch wrapper to one of the two below
- ``_setup_plot_with_external_legend`` — sidecar legend in parent layout
- ``_setup_optimized_single_plot`` — no-legend fallback for high ROI counts

The mixin expects the subclass (LiveTraceExtractor) to provide:
- ``self.ids`` — list[int] of ROI IDs
- ``self._plot_curves`` — dict[int, plot curve] populated here
- ``self._legend`` — set in _setup_single_plot_layout
- ``self.plot_widget`` — assigned in all 4 methods
- ``self._get_unified_roi_color(rid)`` — method returning a color

No behavior change vs the original location.

Safety: 29 smoke tests in ``tests/L3_5_split_first/`` must remain green.
"""

from __future__ import annotations

import pyqtgraph as pg


class LiveTracePlotLayoutsMixin:
    """Plot-layout builders for ``LiveTraceExtractor``.

    Methods set up the pyqtgraph plot widget with one of four legend
    layouts depending on ROI count and parent-widget availability.
    """

    def _setup_single_plot_layout(self, plot_widget, roi_count):

        try:
            self.plot_widget = plot_widget
            self.plot_widget.setBackground('k')
            self.plot_widget.setDownsampling(auto=True, mode='peak')
            self.plot_widget.setClipToView(True)
            self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
            self.plot_widget.setMouseEnabled(x=True, y=True)


            self.plot_widget.setLabel('left', 'Intensity', units='AU')
            self.plot_widget.setLabel('bottom', 'Time Points', units='frames')


            self._legend = self.plot_widget.addLegend(offset=(10, 10))


            for idx, rid in enumerate(self.ids):

                unified_color = self._get_unified_roi_color(int(rid))
                pen = pg.mkPen(unified_color, width=2)

                curve = self.plot_widget.plot(pen=pen)
                self._plot_curves[int(rid)] = curve

            print(f"✅ Single plot layout complete for {roi_count} ROIs")

        except Exception as e:
            print(f"❌ Single plot setup failed: {e}")

    def _setup_multi_plot_layout(self, plot_widget, roi_count):

        try:

            parent_widget = plot_widget.parent() if plot_widget.parent() else plot_widget


            if hasattr(parent_widget, 'layout') or hasattr(parent_widget, 'setLayout'):
                self._setup_plot_with_external_legend(plot_widget, parent_widget, roi_count)
            else:

                self._setup_optimized_single_plot(plot_widget, roi_count)

        except Exception as e:
            print(f"❌ Multi-plot setup failed: {e}")

            self._setup_optimized_single_plot(plot_widget, roi_count)

    def _setup_plot_with_external_legend(self, plot_widget, parent_widget, roi_count):

        try:
            from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QLabel, QScrollArea


            main_layout = QHBoxLayout()


            self.plot_widget = plot_widget
            self.plot_widget.setBackground('k')
            self.plot_widget.setDownsampling(auto=True, mode='peak')
            self.plot_widget.setClipToView(True)
            self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
            self.plot_widget.setMouseEnabled(x=True, y=True)


            self.plot_widget.setLabel('left', 'Intensity', units='AU')
            self.plot_widget.setLabel('bottom', 'Time Points', units='frames')


            legend_widget = QWidget()
            legend_widget.setMaximumWidth(200)
            legend_widget.setMinimumWidth(150)
            legend_layout = QVBoxLayout(legend_widget)


            header_label = QLabel(f"ROI Legend ({roi_count} ROIs)")
            header_label.setStyleSheet("font-weight: bold; color: white; background: #333; padding: 5px;")
            legend_layout.addWidget(header_label)


            scroll_area = QScrollArea()
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)


            for idx, rid in enumerate(self.ids):

                unified_color = self._get_unified_roi_color(int(rid))
                pen = pg.mkPen(unified_color, width=1)


                curve = self.plot_widget.plot(pen=pen)


                if roi_count > 30:
                    curve.setDownsampling(factor=2, auto=True, method='peak')

                self._plot_curves[int(rid)] = curve


                color_hex = unified_color
                legend_entry = QLabel(f"<span style='color: {color_hex}'>●</span> ROI {int(rid)}")
                legend_entry.setStyleSheet("color: white; padding: 2px; font-size: 10px;")
                scroll_layout.addWidget(legend_entry)

            scroll_area.setWidget(scroll_content)
            scroll_area.setWidgetResizable(True)
            legend_layout.addWidget(scroll_area)


            if hasattr(parent_widget, 'layout') and parent_widget.layout():

                parent_layout = parent_widget.layout()
                main_layout.addWidget(self.plot_widget, stretch=3)
                main_layout.addWidget(legend_widget, stretch=1)
                parent_layout.addLayout(main_layout)
            else:
                print("⚠️ Could not create external legend, using optimized single plot")
                self._setup_optimized_single_plot(plot_widget, roi_count)
                return

            print(f"✅ Multi-plot layout with external legend complete for {roi_count} ROIs")

        except Exception as e:
            print(f"❌ External legend setup failed: {e}")
            self._setup_optimized_single_plot(plot_widget, roi_count)

    def _setup_optimized_single_plot(self, plot_widget, roi_count):

        try:
            self.plot_widget = plot_widget
            self.plot_widget.setBackground('k')
            self.plot_widget.setDownsampling(auto=True, mode='peak')
            self.plot_widget.setClipToView(True)
            self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
            self.plot_widget.setMouseEnabled(x=True, y=True)


            self.plot_widget.setLabel('left', 'Intensity', units='AU')
            self.plot_widget.setLabel('bottom', 'Time Points', units='frames')


            print(f"📊 {roi_count} ROIs - using optimized mode without legend")


            for idx, rid in enumerate(self.ids):
                hue_count = min(15, max(8, roi_count))
                color = pg.intColor(idx, hues=hue_count)
                pen = pg.mkPen(color, width=1)

                curve = self.plot_widget.plot(pen=pen)


                if roi_count > 25:
                    curve.setDownsampling(factor=3, auto=True, method='peak')

                self._plot_curves[int(rid)] = curve

            print(f"✅ Optimized single plot complete for {roi_count} ROIs")

        except Exception as e:
            print(f"❌ Optimized plot setup failed: {e}")


__all__ = ["LiveTracePlotLayoutsMixin"]
