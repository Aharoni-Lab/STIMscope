"""Aggregation plot modes extracted from ``live_trace_extractor``.

Stage-0.6 of the 6-module decomposition (sub-module 6/6, sub-mixin
2/3 of the plot-modes responsibility). Extracted from
``live_trace_extractor.py`` (iter 39).

Per the iter-36 / iter-37 sub-split plan:
- iter 37 ✅ ``live_trace_plot_modes.py`` (dispatcher + pygame +
  pyqtgraph entry + skip + ROI color)
- **iter 39 ✅ THIS FILE** ``live_trace_plot_aggregation.py`` —
  expanded / statistical / density-heatmap modes
- iter 41 ⏳ ``live_trace_plot_pagination.py`` — paged trace mode +
  page navigation

The 5 helpers in THIS mixin:
- ``_expand_all_rois()`` — open expanded-view QDialog with all-ROI
  pyqtgraph PlotWidget (large modal dialog, ~170 LOC)
- ``_update_expanded_plot()`` — incremental update for the expanded
  dialog (uses ``_resolve_trace_y`` from parent)
- ``_update_statistical_aggregation_mode()`` — population mean ± std
  + p25/p75 + 3 rotating per-ROI highlight curves
- ``_setup_statistical_plot()`` — build the pyqtgraph curves for the
  statistical mode
- ``_update_density_heatmap_mode()`` — pyqtgraph ImageItem heatmap +
  overall mean ± std summary curves
- ``_setup_density_plot()`` — build the ImageItem + summary curves

The mixin expects the subclass (LiveTraceExtractor) to provide:
- ``self.plot_widget`` (pyqtgraph PlotWidget or None)
- ``self.buffers`` (Dict[int, deque[float]])
- ``self._plot_curves`` (Dict, cleared by _setup_statistical_plot)
- ``self._global_frame_index`` (int counter)
- ``self._max_points_cfg`` (int from config)
- ``self._last_fps_est`` (float)
- ``self._highlight_ids`` (set[int])
- ``self._resolve_trace_y(roi_id)`` (method on parent class)
- ``self._get_unified_roi_color(roi_id)`` (now on PlotModesMixin via MRO)
- ``self._setup_pagination_controls()`` (still on parent class until
  iter 41 pagination extract)

No behavior change vs the original location.

Safety: smoke + sibling chars tests in ``tests/L3_5_split_first/``
must remain green.
"""

from __future__ import annotations

import numpy as np

try:
    import pyqtgraph as pg
    PYQTPGRAPH_AVAILABLE = True
except Exception:
    PYQTPGRAPH_AVAILABLE = False
    pg = None


class LiveTracePlotAggregationMixin:
    """Aggregation plot modes for ``LiveTraceExtractor``."""

    def _expand_all_rois(self):

        try:
            if not self.plot_widget:
                print("⚠️ No plot widget available for expansion")
                return


            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QWidget
            import pyqtgraph as pg

            self._expanded_dialog = QDialog()
            self._expanded_dialog.setWindowTitle(f"All ROIs - Live Trace View ({len(self.buffers)} ROIs)")
            self._expanded_dialog.resize(1400, 900)

            layout = QVBoxLayout(self._expanded_dialog)


            header_layout = QHBoxLayout()
            header_label = QLabel(f"📊 Displaying all {len(self.buffers)} ROIs in real-time")
            header_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")

            close_btn = QPushButton("✖ Close Expanded View")
            close_btn.setMaximumWidth(200)
            close_btn.clicked.connect(self._expanded_dialog.close)

            header_layout.addWidget(header_label)
            header_layout.addStretch()
            header_layout.addWidget(close_btn)
            layout.addLayout(header_layout)


            scroll_area = QScrollArea()
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)


            self._expanded_plot = pg.PlotWidget()
            self._expanded_plot.setMinimumHeight(800)
            self._expanded_plot.setLabel('left', 'Intensity')
            self._expanded_plot.setLabel('bottom', 'Time (frames)')
            self._expanded_plot.showGrid(x=True, y=True, alpha=0.3)
            self._expanded_plot.setTitle(f"All {len(self.buffers)} ROIs - Live Traces (Optimized View)")


            viewbox = self._expanded_plot.getViewBox()
            viewbox.setAspectLocked(False)

            import pyqtgraph as pg
            viewbox.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)


            self._expanded_curves = {}
            active_rois = sorted([rid for rid, buf in self.buffers.items() if len(buf) >= 2])


            if len(active_rois) > 10:

                all_traces = []
                for roi_id in active_rois:
                    buffer = list(self.buffers[roi_id])
                    if len(buffer) >= 2:
                        all_traces.append(np.array(buffer, dtype=np.float32))

                if all_traces:

                    global_min = min(np.min(trace) for trace in all_traces)
                    global_max = max(np.max(trace) for trace in all_traces)
                    trace_range = global_max - global_min if global_max > global_min else 1.0


                    spacing = trace_range * 0.3

                    for i, roi_id in enumerate(active_rois):
                        buffer = list(self.buffers[roi_id])
                        if len(buffer) >= 2:
                            unified_color = self._get_unified_roi_color(roi_id)
                            pen = pg.mkPen(color=unified_color, width=1.0, alpha=0.7)

                            x_data = np.arange(len(buffer), dtype=np.float32)
                            y_data = np.array(buffer, dtype=np.float32)


                            normalized_y = ((y_data - global_min) / trace_range) + (i * spacing)

                            curve = self._expanded_plot.plot(x_data, normalized_y, pen=pen)
                            self._expanded_curves[roi_id] = curve
            else:

                for roi_id in active_rois:
                    y_data = self._resolve_trace_y(roi_id)
                    if len(y_data) >= 2:
                        unified_color = self._get_unified_roi_color(roi_id)
                        base_width = 1.0
                        if roi_id in getattr(self, '_highlight_ids', set()):
                            base_width = 3.0
                        pen = pg.mkPen(color=unified_color, width=base_width, alpha=0.9 if base_width>1 else 0.8)

                        x_data = np.arange(len(y_data), dtype=np.float32)
                        curve = self._expanded_plot.plot(x_data, y_data, pen=pen)
                        self._expanded_curves[roi_id] = curve

            scroll_layout.addWidget(self._expanded_plot)


            legend_label = QLabel("ROI Legend (Colors match unified system):")
            legend_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
            scroll_layout.addWidget(legend_label)


            legend_layout = QHBoxLayout()
            legend_layout.setContentsMargins(10, 5, 10, 5)


            for i, roi_id in enumerate(active_rois):
                color = self._get_unified_roi_color(roi_id)
                legend_item = QLabel(f"● ROI {roi_id}")
                legend_item.setStyleSheet(f"color: {color}; font-weight: bold; margin: 2px; font-size: 10px;")
                legend_layout.addWidget(legend_item)

                if (i + 1) % 10 == 0:
                    scroll_layout.addLayout(legend_layout)
                    legend_layout = QHBoxLayout()
                    legend_layout.setContentsMargins(10, 5, 10, 5)

            if legend_layout.count() > 0:
                scroll_layout.addLayout(legend_layout)

            # Selected IDs legend
            # (D-lta-1fix iter 43: this block was previously
            # duplicated immediately below in another try/except; the
            # duplicate was removed since both blocks did identical work.)
            try:
                selected = sorted(list(getattr(self, '_highlight_ids', set())))
                if selected:
                    sel_label = QLabel(f"Selected (top-5): {selected}")
                    sel_label.setStyleSheet("font-weight: bold; color: #1c1c1e; margin: 5px; font-size: 12px;")
                    scroll_layout.addWidget(sel_label)
            except Exception:
                pass

            total_label = QLabel(f"Total: {len(active_rois)} ROIs displayed")
            total_label.setStyleSheet("font-weight: bold; color: #333; margin: 5px; font-size: 12px;")
            scroll_layout.addWidget(total_label)

            scroll_area.setWidget(scroll_widget)
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area)


            self._expanded_dialog.show()


            self._update_expanded_plot()

            print(f"✅ Expanded view opened with {len(active_rois)} ROIs")

        except Exception as e:
            print(f"❌ Error creating expanded view: {e}")
            import traceback
            traceback.print_exc()

    def _update_expanded_plot(self):

        try:
            if not hasattr(self, '_expanded_dialog') or not hasattr(self, '_expanded_curves'):
                return

            if not self._expanded_dialog.isVisible():
                return


            for roi_id, curve in self._expanded_curves.items():
                y_data = self._resolve_trace_y(roi_id)
                if len(y_data) >= 2:
                    try:
                        start_idx = max(0, self._global_frame_index - len(y_data))
                        if getattr(self, '_x_mode_seconds', False):
                            x_data = (np.arange(start_idx, start_idx + len(y_data), dtype=np.float32)
                                      / max(1e-6, getattr(self, '_last_fps_est', 30.0)))
                        else:
                            x_data = np.arange(start_idx, start_idx + len(y_data), dtype=np.float32)
                        curve.setData(x=x_data, y=y_data, skipFiniteCheck=True)
                        try:
                            pen = curve.opts.get('pen', None)
                            if pen is not None and hasattr(pen, 'setWidth'):
                                pen.setWidth(3 if roi_id in getattr(self, '_highlight_ids', set()) else 1)
                                curve.setPen(pen)
                        except Exception:
                            pass
                    except Exception:
                        pass


            if hasattr(self, '_expand_update_count'):
                self._expand_update_count += 1
            else:
                self._expand_update_count = 0

            # Scroll last window but keep global time/index on x-axis
            try:
                x1 = self._global_frame_index
                x0 = max(0, x1 - self._max_points_cfg)
                if getattr(self, '_x_mode_seconds', False):
                    t0 = x0 / max(1e-6, getattr(self, '_last_fps_est', 30.0))
                    t1 = x1 / max(1e-6, getattr(self, '_last_fps_est', 30.0))
                    self._expanded_plot.setXRange(t0, t1, padding=0.02)
                else:
                    self._expanded_plot.setXRange(x0, x1, padding=0.02)
            except Exception:
                pass

        except Exception:

            pass

    def _update_statistical_aggregation_mode(self):

        try:
            if not hasattr(self, '_stat_curves'):
                self._stat_curves = {}
                self._setup_statistical_plot()


            max_len = max(len(buf) for buf in self.buffers.values() if len(buf) > 0)
            if max_len == 0:
                return


            target_points = min(300, max_len)

            trace_matrix = []
            active_rois = []

            for rid, buf in self.buffers.items():
                if len(buf) < 2:
                    continue


                if len(buf) > target_points:
                    indices = np.linspace(0, len(buf) - 1, target_points, dtype=int)
                    resampled = [buf[i] for i in indices]
                else:
                    resampled = list(buf)

                    while len(resampled) < target_points:
                        resampled.append(resampled[-1])

                trace_matrix.append(resampled)
                active_rois.append(rid)

            if not trace_matrix:
                return


            trace_array = np.array(trace_matrix, dtype=np.float32)
            x_data = np.arange(target_points, dtype=np.float32)


            mean_trace = np.mean(trace_array, axis=0)
            std_trace = np.std(trace_array, axis=0)
            percentile_25 = np.percentile(trace_array, 25, axis=0)
            percentile_75 = np.percentile(trace_array, 75, axis=0)
            percentile_10 = np.percentile(trace_array, 10, axis=0)
            percentile_90 = np.percentile(trace_array, 90, axis=0)


            if 'mean' in self._stat_curves:
                self._stat_curves['mean'].setData(x=x_data, y=mean_trace, skipFiniteCheck=True)

            if 'upper_std' in self._stat_curves and 'lower_std' in self._stat_curves:
                upper_std = mean_trace + std_trace
                lower_std = mean_trace - std_trace
                self._stat_curves['upper_std'].setData(x=x_data, y=upper_std, skipFiniteCheck=True)
                self._stat_curves['lower_std'].setData(x=x_data, y=lower_std, skipFiniteCheck=True)

            if 'p75' in self._stat_curves and 'p25' in self._stat_curves:
                self._stat_curves['p75'].setData(x=x_data, y=percentile_75, skipFiniteCheck=True)
                self._stat_curves['p25'].setData(x=x_data, y=percentile_25, skipFiniteCheck=True)


            if len(active_rois) >= 3:

                if not hasattr(self, '_roi_page_index'):
                    self._roi_page_index = 0
                    self._roi_page_size = 3  # Show 3 traces per page
                    self._roi_total_pages = max(1, len(active_rois))  # One page per ROI for full coverage
                    self._setup_pagination_controls()
                    print(f"📄 ROI Pagination initialized: {self._roi_total_pages} ROIs with manual controls")


                if self._roi_total_pages != len(active_rois):
                    self._roi_total_pages = len(active_rois)
                    self._roi_page_index = min(self._roi_page_index, self._roi_total_pages - 1)


                start_idx = self._roi_page_index
                selected_indices = []


                for i in range(3):
                    roi_idx = (start_idx + i) % len(active_rois)
                    selected_indices.append(roi_idx)


                for i in range(3):
                    curve_key = f'highlight_{i}'
                    if curve_key in self._stat_curves:
                        if i < len(selected_indices):
                            idx = selected_indices[i]
                            if idx < len(trace_array):
                                roi_id = active_rois[idx]
                                self._stat_curves[curve_key].setData(x=x_data, y=trace_array[idx], skipFiniteCheck=True)

                                if hasattr(self._stat_curves[curve_key], 'opts') and 'name' in self._stat_curves[curve_key].opts:
                                    self._stat_curves[curve_key].opts['name'] = f'ROI {roi_id} ({idx+1}/{len(active_rois)})'
                        else:

                            self._stat_curves[curve_key].setData(x=[], y=[])


            all_stats = np.concatenate([mean_trace, percentile_10, percentile_90])
            if len(all_stats) > 0:
                stat_min, stat_max = float(np.min(all_stats)), float(np.max(all_stats))
                if np.isfinite(stat_min) and np.isfinite(stat_max) and stat_max > stat_min:
                    range_pad = 0.15 * (stat_max - stat_min)
                    self.plot_widget.setYRange(stat_min - range_pad, stat_max + range_pad, padding=0.0)

        except Exception as e:
            print(f"❌ Statistical aggregation mode error: {e}")

    def _setup_statistical_plot(self):

        try:
            self._stat_curves = {}


            if hasattr(self, '_plot_curves'):
                for curve in self._plot_curves.values():
                    self.plot_widget.removeItem(curve)
                self._plot_curves.clear()


            mean_pen = pg.mkPen(color='#3498db', width=3, style=pg.QtCore.Qt.SolidLine)
            self._stat_curves['mean'] = self.plot_widget.plot(pen=mean_pen, name='Mean')


            std_pen = pg.mkPen(color='#85c1e8', width=2, style=pg.QtCore.Qt.DashLine)
            self._stat_curves['upper_std'] = self.plot_widget.plot(pen=std_pen, name='Mean + 1σ')
            self._stat_curves['lower_std'] = self.plot_widget.plot(pen=std_pen, name='Mean - 1σ')


            perc_pen = pg.mkPen(color='#2ecc71', width=2, style=pg.QtCore.Qt.DotLine)
            self._stat_curves['p75'] = self.plot_widget.plot(pen=perc_pen, name='75th percentile')
            self._stat_curves['p25'] = self.plot_widget.plot(pen=perc_pen, name='25th percentile')


            highlight_colors = ['#e74c3c', '#f39c12', '#9b59b6']
            for i in range(3):
                highlight_pen = pg.mkPen(color=highlight_colors[i], width=1, alpha=0.7)
                self._stat_curves[f'highlight_{i}'] = self.plot_widget.plot(pen=highlight_pen)

            print("✅ Statistical aggregation plot setup complete")

        except Exception as e:
            print(f"❌ Statistical plot setup error: {e}")

    def _update_density_heatmap_mode(self):

        try:
            if not hasattr(self, '_density_plot'):
                self._setup_density_plot()


            max_len = max(len(buf) for buf in self.buffers.values() if len(buf) > 0)
            if max_len == 0:
                return


            target_points = min(200, max_len)
            roi_count = len([buf for buf in self.buffers.values() if len(buf) > 0])


            density_matrix = np.zeros((roi_count, target_points), dtype=np.float32)

            for i, (rid, buf) in enumerate(self.buffers.items()):
                if len(buf) < 2 or i >= roi_count:
                    continue


                if len(buf) > target_points:
                    indices = np.linspace(0, len(buf) - 1, target_points, dtype=int)
                    resampled = np.array([buf[idx] for idx in indices], dtype=np.float32)
                else:
                    resampled = np.array(list(buf), dtype=np.float32)

                    if len(resampled) < target_points:
                        padding = np.full(target_points - len(resampled), resampled[-1])
                        resampled = np.concatenate([resampled, padding])

                density_matrix[i, :] = resampled


            if hasattr(self, '_density_image'):
                self._density_image.setImage(density_matrix, autoLevels=True, autoDownsample=True)


            if hasattr(self, '_summary_curves'):

                overall_mean = np.mean(density_matrix, axis=0)
                overall_std = np.std(density_matrix, axis=0)

                x_data = np.arange(target_points, dtype=np.float32)

                self._summary_curves['mean'].setData(x=x_data, y=overall_mean, skipFiniteCheck=True)
                self._summary_curves['upper'].setData(x=x_data, y=overall_mean + overall_std, skipFiniteCheck=True)
                self._summary_curves['lower'].setData(x=x_data, y=overall_mean - overall_std, skipFiniteCheck=True)

        except Exception as e:
            print(f"❌ Density heatmap mode error: {e}")

    def _setup_density_plot(self):

        try:

            self.plot_widget.clear()


            self._density_image = pg.ImageItem()
            self.plot_widget.addItem(self._density_image)

            self._summary_curves = {}

            mean_pen = pg.mkPen(color='white', width=2)
            self._summary_curves['mean'] = self.plot_widget.plot(pen=mean_pen, name='Population Mean')

            bound_pen = pg.mkPen(color='yellow', width=1, alpha=0.7)
            self._summary_curves['upper'] = self.plot_widget.plot(pen=bound_pen, name='Mean + 1σ')
            self._summary_curves['lower'] = self.plot_widget.plot(pen=bound_pen, name='Mean - 1σ')

            print("✅ Density heatmap plot setup complete")

        except Exception as e:
            print(f"❌ Density plot setup error: {e}")


__all__ = ["LiveTracePlotAggregationMixin"]
