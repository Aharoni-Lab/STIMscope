"""Pagination + page navigation extracted from ``live_trace_extractor``.

Stage-0.6 of the 6-module decomposition (sub-module 6/6, sub-mixin
3/3 of the plot-modes sub-split). Extracted
from ``live_trace_extractor.py`` (iter 41).

Per the iter-36 / iter-37 sub-split plan, COMPLETE after this iter:
- iter 37 ✅ ``live_trace_plot_modes.py`` (dispatcher + pygame +
  pyqtgraph entry + skip + ROI color)
- iter 39 ✅ ``live_trace_plot_aggregation.py`` (expanded /
  statistical / density-heatmap modes)
- **iter 41 ✅ THIS FILE** ``live_trace_plot_pagination.py`` —
  paged-trace mode + page navigation + pagination controls

The 10 helpers in THIS mixin:
- ``_update_paged_trace_mode()`` — paginate ROI traces across pages
  (~195 LOC, the largest pagination method)
- ``_update_legend_for_page(page_rois)`` — refresh page legend
- ``_setup_pagination_controls()`` — build the Prev/Next QPushButton
  widgets and page-label QLabel (~195 LOC)
- ``_update_page_label_safe()`` — DEFINED TWICE in the original
  parent (D-ltm-1 BUG): Python uses only the 2nd definition. Both
  preserved here forchars discipline; iter-42 chars must
  pin the duplicate, and afix can dedupe.
- ``_prev_roi_page()`` — back-page button handler
- ``_next_roi_page()`` — next-page button handler
- ``restart_after_napari(new_plot_widget=None)`` — clean restart
  hook used by napari integration
- ``_cleanup_pagination_widget()`` — tear down pagination controls
- ``_update_page_label()`` — non-safe variant; updates the QLabel

The mixin expects the subclass (LiveTraceExtractor) to provide:
- ``self.plot_widget`` (pyqtgraph PlotWidget or None)
- ``self.buffers``, ``self._dff_buffers`` (Dict[int, deque])
- ``self.ids`` (np.ndarray[int32])
- ``self._plot_curves`` (Dict[int, curve])
- ``self._is_shutting_down`` (bool, optional)
- ``self._cleanup_event`` (threading.Event, optional)
- ``self._global_frame_index`` (int counter)
- ``self._max_points_cfg``, ``self._last_fps_est``
- ``self._highlight_ids`` (set[int])
- ``self._roi_page_index``, ``self._roi_page_size``,
  ``self._roi_total_pages`` (pagination state, lazily inited)
- ``self._get_unified_roi_color`` (on PlotModesMixin via MRO)
- ``self._setup_statistical_plot``, ``self._update_statistical_aggregation_mode``
  (on PlotAggregationMixin via MRO; used by some pagination
  callbacks that recompute the active mode)
- ``self._update_paged_trace_mode``, ``self._setup_density_plot``,
  ``self._resolve_trace_y`` (some referenced by callbacks)

No behavior change vs the original location.

Safety: smoke + sibling chars tests in ``tests/L3_5_split_first/``
must remain green.

After iter 41 ✅ extract + iter 42 ✅ chars, the L3.5 SPLIT-FIRST
decomposition is COMPLETE — live_trace_extractor.py audit moves
from 🟡 IN PROGRESS to 🟢 DONE provisional with a 7-day window.
"""

from __future__ import annotations

import numpy as np

try:
    import pyqtgraph as pg
    PYQTPGRAPH_AVAILABLE = True
except Exception:
    PYQTPGRAPH_AVAILABLE = False
    pg = None


class LiveTracePlotPaginationMixin:
    """Paginated-trace + navigation helpers for ``LiveTraceExtractor``."""

    def _update_paged_trace_mode(self):
       
        try:

            if getattr(self, '_is_shutting_down', False):
                return
            if hasattr(self, '_cleanup_event') and self._cleanup_event and self._cleanup_event.is_set():
                return

            if not self.plot_widget or not hasattr(self.plot_widget, 'plot'):
                return
            

            try:
                viewbox = self.plot_widget.getViewBox()
                if not viewbox:
                    self._plot_curves.clear()
                    return

                _ = viewbox.viewRange()
            except Exception as viewbox_error:
                print(f"⚠️ Plot widget invalid, clearing curves: {viewbox_error}")
                self._plot_curves.clear()
                return
            

            if not hasattr(self, '_trace_page_index'):
                self._trace_page_index = 0
                self._traces_per_page = 5
                self._setup_pagination_controls()
            

            active_rois = sorted([rid for rid, buf in self.buffers.items() if len(buf) >= 2])
            
            if not active_rois:
                return
            

            total_pages = max(1, (len(active_rois) + self._traces_per_page - 1) // self._traces_per_page)
            self._trace_page_index = min(self._trace_page_index, total_pages - 1)
            

            start_idx = self._trace_page_index * self._traces_per_page
            end_idx = min(start_idx + self._traces_per_page, len(active_rois))
            page_rois = active_rois[start_idx:end_idx]
            

            valid_curves = {}
            for roi_id, curve in list(self._plot_curves.items()):
                try:

                    if (hasattr(curve, 'setData') and 
                        hasattr(curve, 'clear') and 
                        not curve.__class__.__name__.endswith('_deleted')):
                        

                        try:
                            scene = curve.scene()
                            if scene is not None:
                                curve.clear()
                                valid_curves[roi_id] = curve
                            else:

                                pass
                        except Exception as scene_error:
                            if "deleted" not in str(scene_error).lower():
                                print(f"⚠️ Curve for ROI {roi_id}: scene access error: {scene_error}")
                    else:

                        pass
                except Exception as curve_error:
                    if "deleted" not in str(curve_error).lower():
                        print(f"⚠️ Curve error for ROI {roi_id}: {curve_error}")
            
            self._plot_curves = valid_curves
            if len(valid_curves) != len(self._plot_curves):
                print(f"🔄 Curve validation: {len(valid_curves)} valid curves retained")
            

            max_len = 0
            for i, roi_id in enumerate(page_rois):
                y_data = self._resolve_trace_y(roi_id)
                if len(y_data) < 2:
                    continue
                if len(y_data) > max_len:
                    max_len = len(y_data)

                try:
                    if roi_id not in self._plot_curves or not hasattr(self._plot_curves[roi_id], 'setData'):
                        if self.plot_widget and hasattr(self.plot_widget, 'plot'):
                            unified_color = self._get_unified_roi_color(roi_id)
                            pen = pg.mkPen(color=unified_color, width=2)
                            self._plot_curves[roi_id] = self.plot_widget.plot(pen=pen)
                        else:
                            continue
                    start_idx = max(0, self._global_frame_index - len(y_data))
                    if getattr(self, '_x_mode_seconds', False):
                        x_data = (np.arange(start_idx, start_idx + len(y_data), dtype=np.float32)
                                  / max(1e-6, getattr(self, '_last_fps_est', 30.0)))
                    else:
                        x_data = np.arange(start_idx, start_idx + len(y_data), dtype=np.float32)
                    # Emphasize highlighted traces
                    try:
                        if roi_id in getattr(self, '_highlight_ids', set()):
                            pen = self._plot_curves[roi_id].opts.get('pen', None)
                            if pen is not None and hasattr(pen, 'setWidth'):
                                pen.setWidth(3)
                                self._plot_curves[roi_id].setPen(pen)
                        else:
                            # set thinner width for non-highlighted
                            pen = self._plot_curves[roi_id].opts.get('pen', None)
                            if pen is not None and hasattr(pen, 'setWidth'):
                                pen.setWidth(1)
                                self._plot_curves[roi_id].setPen(pen)
                    except Exception:
                        pass
                    self._plot_curves[roi_id].setData(x=x_data, y=y_data)
                    
                except Exception as curve_error:
                    if roi_id in self._plot_curves:
                        del self._plot_curves[roi_id]
                    print(f"⚠️ Curve error for ROI {roi_id}: {curve_error}")
            

            for roi_id, curve in list(self._plot_curves.items()):
                if roi_id not in page_rois:
                    try:
                        if hasattr(curve, 'clear'):
                            curve.clear()
                    except Exception:

                        del self._plot_curves[roi_id]
            

            self._update_page_label_safe()

            self._update_legend_for_page(page_rois)

            # Update trace info label in parent UI if available
            try:
                parent = self.plot_widget.parent() if self.plot_widget else None
                # climb to GPU instance
                gpu = None
                d = 0
                p = parent
                while p is not None and d < 6:
                    if hasattr(p, 'camera') and hasattr(p, 'plot_widget'):
                        gpu = p
                        break
                    p = getattr(p, 'parent', lambda: None)()
                    d += 1
                if gpu is not None and hasattr(gpu, '_trace_info_label') and gpu._trace_info_label is not None:
                    try:
                        fps = getattr(self, '_last_fps_est', 0.0)
                        total = getattr(self, 'total_rois_extracted', len(active_rois))
                        gpu._trace_info_label.setText(f"Traces: {fps:.1f} fps | ROIs: {len(active_rois)}/{total}")
                    except Exception:
                        pass
            except Exception:
                pass
            

            # Update labels and dynamic x range
            try:
                if hasattr(self.plot_widget, 'setLabel'):
                    self.plot_widget.setLabel('left', 'Intensity')
                    self.plot_widget.setLabel('bottom', 'Time (frames)' if not getattr(self, '_x_mode_seconds', False) else 'Time (s)')
            except Exception:
                pass

            if max_len > 1:
                # Show last window but keep axis in global coordinates
                x1 = self._global_frame_index
                x0 = max(0, x1 - self._max_points_cfg)
                if getattr(self, '_x_mode_seconds', False):
                    t0 = x0 / max(1e-6, getattr(self, '_last_fps_est', 30.0))
                    t1 = x1 / max(1e-6, getattr(self, '_last_fps_est', 30.0))
                    try:
                        self.plot_widget.setXRange(t0, t1, padding=0.02)
                    except Exception:
                        pass
                else:
                    try:
                        self.plot_widget.setXRange(x0, x1, padding=0.02)
                    except Exception:
                        pass
            

            self._update_expanded_plot()
            
        except Exception as e:

            if "deleted" not in str(e).lower() and "viewbox" not in str(e).lower():
                print(f"❌ Paged trace mode error: {e}")

    def _update_legend_for_page(self, page_rois):
       
        try:

            if not hasattr(self, '_legend_layout') or not self._legend_layout:
                return
            

            if not hasattr(self, '_combined_legend_label') or self._combined_legend_label is None:
                from PyQt5.QtWidgets import QLabel
                from PyQt5.QtCore import Qt
                self._combined_legend_label = QLabel("ROI Legend")
                self._combined_legend_label.setStyleSheet("""
                    QLabel {
                        font-size: 10px; 
                        padding: 5px; 
                        color: #333;
                        background-color: #f8f8f8;
                        border: 1px solid #ddd;
                        border-radius: 3px;
                    }
                """)

                self._combined_legend_label.setTextFormat(Qt.RichText)
                self._legend_layout.addWidget(self._combined_legend_label)
            

            if page_rois:
                legend_text_parts = []
                for roi_id in page_rois:

                    if roi_id in self._plot_curves and hasattr(self._plot_curves[roi_id], 'opts'):
                        try:
                            curve_pen = self._plot_curves[roi_id].opts.get('pen', None)
                            if curve_pen and hasattr(curve_pen, 'color'):

                                curve_color = curve_pen.color()
                                color_hex = f"#{curve_color.red():02x}{curve_color.green():02x}{curve_color.blue():02x}"
                            else:

                                color_hex = self._get_unified_roi_color(roi_id)
                        except Exception:
                            color_hex = self._get_unified_roi_color(roi_id)
                    else:
                        color_hex = self._get_unified_roi_color(roi_id)
                    
                    legend_text_parts.append(f'<span style="color: {color_hex}; font-weight: bold;">● ROI {roi_id}</span>')
                
                legend_text = " | ".join(legend_text_parts)
            else:
                legend_text = "<span style='color: #666;'>No active traces</span>"
            

            self._combined_legend_label.setText(legend_text)
            
        except Exception as e:
            print(f"⚠️ Legend update error (suppressed): {e}")
            pass

    # Expanded-view dialog (_expand_all_rois + _update_expanded_plot)
    # extracted to live_trace_plot_aggregation.py as
    # LiveTracePlotAggregationMixin. Mixed in via class declaration above.
    # _get_unified_roi_color is on LiveTracePlotModesMixin (iter 37).

    def _setup_pagination_controls(self):
       
        try:
            from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
            from PyQt5.QtCore import Qt
            
            if hasattr(self, '_pagination_widget') and self._pagination_widget is not None:
                try:
                    if self._pagination_widget.isVisible():

                        self._update_page_label_safe()
                        return
                    else:

                        self._cleanup_pagination_widget()
                except Exception:

                    self._cleanup_pagination_widget()
                    

            if not hasattr(self, '_current_page'):
                self._current_page = 0
            if not hasattr(self, '_traces_per_page'):
                self._traces_per_page = 5
                

            if not hasattr(self, '_pagination_widget') or self._pagination_widget is None:

                self._pagination_widget = QWidget()
                main_layout = QVBoxLayout(self._pagination_widget)
                main_layout.setSpacing(5)
                

                nav_widget = QWidget()
                pagination_layout = QHBoxLayout(nav_widget)
                pagination_layout.setContentsMargins(0, 0, 0, 0)
                

                self._prev_button = QPushButton("◀ Prev Traces")
                self._prev_button.setMaximumWidth(120)
                self._prev_button.clicked.connect(self._prev_roi_page)
                pagination_layout.addWidget(self._prev_button)
                

                self._page_label = QLabel("Traces 1-5 (Page 1/1)")
                self._page_label.setAlignment(Qt.AlignCenter)
                self._page_label.setStyleSheet("font-weight: bold; padding: 5px; min-width: 150px;")
                pagination_layout.addWidget(self._page_label)
                

                self._next_button = QPushButton("Next Traces ▶")
                self._next_button.setMaximumWidth(120)
                self._next_button.clicked.connect(self._next_roi_page)
                pagination_layout.addWidget(self._next_button)
                

                self._expand_button = QPushButton("🔍 Expand All ROIs")
                self._expand_button.setMaximumWidth(140)
                self._expand_button.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        font-weight: bold;
                        border-radius: 5px;
                        padding: 6px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
                self._expand_button.clicked.connect(self._expand_all_rois)
                pagination_layout.addWidget(self._expand_button)
                
                main_layout.addWidget(nav_widget)
                

                self._legend_widget = QWidget()
                self._legend_layout = QHBoxLayout(self._legend_widget)
                self._legend_layout.setContentsMargins(5, 5, 5, 5)
                self._legend_layout.setSpacing(10)
                

                legend_title = QLabel("Current ROIs:")
                legend_title.setStyleSheet("font-weight: bold; font-size: 10px;")
                self._legend_layout.addWidget(legend_title)
                

                self._legend_labels = []
                
                main_layout.addWidget(self._legend_widget)
                

                self._pagination_widget.setStyleSheet("""
                    QWidget {
                        background-color: #f8f8f8;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        margin: 2px;
                    }
                    QPushButton {
                        background-color: #e8e8e8;
                        border: 1px solid #ccc;
                        border-radius: 3px;
                        padding: 5px;
                    }
                    QPushButton:hover {
                        background-color: #d8d8d8;
                    }
                """)
                
                try:

                    self._pagination_widget.setWindowTitle("ROI Pagination Controls")
                    self._pagination_widget.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)
                    self._pagination_widget.resize(600, 100)
                    

                    # Position the pagination widget on the SAME screen as
                    # the plot widget's top-level window. Default Qt position
                    # places Qt.Tool windows at "primary" which on STIMscope
                    # is the projector monitor — the widget then appears as
                    # garbage on the projector output.
                    if self.plot_widget:
                        try:
                            top = self.plot_widget.window()
                            top_geom = top.geometry() if top is not None else None
                            screen = (
                                top.screen() if top is not None and hasattr(top, "screen") else None
                            )
                            if screen is None and top_geom is not None:
                                # Fallback: position relative to the plot widget
                                self._pagination_widget.move(
                                    top_geom.x() + 20,
                                    top_geom.y() + top_geom.height() + 10,
                                )
                            elif screen is not None:
                                geom = screen.availableGeometry()
                                # Place just below the main GPU dialog if possible,
                                # else top-left of the same screen.
                                if top_geom is not None and geom.contains(top_geom):
                                    self._pagination_widget.move(
                                        top_geom.x(),
                                        min(top_geom.y() + top_geom.height() + 10,
                                            geom.y() + geom.height() - 120),
                                    )
                                else:
                                    self._pagination_widget.move(geom.x() + 80, geom.y() + 100)
                        except Exception:
                            pass
                    
                    try:
                        from PyQt5.QtCore import Qt
                        self._pagination_widget.setWindowModality(Qt.NonModal)
                        self._pagination_widget.setWindowFlags(
                            Qt.Tool | Qt.WindowStaysOnTopHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint
                        )

                        if self.plot_widget and hasattr(self.plot_widget, 'window') and self.plot_widget.window():
                            main_window = self.plot_widget.window()
                            try:

                                if not hasattr(self, '_pagination_close_connected'):
                                    main_window.destroyed.connect(self._cleanup_pagination_widget)
                                    self._pagination_close_connected = True
                            except Exception:
                                pass
                    except Exception:
                        pass
                    self._pagination_widget.show()
                    print("✅ ROI pagination controls created as standalone widget")
                    
                except Exception as pagination_error:
                    print(f"❌ Pagination creation failed: {pagination_error}")

                    if hasattr(self, '_pagination_widget'):
                        try:
                            self._pagination_widget.setParent(None)
                            self._pagination_widget.deleteLater()
                        except Exception:
                            pass
                        self._pagination_widget = None

        except Exception as e:
            print(f"⚠️ Could not create pagination controls: {e}")
            import traceback
            print(f"   Stack trace: {traceback.format_exc()}")

            try:
                if hasattr(self, '_pagination_widget') and self._pagination_widget is not None:
                    self._pagination_widget.close()
                    self._pagination_widget.deleteLater()
                    self._pagination_widget = None
            except Exception:
                pass

    # D-ltm-1fix iter 43: removed the first (DEAD) definition of
    # `_update_page_label_safe` that was here. Python's class-body
    # rebinding rule meant the second definition (below) was the only
    # one that actually dispatched — the first was dead code. The live
    # second definition (~line 632) remains untouched.

    def _prev_roi_page(self):
       
        try:

            if hasattr(self, '_navigation_in_progress') and self._navigation_in_progress:
                return
            self._navigation_in_progress = True
            
            active_rois = sorted([rid for rid, buf in self.buffers.items() if len(buf) >= 2])
            if not active_rois:
                self._navigation_in_progress = False
                return
            
            if not hasattr(self, '_trace_page_index'):
                self._trace_page_index = 0
                
            if self._trace_page_index > 0:
                self._trace_page_index -= 1
            else:

                total_pages = max(1, (len(active_rois) + self._traces_per_page - 1) // self._traces_per_page)
                self._trace_page_index = total_pages - 1
            self._update_paged_trace_mode()
            self._update_page_label_safe()
            print(f"📄 Trace page: {self._trace_page_index + 1}")
            
            self._navigation_in_progress = False
        except Exception as e:
            print(f"⚠️ Previous page error: {e}")
            self._navigation_in_progress = False
    
    def _next_roi_page(self):
       
        try:

            if hasattr(self, '_navigation_in_progress') and self._navigation_in_progress:
                return
            self._navigation_in_progress = True
            
            active_rois = sorted([rid for rid, buf in self.buffers.items() if len(buf) >= 2])
            if not active_rois:
                self._navigation_in_progress = False
                return
            

            if not hasattr(self, '_trace_page_index'):
                self._trace_page_index = 0
            if not hasattr(self, '_traces_per_page'):
                self._traces_per_page = 5
                
            total_pages = max(1, (len(active_rois) + self._traces_per_page - 1) // self._traces_per_page)
            
            if self._trace_page_index < total_pages - 1:
                self._trace_page_index += 1
            else:

                self._trace_page_index = 0
            self._update_paged_trace_mode()
            self._update_page_label_safe()
            print(f"📄 Trace page: {self._trace_page_index + 1}")
            
            self._navigation_in_progress = False
        except Exception as e:
            print(f"⚠️ Next page error: {e}")
            self._navigation_in_progress = False

    def restart_after_napari(self, new_plot_widget=None):
       
        try:
            print("🔄 Restarting LiveTraceExtractor after Napari...")
            

            if new_plot_widget:
                self.plot_widget = new_plot_widget
                print("✅ Plot widget updated")
            

            if self.plot_widget:

                if hasattr(self, '_pagination_widget'):
                    self._cleanup_pagination_widget()
                

                self._setup_pagination_controls()
                print("✅ Pagination controls reinitialized")
            

            if hasattr(self, 'buffers') and self.buffers:
                self._update_paged_trace_mode()
                print("✅ Live traces resumed")
            
            return True
            
        except Exception as e:
            print(f"❌ Restart after Napari failed: {e}")
            return False

    def _cleanup_pagination_widget(self):
       
        try:
            if hasattr(self, '_pagination_widget') and self._pagination_widget is not None:
                try:
                    self._pagination_widget.close()
                except Exception:
                    pass
                self._pagination_widget.setParent(None)
                self._pagination_widget.deleteLater()
                self._pagination_widget = None
                

            if hasattr(self, '_legend_labels'):
                for label in self._legend_labels:
                    if label:
                        label.setParent(None)
                        label.deleteLater()
                self._legend_labels.clear()
                
        except Exception as e:
            print(f"⚠️ Pagination cleanup warning: {e}")

    def _update_page_label_safe(self):
       
        try:
            if not hasattr(self, '_page_label') or not self._page_label:
                return
                
            active_rois = sorted([rid for rid, buf in self.buffers.items() if len(buf) >= 2])
            if not active_rois:
                self._page_label.setText("No active traces")
                if hasattr(self, '_prev_button'):
                    self._prev_button.setEnabled(False)
                if hasattr(self, '_next_button'):
                    self._next_button.setEnabled(False)
                return
                
            total_pages = max(1, (len(active_rois) + self._traces_per_page - 1) // self._traces_per_page)
            current_page = getattr(self, '_trace_page_index', 0) + 1
            
            start_roi = (getattr(self, '_trace_page_index', 0) * self._traces_per_page) + 1
            end_roi = min(start_roi + self._traces_per_page - 1, len(active_rois))
            
            self._page_label.setText(f"Traces {start_roi}-{end_roi} (Page {current_page}/{total_pages})")
            

            if hasattr(self, '_prev_button'):
                self._prev_button.setEnabled(True)  
            if hasattr(self, '_next_button'):
                self._next_button.setEnabled(True)  
                
        except Exception as e:
            print(f"⚠️ Page label update error: {e}")

    def _update_page_label(self):
       
        try:
            if hasattr(self, '_page_label') and hasattr(self, '_trace_page_index'):

                active_rois = [rid for rid, buf in self.buffers.items() if len(buf) >= 2]
                total_pages = max(1, (len(active_rois) + self._traces_per_page - 1) // self._traces_per_page)
                

                start_idx = self._trace_page_index * self._traces_per_page
                end_idx = min(start_idx + self._traces_per_page, len(active_rois))
                
                self._page_label.setText(f"Traces {start_idx + 1}-{end_idx} (Page {self._trace_page_index + 1}/{total_pages})")
        except Exception as e:
            print(f"⚠️ Page label update error: {e}")

    # _setup_statistical_plot + _update_density_heatmap_mode +
    # _setup_density_plot extracted to live_trace_plot_aggregation.py
    # (iter 39). Accessible via MRO.


    # ROI build + buffer init + GPU/CPU label-array setup + dF/F + state
    # cleanup all extracted to live_trace_processing.py (sub-module 5/6,
    # iter 35). Mixed in above. See live_trace_processing.py
    # for the LiveTraceProcessingMixin contract.


__all__ = ["LiveTracePlotPaginationMixin"]
