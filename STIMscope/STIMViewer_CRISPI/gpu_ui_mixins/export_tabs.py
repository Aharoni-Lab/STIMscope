"""ExportViewerTabsMixin — extracted from gpu_ui.py.

Bundles the five export-viewer tab construction methods:

* ``_add_roi_overview_tab(tab_widget, file_data)`` (~195 LOC) — ROI
  overview table.
* ``_add_interactive_plot_tab(tab_widget, file_data)`` (~205 LOC) —
  interactive trace plot tab.
* ``_add_html_tab(tab_widget, html_file)`` (~25 LOC) — HTML report tab.
* ``_add_plot_preview_tab(tab_widget, trace_file, metadata_file)``
  (~88 LOC) — plot preview tab.
* ``_open_html_in_browser(html_file)`` (~10 LOC) — open report
  externally via webbrowser.

Method bodies are byte-identical to the pre-extraction code at
``gpu_ui.py:278-797`` (commit ``c936acf``); only the surrounding
module-level frame changed.

See ``docs/specs/L5_UI/gpu_ui.md``.
"""

import os
import sys
import time

import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QGuiApplication, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

class ExportViewerTabsMixin:
    """Cluster 7 — export-viewer tab constructors."""

    def _add_roi_overview_tab(self, tab_widget, file_data):
       
        try:
            from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel
            
            widget = QWidget()
            layout = QVBoxLayout(widget)
            

            header_label = QLabel(f"📊 ROI Overview ({len(file_data.get('traces', {}))} ROIs)")
            header_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px; background: #f0f0f0;")
            layout.addWidget(header_label)
            

            table = QTableWidget()
            

            traces = file_data.get('traces', {})
            metadata = file_data.get('metadata', {})
            
            print("🔍 ROI Overview Debug:")
            print(f"   Traces found: {len(traces)} ROIs")
            print(f"   Metadata found: {len(metadata)} entries")
            print(f"   Available file_data keys: {list(file_data.keys())}")
            if traces:
                print(f"   Sample trace keys: {list(traces.keys())[:5]}")
            if metadata:
                print(f"   Sample metadata keys: {list(metadata.keys())[:5]}")

                sample_key = list(metadata.keys())[0] if metadata else None
                if sample_key:
                    sample_meta = metadata[sample_key]
                    print(f"   Sample metadata content: {list(sample_meta.keys()) if isinstance(sample_meta, dict) else type(sample_meta)}")
            

            if not metadata or len(metadata) == 0:
                print("   🔄 Primary metadata empty, trying fallback sources...")
                

                trace_stats = file_data.get('trace_stats', {})
                if trace_stats:
                    print(f"   ✅ Using trace_stats as fallback metadata: {len(trace_stats)} entries")
                    metadata = trace_stats
                

                elif 'export_info' in file_data and isinstance(file_data['export_info'], dict):
                    export_roi_meta = file_data['export_info'].get('roi_metadata', {})
                    if export_roi_meta:
                        print(f"   ✅ Using export_info roi_metadata: {len(export_roi_meta)} entries")
                        metadata = export_roi_meta
                

                elif hasattr(self, 'live_extractor') and self.live_extractor:
                    print("   🔄 Generating metadata from live extractor...")
                    metadata = self._extract_roi_metadata()
                    if metadata:
                        print(f"   ✅ Generated metadata from live extractor: {len(metadata)} entries")
                

                if not metadata and traces:
                    print("   🔄 Creating basic metadata from trace data...")
                    metadata = {}
                    for roi_id, trace_data in traces.items():
                        if hasattr(trace_data, '__len__') and len(trace_data) > 0:
                            trace_array = np.array(trace_data, dtype=np.float32)
                            metadata[roi_id] = {
                                'roi_index': int(roi_id),
                                'average_intensity': float(np.mean(trace_array)),
                                'size_pixels': max(10, len(trace_data) // 10),
                                'centroid': [roi_id * 20, roi_id * 15],  
                                'color': self.get_roi_color(int(roi_id)),
                                'shape_info': {'type': 'estimated', 'aspect_ratio': 1.0},
                                'generated': True
                            }
                    print(f"   ✅ Created basic metadata: {len(metadata)} entries")
                
            if traces:
                roi_ids = sorted(traces.keys())
                table.setRowCount(len(roi_ids))
                table.setColumnCount(7) 
                table.setHorizontalHeaderLabels(['ROI ID', 'Color', 'Location', 'Size', 'Avg Intensity', 'Trace Length', 'Activity'])
                
                import numpy as np
                
                for row, roi_id in enumerate(roi_ids):

                    table.setItem(row, 0, QTableWidgetItem(str(roi_id)))
                    

                    roi_meta = metadata.get(str(roi_id), metadata.get(roi_id, {}))
                    

                    trace_data = traces.get(roi_id, [])
                    

                    color = roi_meta.get('color', None)
                    if not color:

                        color = self.get_roi_color(int(roi_id))
                    
                    color_item = QTableWidgetItem(f"● ROI {roi_id}")
                    from PyQt5.QtGui import QColor
                    try:
                        qcolor = QColor(color)
                        color_item.setForeground(qcolor)

                        bg_color = QColor(color)
                        bg_color.setAlpha(30) 
                        color_item.setBackground(bg_color)
                    except Exception as e:
                        print(f"⚠️ Color setting warning for ROI {roi_id}: {e}")

                        color_item = QTableWidgetItem(f"ROI {roi_id}")
                    table.setItem(row, 1, color_item)
                    

                    centroid = roi_meta.get('centroid', None)
                    if centroid and isinstance(centroid, list) and len(centroid) >= 2:
                        try:

                            x_val = float(centroid[0]) if isinstance(centroid[0], (int, float, str)) and str(centroid[0]).replace('.','').replace('-','').isdigit() else 0
                            y_val = float(centroid[1]) if isinstance(centroid[1], (int, float, str)) and str(centroid[1]).replace('.','').replace('-','').isdigit() else 0
                            location_str = f"({x_val:.0f}, {y_val:.0f})"
                        except Exception:
                            location_str = f"({centroid[0]}, {centroid[1]})"
                    else:

                        location_str = f"ROI {roi_id} (estimated)"
                    table.setItem(row, 2, QTableWidgetItem(location_str))
                    

                    size = roi_meta.get('size_pixels', roi_meta.get('size', None))
                    if size is None or size == 'Unknown' or size == 0:

                        if hasattr(trace_data, '__len__') and len(trace_data) > 0:

                            estimated_size = max(10, len(trace_data) // 2) 
                            size = f"~{estimated_size} px (est.)"
                        else:
                            size = "Unknown"
                    else:
                        size = f"{size} px"
                    table.setItem(row, 3, QTableWidgetItem(str(size)))
                    

                    avg_intensity = roi_meta.get('average_intensity', roi_meta.get('mean', None))
                    if avg_intensity is None and hasattr(trace_data, '__len__') and len(trace_data) > 0:
                        try:
                            trace_array = np.array(trace_data, dtype=np.float32)
                            avg_intensity = float(np.mean(trace_array))
                        except Exception:
                            avg_intensity = 0
                    
                    if avg_intensity is not None:
                        table.setItem(row, 4, QTableWidgetItem(f"{avg_intensity:.2f}"))
                    else:
                        table.setItem(row, 4, QTableWidgetItem("N/A"))
                    

                    trace_length = len(trace_data) if hasattr(trace_data, '__len__') else 0
                    table.setItem(row, 5, QTableWidgetItem(str(trace_length)))
                    

                    activity = "Unknown"
                    if hasattr(trace_data, '__len__') and len(trace_data) > 1:
                        try:
                            trace_array = np.array(trace_data, dtype=np.float32)
                            if len(trace_array) > 1:
                                cv = np.std(trace_array) / np.mean(trace_array) if np.mean(trace_array) > 0 else 0
                                if cv > 0.3:
                                    activity = "High"
                                elif cv > 0.1:
                                    activity = "Moderate"
                                else:
                                    activity = "Low"
                        except Exception:
                            activity = "Unknown"
                    table.setItem(row, 6, QTableWidgetItem(activity))
                

                table.resizeColumnsToContents()
                
            else:
                table.setRowCount(1)
                table.setColumnCount(1)
                table.setHorizontalHeaderLabels(['Status'])
                table.setItem(0, 0, QTableWidgetItem("No ROI data found"))
            
            layout.addWidget(table)
            tab_widget.addTab(widget, "📊 ROI Overview")
            
        except Exception as e:
            error_widget = QLabel(f"Error creating ROI overview: {e}")
            tab_widget.addTab(error_widget, "❌ ROI Overview")

    def _add_interactive_plot_tab(self, tab_widget, file_data):
       
        try:
            import numpy as np
            try:
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.figure import Figure
                matplotlib_available = True
            except ImportError as e:
                print(f"⚠️ Matplotlib import error: {e}")
                matplotlib_available = False
            
            from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QScrollArea, QLabel, QPushButton
            from PyQt5.QtCore import Qt
            
            if not matplotlib_available:
                error_widget = QLabel("Matplotlib not available for interactive plotting")
                tab_widget.addTab(error_widget, "❌ Interactive Plot")
                return
            
            widget = QWidget()
            main_layout = QVBoxLayout(widget)
            

            pagination_widget = QWidget()
            pagination_layout = QHBoxLayout(pagination_widget)
            
            prev_btn = QPushButton("◀ Previous 10 ROIs")
            page_label = QLabel("Page 1/1 (ROIs 1-10)")
            page_label.setAlignment(Qt.AlignCenter)
            page_label.setStyleSheet("font-weight: bold; padding: 5px;")
            next_btn = QPushButton("Next 10 ROIs ▶")
            
            pagination_layout.addWidget(prev_btn)
            pagination_layout.addWidget(page_label)
            pagination_layout.addWidget(next_btn)
            main_layout.addWidget(pagination_widget)
            

            plot_container = QWidget()
            plot_layout = QHBoxLayout(plot_container)
            

            plot_widget = QWidget()
            plot_widget_layout = QVBoxLayout(plot_widget)
            

            fig = Figure(figsize=(12, 8))
            canvas = FigureCanvas(fig)
            plot_widget_layout.addWidget(canvas)
            

            control_widget = QWidget()
            control_widget.setMaximumWidth(200)
            control_layout = QVBoxLayout(control_widget)
            
            control_header = QLabel("Current Page ROIs:")
            control_header.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
            control_layout.addWidget(control_header)
            

            checkbox_widget = QWidget()
            checkbox_layout = QVBoxLayout(checkbox_widget)
            

            traces = file_data.get('traces', {})
            metadata = file_data.get('metadata', {})
            
            if traces:

                roi_ids = sorted(traces.keys())
                rois_per_page = 10
                total_pages = (len(roi_ids) + rois_per_page - 1) // rois_per_page
                current_page = 0
                

                ax = fig.add_subplot(111)
                plot_lines = {}
                checkboxes = {}
                
                def update_plot_page():

                    ax.clear()
                    

                    for cb in checkboxes.values():
                        cb.setParent(None)
                    checkboxes.clear()
                    

                    start_idx = current_page * rois_per_page
                    end_idx = min(start_idx + rois_per_page, len(roi_ids))
                    page_roi_ids = roi_ids[start_idx:end_idx]
                    

                    page_label.setText(f"Page {current_page + 1}/{total_pages} (ROIs {start_idx + 1}-{end_idx})")
                    

                    for idx, roi_id in enumerate(page_roi_ids):
                        trace_data = traces[roi_id]
                        if hasattr(trace_data, '__len__') and len(trace_data) > 0:
                            y_data = np.array(trace_data, dtype=np.float32)
                            x_data = np.arange(len(y_data))

                            color_hex = self.get_roi_color(int(roi_id))
                            color = mcolors.to_rgba(color_hex)
                            
                            line, = ax.plot(x_data, y_data, color=color, label=f"ROI {roi_id}", 
                                          alpha=0.8, linewidth=2)
                            plot_lines[roi_id] = line
                            

                            checkbox = QCheckBox(f"ROI {roi_id}")
                            checkbox.setChecked(True)
                            

                            try:
                                checkbox.setStyleSheet(f"color: {color_hex}; font-weight: bold;")
                            except Exception:
                                pass
                            

                            def make_toggle_function(plot_line, roi_identifier):
                                def toggle_line(checked):
                                    try:
                                        plot_line.set_visible(checked)
                                        canvas.draw()
                                        print(f"🔍 ROI {roi_identifier} visibility: {checked}")
                                    except Exception as e:
                                        print(f"⚠️ Toggle error for ROI {roi_identifier}: {e}")
                                return toggle_line
                            
                            checkbox.toggled.connect(make_toggle_function(line, roi_id))
                            checkboxes[roi_id] = checkbox
                            checkbox_layout.addWidget(checkbox)
                    

                    ax.set_xlabel('Time Points')
                    ax.set_ylabel('Intensity')
                    ax.set_title(f'Interactive ROI Traces - Page {current_page + 1}/{total_pages}')
                    ax.grid(True, alpha=0.3)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                    

                    canvas.draw()
                
                def prev_page():
                    nonlocal current_page
                    if current_page > 0:
                        current_page -= 1
                        update_plot_page()
                        prev_btn.setEnabled(current_page > 0)
                        next_btn.setEnabled(current_page < total_pages - 1)
                
                def next_page():
                    nonlocal current_page
                    if current_page < total_pages - 1:
                        current_page += 1
                        update_plot_page()
                        prev_btn.setEnabled(current_page > 0)
                        next_btn.setEnabled(current_page < total_pages - 1)
                

                prev_btn.clicked.connect(prev_page)
                next_btn.clicked.connect(next_page)
                

                prev_btn.setEnabled(False)
                next_btn.setEnabled(total_pages > 1)
                

                update_plot_page()
                
            else:

                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, 'No trace data available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title('Interactive Plot - No Data')
                page_label.setText("No data")
                prev_btn.setEnabled(False)
                next_btn.setEnabled(False)
                canvas.draw()
            

            scroll_area = QScrollArea()
            scroll_area.setWidget(checkbox_widget)
            scroll_area.setWidgetResizable(True)
            control_layout.addWidget(scroll_area)
            

            plot_layout.addWidget(plot_widget)
            plot_layout.addWidget(control_widget)
            main_layout.addWidget(plot_container)
            
            tab_widget.addTab(widget, "📈 Interactive Plot")

        
        except Exception as e:
            error_widget = QLabel(f"Error creating interactive plot: {e}")
            tab_widget.addTab(error_widget, "❌ Interactive Plot")

    def _add_html_tab(self, tab_widget, html_file):
       
        try:
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            from PyQt5.QtCore import QUrl
            
            web_view = QWebEngineView()
            web_view.load(QUrl.fromLocalFile(os.path.abspath(html_file)))
            
            tab_widget.addTab(web_view, "📋 Visual Summary")
            
        except ImportError:

            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            label = QLabel("Web engine not available for HTML preview.\\nUse 'Open Full Report in Browser' button.")
            label.setStyleSheet("padding: 20px; color: #666;")
            layout.addWidget(label)
            
            tab_widget.addTab(widget, "📋 Visual Summary")
        except Exception as e:
            error_widget = QLabel(f"Error loading HTML: {e}")
            tab_widget.addTab(error_widget, "❌ HTML")

    def _add_plot_preview_tab(self, tab_widget, trace_file, metadata_file):
       
        try:
            import numpy as np
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            widget = QWidget()
            layout = QVBoxLayout(widget)
            

            fig = Figure(figsize=(12, 8))
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            

            # App-generated export files may store object arrays (trace dict /
            # metadata blobs); allow pickle to read them. Trusted local input.
            trace_data = np.load(trace_file, allow_pickle=True)


            roi_colors = {}
            roi_labels = {}
            if metadata_file:
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    roi_metadata = metadata.get('roi_metadata', {})
                    for roi_id, roi_data in roi_metadata.items():
                        roi_colors[int(roi_id)] = roi_data.get('color', '#000000')
                        centroid = roi_data.get('centroid', [0, 0])
                        roi_labels[int(roi_id)] = f"ROI {roi_id} @({centroid[0]}, {centroid[1]})"
                except Exception:
                    pass
            

            if isinstance(trace_data, dict):

                ax = fig.add_subplot(111)
                plotted_count = 0
                
                for key, values in trace_data.items():
                    if isinstance(values, np.ndarray) and len(values) > 0:
                        try:

                            roi_id = None
                            if 'roi' in key.lower():
                                import re
                                match = re.search(r'roi.?(\d+)', key.lower())
                                if match:
                                    roi_id = int(match.group(1))
                            
                            color = roi_colors.get(roi_id, f'C{plotted_count % 10}') if roi_id else f'C{plotted_count % 10}'
                            label = roi_labels.get(roi_id, key) if roi_id else key
                            
                            ax.plot(values, color=color, label=label, alpha=0.8)
                            plotted_count += 1
                            
                            if plotted_count >= 20: 
                                break
                                
                        except Exception as e:
                            print(f"Plot error for {key}: {e}")
                
                ax.set_xlabel('Time Points')
                ax.set_ylabel('Intensity')
                ax.set_title(f'Exported Traces Preview ({plotted_count} traces)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
            else:

                ax = fig.add_subplot(111)
                ax.plot(trace_data)
                ax.set_xlabel('Time Points')
                ax.set_ylabel('Intensity')
                ax.set_title('Exported Trace Preview')
                ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            canvas.draw()
            
            tab_widget.addTab(widget, "📈 Plot Preview")
            
        except Exception as e:
            error_widget = QLabel(f"Error generating plot: {e}")
            tab_widget.addTab(error_widget, "❌ Plot Preview")

    def _open_html_in_browser(self, html_file):
       
        try:
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(html_file)}')
        except Exception as e:
            print(f"❌ Browser open error: {e}")



