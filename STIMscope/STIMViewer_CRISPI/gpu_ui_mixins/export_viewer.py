"""ExportViewerMixin — extracted from ``gpu_ui.py`` per L5 SPLIT-FIRST.

Cluster #7 of the 9-sub-module decomposition (see
``docs/specs/L5_UI/gpu_ui.md`` §0.5). Contains the 6 methods that
implement the **exported-trace VIEWER dialog** core surface — file
loading + statistics, system-info, trace-data, and metadata tabs:

- ``_view_exported_traces()`` — Qt-button slot; spawns a ``QDialog``
  with a ``QTabWidget`` and dispatches tab builders. Cross-cluster
  calls into ``_add_roi_overview_tab`` (cluster #8) and
  ``_add_interactive_plot_tab`` / ``_add_html_tab`` /
  ``_open_html_in_browser`` (cluster #9) through MRO.
- ``_load_export_file(file_path)`` — unified-npz / legacy-npz /
  legacy-npy parser with JSON-metadata sidecar support.
- ``_add_statistics_tab(tab_widget, file_data)`` — global + per-ROI
  trace stats (mean / std / range / CV-based activity classification).
- ``_add_system_info_tab(tab_widget, file_data)`` — machine snapshot
  + session-summary text dump.
- ``_add_trace_data_tab(tab_widget, trace_file)`` — npz/npy data
  structure introspection.
- ``_add_metadata_tab(tab_widget, metadata_file)`` — companion JSON
  metadata renderer.

Pure mixin (does NOT inherit from QWidget). The host class is
expected to be a ``QtWidgets.QWidget`` subclass and to provide:

Required state attributes:
    - none directly; the methods operate on file_data dicts and
      tab_widget references passed as arguments.

Required host methods (provided by sibling mixins resolved via MRO):
    - ``self._add_roi_overview_tab(tab_widget, file_data)`` — cluster
      #8 (iter-7 ``gpu_ui_export_viewer_overview.py``); currently
      still on residual ``GPU`` class.
    - ``self._add_interactive_plot_tab(tab_widget, file_data)`` —
      cluster #9 (iter-8); currently still on residual ``GPU``.
    - ``self._add_html_tab(tab_widget, html_file)`` — cluster #9
      (iter-8); currently still on residual ``GPU``.
    - ``self._open_html_in_browser(html_file)`` — cluster #9
      (iter-8); currently still on residual ``GPU``.

The mixin holds the cohesive "viewer skeleton" — the dialog builder
+ file loader + 4 tab builders — while ROI overview (single 195-LOC
method) and the plot/html tabs (cluster #9) are isolated by
responsibility.
"""

from __future__ import annotations

import os

from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget


class ExportViewerMixin:
    """Exported-trace VIEWER dialog core + 4 tab builders.

    See module docstring for the host-class contract.
    """

    def _view_exported_traces(self):

        try:
            from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                                       QLabel, QPushButton, QFileDialog)
            import os


            file_dialog = QFileDialog()
            trace_file, _ = file_dialog.getOpenFileName(
                self,
                "Select Exported ROI Data File",
                ".",
                "ROI Export files (*.npz);;Legacy files (*.npy);;All files (*.*)"
            )

            if not trace_file:
                return


            file_data = self._load_export_file(trace_file)
            if not file_data:
                return


            dialog = QDialog(self)
            dialog.setWindowTitle("ROI Data Viewer")
            dialog.resize(1200, 800)

            layout = QVBoxLayout(dialog)


            file_format = file_data.get('format', 'unknown')
            info_label = QLabel(f"📁 Viewing: {os.path.basename(trace_file)} ({file_format} format)")
            info_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px; background: #e8f4f8;")
            layout.addWidget(info_label)


            tab_widget = QTabWidget()
            layout.addWidget(tab_widget)


            self._add_roi_overview_tab(tab_widget, file_data)


            self._add_interactive_plot_tab(tab_widget, file_data)


            self._add_statistics_tab(tab_widget, file_data)


            self._add_system_info_tab(tab_widget, file_data)


            html_file = trace_file.replace('.npz', '_summary.html').replace('.npy', '_summary.html')
            if os.path.exists(html_file):
                self._add_html_tab(tab_widget, html_file)


            button_layout = QHBoxLayout()


            if os.path.exists(html_file):
                open_html_btn = QPushButton("🌐 Open Full Report in Browser")
                open_html_btn.clicked.connect(lambda: self._open_html_in_browser(html_file))
                button_layout.addWidget(open_html_btn)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            button_layout.addWidget(close_btn)

            layout.addLayout(button_layout)


            dialog.exec_()

        except Exception as e:
            print(f"❌ View exported traces error: {e}")
            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Error viewing exported traces:\\n{str(e)}")
            msg.exec_()

    def _load_export_file(self, file_path):

        try:
            import numpy as np
            import json

            file_data = {'format': 'unknown', 'traces': {}, 'metadata': {}}

            if file_path.endswith('.npz'):

                # Unified exports store the ROI trace_data dict and the *_json
                # blobs as object arrays (np.savez wraps Python objects), so the
                # reader must allow pickle. These are the app's own local export
                # files, written by export_fast.py — trusted input.
                data = np.load(file_path, allow_pickle=True)


                if 'file_format_version' in data and 'unified' in str(data['file_format_version']):
                    file_data['format'] = 'unified_npz'


                    if 'trace_data' in data:
                        trace_data = data['trace_data'].item()
                        for key, trace_array in trace_data.items():
                            if key.startswith('roi_') and key.endswith('_trace'):
                                roi_id = key.replace('roi_', '').replace('_trace', '')
                                file_data['traces'][int(roi_id)] = trace_array


                    def _parse_stored_json(raw_str):
                        """Parse JSON string, with fallback for legacy str() format."""
                        try:
                            return json.loads(raw_str)
                        except (json.JSONDecodeError, TypeError):
                            import ast
                            return ast.literal_eval(raw_str)

                    try:
                        if 'roi_metadata_json' in data:
                            metadata_str = str(data['roi_metadata_json'][0])
                            file_data['metadata'] = _parse_stored_json(metadata_str)

                        if 'export_info_json' in data:
                            export_info_str = str(data['export_info_json'][0])
                            file_data['export_info'] = _parse_stored_json(export_info_str)

                        if 'machine_snapshot_json' in data:
                            machine_str = str(data['machine_snapshot_json'][0])
                            file_data['machine_info'] = _parse_stored_json(machine_str)

                        if 'session_summary_json' in data:
                            session_str = str(data['session_summary_json'][0])
                            file_data['session_info'] = _parse_stored_json(session_str)

                    except Exception as e:
                        print(f"⚠️ Metadata parsing warning: {e}")

                else:

                    file_data['format'] = 'legacy_npz'

                    for key, value in data.items():
                        if isinstance(value, np.ndarray):

                            file_data['traces'][key] = value

            elif file_path.endswith('.npy'):

                file_data['format'] = 'legacy_npy'
                traces = np.load(file_path, allow_pickle=True)

                if isinstance(traces, dict):
                    file_data['traces'] = traces
                else:
                    file_data['traces'] = {'trace_data': traces}


                metadata_file = file_path.replace('.npy', '_metadata.json')
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            companion_data = json.load(f)
                        file_data['metadata'] = companion_data.get('roi_metadata', {})
                        file_data['export_info'] = companion_data.get('export_info', {})
                        file_data['machine_info'] = companion_data.get('machine_snapshot', {})
                        file_data['session_info'] = companion_data.get('session_summary', {})
                    except Exception as e:
                        print(f"⚠️ Companion metadata loading failed: {e}")

            print(f"✅ Loaded {file_data['format']} file with {len(file_data['traces'])} traces")
            return file_data

        except Exception as e:
            print(f"❌ File loading error: {e}")
            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("File Load Error")
            msg.setText(f"Could not load file:\\n{str(e)}")
            msg.exec_()
            return None

    def _add_statistics_tab(self, tab_widget, file_data):

        try:
            import numpy as np
            from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit

            widget = QWidget()
            layout = QVBoxLayout(widget)

            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QtGui.QFont("Courier", 10))


            traces = file_data.get('traces', {})
            metadata = file_data.get('metadata', {})

            stats_text = "=== Detailed ROI Statistics ===\n\n"

            if traces:
                stats_text += f"Total ROIs: {len(traces)}\n\n"

                all_intensities = []
                all_lengths = []

                for roi_id, trace_data in sorted(traces.items()):
                    if hasattr(trace_data, '__len__') and len(trace_data) > 0:
                        trace_array = np.array(trace_data, dtype=np.float32)

                        roi_meta = metadata.get(str(roi_id), {})

                        stats_text += f"ROI {roi_id}:\n"
                        stats_text += f"  Length: {len(trace_array)} points\n"
                        stats_text += f"  Mean: {np.mean(trace_array):.3f}\n"
                        stats_text += f"  Std: {np.std(trace_array):.3f}\n"
                        stats_text += f"  Min: {np.min(trace_array):.3f}\n"
                        stats_text += f"  Max: {np.max(trace_array):.3f}\n"
                        stats_text += f"  Range: {np.max(trace_array) - np.min(trace_array):.3f}\n"


                        cv = np.std(trace_array) / np.mean(trace_array) if np.mean(trace_array) > 0 else 0
                        activity = 'high' if cv > 0.3 else 'moderate' if cv > 0.1 else 'low'
                        stats_text += f"  Activity: {activity} (CV: {cv:.3f})\n"


                        if roi_meta:
                            centroid = roi_meta.get('centroid', [0, 0])
                            size = roi_meta.get('size_pixels', 0)
                            shape = roi_meta.get('shape_info', {}).get('type', 'unknown')
                            stats_text += f"  Location: ({centroid[0]}, {centroid[1]})\n"
                            stats_text += f"  Size: {size} pixels\n"
                            stats_text += f"  Shape: {shape}\n"

                        stats_text += "\n"

                        all_intensities.extend(trace_array)
                        all_lengths.append(len(trace_array))


                if all_intensities:
                    stats_text += "=== Overall Statistics ===\n"
                    stats_text += f"Total data points: {len(all_intensities)}\n"
                    stats_text += f"Global mean intensity: {np.mean(all_intensities):.3f}\n"
                    stats_text += f"Global std intensity: {np.std(all_intensities):.3f}\n"
                    stats_text += f"Average trace length: {np.mean(all_lengths):.1f}\n"
                    stats_text += f"Min trace length: {np.min(all_lengths)}\n"
                    stats_text += f"Max trace length: {np.max(all_lengths)}\n"
            else:
                stats_text += "No trace data available for analysis.\n"

            text_edit.setPlainText(stats_text)
            layout.addWidget(text_edit)

            tab_widget.addTab(widget, "📈 Statistics")

        except Exception as e:
            error_widget = QLabel(f"Error creating statistics: {e}")
            tab_widget.addTab(error_widget, "❌ Statistics")

    def _add_system_info_tab(self, tab_widget, file_data):

        try:
            from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit

            widget = QWidget()
            layout = QVBoxLayout(widget)

            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QtGui.QFont("Courier", 10))


            info_text = "=== System & Session Information ===\n\n"


            export_info = file_data.get('export_info', {})
            if export_info:
                info_text += "Export Information:\n"
                info_text += f"  Timestamp: {export_info.get('datetime', 'Unknown')}\n"
                info_text += f"  Version: {export_info.get('version', 'Unknown')}\n\n"


            machine_info = file_data.get('machine_info', {}) or file_data.get('machine_snapshot', {})
            if machine_info:
                info_text += "Machine Information:\n"
                system = machine_info.get('system', {})
                if system:
                    info_text += f"  Platform: {system.get('platform', 'Unknown')}\n"
                    info_text += f"  Release: {system.get('release', 'Unknown')}\n"
                    info_text += f"  Machine: {system.get('machine', 'Unknown')}\n"
                    info_text += f"  Hostname: {system.get('hostname', 'Unknown')}\n"

                python = machine_info.get('python', {})
                if python:
                    info_text += f"  Python: {python.get('version', 'Unknown')}\n"

                hardware = machine_info.get('hardware', {})
                if hardware:
                    info_text += f"  CPU Cores: {hardware.get('cpu_count', 'Unknown')}\n"
                    info_text += f"  Memory: {hardware.get('memory_total_gb', 0):.1f} GB\n"
                elif machine_info.get('fast_mode'):

                    info_text += "  Fast Mode: Basic info only\n"

                info_text += "\n"


            session_info = (file_data.get('session_info', {}) or
                           file_data.get('session_summary', {}) or
                           file_data.get('session_data', {}))
            if session_info:
                info_text += "Session Information:\n"
                info_text += f"  Extractor Running: {session_info.get('extractor_running', False)}\n"
                info_text += f"  Frames Processed: {session_info.get('frames_processed', 0)}\n"
                info_text += f"  ROIs File: {session_info.get('rois_file', 'Unknown')}\n"
                info_text += f"  Traces File: {session_info.get('traces_file', 'Unknown')}\n"
                info_text += f"  Session ID: {session_info.get('session_id', 'Unknown')}\n"
                info_text += f"  ROI Count: {session_info.get('roi_count', 0)}\n"

            if not any([export_info, machine_info, session_info]):
                info_text += "No system or session information available.\n"

            text_edit.setPlainText(info_text)
            layout.addWidget(text_edit)

            tab_widget.addTab(widget, "🖥️ System Info")

        except Exception as e:
            error_widget = QLabel(f"Error creating system info: {e}")
            tab_widget.addTab(error_widget, "❌ System Info")

    def _add_trace_data_tab(self, tab_widget, trace_file):

        try:
            import numpy as np


            trace_data = np.load(trace_file, allow_pickle=True)

            widget = QWidget()
            layout = QVBoxLayout(widget)

            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QtGui.QFont("Courier", 10))


            info_text = f"""
=== Trace Data Analysis ===

File: {os.path.basename(trace_file)}
File Size: {os.path.getsize(trace_file) / 1024:.1f} KB

Data Structure:
"""

            if isinstance(trace_data, dict):
                info_text += f"Type: Dictionary with {len(trace_data)} keys\\n\\n"
                for key, value in trace_data.items():
                    if isinstance(value, np.ndarray):
                        info_text += f"'{key}': Array shape {value.shape}, dtype {value.dtype}\\n"
                        if len(value) > 0:
                            info_text += f"   Range: {np.min(value):.3f} to {np.max(value):.3f}\\n"
                            info_text += f"   Mean: {np.mean(value):.3f}, Std: {np.std(value):.3f}\\n"
                    else:
                        info_text += f"'{key}': {type(value).__name__}\\n"
                    info_text += "\\n"
            else:
                info_text += f"Type: {type(trace_data).__name__}\\n"
                if isinstance(trace_data, np.ndarray):
                    info_text += f"Shape: {trace_data.shape}\\n"
                    info_text += f"Data type: {trace_data.dtype}\\n"
                    if trace_data.size > 0:
                        info_text += f"Range: {np.min(trace_data):.3f} to {np.max(trace_data):.3f}\\n"
                        info_text += f"Mean: {np.mean(trace_data):.3f}\\n"

            text_edit.setPlainText(info_text)
            layout.addWidget(text_edit)

            tab_widget.addTab(widget, "📊 Trace Data")

        except Exception as e:
            error_widget = QLabel(f"Error loading trace data: {e}")
            tab_widget.addTab(error_widget, "❌ Trace Data")

    def _add_metadata_tab(self, tab_widget, metadata_file):

        try:
            import json

            widget = QWidget()
            layout = QVBoxLayout(widget)

            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QtGui.QFont("Courier", 10))


            with open(metadata_file, 'r') as f:
                metadata = json.load(f)


            info_text = "=== ROI Metadata Summary ===\\n\\n"


            export_info = metadata.get('export_info', {})
            info_text += f"Export Time: {export_info.get('datetime', 'Unknown')}\\n"
            info_text += f"Version: {export_info.get('version', 'Unknown')}\\n\\n"


            roi_metadata = metadata.get('roi_metadata', {})
            info_text += f"=== ROI Details ({len(roi_metadata)} ROIs) ===\\n\\n"

            for roi_id, roi_data in roi_metadata.items():
                info_text += f"ROI {roi_id}:\\n"
                info_text += f"  Location: {roi_data.get('centroid', 'Unknown')}\\n"
                info_text += f"  Size: {roi_data.get('size_pixels', 'Unknown')} pixels\\n"
                info_text += f"  Shape: {roi_data.get('shape_info', {}).get('type', 'Unknown')}\\n"
                info_text += f"  Avg Intensity: {roi_data.get('average_intensity', 0):.2f}\\n"

                activity = roi_data.get('activity_profile', {})
                if activity.get('status') == 'calculated':
                    info_text += f"(Activity: {activity.get('activity_level', 'unknown')})\\n"
                    info_text += f"(CV: {activity.get('coefficient_of_variation', 0):.3f})\\n"

                info_text += "\\n"


            machine_info = metadata.get('machine_snapshot', {})
            if machine_info:
                info_text += "=== System Information ===\\n"
                system = machine_info.get('system', {})
                info_text += f"Platform: {system.get('platform', 'Unknown')} {system.get('release', '')}\\n"

                hardware = machine_info.get('hardware', {})
                if hardware:
                    info_text += f"CPU Cores: {hardware.get('cpu_count', 'Unknown')}\\n"
                    info_text += f"Memory: {hardware.get('memory_total_gb', 0):.1f} GB\\n"

            text_edit.setPlainText(info_text)
            layout.addWidget(text_edit)

            tab_widget.addTab(widget, "🏷️ ROI Metadata")

        except Exception as e:
            error_widget = QLabel(f"Error loading metadata: {e}")
            tab_widget.addTab(error_widget, "❌ Metadata")
