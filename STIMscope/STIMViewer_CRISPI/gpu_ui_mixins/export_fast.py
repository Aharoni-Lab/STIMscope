"""FastExportMixin — extracted from ``gpu_ui.py`` per L5 SPLIT-FIRST.

Cluster #5 of the 9-sub-module decomposition (see
``docs/specs/L5_UI/gpu_ui.md`` §0.5). Contains the 10 methods that
implement the **FAST export path** — threaded export worker,
comprehensive-export aggregator, ROI color palette, and FAST-mode
versions of each metadata gatherer:

- ``_export_traces()`` — Qt-button slot; spawns a ``QThread`` +
  ``ExportWorker(QObject)`` that runs the unified export off the
  GUI thread, then re-enters the main thread via signals.
- ``_generate_comprehensive_export_data(fast_mode=False)`` —
  aggregator; dispatches to FAST or SLOW gatherers based on
  ``fast_mode``.
- ``_get_unified_roi_colors()`` — 30-entry hex color palette.
- ``get_roi_color(roi_id, total_rois=None)`` — public color lookup.
- ``_get_machine_snapshot_fast()`` — platform + CPU + mem.
- ``_get_camera_info_fast()`` — exposure/gain/fps from camera handle.
- ``_get_calibration_info_fast()`` — homography file path.
- ``_extract_roi_metadata_fast()`` — per-ROI centroid + bbox + color.
- ``_get_session_summary_fast()`` — extractor state summary.
- ``_create_unified_export_file(export_data)`` — packs trace data
  + metadata into a unified ``.npz``, with fallback to a basic
  ``roi_basic_export_*.npz`` on failure.

Pure mixin (does NOT inherit from QWidget). The host class is
expected to be a ``QtWidgets.QWidget`` subclass and to provide the
following host contract:

Required state attributes:
    - ``self.camera`` — IDS Peak camera handle (read-only here)
    - ``self.live_extractor`` — ``LiveTraceExtractor`` or ``None``
      (read: ``buffers``, ``stats``, ``_labels_orig``)
    - ``self.rois_path: str`` — read for session summary

Required host methods (provided by either the residual ``GPU`` class
or sibling mixins):
    - ``self._handle_error(error, context)`` — from residual GPU.
    - SLOW-cluster mirrors when ``fast_mode=False``:
      ``self._get_machine_snapshot``, ``self._get_camera_info``,
      ``self._extract_roi_metadata``, ``self._get_session_summary``,
      ``self._get_calibration_info``, ``self._generate_html_summary``
      — these are provided by ``SlowExportMixin`` once cluster #6
      lands (currently still on the residual ``GPU`` class).

Note on D-gu-4 (spec §12): the FAST/SLOW duplication is intentional
at the extraction stage. The split makes the duplication structurally
visible; stage-5 reconciliation will lift shared helpers up into a
common base.

The mixin does NOT install any ``@pyqtSlot`` decorator on
``_export_traces`` (the residual host wires the "Export Traces"
QPushButton's clicked signal to ``self._export_traces`` directly —
the slot is implicit).
"""

from __future__ import annotations

import os
import time

import numpy as np


class FastExportMixin:
    """FAST trace-export pipeline + ROI color palette.

    See module docstring for the host-class contract.
    """

    def _export_traces(self):

        try:
            if not self.live_extractor:
                print("Live trace extractor is not running.")
                return


            from PyQt5.QtCore import QThread, QObject, pyqtSignal

            class ExportWorker(QObject):
                finished = pyqtSignal(str, str)
                failed = pyqtSignal(str)

                def __init__(self, outer):
                    super().__init__()
                    self.outer = outer

                def run(self):
                    try:
                        print("📊 Generating export metadata (optimized)...")
                        export_data = self.outer._generate_comprehensive_export_data(fast_mode=True)
                        unified_file = self.outer._create_unified_export_file(export_data)
                        print("🌐 Generating detailed HTML summary...")
                        html_export_data = self.outer._generate_comprehensive_export_data(fast_mode=False)
                        html_file = unified_file.replace('.npz', '_summary.html')
                        self.outer._generate_html_summary(html_export_data, html_file)
                        self.finished.emit(unified_file, html_file)
                    except Exception as e:
                        self.failed.emit(str(e))

            self._export_thread = QThread(self)
            self._export_worker = ExportWorker(self)
            self._export_worker.moveToThread(self._export_thread)
            self._export_thread.started.connect(self._export_worker.run)

            def on_finished(unified_file, html_file):
                print("✅ Unified export completed:")
                print(f"   📦 Complete Data: {unified_file}")
                print(f"   🌐 Visual Summary: {html_file}")
                print("   ℹ️  Use 'View Exported Traces' to load the .npz file")
                self._export_thread.quit()
                self._export_thread.wait(100)

            def on_failed(msg):
                self._handle_error(Exception(msg), "Unified trace export")
                self._export_thread.quit()
                self._export_thread.wait(100)

            self._export_worker.finished.connect(on_finished)
            self._export_worker.failed.connect(on_failed)
            self._export_thread.start()

        except Exception as e:
            self._handle_error(e, "Unified trace export")

    def _generate_comprehensive_export_data(self, fast_mode=False):

        import time

        export_data = {
            'export_info': {
                'timestamp': time.time(),
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0.0'
            }
        }

        if fast_mode:

            print("⚡ Fast export mode - essential data only")
            export_data.update({
                'machine_snapshot': self._get_machine_snapshot_fast(),
                'camera_info': self._get_camera_info_fast(),
                'roi_metadata': self._extract_roi_metadata_fast(),
                'session_summary': self._get_session_summary_fast(),
                'calibration_info': self._get_calibration_info_fast()
            })
        else:

            export_data.update({
                'machine_snapshot': self._get_machine_snapshot(),
                'camera_info': self._get_camera_info(),
                'roi_metadata': self._extract_roi_metadata(),
                'session_summary': self._get_session_summary(),
                'calibration_info': self._get_calibration_info()
            })

        return export_data

    def _get_unified_roi_colors(self):


        return [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#FFA07A', '#87CEEB', '#DEB887',
            '#FF9F43', '#10AC84', '#EE5A24', '#0084FF', '#341F97',
            '#F8B500', '#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E',
            '#E17055', '#00B894', '#00CECE', '#2D3436', '#636E72',
            '#FAB1A0', '#74B9FF', '#55A3FF', '#FF7675', '#6C5CE7',
        ]

    def get_roi_color(self, roi_id, total_rois=None):

        colors = self._get_unified_roi_colors()


        color_index = (roi_id - 1) % len(colors)
        return colors[color_index]

    def _get_machine_snapshot_fast(self):

        import platform
        import psutil

        return {
            'fast_mode': True,
            'timestamp': time.time(),
            'system': {
                'platform': platform.system(),
                'release': platform.release(),
                'machine': platform.machine(),
                'hostname': platform.node()
            },
            'python': {
                'version': platform.python_version()
            },
            'hardware': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3)
            }
        }

    def _get_camera_info_fast(self):

        camera_info = {'fast_mode': True}
        try:
            if hasattr(self.camera, 'get_exposure'):
                camera_info['exposure'] = self.camera.get_exposure()
            if hasattr(self.camera, 'get_gain'):
                camera_info['gain'] = self.camera.get_gain()
            if hasattr(self.camera, 'get_fps'):
                camera_info['fps'] = self.camera.get_fps()
        except Exception:
            pass
        return camera_info

    def _get_calibration_info_fast(self):

        return {
            'fast_mode': True,
            'homography_file': getattr(self.camera, 'translation_matrix_path', 'Unknown'),
            'timestamp': time.time()
        }

    def _extract_roi_metadata_fast(self):

        try:
            roi_metadata = {}

            if not self.live_extractor or not hasattr(self.live_extractor, '_labels_orig'):
                return roi_metadata

            labels = self.live_extractor._labels_orig
            unique_ids = np.unique(labels)
            roi_ids = unique_ids[unique_ids > 0]

            colors = self._get_unified_roi_colors()

            for i, roi_id in enumerate(roi_ids):
                roi_mask = (labels == roi_id)
                roi_locations = np.where(roi_mask)

                if len(roi_locations[0]) == 0:
                    continue


                center_y = int(np.mean(roi_locations[0]))
                center_x = int(np.mean(roi_locations[1]))
                size = int(np.sum(roi_mask))


                avg_intensity = 0.0
                if hasattr(self.live_extractor, 'buffers') and roi_id in self.live_extractor.buffers:
                    buffer = list(self.live_extractor.buffers[roi_id])
                    if buffer:
                        avg_intensity = float(np.mean(buffer))


                bbox_height = np.max(roi_locations[0]) - np.min(roi_locations[0]) + 1
                bbox_width = np.max(roi_locations[1]) - np.min(roi_locations[1]) + 1
                aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0

                roi_metadata[int(roi_id)] = {
                    'roi_index': int(roi_id),
                    'centroid': [center_x, center_y],
                    'size_pixels': size,
                    'size': size,
                    'shape_info': {
                        'type': 'compact' if aspect_ratio < 1.5 else 'elongated',
                        'aspect_ratio': aspect_ratio
                    },
                    'color': colors[i % len(colors)],
                    'average_intensity': avg_intensity,
                    'fast_mode': True
                }

            return roi_metadata

        except Exception as e:
            print(f"⚠️ Fast ROI metadata extraction error: {e}")
            return {}

    def _get_session_summary_fast(self):

        try:
            frames_processed = 0
            if self.live_extractor and hasattr(self.live_extractor, 'stats'):
                frames_processed = self.live_extractor.stats.get('frames_processed', 0)

            summary = {
                'extractor_running': self.live_extractor is not None,
                'roi_count': len(self.live_extractor.buffers) if self.live_extractor else 0,
                'frames_processed': frames_processed,
                'rois_file': os.path.basename(self.rois_path) if hasattr(self, 'rois_path') and self.rois_path else 'Unknown',
                'traces_file': 'Live traces (in memory)',
                'fast_mode': True,
                'timestamp': time.time()
            }
            return summary
        except Exception as e:
            print(f"⚠️ Fast session summary error: {e}")
            return {'fast_mode': True, 'error': str(e)}

    def _create_unified_export_file(self, export_data):

        import time
        import json
        import numpy as np


        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unified_file = f"roi_complete_export_{timestamp}.npz"

        try:

            trace_data = {}
            trace_metadata = {}

            if self.live_extractor and hasattr(self.live_extractor, 'buffers'):
                print("📊 Collecting ALL ROI trace data for export...")


                all_roi_ids = sorted(self.live_extractor.buffers.keys())
                collected_count = 0
                empty_count = 0

                for roi_id in all_roi_ids:
                    buffer = self.live_extractor.buffers.get(roi_id, [])

                    if buffer and len(buffer) > 0:

                        trace_array = np.asarray(buffer, dtype=np.float32)
                        trace_data[f'roi_{roi_id}_trace'] = trace_array


                        trace_metadata[f'roi_{roi_id}_info'] = {
                            'length': len(trace_array),
                            'mean': float(trace_array.mean()),
                            'std': float(trace_array.std()),
                            'min': float(trace_array.min()),
                            'max': float(trace_array.max()),
                            'has_data': True
                        }
                        collected_count += 1
                    else:

                        trace_data[f'roi_{roi_id}_trace'] = np.array([], dtype=np.float32)
                        trace_metadata[f'roi_{roi_id}_info'] = {
                            'length': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                            'has_data': False, 'roi_id': int(roi_id)
                        }
                        empty_count += 1

                print(f"✅ Collected ALL {len(trace_data)} ROI traces: {collected_count} with data, {empty_count} empty")


            unified_data = {

                'trace_data': trace_data,
                'trace_stats': trace_metadata,


                'export_info_json': np.array([json.dumps(export_data.get('export_info', {}), default=str)]),
                'machine_snapshot_json': np.array([json.dumps(export_data.get('machine_snapshot', {}), default=str)]),
                'camera_info_json': np.array([json.dumps(export_data.get('camera_info', {}), default=str)]),
                'roi_metadata_json': np.array([json.dumps(export_data.get('roi_metadata', {}), default=str)]),
                'session_summary_json': np.array([json.dumps(export_data.get('session_summary', {}), default=str)]),
                'calibration_info_json': np.array([json.dumps(export_data.get('calibration_info', {}), default=str)]),


                'file_format_version': np.array(['unified_v1.0']),
                'creation_timestamp': np.array([time.time()]),
                'readable_timestamp': np.array([time.strftime('%Y-%m-%d %H:%M:%S')])
            }


            np.savez_compressed(unified_file, **unified_data)

            print(f"✅ Unified file created: {unified_file}")
            print(f"   Contains: {len(trace_data)} ROI traces + complete metadata")

            return unified_file

        except Exception as e:
            print(f"❌ Unified export creation failed: {e}")

            fallback_file = f"roi_basic_export_{timestamp}.npz"
            np.savez_compressed(fallback_file,
                               traces=list(self.live_extractor.buffers.values()) if self.live_extractor else [],
                               roi_ids=list(self.live_extractor.buffers.keys()) if self.live_extractor else [],
                               error_info=str(e))
            return fallback_file
