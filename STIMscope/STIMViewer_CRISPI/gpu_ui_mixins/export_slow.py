"""SlowExportMixin — extracted from ``gpu_ui.py`` per L5 SPLIT-FIRST.

Cluster #6 of the 9-sub-module decomposition (see
``docs/specs/L5_UI/gpu_ui.md`` §0.5). Contains the 9 methods that
implement the **SLOW export path** — full-fidelity machine snapshot,
detailed ROI metadata with shape estimation + activity profiling,
session summary, calibration info, and HTML summary generation:

- ``_get_machine_snapshot()`` — full platform + CPU + memory +
  per-process stats (vs FAST mode's abbreviated version)
- ``_get_camera_info()`` — actual_fps + GenICam node-map reads
  (vs FAST mode's basic exposure/gain/fps reads)
- ``_extract_roi_metadata()`` — per-ROI centroid + size + shape +
  activity profile + mask reference (vs FAST mode's centroid +
  bbox only)
- ``_estimate_roi_shape(roi_locations)`` — bbox + circularity +
  aspect ratio + shape-type classification
- ``_calculate_activity_profile(roi_id)`` — per-ROI trace stats +
  coefficient-of-variation activity-level classification
- ``_get_session_summary()`` — extractor state + per-ROI buffer
  lengths
- ``_get_calibration_info()`` — framework-ready stub
- ``_save_enhanced_metadata(export_data)`` — JSON metadata writer
  + HTML summary dispatcher
- ``_generate_html_summary(export_data, html_file)`` — multi-section
  HTML summary builder (ROI grid + system info + session summary)

Pure mixin (does NOT inherit from QWidget). The host class is
expected to be a ``QtWidgets.QWidget`` subclass and to provide the
following host contract:

Required state attributes:
    - ``self.camera`` — IDS Peak camera handle (read: ``acquisition_running``,
      ``get_actual_fps``, ``node_map`` GenICam interface)
    - ``self.live_extractor`` — ``LiveTraceExtractor`` or ``None``
      (read: ``_labels_orig``, ``buffers``, ``_frame_count``, ``ids``)
    - ``self.rois_path: str`` — ROI NPZ path
    - ``self.trace_path: str`` — trace file path

Required host methods (provided by either the residual ``GPU`` class
or sibling mixins):
    - ``self._get_unified_roi_colors()`` — palette getter from
      ``FastExportMixin`` (cluster #5, iter-4).

D-gu-4 note (spec §12): The SLOW path is now structurally separate
from FAST. Stage-5 reconciliation will lift shared logic (e.g.,
machine snapshot, camera info, session summary) into a common base
helper module after both halves have landed.
"""

from __future__ import annotations

import numpy as np


# Mirror gpu_ui.py module-top constant; the SLOW path's metadata +
# HTML summary path-build references it via string substitution.
# (Reproduced here so the mixin is self-contained and avoids a
# circular import on ``gpu_ui``.)
TRACE_OUT = "live_traces.npy"


class SlowExportMixin:
    """SLOW (full-fidelity) trace-export pipeline + HTML summary.

    See module docstring for the host-class contract.
    """

    def _get_machine_snapshot(self):

        import platform
        import os

        snapshot = {
            'system': {
                'platform': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'hostname': platform.node()
            },
            'python': {
                'version': platform.python_version(),
                'implementation': platform.python_implementation()
            },
            'environment': {
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                'pythonpath': os.environ.get('PYTHONPATH', '')
            }
        }


        try:
            import psutil
            snapshot['hardware'] = {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3)
            }


            process = psutil.Process()
            snapshot['process'] = {
                'memory_mb': process.memory_info().rss / (1024**2),
                'cpu_percent': process.cpu_percent()
            }
        except ImportError:
            snapshot['hardware_note'] = 'psutil not available for detailed hardware info'

        return snapshot

    def _get_camera_info(self):

        camera_info = {
            'acquisition_running': getattr(self.camera, 'acquisition_running', False)
        }


        try:
            if hasattr(self.camera, 'get_actual_fps'):
                camera_info['actual_fps'] = self.camera.get_actual_fps()

            if hasattr(self.camera, 'node_map'):
                try:
                    fps_node = self.camera.node_map.FindNode("AcquisitionFrameRate")
                    if fps_node:
                        camera_info['configured_fps'] = float(fps_node.Value())


                    gain_node = self.camera.node_map.FindNode("Gain")
                    if gain_node:
                        camera_info['gain'] = float(gain_node.Value())
                except Exception:
                    pass
        except Exception:
            pass

        return camera_info

    def _extract_roi_metadata(self):

        roi_metadata = {}

        if not self.live_extractor or not hasattr(self.live_extractor, '_labels_orig'):
            return roi_metadata

        try:
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


                shape_info = self._estimate_roi_shape(roi_locations)


                avg_intensity = 0.0
                if hasattr(self.live_extractor, 'buffers') and roi_id in self.live_extractor.buffers:
                    buffer = list(self.live_extractor.buffers[roi_id])
                    if buffer:
                        avg_intensity = float(np.mean(buffer))


                activity_profile = self._calculate_activity_profile(roi_id)

                roi_metadata[int(roi_id)] = {
                    'roi_index': int(roi_id),
                    'centroid': [center_x, center_y],
                    'size_pixels': size,
                    'shape_info': shape_info,
                    'color': colors[i % len(colors)],
                    'average_intensity': avg_intensity,
                    'activity_profile': activity_profile,
                    'mask_reference': {
                        'main_mask_file': self.rois_path,
                        'roi_id_in_mask': int(roi_id)
                    }
                }

        except Exception as e:
            print(f"⚠️ ROI metadata extraction error: {e}")

        return roi_metadata

    def _estimate_roi_shape(self, roi_locations):

        if len(roi_locations[0]) < 5:
            return {'type': 'small', 'circularity': 0.0, 'aspect_ratio': 1.0}

        try:

            coords = np.column_stack(roi_locations)


            min_y, min_x = np.min(coords, axis=0)
            max_y, max_x = np.max(coords, axis=0)

            width = max_x - min_x + 1
            height = max_y - min_y + 1
            aspect_ratio = float(width) / float(height) if height > 0 else 1.0


            area = len(coords)
            perimeter_approx = 2 * np.sqrt(np.pi * area)
            circularity = 4 * np.pi * area / (perimeter_approx * perimeter_approx) if perimeter_approx > 0 else 0


            shape_type = "irregular"
            if circularity > 0.7:
                shape_type = "circular"
            elif aspect_ratio > 2.0 or aspect_ratio < 0.5:
                shape_type = "elongated"
            else:
                shape_type = "oval"

            return {
                'type': shape_type,
                'circularity': float(circularity),
                'aspect_ratio': float(aspect_ratio),
                'bounding_box': [int(min_x), int(min_y), int(width), int(height)]
            }

        except Exception as e:
            return {'type': 'unknown', 'error': str(e)}

    def _calculate_activity_profile(self, roi_id):

        if not hasattr(self.live_extractor, 'buffers') or roi_id not in self.live_extractor.buffers:
            return {'status': 'no_data'}

        try:
            buffer = list(self.live_extractor.buffers[roi_id])
            if not buffer:
                return {'status': 'empty_buffer'}

            traces = np.array(buffer)
            profile = {
                'status': 'calculated',
                'length': len(traces),
                'mean': float(np.mean(traces)),
                'std': float(np.std(traces)),
                'min': float(np.min(traces)),
                'max': float(np.max(traces)),
                'range': float(np.max(traces) - np.min(traces))
            }


            cv = profile['std'] / profile['mean'] if profile['mean'] > 0 else 0
            if cv < 0.1:
                profile['activity_level'] = 'low'
            elif cv < 0.3:
                profile['activity_level'] = 'moderate'
            else:
                profile['activity_level'] = 'high'

            profile['coefficient_of_variation'] = float(cv)

            return profile

        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def _get_session_summary(self):

        summary = {
            'rois_file': self.rois_path,
            'traces_file': self.trace_path
        }

        if self.live_extractor:
            summary.update({
                'extractor_running': True,
                'frames_processed': getattr(self.live_extractor, '_frame_count', 0),
                'total_rois': len(getattr(self.live_extractor, 'ids', [])),
                'buffer_lengths': {}
            })


            if hasattr(self.live_extractor, 'buffers'):
                for roi_id, buffer in self.live_extractor.buffers.items():
                    summary['buffer_lengths'][roi_id] = len(buffer)
        else:
            summary['extractor_running'] = False

        return summary

    def _get_calibration_info(self):

        return {
            'status': 'framework_ready',
            'note': 'Calibration system ready for implementation'
        }

    def _save_enhanced_metadata(self, export_data):

        import json


        metadata_file = TRACE_OUT.replace('.npy', '_metadata.json')
        try:
            with open(metadata_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"✅ Metadata saved: {metadata_file}")
        except Exception as e:
            print(f"❌ Metadata save error: {e}")


        html_file = TRACE_OUT.replace('.npy', '_summary.html')
        try:
            self._generate_html_summary(export_data, html_file)
            print(f"✅ HTML summary generated: {html_file}")
        except Exception as e:
            print(f"❌ HTML generation error: {e}")

    def _generate_html_summary(self, export_data, html_file):

        import os

        roi_metadata = export_data.get('roi_metadata', {})
        machine_info = export_data.get('machine_snapshot', {})
        session_info = export_data.get('session_summary', {})

        html_content = f"""<!DOCTYPE html>
<html><head><title>ROI Export Summary</title><style>
body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}.container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
h1, h2 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 5px; }}.roi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}.roi-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; background: #f9f9f9; }}.roi-header {{ font-weight: bold; color: #007acc; margin-bottom: 10px; }}.metadata {{ font-family: monospace; font-size: 0.9em; }}.stats {{ background: #e8f4f8; padding: 10px; border-radius: 3px; margin: 10px 0; }}
</style></head><body><div class="container">
<h1>🔬 ROI Trace Export Summary</h1>
<div class="stats">
<strong>Export Time:</strong> {export_data.get('export_info', {}).get('datetime', 'Unknown')}<br/>
<strong>Total ROIs:</strong> {len(roi_metadata)}<br/>
<strong>Traces File:</strong> {os.path.basename(TRACE_OUT)}<br/>
<strong>System:</strong> {machine_info.get('system', {}).get('platform', 'Unknown')} {machine_info.get('system', {}).get('release', '')}
</div><h2>📊 ROI Details</h2><div class="roi-grid">"""


        for roi_id, roi_data in roi_metadata.items():
            activity = roi_data.get('activity_profile', {})
            shape_info = roi_data.get('shape_info', {})

            html_content += f"""<div class="roi-card" style="border-left: 4px solid {roi_data.get('color', '#ccc')}">
<div class="roi-header">ROI {roi_id}</div><div class="metadata">
<strong>Location:</strong> ({roi_data.get('centroid', [0, 0])[0]}, {roi_data.get('centroid', [0, 0])[1]})<br/>
<strong>Size:</strong> {roi_data.get('size_pixels', 0)} pixels<br/>
<strong>Shape:</strong> {shape_info.get('type', 'unknown')} (circularity: {shape_info.get('circularity', 0):.2f})<br/>
<strong>Avg Intensity:</strong> {roi_data.get('average_intensity', 0):.1f}<br/>
<strong>Activity:</strong> {activity.get('activity_level', 'unknown')} (CV: {activity.get('coefficient_of_variation', 0):.3f})
</div></div>"""

        html_content += f"""</div><h2>🖥️ System Information</h2><div class="metadata">
<strong>Platform:</strong> {machine_info.get('system', {}).get('platform', 'Unknown')}<br/>
<strong>Python:</strong> {machine_info.get('python', {}).get('version', 'Unknown')}<br/>
<strong>CPU Cores:</strong> {machine_info.get('hardware', {}).get('cpu_count', 'Unknown')}<br/>
<strong>Memory:</strong> {machine_info.get('hardware', {}).get('memory_total_gb', 0):.1f} GB
</div><h2>📈 Session Summary</h2><div class="metadata">
<strong>Extractor Running:</strong> {session_info.get('extractor_running', False)}<br/>
<strong>Frames Processed:</strong> {session_info.get('frames_processed', 0)}<br/>
<strong>ROIs File:</strong> {session_info.get('rois_file', 'Unknown')}
</div></div></body></html>"""

        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
