"""Initialization helpers extracted from ``live_trace_extractor``.

Stage-0.6 of the 6-module decomposition (sub-module 5 of 6).
Extracted from ``live_trace_extractor.py``.

Contains the 5 init helpers as a mixin class:
- ``_init_roi_processing(label_path, max_rois, max_points)`` — load
  labels, init ROI buffer state
- ``_limit_cuda_pools()`` — cap cupy memory pools at 256MB each
- ``_init_plotting(plot_widget)`` — wire up plot widget + timer
- ``_detect_camera_fps()`` — auto-detect via 5 strategies
- ``_calculate_update_throttle(max_rois)`` — pure plot throttle calc

The mixin expects the subclass (LiveTraceExtractor) to provide:
- ``self.camera`` — camera module (with FPS hooks)
- ``self.use_pygame_plot`` — bool (skip plotting if pygame mode)
- ``self.ids`` — list[int] (writable; populated by _init_roi_processing)
- ``self.update_plot_signal`` — pyqtSignal() (emitted by plot_timer)
- ``self._setup_single_plot_layout`` / ``_setup_multi_plot_layout`` —
  methods from LiveTracePlotLayoutsMixin (already mixed in)
- Plus ~10 other ROI state attrs that `_init_roi_processing` writes

No behavior change vs the original location.

Safety: smoke tests in ``tests/L3_5_split_first/`` must remain green.
"""

from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt

# CUDA availability — same dance as live_trace_extractor + live_trace_ingest.
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False
    cp = None

# pyqtgraph availability — checked at mixin caller's discretion via PYQTPGRAPH_AVAILABLE
try:
    import pyqtgraph as pg  # noqa: F401
    PYQTPGRAPH_AVAILABLE = True
except Exception:
    PYQTPGRAPH_AVAILABLE = False


class LiveTraceInitMixin:
    """Initialization helpers for ``LiveTraceExtractor``."""

    def _init_roi_processing(self, label_path: str, max_rois: int, max_points: int):
        labels = np.load(label_path)["labels"].astype(np.int32)
        if labels.ndim != 2:
            raise ValueError("labels must be 2D")
        self._labels_orig = labels
        self._roi_max = int(labels.max(initial=0))
        self._max_rois_cfg = max_rois
        self._max_points_cfg = max_points

        self._roi_ready = False

        self._ids_gpu = None
        self._roi_sizes_gpu = None
        self._f_gpu = None
        self._roi_sizes_cpu = None
        self._flat_labels_cpu = None
        self._max_label = 0
        self.ids = []

    def _limit_cuda_pools(self):
        try:
            mempool = cp.get_default_memory_pool()
            if hasattr(mempool, "set_limit"):
                mempool.set_limit(size=2**28)  # 256MB
                print("✅ CUDA memory pool limit set to 256MB")
            pmp = cp.get_default_pinned_memory_pool()
            if hasattr(pmp, "set_limit"):
                pmp.set_limit(size=2**28)
                print("✅ CUDA pinned memory pool limit set to 256MB")
        except Exception as e:
            print(f"Could not set CUDA pool limits: {e}")


    def _init_plotting(self, plot_widget=None):
        self._legend = None
        if self.use_pygame_plot:
            return
        if plot_widget is not None and PYQTPGRAPH_AVAILABLE:
            roi_count = len(self.ids)
            print(f"🎨 Setting up optimized plotting for {roi_count} ROIs...")


            if roi_count <= 20:
                self._setup_single_plot_layout(plot_widget, roi_count)
            else:
                self._setup_multi_plot_layout(plot_widget, roi_count)

        from PyQt5.QtCore import QTimer
        self._plot_timer = QTimer(self)


        camera_fps = self._detect_camera_fps()
        self._last_fps_est = camera_fps
        # Cap plot updates at ~15 Hz regardless of camera FPS. The Qt main
        # thread does pyqtgraph setData per ROI here; at camera-matched 30–60 Hz
        # with many ROIs each tick exceeds its budget, which saturates the event
        # loop — that is what causes "STIMviewer not responding" popups and the
        # delayed pagination/dialog appearance during trace extraction. 15 Hz is
        # the human-eye upper bound for following a trace; faster doesn't help.
        # Frame-level processing decimation is independent (_update_every_n).
        plot_interval_ms = max(int(1000 / camera_fps), 67)

        self._plot_timer.setInterval(plot_interval_ms)
        self._plot_timer.timeout.connect(lambda: self.update_plot_signal.emit(), Qt.QueuedConnection)
        self._plot_timer.start()
        print(f"✅ Plot timer: {plot_interval_ms}ms (≈{1000/plot_interval_ms:.1f} Hz, capped from {camera_fps:.1f} fps camera)")

    def _detect_camera_fps(self):

        try:

            if hasattr(self.camera, 'get_actual_fps'):
                fps = self.camera.get_actual_fps()
                if fps and fps > 0:
                    print(f"🎥 Camera FPS detected via get_actual_fps(): {fps:.1f}")
                    return float(fps)


            if hasattr(self.camera, 'node_map') and self.camera.node_map:
                try:
                    fps_node = self.camera.node_map.FindNode("AcquisitionFrameRate")
                    if fps_node and fps_node.IsReadable():
                        fps = float(fps_node.Value())
                        if fps > 0:
                            print(f"🎥 Camera FPS detected via node map: {fps:.1f}")
                            return fps
                except Exception as e:
                    print(f"⚠️ Node map FPS detection failed: {e}")


            fps_attrs = ['fps', 'framerate', 'frame_rate', 'acquisition_fps']
            for attr in fps_attrs:
                if hasattr(self.camera, attr):
                    try:
                        fps = getattr(self.camera, attr)
                        if fps and fps > 0:
                            print(f"🎥 Camera FPS detected via {attr}: {fps:.1f}")
                            return float(fps)
                    except Exception:
                        pass


            if hasattr(self.camera, 'get_fps'):
                try:
                    fps = self.camera.get_fps()
                    if fps and fps > 0:
                        print(f"🎥 Camera FPS detected via get_fps(): {fps:.1f}")
                        return float(fps)
                except Exception:
                    pass


            print("⚠️ Could not detect camera FPS, using 30 fps default")
            return 30.0

        except Exception as e:
            print(f"❌ Camera FPS detection error: {e}, using 30 fps default")
            return 30.0

    def _calculate_update_throttle(self, max_rois):

        if max_rois <= 10:
            return 2
        elif max_rois <= 25:
            return 3
        elif max_rois <= 50:
            return 5
        else:
            return 8


__all__ = ["LiveTraceInitMixin"]
