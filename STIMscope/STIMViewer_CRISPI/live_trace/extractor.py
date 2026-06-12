from __future__ import annotations

import gc
import time
import queue
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Tuple

import numpy as np
import psutil
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "pkg_resources is deprecated", DeprecationWarning)
import pygame
import cv2

from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, Qt
from PyQt5.QtGui import QImage
try:
    import pyqtgraph as pg
    PYQTPGRAPH_AVAILABLE = True
except Exception:
    PYQTPGRAPH_AVAILABLE = False
    pg = None

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False
    cp = None

# Determine if CUDA runtime is actually usable (driver/runtime compatible)
CUDA_USABLE = False
if CUDA_AVAILABLE:
    try:
        # Avoid memory pool calls; just query device count to validate runtime
        import cupy.cuda.runtime as _cur
        ndev = _cur.getDeviceCount()
        if ndev and ndev > 0:
            # Optional light op to catch driver/runtime mismatches without heavy alloc
            _ = cp.arange(1, dtype=cp.int8)
            CUDA_USABLE = True
            print("✅ CUDA runtime usable for live_trace_extractor")
        else:
            print("ℹ️ No CUDA devices found; CPU path will be used")
    except Exception as e:
        CUDA_USABLE = False
        print(f"⚠️ CUDA import succeeded but runtime is unusable; CPU path will be used: {e}")
else:
    print("ℹ️ CUDA not available for live_trace_extractor; CPU path will be used")

# Performance + sync infrastructure extracted to live_trace_perf.py
# (Re-exported
# here for backward compatibility with any caller doing
# `from live_trace.extractor import PerformanceMonitor` etc.
from live_trace.perf import (
    MAX_FRAME_QUEUE_SIZE,
    qimage_to_gray_np,
    PerformanceMonitor,
    SyncState,
    SyncInfo,
    FrameProcessor,
)

THREAD_POOL_SIZE = 1
SYNCHRONIZATION_TIMEOUT = 3.0



from live_trace.plot_layouts import LiveTracePlotLayoutsMixin
from live_trace.ingest import LiveTraceIngestMixin
from live_trace.init import LiveTraceInitMixin
from live_trace.processing import LiveTraceProcessingMixin
from live_trace.plot_modes import LiveTracePlotModesMixin
from live_trace.plot_aggregation import LiveTracePlotAggregationMixin
from live_trace.plot_pagination import LiveTracePlotPaginationMixin


class LiveTraceExtractor(
    LiveTraceInitMixin,
    LiveTraceIngestMixin,
    LiveTraceProcessingMixin,
    LiveTracePlotModesMixin,
    LiveTracePlotAggregationMixin,
    LiveTracePlotPaginationMixin,
    LiveTracePlotLayoutsMixin,
    QObject,
):
    update_plot_signal = pyqtSignal()
    sync_state_changed = pyqtSignal(SyncInfo)
    performance_update = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
         self,
        camera,
        label_path,
        plot_widget=None,
        max_points: int = 300,
        max_rois: int = 6,
        use_pygame_plot: bool = False,
        enable_sync: bool = False,
    ):
        super().__init__()
        
        self.camera = camera
        self.use_pygame_plot = bool(use_pygame_plot)
        self.enable_sync = bool(enable_sync)

        self._camera_signal_refs: List[Tuple[object, callable]] = []
        self._cleanup_event = threading.Event()
        self.plot_widget = None
        self._plot_curves = {}
        self._plot_timer = None
        self._x_mode_seconds = False  # False=frames, True=seconds
        self._last_fps_est = 30.0
        self._global_frame_index = 0  # monotonically increasing sample index for x-axis
        self._oasis_enabled = False
        self._oasis_gamma = 0.95  # default decay; can be tuned
        self._oasis_lambda = 0.0  # default sparsity; 0 -> ML
        self._oasis_prev_c: Dict[int, float] = {}
        self._oasis_prev_s: Dict[int, float] = {}
        self._plot_norm_mode: str = "Raw"   # Raw | ΔF/F₀ | z-score | Spikes
        self._dff_buffers: Dict[int, deque] = {}
        self._spike_buffers: Dict[int, deque] = {}
        self._baseline_window_s: float = 10.0
        self._baseline_percentile: float = 20.0
        self._neuropil_r: float = 0.0
        self._neuropil_inner_gap: int = 2
        self._neuropil_ring_width: int = 10
        self._npil_labels_flat_cpu: Optional[np.ndarray] = None
        self._npil_sizes_cpu: Optional[np.ndarray] = None
        self._npil_labels_gpu = None
        self._npil_sizes_gpu = None

        # IDs highlighted in the per-ROI plots
        self._highlight_ids: Set[int] = set()
        self._labels_gpu = None

        self._frame_count = 0
        
        self._max_rois_cfg = max_rois
        self._update_every_n = self._calculate_update_throttle(max_rois)
        
        if max_rois <= 10:
            self._process_every_n = 1   
        elif max_rois <= 25:
            self._process_every_n = 2   
        elif max_rois <= 50:
            self._process_every_n = 3   
        else:
            self._process_every_n = 5  
        
        print(f"🚀 Performance optimized: update_throttle={self._update_every_n}, process_throttle={self._process_every_n} for {max_rois} ROIs")

        self.start_time = time.time()
        self.stats = {
            "frames_processed": 0,
            "frames_failed": 0,
            "memory_usage_peak": 0.0,
            "uptime_seconds": 0.0,
            "last_frame_time": 0.0,
            "gpu_memory_peak": 0.0,
            "sync_operations": 0,
            "sync_failures": 0,
        }

        self._sync_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._gpu_lock = threading.Lock()

        self._sync_state = SyncState.IDLE
        self._syncprint = SyncInfo(self._sync_state, time.time(), 0, 0.0, 0.0, None)


        self.ids: np.ndarray = np.array([], dtype=np.int32)
        self.buffers: Dict[int, deque] = {}
        self._cpu_masks: Optional[List[np.ndarray]] = None  # list of boolean 1D masks
        self.mask_mat = None       
        self.roi_sizes = None        
        self._f_gpu = None           
        self._H = 0
        self._W = 0

        self.export_counter = 0



        self.update_plot_signal.connect(self._update_plot, Qt.QueuedConnection)
        if self.ids.size == 0:
            print("⚠️ No positive ROI labels found in labels array; running in empty-safe mode")

            self.ids = np.array([], dtype=np.int32)
            self.buffers = {}


        self._init_roi_processing(label_path, max_rois=max_rois, max_points=max_points)


        self._init_plotting(plot_widget)
        # Note: update_plot_signal was already connected above (QueuedConnection).
        # A second connect here meant _update_plot fired twice per timer tick,
        # doubling all main-thread render work. Removed.



        self.frame_processor = FrameProcessor(max_workers=THREAD_POOL_SIZE)
        self.frame_processor.frame_processed.connect(self._on_frame_processed, Qt.QueuedConnection)
        self.frame_processor.error_occurred.connect(self._on_processing_error, Qt.QueuedConnection)
        self.frame_processor.start()

        # Expose counts for UI (total ROIs, plotted cap)
        try:
            self.total_rois_extracted = int(self._roi_max)
        except Exception:
            self.total_rois_extracted = 0



        self._connect_camera_signals()

        self._update_sync_state(SyncState.INITIALIZING)
        print("🚀 LiveTraceExtractor initialized")

    def set_oasis_enabled(self, enabled: bool, gamma: float = None, lam: float = None):
        try:
            self._oasis_enabled = bool(enabled)
            if gamma is not None:
                self._oasis_gamma = float(gamma)
            if lam is not None:
                self._oasis_lambda = float(lam)
            if not self._oasis_enabled:
                self._oasis_prev_c.clear()
                self._oasis_prev_s.clear()
            print(f"[OASIS] enabled={self._oasis_enabled} gamma={self._oasis_gamma} lambda={self._oasis_lambda}")
        except Exception as e:
            print(f"[OASIS] set failed: {e}")

    def set_neuropil(self, r: float = 0.7, inner_gap: int = 2, ring_width: int = 10):
        self._neuropil_r = max(0.0, float(r))
        self._neuropil_inner_gap = int(inner_gap)
        self._neuropil_ring_width = int(ring_width)
        self._roi_ready = False
        print(f"[Neuropil] r={self._neuropil_r} gap={self._neuropil_inner_gap} ring={self._neuropil_ring_width}")

    def set_plot_normalization(self, mode: str):
        try:
            if isinstance(mode, str):
                self._plot_norm_mode = mode
                _labels = {
                    'Raw': ('Intensity', 'AU'),
                    'ΔF/F₀': ('ΔF/F₀', ''),
                    'dF/F': ('ΔF/F₀', ''),
                    'z-score': ('z-score', 'σ'),
                    'Spikes': ('Spike rate', 'AU'),
                }
                lbl, unit = _labels.get(mode, ('Intensity', 'AU'))
                if self.plot_widget and hasattr(self.plot_widget, 'setLabel'):
                    try:
                        self.plot_widget.setLabel('left', lbl, units=unit)
                    except Exception:
                        pass
        except Exception:
            self._plot_norm_mode = "Raw"

    def _resolve_trace_y(self, roi_id: int) -> np.ndarray:
        mode = getattr(self, '_plot_norm_mode', 'Raw')
        if mode == 'ΔF/F₀' or mode == 'dF/F':
            buf = self._dff_buffers.get(roi_id, [])
            if len(buf) < 2:
                return np.array([], dtype=np.float32)
            return np.array(list(buf), dtype=np.float32)
        elif mode == 'Spikes':
            buf = self._spike_buffers.get(roi_id, [])
            if len(buf) < 2:
                return np.array([], dtype=np.float32)
            return np.array(list(buf), dtype=np.float32)
        elif mode.startswith('z-score'):
            buf = self.buffers.get(roi_id, [])
            if len(buf) < 2:
                return np.array([], dtype=np.float32)
            y_raw = np.array(list(buf), dtype=np.float32)
            w = int(max(3, min(len(y_raw), int(max(1, getattr(self, '_last_fps_est', 30.0)) * 10))))
            yw = y_raw[-w:]
            mu = float(np.mean(yw))
            sd = float(np.std(yw)) if np.std(yw) > 1e-6 else 1.0
            return (y_raw - mu) / sd
        else:
            buf = self.buffers.get(roi_id, [])
            if len(buf) < 2:
                return np.array([], dtype=np.float32)
            return np.array(list(buf), dtype=np.float32)

    def set_highlight_ids(self, ids: List[int]):
        try:
            self._highlight_ids = set(int(x) for x in ids)
        except Exception:
            self._highlight_ids = set()



    # Init helpers extracted to live_trace_init.py as LiveTraceInitMixin
    #. Methods accessible
    # via MRO: self._init_roi_processing(), self._limit_cuda_pools(),
    # self._init_plotting(), self._detect_camera_fps(),
    # self._calculate_update_throttle().

    # Plot-layout builders extracted to live_trace_plot_layouts.py as
    # LiveTracePlotLayoutsMixin.
    # Class inherits LiveTracePlotLayoutsMixin above; methods accessible
    # via standard MRO: self._setup_single_plot_layout(...) etc.


    # Camera-frame ingestion + GPU memory monitoring extracted to
    # live_trace_ingest.py as LiveTraceIngestMixin. Mixed in via class declaration above.
    # Methods accessible via MRO: self._connect_camera_signals(),
    # self._on_camera_frame(), self.on_frame(),
    # self._update_performance_stats(), etc.

    # Frame-processing helpers extracted to live_trace_processing.py as
    # LiveTraceProcessingMixin.
    # Mixed in via class declaration above. Methods accessible via MRO:
    # self._on_frame_processed(), self._on_processing_error(),
    # self._build_rois_for_shape(), self._compute_dff(),
    # self._cleanup_existing_rois(), self._initialize_empty_state(),
    # self._initialize_buffers_safely(),
    # self._initialize_processing_structures(),
    # self._initialize_cpu_fallback().

    @pyqtSlot()
    # Top-level dispatcher + pygame renderer + pyqtgraph entry + skip
    # factor extracted to live_trace_plot_modes.py as
    # LiveTracePlotModesMixin. Mixed in via class declaration above.
    # Methods accessible via MRO: self._update_plot(),
    # self._update_pygame_plot(), self._update_pyqtgraph_plot(),
    # self._calculate_skip_factor(), self._get_unified_roi_color().
    # _update_paged_trace_mode + statistical/density/expanded modes
    # remain on this class until iter 39 + iter 41 extraction iters.

    def _update_direct_overlay_mode(self):
       
        try:

            active_buffers = {}
            all_vals = []

            for rid, buf in self.buffers.items():
                if len(buf) == 0:
                    continue
                    

                if len(buf) > 1000:
                    step = max(1, len(buf) // 500)
                    sampled_buf = buf[::step]
                else:
                    sampled_buf = buf
                
                active_buffers[rid] = sampled_buf
                all_vals.extend(sampled_buf)
            

            if len(all_vals) >= 4:
                vals_array = np.array(all_vals, dtype=np.float32)
                global_min, global_max = float(np.min(vals_array)), float(np.max(vals_array))
                
                if np.isfinite(global_min) and np.isfinite(global_max) and global_max > global_min:
                    range_pad = 0.1 * (global_max - global_min)
                    self.plot_widget.setYRange(global_min - range_pad, global_max + range_pad, padding=0.0)
            

            for rid, sampled_buf in active_buffers.items():
                curve = self._plot_curves.get(int(rid))
                if curve is None:
                    continue
                
                y_data = np.asarray(sampled_buf, dtype=np.float32)
                x_data = np.arange(len(y_data), dtype=np.float32)
                

                curve.setData(x=x_data, y=y_data, skipFiniteCheck=True)
                

                alpha = 0.8 if len(self.buffers) <= 10 else 0.6
                pen = curve.opts['pen']
                if hasattr(pen, 'color'):
                    color = pen.color()
                    color.setAlphaF(alpha)
                    pen.setColor(color)
                    curve.setPen(pen)
            
        except Exception as e:
            print(f"❌ Direct overlay mode error: {e}")

    # _update_statistical_aggregation_mode extracted to
    # live_trace_plot_aggregation.py (iter 39). Accessible via MRO.

    # Pagination + page navigation + paged-trace mode + restart-after-
    # napari + pagination controls all extracted to live_trace_plot_pagination.py
    # as LiveTracePlotPaginationMixin. Mixed in via
    # class declaration above. D-ltm-1 BUG preserved: _update_page_label_safe
    # is defined TWICE in the extracted mixin (Python uses only the 2nd).

    def get_performance_stats(self) -> Dict[str, Any]:
        try:
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            mem_mb = 0.0
        uptime = time.time() - self.start_time
        fps = self.stats["frames_processed"] / uptime if uptime > 0 else 0.0
        out = {
            "frames_processed": self.stats["frames_processed"],
            "frames_failed": self.stats["frames_failed"],
            "memory_usage_peak": self.stats["memory_usage_peak"],
            "current_memory_mb": mem_mb,
            "uptime_seconds": uptime,
            "frames_per_second": fps,
            "gpu_memory_peak": self.stats["gpu_memory_peak"],
            "sync_operations": self.stats["sync_operations"],
            "sync_failures": self.stats["sync_failures"],
            "sync_state": self._sync_state.value,
        }
        return out

    def export_traces(self, base_name="live_traces", last_n=100):
        try:
            self.export_counter += 1
            output_path = f"{base_name}_{self.export_counter}.npy"
            dff_output_path = f"{base_name}_dff_{self.export_counter}.npy"
            roiprint_out = f"roiprint_export_{self.export_counter}.npz"

            traces = {}
            dff_traces = {}
            spike_traces = {}
            for rid, buf in self.buffers.items():
                if buf:
                    traces[f"roi_{int(rid)}"] = list(buf)[-last_n:]
            for rid, buf in self._dff_buffers.items():
                if buf:
                    dff_traces[f"roi_{int(rid)}"] = list(buf)[-last_n:]
            for rid, buf in self._spike_buffers.items():
                if buf:
                    spike_traces[f"roi_{int(rid)}"] = list(buf)[-last_n:]
            np.save(output_path, traces)
            np.save(dff_output_path, dff_traces)
            spike_output_path = f"{base_name}_spikes_{self.export_counter}.npy"
            np.save(spike_output_path, spike_traces)

            sizes = (self._roi_sizes_gpu.get() if (CUDA_AVAILABLE and self._roi_sizes_gpu is not None)
                     else np.asarray(self._roi_sizes_cpu))
            np.savez_compressed(roiprint_out,
                                ids=np.asarray(self.ids, dtype=np.int32),
                                roi_sizes=np.asarray(sizes, dtype=np.float32),
                                shape=(self._H, self._W))

            print(f"Traces saved → {output_path}, ΔF/F₀ → {dff_output_path}, ROI info → {roiprint_out}")

        except Exception as e:
            print(f"Trace export error: {e}")
            self.error_occurred.emit(str(e))

    def get_dff_traces(self, last_n: int = 0) -> Dict[int, np.ndarray]:
        """Return ΔF/F₀ traces for all ROIs as {roi_id: ndarray}."""
        out = {}
        for rid, buf in self._dff_buffers.items():
            if buf:
                arr = np.array(list(buf), dtype=np.float32)
                if last_n > 0:
                    arr = arr[-last_n:]
                out[int(rid)] = arr
        return out

    def get_raw_traces(self, last_n: int = 0) -> Dict[int, np.ndarray]:
        out = {}
        for rid, buf in self.buffers.items():
            if buf:
                arr = np.array(list(buf), dtype=np.float32)
                if last_n > 0:
                    arr = arr[-last_n:]
                out[int(rid)] = arr
        return out

    def get_spike_traces(self, last_n: int = 0) -> Dict[int, np.ndarray]:
        out = {}
        for rid, buf in self._spike_buffers.items():
            if buf:
                arr = np.array(list(buf), dtype=np.float32)
                if last_n > 0:
                    arr = arr[-last_n:]
                out[int(rid)] = arr
        return out

    def _update_sync_state(self, state: SyncState, err: Optional[str] = None):
        with self._sync_lock:
            self._sync_state = state
            self._syncprint = SyncInfo(
                state=state,
                timestamp=time.time(),
                frame_count=self.stats["frames_processed"],
                memory_usage=self.stats["memory_usage_peak"],
                gpu_memory_usage=self.stats["gpu_memory_peak"],
                error_message=err,
            )
            self.sync_state_changed.emit(self._syncprint)


    def cleanup(self):
       
        try:
            print("🧹 Starting LiveTraceExtractor cleanup...")
            self._is_shutting_down = True
            self._update_sync_state(SyncState.STOPPING)
            
            if hasattr(self, "_cleanup_event"):
                self._cleanup_event.set()
                print("✅ Cleanup event set - signaling all threads to stop")
            
            if hasattr(self, '_pagination_widget'):
                try:
                    self._cleanup_pagination_widget()
                    print("✅ Pagination controls cleaned up")
                except Exception as e:
                    print(f"⚠️ Pagination cleanup warning: {e}")
                    
            if hasattr(self, '_expanded_dialog'):
                try:
                    if self._expanded_dialog and self._expanded_dialog.isVisible():
                        self._expanded_dialog.close()
                    self._expanded_dialog = None
                    self._expanded_curves = {}
                    print("✅ Expanded view cleaned up")
                except Exception as e:
                    print(f"⚠️ Expanded view cleanup warning: {e}")

            try:
                self._disconnect_camera_signals()
                print("✅ Camera signals disconnected")
            except Exception as e:
                print(f"⚠️ Error disconnecting camera signals: {e}")

            if hasattr(self, "frame_processor") and self.frame_processor is not None:
                try:
                    if self.frame_processor.isRunning():
                        self.frame_processor.stop()
                        if not self.frame_processor.wait(2000):  
                            print("⚠️ Frame processor did not stop gracefully, forcing termination")
                            self.frame_processor.terminate()
                    self.frame_processor.wait(1000)
                    print("✅ Frame processor stopped")
                except Exception as e:
                    print(f"⚠️ Error stopping frame processor: {e}")

            if getattr(self, "_plot_timer", None):
                try:
                    self._plot_timer.stop()
                    self._plot_timer.deleteLater()
                    self._plot_timer = None
                    print("✅ Plot timer stopped")
                except Exception as e:
                    print(f"⚠️ Error stopping plot timer: {e}")

            try:
                if hasattr(self, '_plot_curves'):
                    self._plot_curves.clear()
                if hasattr(self, '_stat_curves'):
                    self._stat_curves.clear()
                if hasattr(self, '_pagination_widget'):
                    try:
                        self._pagination_widget.close()
                        self._pagination_widget.deleteLater()
                        self._pagination_widget = None
                    except Exception:
                        pass
                print("✅ Plot resources cleared")
            except Exception as e:
                print(f"⚠️ Error clearing plot resources: {e}")

            if CUDA_USABLE:
                try:
                    gpu_resources = ['_f_gpu', '_labels_gpu', '_ids_gpu', '_roi_sizes_gpu']
                    for resource in gpu_resources:
                        if hasattr(self, resource) and getattr(self, resource) is not None:
                            try:
                                delattr(self, resource)
                            except Exception:
                                setattr(self, resource, None)

                    cp.get_default_memory_pool().free_all_blocks()
                    print("✅ GPU resources cleaned")
                except Exception as e:
                    print(f"⚠️ GPU cleanup error: {e}")

            if self.use_pygame_plot:
                try:
                    pygame.display.quit()
                    pygame.quit()
                    print("✅ Pygame cleaned up")
                except Exception as e:
                    print(f"⚠️ Pygame cleanup error: {e}")

            try:
                self.buffers.clear()
                self._cpu_masks = None
                self._flat_labels_cpu = None
                self._roi_sizes_cpu = None
                print("✅ Data structures cleared")
            except Exception as e:
                print(f"⚠️ Error clearing data structures: {e}")

            try:
                collected = gc.collect()
                if collected > 0:
                    print(f"✅ Garbage collection freed {collected} objects")
            except Exception as e:
                print(f"⚠️ Garbage collection error: {e}")

            print("✅ LiveTraceExtractor cleanup completed successfully")

        except Exception as e:
            print(f"❌ Critical cleanup error: {e}")
            import traceback
            print(f"   Stack trace: {traceback.format_exc()}")
            try:
                if hasattr(self, 'buffers'):
                    self.buffers.clear()
                gc.collect()
            except Exception:
                pass
            self._update_sync_state(SyncState.IDLE)

        uptime = time.time() - self.start_time
        print("✅ LiveTraceExtractor cleanup complete")
        print(f"📊 Runtime: {uptime:.1f}s, frames: {self.stats['frames_processed']}, "
              f"peak RSS: {self.stats['memory_usage_peak']:.1f} MB")

    def stop(self):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
