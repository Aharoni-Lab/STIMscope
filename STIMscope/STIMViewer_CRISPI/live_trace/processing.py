"""Frame-processing helpers extracted from ``live_trace_extractor``.

Stage-0.6 of the 6-module decomposition (sub-module 5 of 6).
Extracted from ``live_trace_extractor.py``.

Contains the 9 processing helpers as a mixin class:
- ``_on_frame_processed(processed_data)`` — main frame slot: GPU/CPU
  bincount-mean ROI extraction → buffers + dF/F + OASIS spike inference
- ``_on_processing_error(msg)`` — @pyqtSlot(str) error relay to UI
- ``_build_rois_for_shape(H, W)`` — runtime ROI builder triggered by
  the first frame after start (resizes labels if camera shape differs)
- ``_compute_dff(rid_key, raw_val)`` — rolling-percentile baseline dF/F
- ``_cleanup_existing_rois()`` — tear down GPU + CPU ROI structures
- ``_initialize_empty_state()`` — safe-empty fallback when no labels
- ``_initialize_buffers_safely()`` — per-ROI deque allocation + verify
- ``_initialize_processing_structures(resized)`` — GPU/CPU label arrays
  + neuropil rings + plot-curve allocation
- ``_initialize_cpu_fallback(flat)`` — CPU-only label/size init when GPU
  initialization fails

The mixin expects the subclass (LiveTraceExtractor) to provide:
- ``self._labels_orig`` (set by LiveTraceInitMixin._init_roi_processing)
- ``self._max_rois_cfg``, ``self._max_points_cfg`` (config snapshots)
- ``self._neuropil_r``, ``self._neuropil_inner_gap``,
  ``self._neuropil_ring_width`` (neuropil config)
- ``self._baseline_window_s``, ``self._baseline_percentile`` (dF/F config)
- ``self._oasis_enabled``, ``self._oasis_gamma``, ``self._oasis_lambda``,
  ``self._oasis_prev_c`` (OASIS spike inference state)
- ``self._proc_gate``, ``self._process_every_n`` (frame-decimation gate)
- ``self._gpu_lock`` (threading.Lock)
- ``self.ids`` (np.ndarray[int32], filled by _build_rois_for_shape)
- ``self.buffers``, ``self._dff_buffers``, ``self._spike_buffers``
- ``self.stats`` (dict with frames_processed, frames_failed,
  last_frame_time keys)
- ``self._global_frame_index`` (counter)
- ``self.plot_widget``, ``self._plot_curves`` (Qt plotting state)
- ``self._last_fps_est`` (from LiveTraceInitMixin._init_plotting)
- ``self.error_occurred`` (pyqtSignal(str))
- ``self.use_pygame_plot`` (bool) — referenced indirectly by callers

No behavior change vs the original location.

Safety: smoke tests in ``tests/L3_5_split_first/`` must remain green.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import cv2

from PyQt5.QtCore import pyqtSlot

# CUDA availability — same dance as live_trace_extractor.
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False
    cp = None

CUDA_USABLE = False
if CUDA_AVAILABLE:
    try:
        import cupy.cuda.runtime as _cur
        ndev = _cur.getDeviceCount()
        if ndev and ndev > 0:
            _ = cp.arange(1, dtype=cp.int8)
            CUDA_USABLE = True
    except Exception:
        CUDA_USABLE = False

# pyqtgraph availability for the plot-curve allocation branch in
# _initialize_processing_structures.
try:
    import pyqtgraph as pg
    PYQTPGRAPH_AVAILABLE = True
except Exception:
    PYQTPGRAPH_AVAILABLE = False
    pg = None


class LiveTraceProcessingMixin:
    """Frame-processing helpers for ``LiveTraceExtractor``."""

    def _on_frame_processed(self, processed_data: dict):
        try:

            if not isinstance(processed_data, dict) or 'frame' not in processed_data:
                print("⚠️ Invalid frame data received, skipping")
                return

            gray = processed_data['frame']


            if gray is None:
                print("⚠️ Received None frame, skipping")
                return

            if not hasattr(gray, 'shape') or len(gray.shape) < 2:
                print(f"⚠️ Invalid frame shape: {getattr(gray, 'shape', 'no shape')}, skipping")
                return

            H, W = gray.shape[:2]


            if H <= 0 or W <= 0 or H > 10000 or W > 10000:
                print(f"⚠️ Unreasonable frame dimensions {W}x{H}, skipping")
                return


            if not getattr(self, "_roi_ready", False):
                if not hasattr(self, '_labels_orig') or self._labels_orig is None:
                    print("⚠️ No ROI labels loaded, cannot process frame")
                    return

                self._build_rois_for_shape(H, W)
                if not self._roi_ready or self.ids.size == 0:
                    return


            self._proc_gate = (getattr(self, "_proc_gate", -1) + 1) % self._process_every_n
            if self._proc_gate:

                self.stats['last_frame_time'] = time.time()
                return


            flat = gray.ravel().astype(np.float32, copy=False)


            if CUDA_USABLE and hasattr(self, '_labels_gpu') and self._labels_gpu is not None:

                if not hasattr(self, '_roi_sizes_gpu') or self._roi_sizes_gpu is None:
                    print("⚠️ GPU ROI sizes not initialized, falling back to CPU")
                else:
                    with self._gpu_lock:
                        self._f_gpu.set(flat)
                        if not hasattr(self, '_max_label') or self._max_label is None:
                            self._max_label = int(self._labels_gpu.max().get())
                        sums = cp.bincount(
                            self._labels_gpu,
                            weights=self._f_gpu,
                            minlength=self._max_label + 1
                        )
                        den = cp.maximum(self._roi_sizes_gpu, 1e-6)
                        means = (sums[self._ids_gpu] / den)
                        if self._neuropil_r > 0 and self._npil_labels_gpu is not None:
                            npil_sums = cp.bincount(
                                self._npil_labels_gpu,
                                weights=self._f_gpu,
                                minlength=self._max_label + 1,
                            )
                            npil_den = cp.maximum(self._npil_sizes_gpu, 1e-6)
                            means = means - self._neuropil_r * (npil_sums[self._ids_gpu] / npil_den)
                        means = means.get()

                        for val, rid in zip(means, self.ids):
                            rid_key = int(rid)
                            if rid_key not in self.buffers:
                                print(f"⚠️ GPU path: ROI {rid_key} not in buffers, creating...")
                                from collections import deque
                                self.buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                                self._dff_buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                                self._spike_buffers[rid_key] = deque(maxlen=self._max_points_cfg)

                            try:
                                raw_v = float(val)
                                self.buffers[rid_key].append(raw_v)
                                dff_v = self._compute_dff(rid_key, raw_v)
                                self._dff_buffers[rid_key].append(dff_v)
                                spike_v = 0.0
                                if self._oasis_enabled:
                                    c_prev = self._oasis_prev_c.get(rid_key, 0.0)
                                    s_t = dff_v - (self._oasis_gamma * c_prev) - float(self._oasis_lambda)
                                    if s_t < 0.0:
                                        s_t = 0.0
                                    c_t = (self._oasis_gamma * c_prev) + s_t
                                    self._oasis_prev_c[rid_key] = c_t
                                    spike_v = float(s_t)
                                if rid_key not in self._spike_buffers:
                                    self._spike_buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                                self._spike_buffers[rid_key].append(spike_v)
                            except Exception as e:
                                print(f"❌ GPU buffer error for ROI {rid_key}: {e}")

                        # Diagnostic: print extracted means + frame stats every ~5s
                        # so the user can watch values change as they cover/uncover
                        # the sample. Tells us definitively whether the ROIs are
                        # sampling pixels that respond to physical scene changes.
                        try:
                            now_t = time.time()
                            last_t = getattr(self, "_last_extract_log_t", 0.0)
                            if now_t - last_t > 5.0:
                                fmin = float(np.asarray(flat).min())
                                fmax = float(np.asarray(flat).max())
                                fmean = float(np.asarray(flat).mean())
                                m_lo = float(min(means)) if len(means) else 0.0
                                m_hi = float(max(means)) if len(means) else 0.0
                                m_mean = float(np.mean(means)) if len(means) else 0.0
                                m_std = float(np.std(means)) if len(means) else 0.0
                                print(
                                    f"[Extractor] frame: min={fmin:.1f} max={fmax:.1f} mean={fmean:.1f} | "
                                    f"per-ROI means: lo={m_lo:.1f} hi={m_hi:.1f} mean={m_mean:.1f} std={m_std:.1f} "
                                    f"(N={len(means)})"
                                )
                                self._last_extract_log_t = now_t
                        except Exception:
                            pass

                        self.stats['frames_processed'] += 1
                        self.stats['last_frame_time'] = time.time()
                        self._global_frame_index += 1
                        return
            else:

                if not hasattr(self, '_flat_labels_cpu') or self._flat_labels_cpu is None:
                    print("⚠️ CPU labels not initialized, skipping frame")
                    return
                if not hasattr(self, '_roi_sizes_cpu') or self._roi_sizes_cpu is None:
                    print("⚠️ CPU ROI sizes not initialized, attempting to initialize...")
                    try:
                        if hasattr(self, '_flat_labels_cpu') and self._flat_labels_cpu is not None:
                            if not hasattr(self, '_max_label') or self._max_label is None:
                                self._max_label = int(self._flat_labels_cpu.max(initial=0))
                            counts = np.bincount(self._flat_labels_cpu, minlength=self._max_label + 1)
                            self._roi_sizes_cpu = counts[self.ids].astype(np.float32)
                            print("✅ CPU ROI sizes initialized")
                        else:
                            print("⚠️ Cannot initialize ROI sizes, skipping frame")
                            return
                    except Exception as e:
                        print(f"⚠️ Failed to initialize ROI sizes: {e}, skipping frame")
                        return

                sums = np.bincount(
                    self._flat_labels_cpu,
                    weights=flat,
                    minlength=self._max_label + 1
                )
                if self._roi_sizes_cpu is None:
                    print("⚠️ CPU ROI sizes still None after initialization attempt, skipping frame")
                    return
                den = np.maximum(self._roi_sizes_cpu, 1e-6)
                means = (sums[self.ids] / den)
                if self._neuropil_r > 0 and self._npil_labels_flat_cpu is not None:
                    npil_sums = np.bincount(
                        self._npil_labels_flat_cpu,
                        weights=flat,
                        minlength=self._max_label + 1,
                    )
                    npil_den = np.maximum(self._npil_sizes_cpu, 1e-6)
                    means = means - self._neuropil_r * (npil_sums[self.ids] / npil_den)


            for val, rid in zip(means, self.ids):
                rid_key = int(rid)
                if rid_key not in self.buffers:
                    print(f"⚠️ ROI {rid_key} not in buffers, reinitializing buffers...")

                    from collections import deque
                    for missing_rid in self.ids:
                        missing_key = int(missing_rid)
                        if missing_key not in self.buffers:
                            self.buffers[missing_key] = deque(maxlen=self._max_points_cfg)
                            self._dff_buffers[missing_key] = deque(maxlen=self._max_points_cfg)
                            self._spike_buffers[missing_key] = deque(maxlen=self._max_points_cfg)
                            print(f"   ✅ Created buffer for ROI {missing_key}")

                try:
                    raw_v = float(val)
                    self.buffers[rid_key].append(raw_v)
                    dff_v = self._compute_dff(rid_key, raw_v)
                    self._dff_buffers[rid_key].append(dff_v)
                    spike_v = 0.0
                    if self._oasis_enabled:
                        c_prev = self._oasis_prev_c.get(rid_key, 0.0)
                        s_t = dff_v - (self._oasis_gamma * c_prev) - float(self._oasis_lambda)
                        if s_t < 0.0:
                            s_t = 0.0
                        c_t = (self._oasis_gamma * c_prev) + s_t
                        self._oasis_prev_c[rid_key] = c_t
                        spike_v = float(s_t)
                    if rid_key not in self._spike_buffers:
                        self._spike_buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                    self._spike_buffers[rid_key].append(spike_v)
                except KeyError as e:
                    print(f"❌ Still missing buffer for ROI {rid_key}: {e}")

                    from collections import deque
                    self.buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                    self._dff_buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                    self._spike_buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                    self.buffers[rid_key].append(float(val))
                    self._dff_buffers[rid_key].append(0.0)
                    self._spike_buffers[rid_key].append(0.0)
                    print(f"   🔧 Emergency buffer created for ROI {rid_key}")
                except Exception as e:
                    print(f"❌ Unexpected buffer error for ROI {rid_key}: {e}")


            self.stats['frames_processed'] += 1
            self.stats['last_frame_time'] = time.time()
            self._global_frame_index += 1

        except Exception as e:
            self.stats['frames_failed'] += 1
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"❌ Frame processing error [{error_type}]: {error_msg}")


            if hasattr(self, '_labels_orig') and self._labels_orig is not None:
                print(f"   Labels shape: {self._labels_orig.shape}")
            if hasattr(self, 'ids') and self.ids is not None:
                print(f"   Active ROIs: {len(self.ids)}")
            if hasattr(gray, 'shape'):
                print(f"   Frame shape: {gray.shape}")


            if "index" in error_msg.lower() or "shape" in error_msg.lower():
                print("🔧 Attempting ROI reinitialization due to indexing/shape error...")
                try:
                    if hasattr(gray, 'shape') and len(gray.shape) >= 2:
                        self._build_rois_for_shape(gray.shape[0], gray.shape[1])
                        print("✅ ROI reinitialization successful")
                        return
                except Exception as recovery_error:
                    print(f"❌ ROI recovery failed: {recovery_error}")


            if self.stats['frames_failed'] % 10 == 0:
                self.error_occurred.emit(f"Frame processing error [{error_type}]: {error_msg}")

    @pyqtSlot(str)
    def _on_processing_error(self, msg: str):
        print(f"Processing error: {msg}")
        self.error_occurred.emit(msg)

    def _build_rois_for_shape(self, H: int, W: int):

        try:
            print(f"🔄 Building ROIs for frame shape {W}x{H}...")

            self._cleanup_existing_rois()


            if (self._labels_orig.shape[0], self._labels_orig.shape[1]) != (H, W):
                resized = cv2.resize(self._labels_orig, (W, H), interpolation=cv2.INTER_NEAREST)
                print(f"📐 Resized labels from {self._labels_orig.shape} to {resized.shape}")
            else:
                resized = self._labels_orig

            ids = np.unique(resized)
            ids = ids[ids > 0]
            if ids.size == 0:
                print("⚠️ No positive ROI labels found after resize; running in empty-safe mode")
                self._initialize_empty_state()

                return

            self.ids = ids[: self._max_rois_cfg].astype(np.int32)
            self._H, self._W = H, W


            self._initialize_buffers_safely()


            self._initialize_processing_structures(resized)

            self._roi_ready = True
            print(f"✅ ROIs ready for frame shape {W}x{H} with {len(self.ids)} labels")

        except Exception as e:
            print(f"❌ Error building ROIs: {e}")
            import traceback
            print(f"   Stack trace: {traceback.format_exc()}")
            self._initialize_empty_state()

    def _compute_dff(self, rid_key: int, raw_val: float) -> float:
        buf = self.buffers.get(rid_key)
        if buf is None or len(buf) < 3:
            return 0.0
        fps = max(1.0, getattr(self, '_last_fps_est', 30.0))
        win = int(min(len(buf), fps * self._baseline_window_s))
        if win < 3:
            return 0.0
        from itertools import islice
        start = max(0, len(buf) - win)
        recent = np.fromiter(islice(buf, start, len(buf)), dtype=np.float32, count=win)
        f0 = float(np.percentile(recent, self._baseline_percentile))
        if abs(f0) < 1e-6:
            f0 = 1.0
        return (raw_val - f0) / f0

    def _cleanup_existing_rois(self):

        try:

            if hasattr(self, 'buffers'):
                self.buffers.clear()
            if hasattr(self, '_dff_buffers'):
                self._dff_buffers.clear()


            if CUDA_AVAILABLE:
                if hasattr(self, '_labels_gpu') and self._labels_gpu is not None:
                    del self._labels_gpu
                if hasattr(self, '_ids_gpu') and self._ids_gpu is not None:
                    del self._ids_gpu
                if hasattr(self, '_roi_sizes_gpu') and self._roi_sizes_gpu is not None:
                    del self._roi_sizes_gpu
                if hasattr(self, '_f_gpu') and self._f_gpu is not None:
                    del self._f_gpu


            self._flat_labels_cpu = None
            self._roi_sizes_cpu = None


            if hasattr(self, '_plot_curves'):
                self._plot_curves.clear()

            print("🧹 Existing ROI structures cleaned up")

        except Exception as e:
            print(f"⚠️ Error during ROI cleanup: {e}")

    def _initialize_empty_state(self):

        self.ids = np.array([], dtype=np.int32)
        self.buffers = {}
        self._dff_buffers = {}
        self._roi_ready = False
        self._labels_gpu = None
        self._ids_gpu = None
        self._roi_sizes_gpu = None
        self._f_gpu = None
        self._flat_labels_cpu = None
        self._roi_sizes_cpu = None

    def _initialize_buffers_safely(self):

        from collections import deque

        self.buffers = {}
        self._dff_buffers = {}
        self._spike_buffers = {}
        for r in self.ids:
            rid_key = int(r)
            self.buffers[rid_key] = deque(maxlen=self._max_points_cfg)
            self._dff_buffers[rid_key] = deque(maxlen=self._max_points_cfg)
            self._spike_buffers[rid_key] = deque(maxlen=self._max_points_cfg)


        print(f"📊 Initialized buffers for ROI IDs: {sorted(self.buffers.keys())}")
        if len(self.buffers) != len(self.ids):
            print(f"⚠️ Buffer count mismatch: {len(self.buffers)} buffers vs {len(self.ids)} ROIs")

            for r in self.ids:
                rid_key = int(r)
                if rid_key not in self.buffers:
                    self.buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                    self._dff_buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                    self._spike_buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                    print(f"   🔧 Added missing buffer for ROI {rid_key}")

        print(f"✅ Buffer verification complete: {len(self.buffers)} buffers for {len(self.ids)} ROIs")

    def _initialize_processing_structures(self, resized):

        flat = resized.ravel().astype(np.int32)
        self._flat_labels_cpu = flat
        self._max_label = int(flat.max(initial=0))

        self._npil_labels_flat_cpu = None
        self._npil_sizes_cpu = None
        self._npil_labels_gpu = None
        self._npil_sizes_gpu = None
        if self._neuropil_r > 0:
            try:
                from trace_extractor import build_neuropil_labels
                npil_2d = build_neuropil_labels(
                    resized, self.ids.tolist(),
                    inner_gap=self._neuropil_inner_gap,
                    ring_width=self._neuropil_ring_width,
                )
                self._npil_labels_flat_cpu = npil_2d.ravel().astype(np.int32)
                npil_counts = np.bincount(self._npil_labels_flat_cpu, minlength=self._max_label + 1)
                self._npil_sizes_cpu = np.maximum(npil_counts[self.ids].astype(np.float32), 1e-6)
                print(f"✅ Neuropil rings built (r={self._neuropil_r})")
            except Exception as e:
                print(f"⚠️ Neuropil ring build failed: {e}")
                self._neuropil_r = 0.0

        if CUDA_USABLE:
            try:
                self._labels_gpu = cp.asarray(flat)
                self._ids_gpu = cp.asarray(self.ids)
                counts = cp.bincount(self._labels_gpu, minlength=self._max_label + 1)
                self._roi_sizes_gpu = counts[self._ids_gpu].astype(cp.float32)
                self._f_gpu = cp.empty(len(flat), dtype=cp.float32)
                self._roi_sizes_cpu = None
                if self._npil_labels_flat_cpu is not None:
                    self._npil_labels_gpu = cp.asarray(self._npil_labels_flat_cpu)
                    self._npil_sizes_gpu = cp.asarray(self._npil_sizes_cpu)
                print(f"✅ GPU processing structures initialized for {len(self.ids)} ROIs")
            except Exception as e:
                print(f"⚠️ GPU initialization failed, falling back to CPU: {e}")
                self._initialize_cpu_fallback(flat)
        else:
            self._initialize_cpu_fallback(flat)


        if self.plot_widget is not None and PYQTPGRAPH_AVAILABLE:
            for rid in self.ids:
                if rid not in self._plot_curves:
                    pen = pg.mkPen(pg.intColor(len(self._plot_curves), hues=max(8, len(self.ids))), width=1)
                    self._plot_curves[int(rid)] = self.plot_widget.plot(pen=pen)

    def _initialize_cpu_fallback(self, flat):

        try:
            counts = np.bincount(flat, minlength=self._max_label + 1)
            self._roi_sizes_cpu = counts[self.ids].astype(np.float32)
            self._labels_gpu = None
            self._ids_gpu = None
            self._roi_sizes_gpu = None
            self._f_gpu = None
            print(f"✅ CPU processing structures initialized for {len(self.ids)} ROIs")
        except Exception as e:
            print(f"❌ CPU initialization also failed: {e}")
            self._initialize_empty_state()


__all__ = ["LiveTraceProcessingMixin"]
