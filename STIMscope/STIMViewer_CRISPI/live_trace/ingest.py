"""Camera-frame ingestion for live trace extraction.

Stage-0.6 of the 6-module decomposition (sub-module 4 of 6).
Extracted from ``live_trace_extractor.py``.

Contains the camera-frame intake path as a mixin class. The intake path
is the L4↔L3.5 hot signal seam — camera frames arrive via Qt signal
(`_on_camera_frame`, `_on_camera_qimage`) or direct callback (`on_frame`)
and get queued on `self.frame_processor` for downstream processing.

Methods:
- ``_connect_camera_signals`` — auto-detect camera's frame signal
- ``_disconnect_camera_signals`` — tear down at cleanup
- ``_on_camera_frame`` — @pyqtSlot(object) wrapper
- ``_on_camera_qimage`` — @pyqtSlot(QImage) wrapper
- ``on_frame`` — main frame intake point (public API)
- ``_update_performance_stats`` — emit performance_update signal

The mixin expects the subclass (LiveTraceExtractor) to provide:
- ``self.camera`` — camera module with a frame signal (object or QImage)
- ``self._camera_signal_refs`` — list[(sig, slot)] for disconnect
- ``self.frame_processor`` — FrameProcessor with add_frame(frame)
- ``self.error_occurred`` — pyqtSignal(str) for error reporting
- ``self.performance_update`` — pyqtSignal(dict) for periodic stats
- ``self.stats`` — dict with "gpu_memory_peak", "memory_usage_peak",
  "uptime_seconds" keys
- ``self.start_time`` — float (time.time())

No behavior change vs the original location.

Safety: smoke tests in ``tests/L3_5_split_first/`` must remain green.
"""

from __future__ import annotations

import time

import psutil

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage

from live_trace.perf import qimage_to_gray_np


# CUDA availability — same import dance as in live_trace_extractor.py.
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


class LiveTraceIngestMixin:
    """Camera-frame intake + GPU memory monitoring for ``LiveTraceExtractor``."""

    def _connect_camera_signals(self):
        """
        Try several common signal names; prefer connecting to the generic on_frame(Object)
        to avoid Qt signature mismatches. Fall back to QImage-typed slot if needed.
        """
        connected = False

        candidates = (
            "image_update_signal", "frame_numpy", "frame_np",
            "frame_ready", "newFrame", "frame_signal", "new_qimage", "frame_qimage"
        )

        for name in candidates:
            try:
                sig = getattr(self.camera, name, None)
            except Exception:
                sig = None
            if sig is None:
                continue


            try:
                sig.connect(self.on_frame, Qt.QueuedConnection)
                self._camera_signal_refs.append((sig, self.on_frame))
                print(f"LiveTraceExtractor: connected to camera signal '{name}' → on_frame(object)")
                connected = True
                break
            except Exception:
                pass


            try:
                sig.connect(self._on_camera_qimage, Qt.QueuedConnection)
                self._camera_signal_refs.append((sig, self._on_camera_qimage))
                print(f"LiveTraceExtractor: connected to camera signal '{name}' → _on_camera_qimage(QImage)")
                connected = True
                break
            except Exception:
                pass


        if not connected:
            # D-lti-1fix iter 44: wrap getattr in try/except for
            # symmetry with the signal-name candidates loop above (lines
            # 92-96). Pre-fix, a camera that raised RuntimeError from
            # __getattr__ would crash here even though the candidate
            # loop would swallow the same exception. Now both probes
            # use identical defensive coding.
            try:
                cb = getattr(self.camera, "register_consumer", None)
            except Exception:
                cb = None
            if callable(cb):
                try:
                    cb(self.on_frame)
                    print("LiveTraceExtractor: registered camera consumer callback")
                    connected = True
                except Exception as e:
                    print(f"register_consumer failed: {e}")

        if not connected:
            print("LiveTraceExtractor: could not connect to camera; waiting for manual feed (on_frame)")


    def _disconnect_camera_signals(self):
        for sig, slot in list(getattr(self, "_camera_signal_refs", [])):
            try:
                sig.disconnect(slot)
            except Exception:
                pass
        if hasattr(self, "_camera_signal_refs"):
            self._camera_signal_refs.clear()



    @pyqtSlot(object)
    def _on_camera_frame(self, frame_obj: object):
        self.on_frame(frame_obj)

    @pyqtSlot(QImage)
    def _on_camera_qimage(self, qimg: QImage):
        try:
            arr = qimage_to_gray_np(qimg)
            self.on_frame(arr)
        except Exception as e:
            print(f"QImage→np conversion failed: {e}")

    def on_frame(self, frame):
        # Diagnostic: prove frames are reaching the extractor at all.
        if not getattr(self, "_first_frame_logged", False):
            try:
                ftype = type(frame).__name__
                shape = getattr(frame, "shape", None)
                wh = None
                if hasattr(frame, "Width"):
                    try:
                        wh = (frame.Width(), frame.Height())
                    except Exception:
                        wh = None
                print(f"[LiveTraceExtractor] FIRST frame received: type={ftype} shape={shape} (W,H)={wh}")
                self._first_frame_logged = True
            except Exception:
                pass
        try:
            self.frame_processor.add_frame(frame)
        except Exception as e:
            print(f"Error queueing frame: {e}")
            self.error_occurred.emit(str(e))


    def _update_performance_stats(self):
        self.stats["uptime_seconds"] = time.time() - self.start_time
        try:
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.stats["memory_usage_peak"] = max(self.stats["memory_usage_peak"], mem_mb)
        except Exception:
            pass
        self.performance_update.emit(self.stats.copy())


__all__ = ["LiveTraceIngestMixin"]
