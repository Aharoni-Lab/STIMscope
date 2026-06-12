"""Performance + sync infrastructure for live trace extraction.

Extracted from ``live_trace_extractor.py``.

Contains:
- ``qimage_to_gray_np`` — QImage → grayscale numpy array helper
- ``PerformanceMonitor`` — wall-clock + memory delta timer
- ``SyncState`` enum + ``SyncInfo`` dataclass — pipeline sync state machine
- ``FrameProcessor`` — QThread that processes a queue of camera frames

Module constants (originally at top of ``live_trace_extractor.py``):
- ``MAX_FRAME_QUEUE_SIZE`` — capacity bound for the frame queue

No behavior change vs the original location. ``live_trace_extractor.py``
re-exports these names for backward-compat with existing callers.

"""

from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import psutil

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


MAX_FRAME_QUEUE_SIZE = 8


def qimage_to_gray_np(qimg: QImage) -> np.ndarray:

    if qimg.isNull():
        raise ValueError("Null QImage")
    fmt = qimg.format()
    if fmt not in (QImage.Format_Grayscale8, QImage.Format_RGB888, QImage.Format_ARGB32, QImage.Format_RGBA8888):
        qimg = qimg.convertToFormat(QImage.Format_ARGB32)
        fmt = qimg.format()

    width = qimg.width()
    height = qimg.height()
    # D-ltp-1fix iter 44: Qt aligns image rows to 4-byte
    # boundaries, so `bytesPerLine()` ≥ `width * bytes_per_pixel`.
    # The previous code reshaped using `width` directly, which
    # crashed on non-4-aligned widths (e.g. 6-pixel-wide Grayscale8
    # has 8-byte rows, 4 bytes of padding per row). Reshape by
    # bytesPerLine, then slice to the real width.
    bpl = qimg.bytesPerLine()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    buf = np.frombuffer(ptr, dtype=np.uint8)

    if fmt == QImage.Format_Grayscale8:
        arr = buf.reshape((height, bpl))
        return arr[:, :width].copy()

    if fmt in (QImage.Format_ARGB32, QImage.Format_RGBA8888):
        arr = buf.reshape((height, bpl // 4, 4))
        return arr[:, :width, 1].copy()

    if fmt == QImage.Format_RGB888:
        arr = buf.reshape((height, bpl // 3, 3))
        return arr[:, :width, 1].copy()

    qimg = qimg.convertToFormat(QImage.Format_Grayscale8)
    bpl2 = qimg.bytesPerLine()
    ptr = qimg.bits(); ptr.setsize(qimg.byteCount())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((qimg.height(), bpl2))
    return arr[:, :qimg.width()].copy()


class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.memory_before = 0.0

    def start(self):
        self.start_time = time.perf_counter()
        try:
            self.memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            self.memory_before = 0.0

    def end(self, label: str):
        if self.start_time is None:
            return
        dt = time.perf_counter() - self.start_time
        try:
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"⏱️ {label}: {dt:.3f}s, ΔMem {mem_after - self.memory_before:+.1f} MB")
        except Exception:
            print(f"⏱️ {label}: {dt:.3f}s")
        self.start_time = None


class SyncState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    RECORDING = "recording"
    PROCESSING = "processing"
    PROJECTING = "projecting"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class SyncInfo:
    state: SyncState
    timestamp: float
    frame_count: int
    memory_usage: float
    gpu_memory_usage: float
    error_message: Optional[str] = None


class FrameProcessor(QThread):
    frame_processed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, max_workers: int = 1):
        super().__init__()
        self.frame_queue: "queue.Queue[Any]" = queue.Queue(maxsize=MAX_FRAME_QUEUE_SIZE)
        self.running = True
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="FrameProc")
        self.perf = PerformanceMonitor()
        self._frames = 0

    def add_frame(self, frame: Any):
        try:
            if self.frame_queue.qsize() > int(MAX_FRAME_QUEUE_SIZE * 0.8):
                drop = max(1, self.frame_queue.qsize() // 4)
                for _ in range(drop):
                    try: self.frame_queue.get_nowait()
                    except queue.Empty: break
                print(f"Frame queue high-watermark; dropped {drop} frames")
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            print("Frame queue full; skipping frame")
        except Exception as e:
            self.error_occurred.emit(f"Queue add error: {e}")

    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                fut = self.pool.submit(self._process_one, frame)
                fut.add_done_callback(self._on_done)
            except queue.Empty:
                continue
            except Exception as e:
                self.error_occurred.emit(f"FrameProcessor error: {e}")

    def _process_one(self, frame: Any) -> dict:
        # Diagnostic: prove _process_one is being called.
        if not getattr(self, "_first_process_logged", False):
            print(f"[FrameProcessor] FIRST _process_one called, frame type={type(frame).__name__}")
            self._first_process_logged = True
        self.perf.start()
        try:
            if hasattr(frame, "get_numpy_1D"):
                h, w = frame.Height(), frame.Width()
                arr4 = np.array(frame.get_numpy_1D(), dtype=np.uint8).reshape((h, w, 4))
                # Use green channel for fluorescence
                gray = arr4[..., 1]
            elif isinstance(frame, np.ndarray):
                if frame.ndim == 2:
                    gray = frame
                elif frame.ndim == 3 and frame.shape[2] >= 3:
                    # Use green channel for fluorescence
                    gray = frame[..., 1]
                else:
                    raise ValueError("Unsupported ndarray shape")
            elif isinstance(frame, QImage):
                gray = qimage_to_gray_np(frame)
            else:
                raise ValueError("Unsupported frame type")

            self._frames += 1
            return {"frame": gray, "timestamp": time.time(), "frame_id": self._frames}
        finally:
            pass

    def _on_done(self, fut):
        try:
            res = fut.result()
            self.frame_processed.emit(res)
        except Exception as e:
            self.error_occurred.emit(f"Processing failure: {e}")

    def stop(self):
        self.running = False
        try:
            self.pool.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass


__all__ = [
    "MAX_FRAME_QUEUE_SIZE",
    "qimage_to_gray_np",
    "PerformanceMonitor",
    "SyncState",
    "SyncInfo",
    "FrameProcessor",
]
