"""Unified trace extraction for the CRISPI platform.

One implementation, used by all three callers:
  - Trace Test dialog (qt_interface.py: _open_trace_test_dialog)
  - Real-Time Trace Extraction (live_trace_extractor.py)
  - GUI hardware mode (the GUI entry point)

The core operation is the same everywhere: given a labeled ROI map
and a camera frame, compute the mean pixel value per ROI.

Prefers CuPy when available for GPU-vectorised bincount; falls back
to numpy transparently. Caches per-instance GPU arrays across calls
so the hot path is just `set(flat) + bincount + divide`.

Authoritative API:
    extractor = TraceExtractor(labels, roi_ids)
    means = extractor.extract(frame)   # returns np.ndarray shape (len(roi_ids),)
    extractor.close()                  # releases GPU memory

For single-ROI callers (e.g. Trace Test), pass labels with two classes
(0 = background, 1 = ROI pixels) and roi_ids=[1].
"""
from __future__ import annotations

import threading
from typing import List, Optional, Sequence

import numpy as np

try:
    import cupy as _cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False


def _is_cupy_runtime_usable() -> bool:
    """CuPy imports successfully on Jetson but CUDA may not be available.
    Do a small allocation test to confirm the runtime works before we
    commit to the GPU path."""
    if not _HAS_CUPY:
        return False
    try:
        _ = _cp.zeros(1, dtype=_cp.float32)
        return True
    except Exception:
        return False


_CUPY_USABLE = _is_cupy_runtime_usable()


def build_neuropil_labels(
    labels: np.ndarray,
    roi_ids: Sequence[int],
    inner_gap: int = 2,
    ring_width: int = 10,
) -> np.ndarray:
    """Build a neuropil ring label map from ROI labels.
    For each ROI id, the neuropil ring consists of pixels within
    [inner_gap+1, inner_gap+ring_width] pixels of the ROI boundary
    that do not belong to any ROI. Returns int32 array same shape as
    labels, where pixel value = ROI id if it's in that ROI's neuropil
    ring, 0 otherwise. Overlapping rings are assigned to the nearest ROI."""
    from scipy.ndimage import binary_dilation
    labels_2d = labels.reshape(labels.shape) if labels.ndim == 2 else labels
    h, w = labels_2d.shape
    npil = np.zeros((h, w), dtype=np.int32)
    any_roi = labels_2d > 0
    for rid in roi_ids:
        roi_mask = labels_2d == rid
        outer = binary_dilation(roi_mask, iterations=inner_gap + ring_width)
        inner = binary_dilation(roi_mask, iterations=inner_gap)
        ring = outer & ~inner & ~any_roi
        npil[ring & (npil == 0)] = rid
    return npil


class TraceExtractor:
    """Label-based mean-intensity extractor with CuPy/numpy backends.

    Thread-safe for serial calls from one consumer thread. If multiple
    threads will call extract() concurrently, wrap in an external lock.
    """

    def __init__(
        self,
        labels: np.ndarray,
        roi_ids: Optional[Sequence[int]] = None,
        *,
        prefer_gpu: bool = True,
        neuropil_r: float = 0.0,
        neuropil_inner_gap: int = 2,
        neuropil_ring_width: int = 10,
    ):
        """
        labels  : int array (H,W) or flat (H*W,). 0 = background.
        roi_ids : ordered sequence of label IDs to extract. If None,
                  uses all unique non-zero labels in ascending order.
        prefer_gpu : if False, forces CPU path even if CuPy is present.
        neuropil_r : subtraction coefficient (0 = disabled, 0.7 typical).
        """
        labels = np.asarray(labels)
        if labels.dtype not in (np.int32, np.int64, np.uint32, np.uint16):
            labels = labels.astype(np.int32, copy=False)
        self._labels_shape: Optional[tuple] = (
            tuple(labels.shape) if labels.ndim == 2 else None
        )
        self._labels_flat = np.ascontiguousarray(labels.reshape(-1))
        if roi_ids is None:
            ids = np.unique(self._labels_flat)
            ids = ids[ids != 0]
            roi_ids = ids.tolist()
        self.roi_ids: List[int] = [int(i) for i in roi_ids]
        self._ids_np = np.asarray(self.roi_ids, dtype=np.int64)
        self._max_label = (
            int(self._labels_flat.max(initial=0))
            if self._labels_flat.size
            else 0
        )

        counts = np.bincount(self._labels_flat, minlength=self._max_label + 1)
        self._roi_sizes_np = np.maximum(counts[self._ids_np].astype(np.float32), 1e-6)

        self._neuropil_r = float(neuropil_r)
        self._npil_labels_flat = None
        self._npil_sizes_np = None
        self._npil_labels_gpu = None
        self._npil_sizes_gpu = None
        if self._neuropil_r > 0 and self._labels_shape is not None:
            npil_2d = build_neuropil_labels(
                labels, self.roi_ids,
                inner_gap=neuropil_inner_gap,
                ring_width=neuropil_ring_width,
            )
            self._npil_labels_flat = np.ascontiguousarray(npil_2d.reshape(-1))
            npil_counts = np.bincount(
                self._npil_labels_flat, minlength=self._max_label + 1
            )
            self._npil_sizes_np = np.maximum(
                npil_counts[self._ids_np].astype(np.float32), 1e-6
            )

        self._use_gpu = bool(prefer_gpu and _CUPY_USABLE)
        self._lock = threading.Lock()

        self._labels_gpu = None
        self._roi_sizes_gpu = None
        self._ids_gpu = None
        self._frame_gpu = None
        self._gpu_n_pixels = 0

        if self._use_gpu:
            try:
                self._labels_gpu = _cp.asarray(self._labels_flat, dtype=_cp.int32)
                self._roi_sizes_gpu = _cp.asarray(self._roi_sizes_np, dtype=_cp.float32)
                self._ids_gpu = _cp.asarray(self._ids_np, dtype=_cp.int64)
                self._gpu_n_pixels = int(self._labels_gpu.size)
                if self._npil_labels_flat is not None:
                    self._npil_labels_gpu = _cp.asarray(
                        self._npil_labels_flat, dtype=_cp.int32
                    )
                    self._npil_sizes_gpu = _cp.asarray(
                        self._npil_sizes_np, dtype=_cp.float32
                    )
            except Exception:
                self._use_gpu = False
                self._labels_gpu = None
                self._roi_sizes_gpu = None
                self._ids_gpu = None

    @property
    def backend(self) -> str:
        return "cupy" if self._use_gpu else "numpy"

    @property
    def n_rois(self) -> int:
        return len(self.roi_ids)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Return per-ROI mean intensity as a float32 np.ndarray shape (n_rois,).
        frame may be 2D (H,W) or already flat 1D (H*W,). If shape differs
        from the labels shape, a nearest-neighbour resize is applied to
        the frame to match labels. Multi-channel frames are collapsed to
        grayscale by averaging channels."""
        frame = np.asarray(frame)
        if frame.ndim == 3:
            # Collapse channels — equal-weight gray; callers who care about
            # weighting (e.g. green-channel-only for GCaMP) should do it
            # upstream before passing.
            frame = frame.mean(axis=2)
        if frame.ndim == 2 and self._labels_shape is not None:
            if frame.shape != self._labels_shape:
                frame = _resize_nn(frame, self._labels_shape)
        flat = np.ascontiguousarray(
            frame.reshape(-1).astype(np.float32, copy=False)
        )
        if flat.size != self._labels_flat.size:
            # last-ditch size match: reshape to labels size by linear
            # interpolation. Rare path — should have been caught above.
            flat = np.resize(flat, self._labels_flat.size).astype(np.float32)

        with self._lock:
            if self._use_gpu:
                return self._extract_gpu(flat)
            return self._extract_cpu(flat)

    def _extract_cpu(self, flat: np.ndarray) -> np.ndarray:
        sums = np.bincount(
            self._labels_flat, weights=flat, minlength=self._max_label + 1
        )
        means = (sums[self._ids_np] / self._roi_sizes_np).astype(np.float32)
        if self._neuropil_r > 0 and self._npil_labels_flat is not None:
            npil_sums = np.bincount(
                self._npil_labels_flat, weights=flat, minlength=self._max_label + 1
            )
            npil_means = (npil_sums[self._ids_np] / self._npil_sizes_np).astype(np.float32)
            means = means - self._neuropil_r * npil_means
        return means

    def _extract_gpu(self, flat: np.ndarray) -> np.ndarray:
        if self._frame_gpu is None or self._frame_gpu.size != flat.size:
            self._frame_gpu = _cp.empty(flat.size, dtype=_cp.float32)
        self._frame_gpu.set(flat)
        sums = _cp.bincount(
            self._labels_gpu,
            weights=self._frame_gpu,
            minlength=self._max_label + 1,
        )
        means = sums[self._ids_gpu] / self._roi_sizes_gpu
        if self._neuropil_r > 0 and self._npil_labels_gpu is not None:
            npil_sums = _cp.bincount(
                self._npil_labels_gpu,
                weights=self._frame_gpu,
                minlength=self._max_label + 1,
            )
            npil_means = npil_sums[self._ids_gpu] / self._npil_sizes_gpu
            means = means - self._neuropil_r * npil_means
        return _cp.asnumpy(means).astype(np.float32, copy=False)

    def extract_dff(
        self,
        frame: np.ndarray,
        baseline: np.ndarray,
        percentile: float = 20.0,
    ) -> np.ndarray:
        """Return per-ROI ΔF/F₀ where F₀ is computed from baseline array.
        baseline: 2D array (n_frames, n_rois) of prior raw means.
        Returns float32 array shape (n_rois,)."""
        raw = self.extract(frame)
        if baseline.size == 0 or baseline.shape[0] < 3:
            return np.zeros_like(raw)
        f0 = np.percentile(baseline, percentile, axis=0).astype(np.float32)
        f0 = np.where(np.abs(f0) < 1e-6, 1.0, f0)
        return ((raw - f0) / f0).astype(np.float32)

    def close(self) -> None:
        """Release GPU arrays. Safe to call multiple times."""
        with self._lock:
            self._labels_gpu = None
            self._roi_sizes_gpu = None
            self._ids_gpu = None
            self._frame_gpu = None
            if self._use_gpu:
                try:
                    _cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass


def _resize_nn(frame: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Nearest-neighbour resize a 2D frame to target_shape (H,W). Pure numpy
    so we don't pull cv2 into the hot path."""
    th, tw = target_shape
    sh, sw = frame.shape
    if (sh, sw) == (th, tw):
        return frame
    ys = (np.arange(th) * sh // th).astype(np.int64)
    xs = (np.arange(tw) * sw // tw).astype(np.int64)
    return frame[ys[:, None], xs[None, :]]


def extract_single_roi(frame: np.ndarray, roi_mask: np.ndarray) -> float:
    """Convenience for single-ROI callers (Trace Test dialog).
    roi_mask is a boolean 2D array of the same shape as frame (after any
    channel collapse). Returns mean pixel value inside the mask.
    No CuPy path — a single-ROI mean is cheap on CPU."""
    frame = np.asarray(frame)
    if frame.ndim == 3:
        frame = frame.mean(axis=2)
    m = np.asarray(roi_mask, dtype=bool)
    if m.shape != frame.shape:
        m = _resize_nn(m.astype(np.uint8), frame.shape).astype(bool)
    if not m.any():
        return 0.0
    return float(frame[m].mean())


__all__ = ["TraceExtractor", "extract_single_roi", "build_neuropil_labels"]
