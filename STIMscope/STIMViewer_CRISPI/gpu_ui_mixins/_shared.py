"""Shared module-level constants for the gpu_ui mixin package.

Mirrors the CUDA detection block at the top of gpu_ui.py so mixin
methods can read CUDA_AVAILABLE / CUDA_USABLE / cp without those
names having to be in the parent gpu_ui module namespace.

alongside the qt_interface_mixins/_shared.py
pattern after the folder reorg surfaced NameError crashes in
mixin method bodies.
"""
from __future__ import annotations

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    CUDA_AVAILABLE = False

# Validate CUDA runtime usability (driver/runtime compat), not just import.
# Mirror of gpu_ui.py:37-49 — kept in sync so behavior is identical
# whether the consumer imports from gpu_ui or from this shared module.
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
