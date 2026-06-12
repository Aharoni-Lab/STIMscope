"""Shared module-level constants for the qt_interface mixin package.

Holds names that previously lived at the top of ``qt_interface.py`` and
were referenced from mixin method bodies. After the  folder
reorg (qt_interface_*.py → qt_interface_mixins/), those names were no
longer in the same module as the methods that used them, causing
``NameError`` at runtime. Centralizing them here gives every mixin a
single canonical import target.

If you add a new name that >=2 files need, put it here.
"""
from __future__ import annotations

from pathlib import Path

# Repository assets directory (PNG icons, generated calibration files, etc.).
# Resolved relative to this file's parent dir so the path is stable no
# matter where the mixin package is imported from.
ASSETS = (Path(__file__).resolve().parent.parent / "Assets").resolve()

# Whether the GPU sub-window can be enabled. Currently unconditional;
# real CUDA-runtime detection lives in gpu_ui.py and is checked at
# GPU sub-window construction time. Kept as a module-level flag so the
# main-window button can be disabled in environments where GPU import
# is known to fail at startup.
_GPU_AVAILABLE = True
