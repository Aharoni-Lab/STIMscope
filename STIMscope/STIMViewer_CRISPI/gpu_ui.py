
import os
import time
import gc
import signal
import atexit
import psutil
import sys
import threading
from collections import deque
from typing import Optional

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import subprocess

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout, QTextEdit, QLabel
)

from PyQt5.QtCore import QTimer, pyqtSignal, pyqtSlot

PLOT_WITH_PYQTGRAPH = True  
ENABLE_GPUUI_HTMLprint = False  

def _noop(*a, **kw): pass

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False

# Validate CUDA runtime usability (driver/runtime compatibility), not just import
CUDA_USABLE = False
if CUDA_AVAILABLE:
    try:
        import cupy.cuda.runtime as _cur
        ndev = _cur.getDeviceCount()
        if ndev and ndev > 0:
            _ = cp.arange(1, dtype=cp.int8)
            CUDA_USABLE = True
        else:
            print("ℹ️ No CUDA devices detected; GPU features disabled")
    except Exception as _e_rt:
        CUDA_USABLE = False
        print(f"⚠️ CUDA runtime unusable; GPU features disabled: {_e_rt}")

TRACE_OUT = "live_traces.npy"
ROIprint_OUT = "roiprint_export.npz"

CAMERA_AVAILABLE = True
Camera = None 

from live_trace.extractor import LiveTraceExtractor
from gpu_ui_mixins.roi_discovery import ROIDiscoveryMixin
from gpu_ui_mixins.traces import LiveTracesMixin
from gpu_ui_mixins.napari import NapariViewerMixin
from gpu_ui_mixins.export_fast import FastExportMixin
from gpu_ui_mixins.export_slow import SlowExportMixin
from gpu_ui_mixins.export_viewer import ExportViewerMixin
from gpu_ui_mixins.export_tabs import ExportViewerTabsMixin
from gpu_ui_mixins.health import HealthMonitoringMixin

__all__ = ["GPU"]

class GPU(FastExportMixin, SlowExportMixin, ExportViewerMixin, ExportViewerTabsMixin, NapariViewerMixin, LiveTracesMixin, ROIDiscoveryMixin, HealthMonitoringMixin, QtWidgets.QWidget):


    closed = pyqtSignal()

    refineRequested = pyqtSignal(object, object)
    requestStartLiveTraces = pyqtSignal()
    requestStopLiveTraces = pyqtSignal()

    instance: Optional["GPU"] = None

    export_count = 0

    def __init__(self, camera: Camera,parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        if camera is None:
            raise ValueError("GPU UI requires a Camera instance")
        self.camera = camera
        GPU.instance = self
        self._shutting_down = False

        self.setWindowTitle("Real-Time Trace Extraction")
        self.resize(800, 560)


        self.requestStartLiveTraces.connect(self.start_live_traces, QtCore.Qt.QueuedConnection)
        self.requestStopLiveTraces.connect(self.stop_live_traces, QtCore.Qt.QueuedConnection)

        self.refineRequested.connect(self._launch_napari_viewer)

        self.layout = QVBoxLayout(self)


        self.plot_widget = None
        if PLOT_WITH_PYQTGRAPH:
            try:
                import pyqtgraph as pg
                self.plot_widget = pg.PlotWidget()
                self.plot_widget.setBackground('k')
                self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
                self.plot_widget.setMouseEnabled(x=False, y=False)
                self.plot_widget.setYRange(0, 255)
                try:
                    self.plot_widget.setLabel('left', 'Intensity')
                    self.plot_widget.setLabel('bottom', 'Time (frames)')
                except Exception:
                    pass
                self.layout.addWidget(self.plot_widget)
            except Exception as e:
                print(f"pyqtgraph unavailable, continuing without on-screen traces: {e}")

        self._trace_mode_combo = QtWidgets.QComboBox()
        self._trace_mode_combo.addItems(["Raw", "ΔF/F₀", "z-score", "Spikes"])
        self._trace_mode_combo.setToolTip("Trace display mode: Raw intensity, ΔF/F₀, z-score, or OASIS spikes")
        self._trace_mode_combo.setFixedWidth(120)
        self._trace_mode_combo.currentTextChanged.connect(self._on_trace_mode_changed)
        self.layout.addWidget(self._trace_mode_combo)

        self.paused = False


        self.video_path = None
        self.proj_display = None
        # Persistent paths under STIM_SAVE_DIR (the launcher mounts this from
        # the host so artifacts survive container --rm). Falls back to CWD for
        # ad-hoc runs without the env var.
        _save_dir = os.environ.get("STIM_SAVE_DIR") or "."
        try:
            os.makedirs(_save_dir, exist_ok=True)
        except Exception:
            pass
        self.memmap_path = os.path.join(_save_dir, "movie_mmap.npy")
        self.rois_path = os.path.join(_save_dir, "rois.npz")
        self.trace_path = os.path.join(_save_dir, "traces_live.npy")
        self._discover_method = "OTSU"


        from live_trace.extractor import LiveTraceExtractor
        self.live_extractor: Optional[LiveTraceExtractor] = None

        self._build_pipeline_buttons()

        self._setup_long_term_stability()


    def _build_pipeline_buttons(self):
        grid = QtWidgets.QGridLayout()
        row = 0


        btn = QtWidgets.QPushButton("🖼 Select Video…")
        btn.clicked.connect(self._select_video)
        grid.addWidget(btn, row, 0)


        btn = QtWidgets.QPushButton("➤ Make Memmap")
        btn.clicked.connect(self._run_make_memmap)
        grid.addWidget(btn, row, 1)


        dd = QtWidgets.QToolButton()
        dd.setText("➤ Discover Mask")
        dd.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(dd)
        for method in ("Cellpose", "CNMF", "Custom", "OTSU"):
            act = QtWidgets.QAction(method, dd)
            act.triggered.connect(lambda _=False, m=method: self._run_discover_rois(m))
            menu.addAction(act)
        dd.setMenu(menu)
        grid.addWidget(dd, row, 2)


        # Manual Mask Editor button removed:
        # the manual mask editing workflow is incomplete and the
        # `_run_refine_rois` handler is a stub. To be reimplemented as a
        # future feature (tracked in docs/specs/L5_UI/gpu_ui.md §12 D-gu-MM).
        # Handler kept for now to avoid breaking any other callers.


        btn = QtWidgets.QPushButton("📂 Load ROI File…")
        btn.setToolTip(
            "Load an existing ROI file (NPZ with 'labels' array). "
            "Use this to pull segmented neurons from Offline Setup into live "
            "trace extraction. Expected keys: 'labels' (int H×W), optional "
            "'neuron_ids', 'centroids'.")
        btn.clicked.connect(self._load_roi_file)
        grid.addWidget(btn, row, 4)


        btn = QtWidgets.QPushButton("▶ Export Traces")
        btn.clicked.connect(self._export_traces)
        grid.addWidget(btn, row, 5)


        row += 1
        btn = QtWidgets.QPushButton("👁️ View Exported Traces")
        btn.clicked.connect(self._view_exported_traces)
        grid.addWidget(btn, row, 0, 1, 2)  # Span 2 columns

        # OASIS (Online) toggle under Discover Mask
        try:
            self._button_oasis_online = QtWidgets.QPushButton("OASIS (Online)")
            self._button_oasis_online.setCheckable(True)
            self._button_oasis_online.setChecked(False)
            self._button_oasis_online.setToolTip("Apply fast online OASIS deconvolution to ROI traces (enabled only when pressed)")
            self._button_oasis_online.toggled.connect(self._toggle_oasis)
            grid.addWidget(self._button_oasis_online, row, 2)
        except Exception:
            pass

        self.layout.addLayout(grid)






