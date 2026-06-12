import sys
import time
from typing import Optional
import os
import cv2
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, pyqtSlot as Slot
from PyQt5.QtGui import QGuiApplication, QPixmap
import numpy as np
from ids_peak import ids_peak
from camera import Camera

from PyQt5.QtWidgets import (
    QLabel, QVBoxLayout, QWidget, QFrame, QSizePolicy
)
from pathlib import Path

from qt_interface_mixins.camera_controls import CameraControlsMixin
from qt_interface_mixins.hw_acq import HardwareAcqMixin
from qt_interface_mixins.led_and_procs import LEDAndProcessMixin
from qt_interface_mixins.mask_ops import MaskOpsMixin
from qt_interface_mixins.overlay_probe import OverlayProbeMixin
from qt_interface_mixins.sensor_settings import SensorSettingsMixin
from qt_interface_mixins.trace_test import TraceTestMixin
from qt_interface_mixins.trig_params import TrigParamsMixin
from qt_interface_mixins.troubleshoot import TroubleshootMixin
from qt_interface_mixins.offline_setup import OfflineSetupDialogMixin
from qt_interface_mixins.button_bar import ButtonBarMixin
from qt_interface_mixins.i2c_dialog import I2CDialogMixin
from qt_interface_mixins.triggers import TriggerControlsMixin
from qt_interface_mixins.sl_calibrate import SLCalibrateMixin
from qt_interface_mixins.image_received import ImageReceivedMixin
from qt_interface_mixins.calib_projector import CalibrationProjectorMixin
from qt_interface_mixins.window_lifecycle import WindowLifecycleMixin
from qt_interface_mixins.startup_window import StartupWindowMixin
from qt_interface_mixins.projection_controls import ProjectionControlsMixin

# ASSETS + _GPU_AVAILABLE moved to qt_interface_mixins/_shared.py
#  so the mixin package can share them without circular imports.
from qt_interface_mixins._shared import ASSETS, _GPU_AVAILABLE  # noqa: F401


class _TiffViewer(QtWidgets.QMainWindow):
    """Lightweight viewer for multi-page TIFF recordings with frame slider and auto-contrast."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        import tifffile
        self.setWindowTitle(f"TIFF Viewer — {os.path.basename(path)}")
        self._path = path
        self._tif = tifffile.TiffFile(path)
        self._n = len(self._tif.pages)
        self._current = 0
        self._auto_contrast = True

        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        v = QtWidgets.QVBoxLayout(w)
        self._label = QtWidgets.QLabel()
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        self._label.setMinimumSize(800, 600)
        self._label.setStyleSheet("background-color: #000;")
        v.addWidget(self._label, 1)

        h = QtWidgets.QHBoxLayout()
        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, self._n - 1))
        self._slider.valueChanged.connect(self._show_frame)
        self._info = QtWidgets.QLabel()
        self._check = QtWidgets.QCheckBox("Auto-contrast")
        self._check.setChecked(True)
        self._check.toggled.connect(self._toggle_contrast)
        h.addWidget(QtWidgets.QLabel("Frame:"))
        h.addWidget(self._slider, 1)
        h.addWidget(self._info)
        h.addWidget(self._check)
        v.addLayout(h)
        self.resize(1000, 750)
        if self._n > 0:
            self._show_frame(0)
        else:
            self._info.setText("(no frames)")

    def _toggle_contrast(self, checked):
        self._auto_contrast = bool(checked)
        self._show_frame(self._current)

    def _show_frame(self, idx):
        self._current = int(idx)
        try:
            arr = self._tif.pages[self._current].asarray()
        except Exception as e:
            self._info.setText(f"Frame {self._current + 1}/{self._n}: read error: {e}")
            return
        if arr.ndim == 3:
            arr = arr.mean(axis=2) if arr.shape[2] > 1 else arr.squeeze()
        raw_min = int(arr.min()); raw_max = int(arr.max()); raw_mean = float(arr.mean())
        if self._auto_contrast:
            lo, hi = np.percentile(arr, (1, 99))
            if hi > lo:
                disp = np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
            else:
                disp = arr.astype(np.uint8, copy=False)
        else:
            if arr.dtype != np.uint8:
                disp = (arr.astype(np.float32) / max(1.0, float(arr.max())) * 255.0).astype(np.uint8)
            else:
                disp = arr
        disp = np.ascontiguousarray(disp)
        h, w = disp.shape
        img = QtGui.QImage(disp.tobytes(), w, h, w, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(img).scaled(
            self._label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._label.setPixmap(pix)
        self._info.setText(
            f"{self._current + 1}/{self._n}  raw min={raw_min} max={raw_max} mean={raw_mean:.1f}")

    def closeEvent(self, event):
        try:
            self._tif.close()
        except Exception:
            pass
        super().closeEvent(event)


class Interface(CameraControlsMixin, HardwareAcqMixin, LEDAndProcessMixin, MaskOpsMixin, OverlayProbeMixin, SensorSettingsMixin, TraceTestMixin, TrigParamsMixin, TroubleshootMixin, OfflineSetupDialogMixin, ButtonBarMixin, I2CDialogMixin, TriggerControlsMixin, SLCalibrateMixin, ImageReceivedMixin, CalibrationProjectorMixin, WindowLifecycleMixin, StartupWindowMixin, ProjectionControlsMixin, QtWidgets.QMainWindow):
  

    messagebox_pyqtSignal = QtCore.pyqtSignal(str, str)
    image_update_signal = QtCore.pyqtSignal(object)
    fps_update_signal = QtCore.pyqtSignal(float)
    sl_decode_done = QtCore.pyqtSignal(bool, str)
    from camera import Camera

    def __init__(self, cam_module: Optional[Camera] = None):

        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        self._qt_instance = app

        super().__init__()  # only after app exists
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        QtWidgets.QApplication.setQuitOnLastWindowClosed(True)
        self._closing = False
        
        # (Reverted) Global modern styling disabled to restore default compact widgets

        if cam_module is None:
            try:
                self._camera = Camera(ids_peak.DeviceManager.Instance(), self)
            except Exception as e:
                print("WARN", f"Camera not available: {e}")
                print("WARN", "Running without camera — simulation and offline features still work")
                self._camera = None
        else:
            self._camera = cam_module


        from video_recorder import VideoRecorder

        def _notify_finalized(path: str):
            QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.information(
                self, "Recording Complete", f"Saved video:\n{path}"
            ))

        if self._camera is not None and (not hasattr(self._camera, "video_recorder") or self._camera.video_recorder is None):
            self._camera.video_recorder = VideoRecorder(interface=self, on_finalized=_notify_finalized)

        # Default camera type (can be changed in GUI)
        self.selected_camera_type = "IDS_Peak"

        self.last_frame_time = time.time()
        self.gpu_ui = None
        
        self.gui_init()
        
        # Read back camera's actual exposure for the text field
        if hasattr(self, '_exp_line'):
            try:
                nm = getattr(self._camera, "node_map", None)
                if nm is not None:
                    actual = nm.FindNode("ExposureTime").Value()
                    self._exp_line.setText(f"{actual:.3f}")
            except Exception:
                pass
        
        self._qt_instance.aboutToQuit.connect(self._close)
        try:
            self.sl_decode_done.connect(self._on_sl_decode_done, QtCore.Qt.QueuedConnection)
        except Exception:
            pass

        # No minimum size restriction - allow window to be resized freely
        self.setWindowTitle("STIMscope")
        
        # Set window icon if available
        icon_path = self._findprinto()
        if icon_path:
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))
        # Contrast/preview defaults: disable software contrast for performance; enable only if explicitly set
        try:
            self._soft_contrast_active = False
            self._has_hw_contrast = False
            self._contrast_factor = 1.0
            self._contrast_lut = None
            self._contrast_lut_factor = 1.0
        except Exception:
            pass
    @staticmethod
    def _findprinto():
        candidates = [
            ASSETS / "stimviewer-load.png",
            ASSETS / "UI" / "stimviewer-load.png",
            ASSETS / "Images" / "stimviewer-load.png",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None



    def gui_init(self):
        container = QWidget()

        self._layout = QVBoxLayout(container)
        self.setCentralWidget(container)
        from display import Display

        self.display = Display()
        # Let the display resize freely; fixed max can stress layout/paint
        # Keep a reasonable minimum, but no artificial maximum
        self.display.setMinimumSize(320, 240)
        self.display.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._layout.addWidget(self.display)
        self.projection = None
        self._projection_active = False  # Track projection state
        self.acquisition_thread = None


        self._button_software_trigger = None
        self._button_start_hardware_acquisition = None
        self._hardware_status = False #False = Display Start, False = End
        self._recording_status = False #False = Display Start, False = End
        # External process handles (non-blocking)
        self._proc_i2c = None
        self._proc_masks = None
        self._proc_projector = None




        self._dropdown_pixel_format = None
        self._dropdown_trigger_line = None # Dropdown for hardware trigger line





        
        self._button_show_gpu_ui = None

        self.messagebox_pyqtSignal.connect(self.message)
        for sig, slot in (("recordingStarted", self._on_recording_started),
                          ("recordingStopped", self._on_recording_stopped),
                          ("autoStartRecording", self._on_auto_start_recording)):
            try:
                getattr(self._camera, sig).connect(slot)
            except Exception:
                pass

        self._frame_count = 0
        self._gain_label = None

        self._gain_slider = None
        # Contrast control defaults
        self._has_hw_contrast = False
        self._soft_contrast_active = True
        self._contrast_factor = 1.0



    def is_gui(self):
        return True
    
    def set_camera(self, cam_module):
        self._camera = cam_module
    
    def _set_compact_width_to_text(self, widget, extra_px: int = 24):
        try:
            fm = widget.fontMetrics()
            text = widget.currentText() if hasattr(widget, 'currentText') else widget.text()
            width = fm.horizontalAdvance(text) + extra_px
            if width > 0:
                widget.setFixedWidth(width)
        except Exception:
            pass
    

    def _ensure_qprocess(self):
        # Lazy import to avoid startup penalty if unused
        from PyQt5.QtCore import QProcess
        return QProcess

    # _maybe_build_projector + _helper_python_path_for_masks +
    # _on_mask_pattern_changed + _browse_mask_pattern_path extracted to
    # qt_interface_mask_ops.py (MaskOpsMixin) per L5 §0.5 decomposition (iter-4).

    # _on_led_color_changed_live + _apply_led_color_live extracted to
    # qt_interface_led_and_procs.py (LEDAndProcessMixin) per L5 §0.5
    # decomposition (iter-3).

    # _toggle_send_masks extracted to qt_interface_mask_ops.py
    # (MaskOpsMixin) per L5 §0.5 decomposition (iter-4).

    # _on_proc_finished + _terminate_external_processes extracted to
    # qt_interface_led_and_procs.py (LEDAndProcessMixin) per L5 §0.5
    # decomposition (iter-3).

    # _trigger_sw_trigger + _start_hardware_acquisition + _start_recording
    # extracted to qt_interface_hw_acq.py (HardwareAcqMixin) per L5 §0.5
    # decomposition (iter-2).

    def _apply_modern_style(self):
        # Styling intentionally disabled for revert.
        return

    # _on_camera_type_changed + change_pixel_format +
    # change_hardware_trigger_line extracted to
    # qt_interface_camera_controls.py (CameraControlsMixin) per L5 §0.5
    # decomposition (iter-8).

    @QtCore.pyqtSlot(object)
    def warning(self, message: str):
        self.messagebox_pyqtSignal.emit("Warning", message)

    def information(self, message: str):
        self.messagebox_pyqtSignal.emit("Information", message)



    def show_gpu_ui(self):
        import time as _t
        _t0 = _t.time()
        print("[show_gpu_ui] click handler entered")
        try:
            from gpu_ui import GPU
            print(f"[show_gpu_ui] gpu_ui imported in {_t.time()-_t0:.2f}s")
        except ImportError as e:
            print(f"Trace extraction UI not available: {e}")
            return

        if not _GPU_AVAILABLE:
            print("Trace extraction UI not available in this environment.")
            return
        if self.gpu_ui is None:
            print("[show_gpu_ui] constructing GPU(...) — first time")
            _t1 = _t.time()
            try:
                self.gpu_ui = GPU(camera=self._camera, parent=self)
            except TypeError:
                self.gpu_ui = GPU(camera=self._camera)
                self.gpu_ui.setParent(self)
            except Exception as e:
                import traceback
                print(f"[show_gpu_ui] GPU(...) raised: {e}")
                traceback.print_exc()
                return
            print(f"[show_gpu_ui] GPU constructed in {_t.time()-_t1:.2f}s")
            # Free memory on close: destroy the window on close and drop our
            # reference so the next open reconstructs a fresh instance
            # (~0.04 s). Without this the trace extractor, buffers, and the
            # health-monitor QTimer chains would outlive the closed window —
            # the source of post-close "high memory" warnings + gc.collect churn.
            try:
                self.gpu_ui.setAttribute(Qt.WA_DeleteOnClose, True)
                self.gpu_ui.closed.connect(lambda: setattr(self, "gpu_ui", None))
            except Exception as _e:
                print(f"[show_gpu_ui] could not wire close-cleanup: {_e}")
        else:
            print("[show_gpu_ui] reusing existing GPU instance")
        try:
            self.gpu_ui.setWindowFlags(Qt.Tool)
            # Place the dialog on the SAME screen as the main window — not
            # the "primary" screen, which on a STIMscope setup is often the
            # projector/DMD monitor at x>=1920. self.screen() returns the
            # screen the main window is currently on.
            try:
                screen = self.screen()
            except Exception:
                screen = None
            if screen is None:
                screen = QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                geom = screen.availableGeometry()
                self.gpu_ui.move(geom.x() + 80, geom.y() + 80)
            self.gpu_ui.show()
            self.gpu_ui.raise_()
            self.gpu_ui.activateWindow()
            print(f"[show_gpu_ui] show()+raise()+activate() done. visible={self.gpu_ui.isVisible()} "
                  f"geo={self.gpu_ui.geometry()} total {_t.time()-_t0:.2f}s")
        except Exception as e:
            import traceback
            print(f"[show_gpu_ui] show() raised: {e}")
            traceback.print_exc()




    @Slot(str, str)
    def message(self, typ: str, message: str):
        if typ == "Warning":
            QtWidgets.QMessageBox.warning(
                self, "Warning", message, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.information(
                self, "Information", message, QtWidgets.QMessageBox.Ok)


    # change_slider_gain + _update_gain + change_slider_dgain +
    # _update_dgain + _set_camera_contrast + _make_contrast_lut +
    # _apply_exposure_from_text extracted to qt_interface_camera_controls.py
    # (CameraControlsMixin) per L5 §0.5 decomposition (iter-8).

    # _open_sensor_settings extracted to qt_interface_sensor_settings.py
    # (SensorSettingsMixin) per L5 §0.5 decomposition (iter-6).


    # Zoom slider methods removed - using mouse wheel zoom instead

    # ── Trace Extraction Test ─────────────────────────────────────────
    # _open_trace_test_dialog extracted to qt_interface_trace_test.py
    # (TraceTestMixin) per L5 §0.5 decomposition (iter-7).

