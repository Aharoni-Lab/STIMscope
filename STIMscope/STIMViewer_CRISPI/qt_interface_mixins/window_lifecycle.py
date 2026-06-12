"""WindowLifecycleMixin — extracted from qt_interface.py.

Bundles the six window-scaffolding / lifecycle methods:

* ``_create_statusbar()`` (~80 LOC) — builds the bottom status bar
  with FPS / queue / preview-toggle indicators.
* ``_tick_fps_refresh()`` (~13 LOC) — timer-driven GUI FPS sampler.
* ``_set_gui_fps(fps)`` (~16 LOC) — updates the GUI-side FPS label.
* ``_close()`` (~12 LOC) — request shutdown of cooperating windows.
* ``_on_sl_decode_done(ok, msg)`` (~11 LOC) — structured-light decode
  completion handler routing to message popup + status update.
* ``closeEvent(event)`` (~33 LOC) — Qt close handler with
  terminate_external_processes + accept().

Method bodies are byte-identical to the pre-extraction code at
``qt_interface.py:323-487`` (commit ``3079403``); only the
surrounding module-level frame changed.

Mixin contract (Interface attributes the method reads/writes):
  * ``self._sl_progress``, ``self._sl_status`` — created in button
    bar, populated here on the status row.
  * ``self.acq_label``, ``self.queue_label``, ``self.fps_label`` —
    status-bar QLabel refs created here.
  * ``self._fps_timer`` — QTimer for the FPS sampler.
  * ``self._terminate_external_processes`` — provided by
    LEDAndProcessMixin.
  * ``self.message`` — for SL-decode-done popup.

See ``docs/specs/L5_UI/qt_interface.md``.
"""

import os
import sys
import time

import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QGuiApplication, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

class WindowLifecycleMixin:
    """Cluster 18 — main-window status bar + FPS + close lifecycle."""

    def _create_statusbar(self):
       
        status_bar = QtWidgets.QWidget(self.centralWidget())
        status_bar.setMaximumHeight(30)
        try:
            status_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        except Exception:
            pass
        status_bar_layout = QtWidgets.QHBoxLayout()
        status_bar_layout.setContentsMargins(5, 2, 5, 2)  # Smaller margins


        separator = QFrame(self)
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._layout.addWidget(separator)


        self.acq_label = QLabel("Acquisition Mode: RealTime", self)
        self.acq_label.setStyleSheet("font-size: 11px; color: #1c1c1e;")
        self.acq_label.setAlignment(Qt.AlignLeft)
        self.acq_label.setToolTip("Current Acquisition Mode")

        # Projector status
        screens = QGuiApplication.screens()
        self.projector_status_label = QLabel(self)
        if len(screens) > 1:
            self.projector_status_label.setText("✅ Projector Connected")
            self.projector_status_label.setStyleSheet("font-size: 11px; color: #27ae60;")
        else:
            self.projector_status_label.setText("❌ No Projector Found")
            self.projector_status_label.setStyleSheet("font-size: 11px; color: #e74c3c;")
        self.projector_status_label.setAlignment(Qt.AlignCenter)
        self.projector_status_label.setToolTip("Projector connection status")

        self.GUIfps_label = QLabel("FPS: 0", self)
        self.GUIfps_label.setStyleSheet("font-size: 11px; color: #1c1c1e;")
        self.GUIfps_label.setAlignment(Qt.AlignRight)
        self.GUIfps_label.setToolTip(
            "Live frame rate the camera is actually delivering, averaged over the "
            "last 2 s — NOT the configured trigger / max rate. If this stays below "
            "the configured rate, the camera's frame time (exposure + sensor readout) "
            "is exceeding the trigger period, so triggers are silently missed. "
            "Reduce exposure or pixel-format bit-depth to recover the rate."
        )
        try:
            self.fps_update_signal.connect(self._set_gui_fps, QtCore.Qt.QueuedConnection)
        except Exception:
            pass
        # Periodic FPS refresh — decays label to 0 when no frames arrive
        # (previously the label froze at last-measured value)
        try:
            self._fps_refresh_timer = QtCore.QTimer(self)
            self._fps_refresh_timer.setInterval(250)  # 4 Hz — responsive without thrashing
            self._fps_refresh_timer.timeout.connect(self._tick_fps_refresh)
            self._fps_refresh_timer.start()
        except Exception:
            pass
        # SL progress widgets in status row
        try:
            self._sl_progress = QtWidgets.QProgressBar(self)
            self._sl_progress.setRange(0, 0)  # indeterminate by default
            self._sl_progress.setVisible(False)
            self._sl_progress.setMaximumWidth(160)
            self._sl_status = QLabel("", self)
            self._sl_status.setStyleSheet("font-size: 11px; color: #1c1c1e;")
        except Exception:
            self._sl_progress = None
            self._sl_status = None

        status_bar_layout.addWidget(self.acq_label)
        status_bar_layout.addSpacing(12)
        status_bar_layout.addWidget(self.projector_status_label)
        status_bar_layout.addSpacing(12)
        if getattr(self, '_sl_progress', None):
            status_bar_layout.addWidget(self._sl_progress)
        if getattr(self, '_sl_status', None):
            status_bar_layout.addWidget(self._sl_status)
        # Push FPS all the way to the right
        status_bar_layout.addStretch(1)
        status_bar_layout.addWidget(self.GUIfps_label)

        status_bar.setLayout(status_bar_layout)
        self._layout.addWidget(status_bar)

    def _tick_fps_refresh(self):
        """Pull current FPS from camera and push to the label. Runs on a QTimer
        so the label decays to 0 when frames stop arriving (e.g., wrong trigger line)."""
        try:
            cam = getattr(self, "_camera", None)
            if cam is None or not hasattr(cam, "get_actual_fps"):
                return
            fps = float(cam.get_actual_fps())
            self.fps_update_signal.emit(fps)
        except Exception:
            pass

    @QtCore.pyqtSlot(float)
    def _set_gui_fps(self, fps: float):
        try:
            capped = getattr(self, "_fps_capped", False)
            cap_value = getattr(self, "_fps_cap_value", 30)
            if capped:
                self.GUIfps_label.setText(
                    f"FPS: {int(round(fps))}  (capped at {cap_value})")
                self.GUIfps_label.setStyleSheet(
                    "font-size: 11px; color: #b26b00; font-weight: bold;")
            else:
                self.GUIfps_label.setText(f"FPS: {int(round(fps))}")
                self.GUIfps_label.setStyleSheet(
                    "font-size: 11px; color: #1c1c1e;")
        except Exception:
            pass

    def _close(self):
        try:
            # Stop helper processes first
            try:
                self._terminate_external_processes()
            except Exception:
                pass
            self._camera.shutdown()
        except Exception:
            pass

    @QtCore.pyqtSlot(bool, str)
    def _on_sl_decode_done(self, ok: bool, msg: str):
        try:
            if getattr(self, '_sl_progress', None):
                self._sl_progress.setVisible(False)
            if getattr(self, '_sl_status', None):
                self._sl_status.setText("✅ SL ready" if ok else f"❌ SL failed: {msg}")
            if hasattr(self, '_button_sl_project_reg') and self._button_sl_project_reg is not None:
                self._button_sl_project_reg.setEnabled(ok)
        except Exception:
            pass

    def closeEvent(self, event):
        try:

            if getattr(self, 'gpu_ui', None) is not None:
                try: self.gpu_ui.shutdown()
                except Exception: pass

            try: self._camera.shutdown()
            except Exception: pass


            try:
                if hasattr(self._camera, "frame_ready"):
                    self._camera.frame_ready.disconnect(self.on_image_received)
                if hasattr(self._camera, "image_ready"):
                    self._camera.image_ready.disconnect(self.on_image_received)
                iface = getattr(self._camera, "_interface", None)
                if iface is not None and hasattr(iface, "frame_ready"):
                    iface.frame_ready.disconnect(self.on_image_received)
            except Exception:
                pass

            if self.projection is not None:
                try: self.projection.close()
                except Exception: pass
            try:
                self._terminate_external_processes()
            except Exception:
                pass
        finally:
            event.accept()

           

