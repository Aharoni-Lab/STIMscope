"""ProjectionControlsMixin — extracted from qt_interface.py.

Bundles five projection-control methods:

* ``_calibrate()`` (~53 LOC) — initial homography calibration via
  manual point picking.
* ``_update_project_intensity()`` (~9 LOC) — slider→projection
  intensity update.
* ``_project_on()`` (~14 LOC) — turn projector RGB output on.
* ``_project_off()`` (~13 LOC) — turn projector RGB output off.
* ``_project_with_intensity(intensity)`` (~12 LOC) — project a
  solid color at the given intensity.

Method bodies are byte-identical to the pre-extraction code at
``qt_interface.py:504-604`` (commit ``3fb0ab2``); only the
surrounding module-level frame changed.

Mixin contract:
  * ``self._ensure_projection`` — provided by StartupWindowMixin
  * ``self.projection`` — second-monitor window
  * ``self._projection_active`` — bool flag
  * ``self.project_intensity_slider`` — QSlider for value reads

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

from qt_interface_mixins._shared import ASSETS
from pathlib import Path


class ProjectionControlsMixin:
    """Cluster 20 — calibrate + project-on/off + intensity controls."""

    def _calibrate(self):

        if not self._ensure_projection():
            print("Calibration aborted: projection window unavailable.")
            return
        try:
            img_path = ASSETS / "Generated" / "custom_registration_image.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            scr = self.projection.windowHandle().screen() if self.projection.windowHandle() else None
            geo = scr.geometry() if scr else None
            target_w = geo.width() if geo else 1920
            target_h = geo.height() if geo else 1080

            # Build the projected registration pattern from the ChArUco board.
            # Prefer the bundled (or operator-supplied) board; if it is somehow
            # missing, generate one on the fly. This implements the previously
            # unimplemented create_charuco_registration_image so calibration is
            # self-contained — no dev-machine-specific board file required.
            from calibration import CHARUCO_BOARD_IMG, generate_registration_board
            board_src = CHARUCO_BOARD_IMG
            if board_src.exists():
                probe = cv2.imread(str(board_src), cv2.IMREAD_COLOR)
                if probe is not None:
                    ph, pw = probe.shape[:2]
                    if pw != target_w or ph != target_h:
                        probe = cv2.resize(probe, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(str(img_path), probe)
                    print(f"Calibration board loaded from {board_src}")
            if not img_path.exists():
                if generate_registration_board(img_path, target_w, target_h):
                    print(f"Generated ChArUco registration board ({target_w}x{target_h})")

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Calibration image not readable: {img_path}")
                return

            # Respect current warp mode: H uses homography, LUT uses prewarped content (no H)
            if getattr(self, '_proj_warp_mode', 'H') == 'H':
                self.projection.show_image_fullscreen_on_second_monitor(
                    img,
                    getattr(self._camera, "translation_matrix", None)
                )
            else:
                self.projection.show_image_fullscreen_on_second_monitor(
                    img,
                    None
                )

            # Allow time for projector to refresh and camera to capture a few frames
            QtCore.QTimer.singleShot(250, lambda: getattr(self._camera, "start_calibration", lambda: None)())
        except Exception as e:
            print(f"Calibration start failed: {e}")

    
    def _update_project_intensity(self):
        """Update the intensity value label when slider changes."""
        intensity = self._project_intensity_slider.value()
        self._project_intensity_value_label.setText(str(intensity))
        
        # If projection is currently on, update it with new intensity
        if hasattr(self, '_projection_active') and self._projection_active:
            self._project_with_intensity(intensity)
    
    def _project_on(self):
        """Turn on projection with current intensity setting."""
        try:
            if not self._ensure_projection():
                print("Projection window unavailable.")
                return
                
            intensity = self._project_intensity_slider.value()
            self._project_with_intensity(intensity)
            self._projection_active = True
            
        except Exception as e:
            print(f"_project_on failed: {e}")
    
    def _project_off(self):
        """Turn off projection (black screen)."""
        try:
            if not self._ensure_projection():
                print("Projection window unavailable.")
                return
                
            self.projection.show_solid_fullscreen((0, 0, 0))
            self._projection_active = False
            
        except Exception as e:
            print(f"_project_off failed: {e}")

    def _project_with_intensity(self, intensity):
        """Project a solid color with the specified intensity (0-255)."""
        try:
            if not self._ensure_projection():
                print("Projection window unavailable.")
                return
                
            # Use the intensity value for all RGB channels (grayscale)
            self.projection.show_solid_fullscreen((intensity, intensity, intensity))
            
        except Exception as e:
            print(f"_project_with_intensity failed: {e}")

