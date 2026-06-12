"""CameraControlsMixin — extracted from qt_interface.py per L5 §0.5 decomposition.

Cluster 6 + 7 subset (camera control surface).
14 methods, ~265 LOC.

Methods:
- ``_on_camera_type_changed(t)``    — store the selected camera type
  (effect on next restart only)
- ``change_pixel_format(*_)``       — apply dropdown pixel format
- ``change_hardware_trigger_line(*_)`` — apply trigger-line dropdown
- ``change_slider_gain(val)``       — float-slider → int-slider scaling
- ``_update_gain(val)``             — write AnalogAll gain to camera
- ``change_slider_dgain(val)``      — float-slider → int-slider for digital
- ``_update_dgain(val)``            — write DigitalAll gain to camera
- ``_set_camera_contrast(value)``   — hardware contrast via API or node map
- ``_make_contrast_lut(factor)``    — build 256-entry preview LUT
- ``_apply_exposure_from_text()``   — write ExposureTime from QLineEdit
- ``_select_warp_h()``              — toggle H-matrix warp mode
- ``_select_warp_lut()``            — toggle LUT warp mode
- ``_on_warp_h_toggled(checked)``   — H-button checkbox handler
- ``_on_warp_lut_toggled(checked)`` — LUT-button checkbox handler

Mixin contract — subclass provides:
    self.selected_camera_type, self._dropdown_pixel_format,
    self._dropdown_trigger_line
    self._gain_slider, self._gain_value_label
    self._dgain_slider, self._dgain_value_label
    self._exp_line                    : QLineEdit
    self._camera                      : OptimizedCamera-like (with .node_map,
                                        .change_pixel_format,
                                        .change_hardware_trigger_line,
                                        .set_gain, .set_contrast (optional))
    self._proj_warp_mode              : str
    self._button_req_hmatrix          : QPushButton (checkable)
    self._button_use_lut              : QPushButton (checkable)
    self._send_hmatrix_to_projector() : Interface helper

Pure hoist — no behavior change vs. monolith.
"""

from __future__ import annotations

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot as Slot


class CameraControlsMixin:
    """Cluster 6+7 subset — camera control surface (gain/exposure/contrast/
    pixel-format/trigger-line/warp)."""

    def _on_camera_type_changed(self, camera_type):
        """Handle camera type selection change."""
        self.selected_camera_type = camera_type
        print(f"Camera type changed to: {camera_type}")
        # Note: Camera type change will take effect on next restart

    def change_pixel_format(self, *_):
        pixel_format = self._dropdown_pixel_format.currentText()
        self._camera.change_pixel_format(pixel_format)

    def change_hardware_trigger_line(self, *_):
        chosen_line = self._dropdown_trigger_line.currentText()
        print(f"Chosen hardware trigger line: {chosen_line}")

        self._camera.change_hardware_trigger_line(chosen_line)

    @Slot(float)
    def change_slider_gain(self, val):
        self._gain_slider.setValue(int(val * 100))

    @Slot(int)
    def _update_gain(self, val):
        value = val / 100
        self._gain_value_label.setText(f"{value:.2f}")
        try:

            self._camera.node_map.FindNode("GainSelector").SetCurrentEntry("AnalogAll")
        except Exception:
            pass
        self._camera.set_gain(value)

    @Slot(float)
    def change_slider_dgain(self, val):
        self._dgain_slider.setValue(int(val * 100))

    @Slot(int)
    def _update_dgain(self, val):
        value = val / 100
        self._dgain_value_label.setText(f"{value:.2f}")
        try:
            self._camera.node_map.FindNode("GainSelector").SetCurrentEntry("DigitalAll")
        except Exception:
            pass
        self._camera.set_gain(value)

    def _set_camera_contrast(self, value: float):
        """Apply contrast to the camera if supported. Tries camera API first, then node map."""
        try:
            # Preferred: explicit camera method if available
            if hasattr(self._camera, "set_contrast"):
                try:
                    self._camera.set_contrast(value)
                    print(f"[CAM] Applied Contrast (method) = {float(value):.4f}")
                    return
                except Exception:
                    pass
            # Fallback to GenICam node map (Contrast or Gamma)
            nm = getattr(self._camera, "node_map", None)
            if nm is None:
                return
            node = None
            used_gamma = False
            # Try contrast nodes first
            for name in ("Contrast", "ContrastAbsolute"):
                try:
                    node = nm.FindNode(name)
                    if node is not None:
                        break
                except Exception:
                    node = None
            # If no contrast nodes, try gamma nodes
            if node is None:
                for name in ("Gamma", "GammaCorrection", "GammaValue"):
                    try:
                        node = nm.FindNode(name)
                        if node is not None:
                            used_gamma = True
                            # Some cameras require enabling gamma
                            try:
                                ge = nm.FindNode("GammaEnable")
                                if ge is not None:
                                    try:
                                        ge.SetValue(True)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            break
                    except Exception:
                        node = None
            if node is None:
                return
            # Try float, then int coercion if needed
            try:
                v = float(value)
                # Clamp gamma to a narrow, stable range to avoid large brightness shifts
                if used_gamma:
                    try:
                        v = max(0.7, min(1.3, v))
                    except Exception:
                        pass
                node.SetValue(v)
            except Exception:
                try:
                    v = int(round(value))
                    node.SetValue(v)
                except Exception:
                    return
            try:
                print(f"[CAM] Applied Contrast/Gamma (node) = {float(value):.4f}")
            except Exception:
                pass
        except Exception:
            pass

    def _make_contrast_lut(self, factor: float):
        """Build a 256-entry LUT for fast contrast application in preview."""
        try:
            import numpy as _np
            f = float(factor)
            x = _np.arange(256, dtype=_np.float32)
            y = (x - 127.5) * f + 127.5
            return _np.clip(y, 0, 255).astype(_np.uint8)
        except Exception:
            return None

    def _apply_exposure_from_text(self):
        try:
            txt = self._exp_line.text().strip()
            if not txt:
                return
            exp_us = float(txt)
            if not (exp_us > 0):
                return
            nm = getattr(self._camera, "node_map", None)
            if nm is None:
                return

            # IDS Peak: AcquisitionFrameRate caps the max ExposureTime.
            # Lower FPS first to make room, then set exposure, then raise
            # FPS back to the fastest rate the new exposure allows.
            fps_node = None
            try:
                fps_node = nm.FindNode("AcquisitionFrameRate")
            except Exception:
                pass

            if fps_node is not None:
                try:
                    needed_fps = 1_000_000.0 / exp_us
                    if needed_fps < fps_node.Value():
                        fps_node.SetValue(max(fps_node.Minimum(), needed_fps))
                except Exception:
                    pass

            try:
                nm.FindNode("ExposureTime").SetValue(exp_us)
            except Exception:
                pass

            # Raise FPS back to fastest rate this exposure allows
            if fps_node is not None:
                try:
                    max_fps = min(fps_node.Maximum(), 1_000_000.0 / exp_us)
                    fps_node.SetValue(max(fps_node.Minimum(), max_fps))
                except Exception:
                    pass

            # Read back what the camera actually accepted
            try:
                actual = nm.FindNode("ExposureTime").Value()
                if abs(actual - exp_us) > 1.0:
                    print(f"[CAM] Exposure requested {exp_us:.0f} µs, camera accepted {actual:.0f} µs")
                    self._exp_line.setText(f"{actual:.3f}")
                else:
                    print(f"[CAM] Exposure set to {actual:.0f} µs")
            except Exception:
                print(f"[CAM] Exposure set to {exp_us:.0f} µs (readback failed)")
        except Exception as e:
            print(f"Exposure apply failed: {e}")

    def _select_warp_h(self):
        # Toggle behavior: if already active, turn off; else activate H and deactivate LUT
        try:
            if getattr(self, '_proj_warp_mode', 'H') == 'H' and self._button_req_hmatrix.isChecked():
                # Deactivate
                self._proj_warp_mode = "NONE"
                self._button_req_hmatrix.setChecked(False)
                print("[PROJ] Warp mode: None (no H applied)")
            else:
                self._proj_warp_mode = "H"
                if hasattr(self, '_button_req_hmatrix'):
                    self._button_req_hmatrix.setChecked(True)
                if hasattr(self, '_button_use_lut'):
                    self._button_use_lut.setChecked(False)
                # Send H to projector immediately
                self._send_hmatrix_to_projector()
                print("[PROJ] Warp mode: Homography (H)")
        except Exception as e:
            print(f"Warp H select failed: {e}")

    def _select_warp_lut(self):
        # Toggle behavior: if already active, turn off; else activate LUT and deactivate H
        try:
            if getattr(self, '_proj_warp_mode', 'H') == 'LUT' and self._button_use_lut.isChecked():
                self._proj_warp_mode = "NONE"
                self._button_use_lut.setChecked(False)
                print("[PROJ] Warp mode: None (no H; content not assumed prewarped)")
            else:
                self._proj_warp_mode = "LUT"
                if hasattr(self, '_button_req_hmatrix'):
                    self._button_req_hmatrix.setChecked(False)
                if hasattr(self, '_button_use_lut'):
                    self._button_use_lut.setChecked(True)
                print("[PROJ] Warp mode: LUT (engine will display prewarped content)")
        except Exception as e:
            print(f"Warp LUT select failed: {e}")

    def _on_warp_h_toggled(self, checked: bool):
        if checked:
            # activate H
            self._proj_warp_mode = "H"
            try:
                if hasattr(self, '_button_use_lut'):
                    self._button_use_lut.setChecked(False)
            except Exception:
                pass
            self._send_hmatrix_to_projector()
            print("[PROJ] Warp mode: Homography (H)")
        else:
            # if H turned off and LUT not active → NONE
            if (getattr(self, '_button_use_lut', None) is None) or (not self._button_use_lut.isChecked()):
                self._proj_warp_mode = "NONE"
                print("[PROJ] Warp mode: None")

    def _on_warp_lut_toggled(self, checked: bool):
        if checked:
            self._proj_warp_mode = "LUT"
            try:
                if hasattr(self, '_button_req_hmatrix'):
                    self._button_req_hmatrix.setChecked(False)
            except Exception:
                pass
            print("[PROJ] Warp mode: LUT (engine will display prewarped content)")
        else:
            if (getattr(self, '_button_req_hmatrix', None) is None) or (not self._button_req_hmatrix.isChecked()):
                self._proj_warp_mode = "NONE"
                print("[PROJ] Warp mode: None")
