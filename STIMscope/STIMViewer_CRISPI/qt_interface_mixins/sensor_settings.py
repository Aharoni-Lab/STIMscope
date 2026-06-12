"""SensorSettingsMixin — extracted from qt_interface.py per L5 §0.5 decomposition.

Cluster 7 subset (camera sensor-settings popup dialog).
1 method, ~273 LOC.

Method:
- ``_open_sensor_settings()``     — Build and show the modeless "Sensor
  Settings" QDialog (Analog/Digital Gain sliders, typed Exposure
  input, hardware Contrast/Gamma slider with hot-swap detection). Two-way syncs
  to the main-window gain sliders so dialog state does not drift.

Mixin contract — subclass provides:
    self._gain_slider, self._dgain_slider, self._gain_value_label,
    self._dgain_value_label                   : main-window widgets
    self._exp_line                            : QLineEdit
    self._camera                              : OptimizedCamera-like
                                               (with .node_map, optional
                                                .get_contrast / .set_contrast /
                                                .get_contrast_range)
    self._apply_exposure_from_text()          : Camera-control helper
    self._make_contrast_lut(factor)           : LUT builder helper
    self._set_camera_contrast(factor)         : Hardware-contrast setter

Writes:
    self._sensor_settings_dlg                 : holds dialog alive (modeless)
    self._has_hw_contrast                     : bool — hardware contrast detected
    self._soft_contrast_active                : bool — software preview flag
    self._contrast_factor                     : float — current factor
    self._contrast_lut, self._contrast_lut_factor : LUT cache

Pure hoist — no behavior change vs. monolith.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets


class SensorSettingsMixin:
    """Cluster 7 subset — Sensor Settings dialog (gain/exposure/contrast)."""

    def _open_sensor_settings(self):
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGridLayout, QPushButton
            dlg = QDialog(self)
            dlg.setWindowTitle("Sensor Settings")
            # Make it a movable, modeless top-level window
            try:
                dlg.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
                dlg.setModal(False)
                dlg.setWindowModality(QtCore.Qt.NonModal)
                dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            except Exception:
                pass
            lay = QVBoxLayout(dlg)
            grid = QGridLayout()

            # Reuse existing widgets by creating new controls bound to the
            # MAIN sliders (not to the slots directly). Previously this dialog
            # wired its own sliders straight to _update_gain / _update_dgain,
            # which updated the camera but left the main-window slider
            # position stale. When the dialog closed, any later interaction
            # with the main slider re-applied its stale value → gain "reset"
            # bug. Two-way sync via the main slider fixes this.
            # AG slider
            ag_label = QtWidgets.QLabel("Analog Gain")
            ag_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            ag_slider.setRange(self._gain_slider.minimum(), self._gain_slider.maximum())
            ag_slider.setValue(self._gain_slider.value())
            ag_slider.valueChanged.connect(lambda v: self._gain_slider.setValue(v))
            ag_val = QtWidgets.QLabel(self._gain_value_label.text())
            ag_slider.valueChanged.connect(lambda v: ag_val.setText(f"{v/100:.2f}"))

            grid.addWidget(ag_label, 0, 0)
            grid.addWidget(ag_slider, 0, 1)
            grid.addWidget(ag_val, 0, 2)

            # DG slider (same two-way sync as AG)
            dg_label = QtWidgets.QLabel("Digital Gain")
            dg_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            dg_slider.setRange(self._dgain_slider.minimum(), self._dgain_slider.maximum())
            dg_slider.setValue(self._dgain_slider.value())
            dg_slider.valueChanged.connect(lambda v: self._dgain_slider.setValue(v))
            dg_val = QtWidgets.QLabel(self._dgain_value_label.text())
            dg_slider.valueChanged.connect(lambda v: dg_val.setText(f"{v/100:.2f}"))

            grid.addWidget(dg_label, 1, 0)
            grid.addWidget(dg_slider, 1, 1)
            grid.addWidget(dg_val, 1, 2)

            # Exposure (typed input — no slider). Writes ExposureTime via the
            # main-window exposure field so the dialog and main window stay in
            # sync. Keep exposure low enough to preserve the sensor readout
            # margin under the 30 Hz hardware trigger — too-high exposure drops
            # every other trigger and halves realized recording FPS.
            exp_label = QtWidgets.QLabel("Exposure (µs)")
            # Read the live ExposureTime from the camera node so what's shown =
            # what's actually running. self._exp_line is empty until the
            # operator has Applied an exposure elsewhere; pre-populating from
            # that stale value is what caused the "set 33333 expecting no
            # change but the image got brighter" surprise — the field claimed
            # one value while the camera was at a different one. Mirror the
            # live value back to the main exposure field so the rest of the
            # GUI is truthful too.
            _current_exp_str = ""
            try:
                _nm = getattr(self._camera, "node_map", None)
                if _nm is not None:
                    _node = _nm.FindNode("ExposureTime")
                    if _node is not None:
                        _current_exp_str = f"{float(_node.Value()):.3f}"
                        try:
                            self._exp_line.setText(_current_exp_str)
                        except Exception:
                            pass
            except Exception as _e:
                print(f"[SensorSettings] live ExposureTime read failed: {_e}")
            if not _current_exp_str:
                _current_exp_str = self._exp_line.text() or ""
            exp_line = QtWidgets.QLineEdit(_current_exp_str)
            exp_line.setValidator(QtGui.QDoubleValidator(1.0, 1e9, 3))

            def _apply_local_exp():
                try:
                    self._exp_line.setText(exp_line.text())
                    self._apply_exposure_from_text()
                except Exception as _e:
                    print(f"[SensorSettings] exposure apply failed: {_e}")

            exp_line.returnPressed.connect(_apply_local_exp)
            exp_set_btn = QPushButton("Set")
            exp_set_btn.clicked.connect(_apply_local_exp)

            grid.addWidget(exp_label, 2, 0)
            grid.addWidget(exp_line, 2, 1)
            grid.addWidget(exp_set_btn, 2, 2)

            # Contrast/Gamma control (hardware if available)
            cnt_label = QtWidgets.QLabel("")
            cnt_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            try:
                # Avoid continuous valueChanged signals while dragging; update on release
                cnt_slider.setTracking(False)
            except Exception:
                pass
            cnt_val = QtWidgets.QLabel("")
            # Detect node and range
            contrast_min, contrast_max, contrast_cur = 0.1, 4.0, 1.0
            # Predefine node map and node so later checks are safe even if probing fails
            nm = getattr(self._camera, "node_map", None)
            node = None
            node_name = None
            try:
                if nm is not None:
                    # Prefer hardware contrast; fall back to gamma family
                    for _name in ("Contrast", "ContrastAbsolute", "Gamma", "GammaCorrection", "GammaValue"):
                        try:
                            node = nm.FindNode(_name)
                            if node is not None:
                                node_name = _name
                                break
                        except Exception:
                            node = None
                def _try_get(method_names):
                    for mn in method_names:
                        try:
                            f = getattr(node, mn, None)
                            if callable(f):
                                v = f()
                                if v is not None:
                                    return float(v)
                        except Exception:
                            continue
                    return None
                if node is not None:
                    vmin = _try_get(["Minimum", "GetMinimum", "Min", "GetMin", "GetLower", "GetMinValue"])
                    vmax = _try_get(["Maximum", "GetMaximum", "Max", "GetMax", "GetUpper", "GetMaxValue"])
                    vcur = None
                    for gn in ("Value", "GetValue"):
                        try:
                            gf = getattr(node, gn, None)
                            if callable(gf):
                                gv = gf()
                                if gv is not None:
                                    vcur = float(gv)
                                    break
                        except Exception:
                            pass
                    if vmin is not None and vmax is not None and float(vmax) > float(vmin):
                        contrast_min, contrast_max = float(vmin), float(vmax)
                    if vcur is not None:
                        contrast_cur = float(vcur)
                    # If using gamma, compress UI range to a stable window around 1.0
                    try:
                        if node_name in ("Gamma", "GammaCorrection", "GammaValue"):
                            contrast_min, contrast_max = 0.7, 1.3
                            if not (contrast_min <= contrast_cur <= contrast_max):
                                contrast_cur = 1.0
                    except Exception:
                        pass
                # Optional helpers on camera
                if hasattr(self._camera, "get_contrast_range"):
                    try:
                        rng = self._camera.get_contrast_range()
                        if isinstance(rng, (tuple, list)) and len(rng) >= 2:
                            mn, mx = float(rng[0]), float(rng[1])
                            if mx > mn:
                                contrast_min, contrast_max = mn, mx
                    except Exception:
                        pass
                if hasattr(self._camera, "get_contrast"):
                    try:
                        contrast_cur = float(self._camera.get_contrast())
                    except Exception:
                        pass
            except Exception:
                pass
            # Decide whether hardware contrast is available; set preview fallback flags
            try:
                has_hw = bool(((nm is not None) and (node is not None)) or hasattr(self._camera, "set_contrast"))
            except Exception:
                try:
                    has_hw = bool(hasattr(self._camera, "set_contrast"))
                except Exception:
                    has_hw = False
            try:
                self._has_hw_contrast = bool(has_hw)
                # Disable software contrast for performance on Jetson unless explicitly enabled elsewhere
                self._soft_contrast_active = False
                self._contrast_factor = float(contrast_cur)
                # Build initial LUT for current factor (cheap, 256 entries)
                self._contrast_lut = self._make_contrast_lut(self._contrast_factor)
                self._contrast_lut_factor = self._contrast_factor
            except Exception:
                pass
            # Label/tooltip according to underlying control
            try:
                if node_name in ("Contrast", "ContrastAbsolute"):
                    cnt_label.setText("Contrast")
                    cnt_label.setToolTip("Hardware Contrast (camera control). 1.0 is neutral on most cameras.")
                elif node_name in ("Gamma", "GammaCorrection", "GammaValue"):
                    cnt_label.setText("Gamma")
                    cnt_label.setToolTip("Hardware Gamma (brightness curve). 1.0 is neutral; <1 brightens, >1 darkens.")
                else:
                    cnt_label.setText("Contrast")
                    cnt_label.setToolTip("Contrast not exposed by camera; consider a software preview option if needed.")
            except Exception:
                pass
            # Clip current to range
            try:
                if not (contrast_min <= contrast_cur <= contrast_max):
                    contrast_cur = max(contrast_min, min(contrast_cur, contrast_max))
            except Exception:
                contrast_cur = 1.0
            # Slider ticks and mapping
            ticks = 1000
            try:
                cnt_slider.setRange(0, ticks)
            except Exception:
                pass
            def _to_pos(v):
                try:
                    return int(round((float(v) - contrast_min) / max(1e-12, (contrast_max - contrast_min)) * ticks))
                except Exception:
                    return 0
            def _to_val(p):
                try:
                    frac = float(p) / float(ticks)
                    return (contrast_min + frac * (contrast_max - contrast_min))
                except Exception:
                    return contrast_min
            try:
                cnt_slider.setValue(_to_pos(contrast_cur))
            except Exception:
                pass
            try:
                cnt_val.setText(f"{contrast_cur:.2f}")
            except Exception:
                pass
            def _on_cnt_change(p, _has_hw=has_hw):
                try:
                    v = float(_to_val(p))
                    cnt_val.setText(f"{v:.2f}")
                    # Store factor (no heavy preview updates here)
                    self._contrast_factor = float(v)
                except Exception:
                    pass
            try:
                cnt_slider.valueChanged.connect(_on_cnt_change)
            except Exception:
                pass
            # Apply hardware on slider release only (prevents camera stalls while dragging)
            try:
                cnt_slider.sliderReleased.connect(lambda: self._set_camera_contrast(float(getattr(self, "_contrast_factor", 1.0))) if bool(getattr(self, "_has_hw_contrast", False)) else None)
            except Exception:
                pass

            grid.addWidget(cnt_label, 4, 0)
            grid.addWidget(cnt_slider, 4, 1)
            grid.addWidget(cnt_val, 4, 2)

            lay.addLayout(grid)
            btns = QtWidgets.QHBoxLayout()
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dlg.accept)
            btns.addStretch(1)
            btns.addWidget(close_btn)
            lay.addLayout(btns)
            # Keep a reference so it stays alive when shown modelessly
            self._sensor_settings_dlg = dlg
            try:
                dlg.show()
                dlg.raise_()
                dlg.activateWindow()
            except Exception:
                dlg.show()
        except Exception as e:
            print(f"Sensor Settings UI error: {e}")
