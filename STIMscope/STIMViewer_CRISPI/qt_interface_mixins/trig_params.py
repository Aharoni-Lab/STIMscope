"""TrigParamsMixin — extracted from qt_interface.py per L5 §0.5 decomposition.

Cluster 9 subset (camera trigger parameters dialog + DMD sequence-type
dispatch).
3 methods, ~265 LOC.

Methods:
- ``_open_trig_params_dialog()``            — Build & show the modeless
  "Trigger Parameters" QDialog (delay / exposure / activation edge,
  presets, status readout, Apply / Close).
- ``_apply_trig_params_to_camera()``        — Apply stored _trig_*
  attributes onto the live IDS Peak NodeMap (TriggerDelay, ExposureTime,
  TriggerActivation). Adjusts AcquisitionFrameRate to keep exposure
  feasible. Updates Sensor Settings exposure read-out widget.
- ``_on_seq_type_changed(text)``            — Log handler for the I²C
  sequence-type dropdown; prints the parsed seq_first byte.

Mixin contract — subclass provides:
    self._camera                      : OptimizedCamera-like (with .node_map,
                                        .acquisition_running, .acquisition_mode)
    self._trig_delay_enabled,
    self._trig_delay_us,
    self._trig_exp_enabled,
    self._trig_exp_us,
    self._trig_activation             : runtime-stored trigger state
    self._exp_line                    : QLineEdit (optional)

Pure hoist — no behavior change vs. monolith.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets


class TrigParamsMixin:
    """Cluster 9 subset — Trigger Parameters dialog + sequence-type handler."""

    def _open_trig_params_dialog(self):
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QCheckBox, QPushButton, QComboBox
            dlg = QDialog(self)
            dlg.setWindowTitle("Trigger Parameters")
            try:
                dlg.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
                dlg.setModal(False)
                dlg.setWindowModality(QtCore.Qt.NonModal)
                dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            except Exception:
                pass

            lay = QVBoxLayout(dlg)

            # R5: Protocol presets for common STIMscope configurations.
            # Blue-sub-frame capture: delay ~11 ms (skip red + green sub-frames
            # within the 16.67 ms HDMI frame), expose ~5 ms (blue sub-frame
            # window only). See docs/hardware/DMD_RED_BLUE_WORKFLOW.md §0.
            # Full-frame: capture the entire trigger period (~30 ms exposure,
            # zero delay) — useful for debug / alignment / imaging without
            # the stim/observe protocol.
            preset_label = QLabel("<b>Preset:</b>")
            lay.addWidget(preset_label)
            preset_row = QtWidgets.QHBoxLayout()
            btn_preset_blue = QPushButton("Blue sub-frame (delay=11000, exp=5000)")
            btn_preset_full = QPushButton("Full frame (delay=0, exp=33333.33)")
            btn_preset_blue.setToolTip(
                "STIMscope stim/observe protocol. Camera skips red-stim "
                "and green dead-time sub-frames, exposes only on blue "
                "sub-frame for GCaMP emission capture.")
            btn_preset_full.setToolTip(
                "Debug / alignment preset. Exposure spans most of the "
                "33.3 ms trigger period. No sub-frame sync.")
            preset_row.addWidget(btn_preset_blue)
            preset_row.addWidget(btn_preset_full)
            preset_row.addStretch(1)
            lay.addLayout(preset_row)

            grid = QGridLayout()

            # Enable toggles and inputs
            chk_delay = QCheckBox("Enable TriggerDelay (µs)")
            edt_delay = QLineEdit()
            edt_delay.setPlaceholderText("e.g. 11000 (blue sub-frame) or 0")
            chk_exp = QCheckBox("Enable ExposureTime (µs)")
            edt_exp = QLineEdit()
            edt_exp.setPlaceholderText("e.g. 5000 (blue sub-frame) or 33333.33 (full)")

            # TriggerActivation edge
            act_label = QLabel("Trigger Activation")
            cmb_act = QComboBox()
            cmb_act.addItems(["RisingEdge", "FallingEdge", "LevelHigh", "LevelLow"])

            # Populate from current camera node map where possible; fall back
            # to stored values. Previously this dialog only read from stored
            # attributes, so the displayed values could drift from reality if
            # another code path wrote the nodes (e.g. HW-mode 30ms default).
            nm = getattr(self._camera, 'node_map', None)
            def _node_val(name, fallback=None):
                try:
                    n = nm.FindNode(name) if nm is not None else None
                    return float(n.Value()) if n is not None else fallback
                except Exception:
                    return fallback
            def _node_enum(name, fallback=""):
                try:
                    n = nm.FindNode(name) if nm is not None else None
                    return n.CurrentEntry().SymbolicValue() if n is not None else fallback
                except Exception:
                    return fallback

            cur_delay = _node_val("TriggerDelay", getattr(self, '_trig_delay_us', 0.0))
            cur_exp = _node_val("ExposureTime", getattr(self, '_trig_exp_us', 30000.0))
            cur_act = _node_enum("TriggerActivation", getattr(self, '_trig_activation', "RisingEdge"))

            try:
                if getattr(self, '_trig_delay_enabled', False):
                    chk_delay.setChecked(True)
                edt_delay.setText(f"{cur_delay:.0f}" if cur_delay is not None else "")
            except Exception:
                pass
            try:
                if getattr(self, '_trig_exp_enabled', False):
                    chk_exp.setChecked(True)
                edt_exp.setText(f"{cur_exp:.0f}" if cur_exp is not None else "")
            except Exception:
                pass
            try:
                idx = cmb_act.findText(cur_act)
                if idx >= 0:
                    cmb_act.setCurrentIndex(idx)
            except Exception:
                pass

            grid.addWidget(chk_delay, 0, 0)
            grid.addWidget(edt_delay, 0, 1)
            grid.addWidget(chk_exp,   1, 0)
            grid.addWidget(edt_exp,   1, 1)
            grid.addWidget(act_label, 2, 0)
            grid.addWidget(cmb_act,   2, 1)

            lay.addLayout(grid)

            # Status readout — visible current node values, refresh on Apply
            status_lbl = QLabel("")
            status_lbl.setStyleSheet("font-size: 11px; color: #555;")
            lay.addWidget(status_lbl)
            def _refresh_status():
                try:
                    d = _node_val("TriggerDelay", None)
                    e = _node_val("ExposureTime", None)
                    a = _node_enum("TriggerActivation", "?")
                    parts = []
                    if d is not None: parts.append(f"TriggerDelay={d:.0f} µs")
                    if e is not None: parts.append(f"ExposureTime={e:.0f} µs")
                    parts.append(f"Activation={a}")
                    status_lbl.setText("Current camera values: " + ", ".join(parts))
                except Exception:
                    status_lbl.setText("Current camera values: (unavailable)")
            _refresh_status()

            btn_apply = QPushButton("Apply")
            btn_close = QPushButton("Close")
            row = QtWidgets.QHBoxLayout()
            row.addStretch(1)
            row.addWidget(btn_apply)
            row.addWidget(btn_close)
            lay.addLayout(row)

            def _load_preset(delay_us: float, exp_us: float):
                chk_delay.setChecked(True)
                edt_delay.setText(str(int(delay_us)))
                chk_exp.setChecked(True)
                edt_exp.setText(str(int(exp_us)))
            btn_preset_blue.clicked.connect(lambda: _load_preset(11000, 5000))
            btn_preset_full.clicked.connect(lambda: _load_preset(0, 33333.33))

            def _apply():
                try:
                    self._trig_delay_enabled = bool(chk_delay.isChecked())
                    self._trig_exp_enabled   = bool(chk_exp.isChecked())
                    try:
                        self._trig_delay_us = float(edt_delay.text()) if edt_delay.text().strip() else None
                    except Exception:
                        self._trig_delay_us = None
                    try:
                        self._trig_exp_us = float(edt_exp.text()) if edt_exp.text().strip() else None
                    except Exception:
                        self._trig_exp_us = None
                    self._trig_activation = cmb_act.currentText()

                    # Sanity check — warn if delay+exposure exceeds trigger
                    # period (33333 µs at 30 Hz). Don't block; user may
                    # intentionally oversample with a slower trigger source.
                    try:
                        d = float(self._trig_delay_us or 0)
                        e = float(self._trig_exp_us or 0)
                        if self._trig_delay_enabled and self._trig_exp_enabled and (d + e) > 33333:
                            print(f"[CAM] ⚠ delay ({d:.0f}) + exposure ({e:.0f}) = {d+e:.0f} µs "
                                  f"exceeds 33333 µs 30 Hz trigger period. Frames will drop.")
                    except Exception:
                        pass

                    print(f"[CAM] Trig params set: delay_en={self._trig_delay_enabled} "
                          f"delay_us={self._trig_delay_us} exp_en={self._trig_exp_enabled} "
                          f"exp_us={self._trig_exp_us} activation={self._trig_activation}")

                    applied_now = False
                    try:
                        if getattr(self._camera, 'acquisition_running', False) and getattr(self._camera, 'acquisition_mode', 0) == 1:
                            self._apply_trig_params_to_camera()
                            applied_now = True
                    except Exception:
                        pass
                    if applied_now:
                        print("[CAM] Trig params applied to camera now (hardware trigger mode active).")
                    else:
                        print("[CAM] Trig params STORED — will apply when you click Start Hardware Acquisition.")
                    _refresh_status()
                except Exception as e:
                    print(f"Failed to apply trig params: {e}")

            btn_apply.clicked.connect(_apply)
            btn_close.clicked.connect(dlg.close)

            dlg.show()
        except Exception as e:
            print(f"Failed to open Trigger Parameters dialog: {e}")

    def _apply_trig_params_to_camera(self):
        try:
            nm = getattr(self._camera, 'node_map', None)
            if nm is None:
                return
            # Apply TriggerDelay if enabled and value is valid
            if getattr(self, '_trig_delay_enabled', False) and getattr(self, '_trig_delay_us', None) is not None:
                try:
                    nm.FindNode("TriggerDelay").SetValue(float(self._trig_delay_us))
                    print(f"[CAM] Applied TriggerDelay = {float(self._trig_delay_us)} µs")
                except Exception as e:
                    print(f"[CAM] Failed to set TriggerDelay: {e}")
            # Apply ExposureTime if enabled and value is valid
            if getattr(self, '_trig_exp_enabled', False) and getattr(self, '_trig_exp_us', None) is not None:
                try:
                    nm.FindNode("ExposureAuto").SetCurrentEntry("Off")
                except Exception:
                    pass
                exp_val = float(self._trig_exp_us)
                fps_node = None
                try:
                    fps_node = nm.FindNode("AcquisitionFrameRate")
                    if fps_node is not None:
                        needed_fps = 1_000_000.0 / exp_val
                        if needed_fps < fps_node.Value():
                            fps_node.SetValue(max(fps_node.Minimum(), needed_fps))
                except Exception:
                    pass
                try:
                    nm.FindNode("ExposureTime").SetValue(exp_val)
                except Exception:
                    pass
                if fps_node is not None:
                    try:
                        max_fps = min(fps_node.Maximum(), 1_000_000.0 / exp_val)
                        fps_node.SetValue(max(fps_node.Minimum(), max_fps))
                    except Exception:
                        pass
                try:
                    actual = nm.FindNode("ExposureTime").Value()
                    print(f"[CAM] Applied ExposureTime = {actual:.0f} µs")
                    # Sync _exp_line so Sensor Settings dialog reflects the
                    # applied exposure (previously wrote to camera but left
                    # the GUI line edit stale — user saw mismatch).
                    try:
                        if hasattr(self, '_exp_line'):
                            self._exp_line.setText(f"{float(self._trig_exp_us):.3f}")
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[CAM] Failed to set ExposureTime: {e}")
            # R5: Apply TriggerActivation edge (rising/falling/level). Previously
            # hard-coded to RisingEdge in camera.py:868; now user-selectable.
            act = getattr(self, '_trig_activation', None)
            if act:
                try:
                    nm.FindNode("TriggerActivation").SetCurrentEntry(str(act))
                    print(f"[CAM] Applied TriggerActivation = {act}")
                except Exception as e:
                    print(f"[CAM] Failed to set TriggerActivation: {e}")
        except Exception:
            pass

    def _on_seq_type_changed(self, text: str):
        try:
            sel = text
            if "0x03" in sel or sel.startswith("8-bit RGB"):
                seq_first = "0x03"
            elif "0x02" in sel or sel.startswith("8-bit Mono"):
                seq_first = "0x02"
            elif "0x00" in sel or sel.startswith("1-bit Mono"):
                seq_first = "0x00"
            else:
                seq_first = "0x01"  # 1-bit RGB
            print(f"[I2C] Sequence type changed: {sel} -> {seq_first}")
        except Exception:
            pass
