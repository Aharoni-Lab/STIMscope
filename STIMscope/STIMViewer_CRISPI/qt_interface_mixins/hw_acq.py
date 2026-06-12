"""HardwareAcqMixin — extracted from qt_interface.py per L5 §0.5 decomposition.

Cluster 6 (recording / capture / TIFF subset) + part of cluster 7
(hardware acquisition). 7 methods, ~167 LOC.

Methods:
- ``_update_recording_button_text()`` — refresh recording-button label
  from camera.is_recording / is_armed.
- ``_on_recording_started()``         — Qt slot when recording begins.
- ``_on_recording_stopped()``         — Qt slot when recording stops.
- ``_on_auto_start_recording()``      — Qt slot when MCU auto-arm starts
  a recording.
- ``_trigger_sw_trigger()``           — operator-click snapshot path.
- ``_start_hardware_acquisition()``   — toggle MCU-trigger / real-time
  acquisition mode on the IDS camera.
- ``_start_recording()``              — operator-click record button:
  start / stop / arm / disarm depending on current state and HW mode.

Mixin contract — subclass provides:
    self._camera                         — OptimizedCamera (L3-audited)
    self._button_start_recording         — QPushButton
    self._button_start_hardware_acquisition — QPushButton
    self._dropdown_trigger_line          — QComboBox
    self._exp_line                       — QLineEdit (optional)
    self.acq_label                       — QLabel (statusbar)
    self._recording_status               — bool
    self._hardware_status                — bool
    self.warning(msg)                    — Interface helper (modal warning)

Pure hoist — no behavior change vs. monolith.
"""

from __future__ import annotations

import os

from PyQt5 import QtCore


class HardwareAcqMixin:
    """Cluster 6/7 — hardware acquisition + recording lifecycle."""

    def _update_recording_button_text(self):
        """Update the recording button text based on current state"""
        is_recording = getattr(self._camera, "is_recording", False)
        is_armed = getattr(self._camera, "is_armed", False)

        print(f"🔍 Updating button text - recording: {is_recording}, armed: {is_armed}")

        if is_recording:
            self._button_start_recording.setText("Stop Recording")
        elif is_armed:
            self._button_start_recording.setText("Disarm Recording")
        else:
            self._button_start_recording.setText("Start Recording")

    @QtCore.pyqtSlot()
    def _on_recording_started(self):
        self._recording_status = True
        self._button_start_recording.setText("Stop Recording")
        self._button_start_hardware_acquisition.setEnabled(False)
        self._dropdown_trigger_line.setEnabled(False)

    @QtCore.pyqtSlot()
    def _on_recording_stopped(self):
        self._recording_status = False
        self._update_recording_button_text()
        self._button_start_hardware_acquisition.setEnabled(True)
        if not self._hardware_status:
            self._dropdown_trigger_line.setEnabled(True)

    @QtCore.pyqtSlot()
    def _on_auto_start_recording(self):
        """Handle automatic recording start from hardware trigger"""
        try:
            self._camera.start_recording()
        except Exception as e:
            print(f"Auto-start recording failed: {e}")

    def _trigger_sw_trigger(self):

        try:
            if not self._camera:
                self.warning("No camera available for snapshot")
                return


            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.png"


            save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)


            if hasattr(self._camera, "snapshot"):
                success = self._camera.snapshot(filepath)
                if success:
                    pass  # camera.py already logged "Snapshot saved: <path>"
                else:
                    self.warning("Snapshot failed - check camera status")
                    print("❌ Snapshot failed")
            elif hasattr(self._camera, "save_image"):
                self._camera.save_image = True
                print("📸 Legacy snapshot triggered")
            elif hasattr(self._camera, "software_trigger"):
                self._camera.software_trigger()
                print("📸 Software trigger sent")
            else:
                self.warning("No snapshot method available")
                print("❌ No snapshot method available")

        except Exception as e:
            error_msg = f"Snapshot error: {e}"
            self.warning(error_msg)
            print(f"❌ {error_msg}")


    def _start_hardware_acquisition(self):
        if not self._hardware_status:
            self._camera.stop_realtime_acquisition()
            self._camera.start_hardware_acquisition()

            # HW-trigger mode REQUIRES a short exposure. In slave/triggered mode
            # each trigger starts a fresh exposure, so exposure + sensor readout
            # must fit inside one trigger period (33.3 ms at 30 Hz). The camera's
            # free-run open-default is 33,333 µs — inheriting that here leaves
            # ZERO readout margin, so the sensor misses every other trigger and
            # the recording drops to ~15 fps. Bench-confirmed :
            # lowering exposure to 10 ms restored 30.8 fps (and the DMD's
            # sequence_abort was irrelevant to FPS — exposure was the cap).
            #
            # So CAP the exposure at a HW-safe value on entry. NOTE the prior
            # guidance got this backwards: forcing a *long* exposure (30000/
            # 33333 µs) is what caused the old 15 fps; capping to a *short* one
            # is the fix. Tunable via STIM_HW_EXP_US (default 15000 µs ≈ half the
            # 30 Hz period, leaving readout margin). We only LOWER (never raise)
            # so a deliberately-short setting (e.g. the Mode B blue-sub-frame
            # 5000 µs exposure) is preserved. User can still raise it afterward
            # via Sensor Settings (accepting frame drops).
            try:
                hw_exp_cap = float(os.environ.get("STIM_HW_EXP_US", "15000"))
            except Exception:
                hw_exp_cap = 15000.0
            try:
                exp_node = self._camera.node_map.FindNode("ExposureTime")
                current_exp = float(exp_node.Value()) if exp_node is not None else 0.0
                if exp_node is not None and current_exp > hw_exp_cap:
                    mn, mx = exp_node.Minimum(), exp_node.Maximum()
                    target = max(mn, min(mx, hw_exp_cap))
                    exp_node.SetValue(target)
                    applied = float(exp_node.Value())
                    print(f"[CAM] HW mode: capped exposure {current_exp:.0f} -> {applied:.0f} µs "
                          f"for readout margin under the 30 Hz trigger (-> ~30 fps). "
                          f"Raise via Sensor Settings / tune with STIM_HW_EXP_US.")
                    current_exp = applied
                else:
                    print(f"[CAM] HW mode: exposure {current_exp:.0f} µs already within the "
                          f"HW-safe cap ({hw_exp_cap:.0f} µs) — left as-is.")
                if hasattr(self, '_exp_line'):
                    self._exp_line.setText(f"{current_exp:.3f}")
            except Exception as e:
                print(f"[CAM] HW mode exposure cap failed: {e}")

            try:
                node_map = self._camera.node_map
                mode_node = node_map.FindNode("TriggerMode")
                source_node = node_map.FindNode("TriggerSource")
                act_node = node_map.FindNode("TriggerActivation")

                print("TriggerMode =", mode_node.CurrentEntry().SymbolicValue() if mode_node else "None")
                print("TriggerSource =", source_node.CurrentEntry().SymbolicValue() if source_node else "None")
                print("TriggerActivation =", act_node.CurrentEntry().SymbolicValue() if act_node else "None")
            except Exception as e:
                print(f"Failed to read trigger nodes: {e}")

            self._dropdown_trigger_line.setEnabled(False)
            self.acq_label.setText("Acquisition Mode: Hardware")
            self._button_start_hardware_acquisition.setText("Stop Hardware Acquisition")
            # Reset armed state and update button text for hardware mode
            if hasattr(self._camera, 'is_armed'):
                self._camera.is_armed = False
            self._update_recording_button_text()
        else:
            # Disarm if armed when stopping hardware acquisition
            if getattr(self._camera, "is_armed", False):
                self._camera.disarm_recording()

            self._camera.stop_hardware_acquisition()
            self._camera.start_realtime_acquisition()

            # Read back current exposure and reflect in GUI
            try:
                nm = getattr(self._camera, "node_map", None)
                if nm is not None:
                    exp_node = nm.FindNode("ExposureTime")
                    if exp_node is not None and hasattr(self, '_exp_line'):
                        self._exp_line.setText(f"{float(exp_node.Value()):.3f}")
            except Exception:
                pass

            self.acq_label.setText("Acquisition Mode: RealTime")
            self._button_start_hardware_acquisition.setText("Start Hardware Acquisition")
            if not self._recording_status:
                self._dropdown_trigger_line.setEnabled(True)
            # Update recording button text for realtime mode
            self._update_recording_button_text()

        self._hardware_status = not self._hardware_status


    def _start_recording(self):
        try:
            if getattr(self._camera, "is_recording", False):
                # Currently recording, stop it
                self._camera.stop_recording()
            elif getattr(self._camera, "is_armed", False):
                # Currently armed, disarm it
                self._camera.disarm_recording()
                self._update_recording_button_text()
            else:
                # Not recording and not armed
                if self._hardware_status:
                    # In hardware mode, arm the system. First force the DMD to a
                    # clean Standby so a lingering 'triggering' state (left by a
                    # prior run or the I2C Burst Sender) cannot instantly
                    # auto-start recording — the intermittent-arming race. This
                    # guarantees arming WAITS until you press Start Projector
                    # Trigger, regardless of prior DMD state.
                    try:
                        self._force_dmd_standby()
                    except Exception as _e:
                        print(f"[arm] force-standby skipped: {_e}")
                    if self._camera.arm_recording():
                        self._update_recording_button_text()
                else:
                    # In realtime mode, start recording directly
                    self._camera.start_recording()
        except Exception as e:
            print(f"Recording toggle failed: {e}")
