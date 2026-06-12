"""LEDAndProcessMixin — extracted from qt_interface.py per L5 §0.5 decomposition.

Cluster 2 subset (LED live-change + external-process lifecycle).
4 methods, ~213 LOC.

Methods:
- ``_on_led_color_changed_live(text)``    — LED dropdown change handler;
  debounces rapid changes through a 250 ms single-shot QTimer.
- ``_apply_led_color_live()``             — debounced handler that spawns
  i2c_test_send_commands.py boot with the current dropdown values.
- ``_on_proc_finished(which)``            — Qt slot routed from finished/
  errorOccurred signals on each helper QProcess; cleans up the right
  field + button label per process kind.
- ``_terminate_external_processes()``     — invoked from closeEvent; kills
  all 3 helper QProcesses, waits for them, restores button labels.

Mixin contract — subclass provides:
    self._dmd_sequencer_running         : bool
    self._led_color_dropdown            : QComboBox
    self._seq_type_dropdown             : QComboBox
    self._proc_i2c, self._proc_masks,
    self._proc_projector,
    self._proc_i2c_live_led             : QProcess | None
    self._button_send_triggers,
    self._button_send_masks,
    self._button_start_projector        : QPushButton
    self._ensure_qprocess()             — Interface helper returning QProcess
    self._attach_proc_signals(proc, tag) — Interface helper

Pure hoist — no behavior change vs. monolith.
"""

from __future__ import annotations

import os
from pathlib import Path

from PyQt5 import QtCore


class LEDAndProcessMixin:
    """Cluster 2 subset — LED live-change + external-process lifecycle."""

    def _on_led_color_changed_live(self, _text: str):
        """LED dropdown changed. If the projector trigger is currently running,
        debounce rapid changes through a QTimer, then kick off a full boot
        subprocess with the newly-selected color. Sequential fast changes
        (e.g. user clicks-scrolls through the dropdown) collapse to one boot
        at the FINAL value instead of chaining multiple I²C bus conflicts
        that can freeze the DLPC.
        """
        if not getattr(self, "_dmd_sequencer_running", False):
            return  # not running — selection takes effect on next Start click
        # Lazy-init the debounce timer (single-shot, 250 ms window)
        if not hasattr(self, "_led_live_debounce_timer"):
            self._led_live_debounce_timer = QtCore.QTimer(self)
            self._led_live_debounce_timer.setSingleShot(True)
            self._led_live_debounce_timer.setInterval(250)
            self._led_live_debounce_timer.timeout.connect(
                self._apply_led_color_live)
        # Restart the timer — if the user keeps changing the dropdown, we
        # keep pushing the deadline out so only the final value fires.
        self._led_live_debounce_timer.start()

    def _apply_led_color_live(self):
        """Debounced handler — runs 250 ms after the last dropdown change.
        Spawns i2c_test_send_commands.py boot with the current dropdown
        values. Kills any in-flight live-change subprocess first to avoid
        two boots contending for the I²C bus (which was causing freezes).
        """
        QProcess = self._ensure_qprocess()
        # Translate the *current* dropdown value to an illum bitmask.
        try:
            sel = self._led_color_dropdown.currentText()
        except Exception:
            return
        if "0x01" in sel:
            illum = "0x01"
        elif "0x02" in sel:
            illum = "0x02"
        elif "0x04" in sel:
            illum = "0x04"
        elif "0x07" in sel:
            illum = "0x07"
        elif "0x05" in sel:
            illum = "0x05"
        elif "0x03" in sel:
            illum = "0x03"
        else:
            return
        try:
            stxt = self._seq_type_dropdown.currentText()
        except Exception:
            stxt = ""
        if "0x03" in stxt or stxt.startswith("8-bit RGB"):
            seq_type = "3"
        elif "0x02" in stxt or stxt.startswith("8-bit Mono"):
            seq_type = "2"
        elif "0x01" in stxt or stxt.startswith("1-bit RGB"):
            seq_type = "1"
        else:
            seq_type = "0"

        # I²C bus mutex: if a previous live-change boot is still running,
        # kill it before starting the new one. Two concurrent boots on the
        # same I²C bus cause the DLPC to freeze.
        prev = getattr(self, "_proc_i2c_live_led", None)
        if prev is not None:
            try:
                if prev.state() != QProcess.NotRunning:
                    prev.kill()
                    prev.waitForFinished(500)
            except Exception:
                pass
            try:
                prev.deleteLater()
            except Exception:
                pass
            self._proc_i2c_live_led = None

        try:
            work_dir = str(Path(__file__).resolve().parents[2])
            script = os.path.join(work_dir, "ZMQ_sender_mask",
                                  "i2c_test_send_commands.py")
            py = "/usr/bin/python3"
            print(f"[I2C] LED live-change → {sel} (illum={illum}) — "
                  f"re-boot")
            proc = QProcess(self)
            proc.setWorkingDirectory(work_dir)
            try:
                if hasattr(self, "_attach_proc_signals"):
                    self._attach_proc_signals(proc, "i2c-led-live")
            except Exception:
                pass

            def _cleanup(*_):
                try:
                    if self._proc_i2c_live_led is proc:
                        self._proc_i2c_live_led = None
                except Exception:
                    pass
                try:
                    proc.deleteLater()
                except Exception:
                    pass

            proc.finished.connect(_cleanup)
            proc.errorOccurred.connect(_cleanup)
            self._proc_i2c_live_led = proc
            proc.start(py, [script, "boot", "--illum", illum, "--seq-type",
                             seq_type, "--no-validate"])
        except Exception as e:
            print(f"[I2C] LED live-change failed: {e}")
            self._proc_i2c_live_led = None

    def _on_proc_finished(self, which: str):
        if which == 'i2c':
            try:
                if self._proc_i2c is not None:
                    self._proc_i2c.deleteLater()
            except Exception:
                pass
            self._proc_i2c = None
            if hasattr(self, '_button_send_triggers') and self._button_send_triggers is not None:
                # Set button text according to DMD sequencer state, not just to
                # a generic "Send …" label. The I2C subprocess exits after its
                # one-shot writes but the DMD sequencer keeps running.
                if getattr(self, "_dmd_sequencer_running", False):
                    self._button_send_triggers.setText("Stop Projector Trigger")
                else:
                    self._button_send_triggers.setText("Start Projector Trigger")
        else:
            if which == 'masks':
                try:
                    if self._proc_masks is not None:
                        self._proc_masks.deleteLater()
                except Exception:
                    pass
                self._proc_masks = None
                if hasattr(self, '_button_send_masks') and self._button_send_masks is not None:
                    self._button_send_masks.setText("Send Masks")
            elif which == 'projector':
                try:
                    if self._proc_projector is not None:
                        self._proc_projector.deleteLater()
                except Exception:
                    pass
                self._proc_projector = None
                if hasattr(self, '_button_start_projector') and self._button_start_projector is not None:
                    self._button_start_projector.setText("Start Projection Engine")

    def _terminate_external_processes(self):
        # Ensure spawned helper scripts are stopped when GUI closes
        try:
            if self._proc_i2c is not None:
                try:
                    self._proc_i2c.kill()
                except Exception:
                    pass
                try:
                    self._proc_i2c.waitForFinished(1000)
                except Exception:
                    pass
        finally:
            self._proc_i2c = None
            try:
                if hasattr(self, '_button_send_triggers') and self._button_send_triggers is not None:
                    # State-aware button label — respects _dmd_sequencer_running
                    if getattr(self, "_dmd_sequencer_running", False):
                        self._button_send_triggers.setText("Stop Projector Trigger")
                    else:
                        self._button_send_triggers.setText("Start Projector Trigger")
            except Exception:
                pass

        try:
            if self._proc_masks is not None:
                try:
                    self._proc_masks.kill()
                except Exception:
                    pass
                try:
                    self._proc_masks.waitForFinished(1000)
                except Exception:
                    pass
        finally:
            self._proc_masks = None
            try:
                if hasattr(self, '_button_send_masks') and self._button_send_masks is not None:
                    self._button_send_masks.setText("Send Masks")
            except Exception:
                pass

        try:
            if self._proc_projector is not None:
                try:
                    self._proc_projector.kill()
                except Exception:
                    pass
                try:
                    self._proc_projector.waitForFinished(2000)
                except Exception:
                    pass
        finally:
            self._proc_projector = None
            try:
                if hasattr(self, '_button_start_projector') and self._button_start_projector is not None:
                    self._button_start_projector.setText("Start Projection Engine")
            except Exception:
                pass
