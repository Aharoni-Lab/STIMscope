"""TriggerControlsMixin — extracted from qt_interface.py.

Bundles the four projector / hardware trigger control methods:

* ``_toggle_hw_trigger_out(checked)`` — enable/disable GPIO trigger
  out on Jetson J30 pin 22 (~80 LOC).
* ``_test_hw_trigger_pulse()`` — fire a one-shot test pulse (~19 LOC).
* ``_toggle_send_triggers()`` — start/stop the DMD 60 Hz GPIO trigger
  stream (the I²C-burst boot+standby toggle) (~207 LOC).
* ``_toggle_start_projector()`` — launch/kill the projector engine
  subprocess (~68 LOC).

Method bodies are byte-identical to the pre-extraction code at
``qt_interface.py:308-683`` (commit ``a9d18ab``); only the
surrounding module-level frame changed.

Mixin contract (Interface attributes the method reads/writes):
  * ``self._ensure_qprocess`` — lazy QProcess import (stays on Interface)
  * ``self._proc_projector`` / ``self._proc_dlpc`` — QProcess refs
  * ``self._helper_python_path_for_i2c`` — provided by I2CDialogMixin
  * ``self._on_proc_finished`` — provided by LEDAndProcessMixin
  * ``self.warning`` — error-surfacing helper

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
from pathlib import Path

class TriggerControlsMixin:
    """Cluster 14 — hardware-trigger + projector-engine toggles."""

    def _toggle_hw_trigger_out(self, checked: bool):
        """Enable/disable GPIO trigger out on Jetson BOARD pin 22.
        When enabled, each engine frame send will emit a short pulse.
        """
        try:
            import Jetson.GPIO as GPIO
            pin = 22  # J30 pin 22 -> GPIO17
            if checked:
                GPIO.setmode(GPIO.BOARD)
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                self._hw_trig_pin = pin
                self._hw_trig_enabled = True
                print("[HWTRIG] Enabled on BOARD pin 22")
                # Start background subscriber that pulses on every projector visibility event
                try:
                    import threading as _th
                    import zmq as _zmq
                    self._hw_trig_stop = _th.Event()

                    def _loop():
                        last_pidx = 0
                        try:
                            ctx = _zmq.Context.instance()
                            sub = ctx.socket(_zmq.SUB)
                            sub.setsockopt(_zmq.LINGER, 0)
                            sub.setsockopt_string(_zmq.SUBSCRIBE, "")
                            sub.connect("tcp://127.0.0.1:5562")
                        except Exception as _e:
                            print(f"[HWTRIG] SUB init error: {_e}")
                            return
                        while not self._hw_trig_stop.is_set():
                            try:
                                msg = sub.recv(flags=_zmq.NOBLOCK)
                                s = msg.decode('utf-8', errors='ignore')
                                # Minimal JSON parse
                                pidx = None
                                vis = None
                                try:
                                    import json as _json
                                    d = _json.loads(s)
                                    pidx = int(d.get('pidx', 0))
                                    vis = int(d.get('vis_id', -1))
                                except Exception:
                                    pass
                                if pidx is not None and pidx > last_pidx and vis is not None and vis >= 0:
                                    try:
                                        GPIO.output(pin, GPIO.HIGH)
                                        import time as _t
                                        _t.sleep(0.001)
                                        GPIO.output(pin, GPIO.LOW)
                                    except Exception as _e:
                                        print(f"[HWTRIG] Pulse error: {_e}")
                                    last_pidx = pidx
                            except Exception:
                                # No message yet
                                import time as _t
                                _t.sleep(0.005)

                    self._hw_trig_thread = _th.Thread(target=_loop, daemon=True)
                    self._hw_trig_thread.start()
                except Exception as _e:
                    print(f"[HWTRIG] Subscriber start error: {_e}")
            else:
                try:
                    GPIO.output(getattr(self, '_hw_trig_pin', pin), GPIO.LOW)
                    GPIO.cleanup(getattr(self, '_hw_trig_pin', pin))
                except Exception:
                    pass
                self._hw_trig_enabled = False
                print("[HWTRIG] Disabled and cleaned up")
                # Stop background subscriber
                try:
                    if hasattr(self, '_hw_trig_stop') and self._hw_trig_stop is not None:
                        self._hw_trig_stop.set()
                    if hasattr(self, '_hw_trig_thread') and self._hw_trig_thread is not None:
                        self._hw_trig_thread.join(timeout=0.5)
                except Exception:
                    pass
        except Exception as e:
            print(f"[HWTRIG] Setup error: {e}")

    def _test_hw_trigger_pulse(self):
        try:
            import Jetson.GPIO as GPIO
            import time as _t
            pin = 22
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            print("[HWTRIG] Test: 5 pulses on BOARD 22")
            for _ in range(5):
                GPIO.output(pin, GPIO.HIGH); _t.sleep(0.01)
                GPIO.output(pin, GPIO.LOW);  _t.sleep(0.01)
            # leave low
        except Exception as e:
            print(f"[HWTRIG] Test pulse error: {e}")

    # _open_trig_params_dialog + _apply_trig_params_to_camera +
    # _on_seq_type_changed extracted to qt_interface_trig_params.py
    # (TrigParamsMixin) per L5 §0.5 decomposition (iter-5).

    def _force_dmd_standby(self):
        """Force the DLPC/DMD to a known-clean Standby (0x05 0xFF) so no
        TRIG_OUT_2 pulses are flowing.

        Called on ARM so a lingering 'triggering' state — left by a prior run
        or by the I2C Burst Sender — cannot immediately auto-start recording
        (the intermittent-arming race). Mirrors the demo shell's Step 0a
        discipline ("clean state regardless of prior run"). Synchronous +
        best-effort; never raises.
        """
        import subprocess
        QProcess = self._ensure_qprocess()
        # Kill any in-flight I2C subprocess first (I2C bus mutex).
        try:
            if getattr(self, "_proc_i2c", None) is not None and \
                    self._proc_i2c.state() != QProcess.NotRunning:
                self._proc_i2c.kill()
                self._proc_i2c.waitForFinished(500)
        except Exception:
            pass
        # Stop the Temporal R/B alternator if running so its last I2C write
        # doesn't race the standby.
        try:
            self._stop_temporal_alt_thread()
        except Exception:
            pass
        try:
            work_dir = str(Path(__file__).resolve().parents[2])
            stop_script = os.path.join(
                work_dir, "ZMQ_sender_mask", "i2c_test_send_commands.py")
            print("[I2C] Arm: forcing DLPC -> Standby (0x05 0xFF) for a clean "
                  "trigger state before arming")
            subprocess.run(["/usr/bin/python3", stop_script, "stop"],
                           cwd=work_dir, timeout=3, check=False)
        except Exception as e:
            print(f"[I2C] force-standby on arm failed (continuing): {e}")
        finally:
            self._dmd_sequencer_running = False
            try:
                if getattr(self, "_button_send_triggers", None) is not None:
                    self._button_send_triggers.setText("Start Projector Trigger")
            except Exception:
                pass

    def _toggle_send_triggers(self):
        """Proper toggle for the DMD pattern sequencer.
        - When OFF → runs full I2C init (i2c_test_send_commands.py boot), DMD
          starts firing 60 Hz GPIO triggers. Button text: 'Stop Projector
          Trigger'.
        - When ON  → sends Standby (0x05 0xFF) via `i2c_test_send_commands.py
          stop`, DMD stops firing triggers. Button text: 'Start Projector
          Trigger'.
        State tracked on self._dmd_sequencer_running so it survives across
        completed I2C subprocesses (which exit after one-shot writes).

        Note: docstring previously
        said "Seq Stop (0x07 0x00) via i2c_send_custom_cmd.py" which was
        the pre-Stream-H mechanism. Actual code uses `stop` subcommand
        which writes 0x05 0xFF correctly (see line ~3478)."""
        QProcess = self._ensure_qprocess()
        try:
            # Always kill any in-flight I2C subprocess first (mutex on the bus)
            if self._proc_i2c is not None:
                try:
                    if self._proc_i2c.state() != QProcess.NotRunning:
                        self._proc_i2c.kill()
                        self._proc_i2c.waitForFinished(500)
                except Exception:
                    pass
                try:
                    self._proc_i2c.deleteLater()
                except Exception:
                    pass
                self._proc_i2c = None

            sequencer_running = bool(getattr(self, "_dmd_sequencer_running", False))

            if sequencer_running:
                # First kill the frame scheduler if it's running — otherwise it
                # will keep firing 0x96 writes after the DLPC has gone to Standby
                # and generate spurious "no ack" errors.
                if getattr(self, "_proc_scheduler", None) is not None:
                    try:
                        if self._proc_scheduler.state() != QProcess.NotRunning:
                            print("[scheduler] killing because Stop Projector Trigger was clicked")
                            self._proc_scheduler.kill()
                            self._proc_scheduler.waitForFinished(1000)
                    except Exception:
                        pass
                    try:
                        self._proc_scheduler.deleteLater()
                    except Exception:
                        pass
                    self._proc_scheduler = None

                # STOP branch — issue 0x05 0xFF (Standby) via the datasheet-correct
                # `stop` subcommand. Replaces the old `--cmd 0x07 --data 0x00` which
                # wrote an invalid parameter to External Video Source Format Select
                # (see docs/hardware/FINDINGS_.md finding #3).
                work_dir = str(Path(__file__).resolve().parents[2])
                stop_script = os.path.join(work_dir, "ZMQ_sender_mask", "i2c_test_send_commands.py")
                py = "/usr/bin/python3"
                self._proc_i2c = QProcess(self)
                self._proc_i2c.setWorkingDirectory(work_dir)
                self._attach_proc_signals(self._proc_i2c, 'i2c')
                self._proc_i2c.finished.connect(lambda *_: self._on_proc_finished('i2c'))
                self._proc_i2c.errorOccurred.connect(lambda *_: self._on_proc_finished('i2c'))
                print("[I2C] Stop Projector Trigger: DLPC → Standby (0x05 0xFF)")
                print(f"[I2C] Launch: {py} {stop_script} stop")
                # Stop the Temporal R/B alternator (no-op if not running, e.g.
                # if the trigger was in Simultaneous/Mode B). Must stop BEFORE
                # the DLPC goes to Standby so the alternator's last I²C call
                # doesn't race with the standby write.
                try:
                    self._stop_temporal_alt_thread()
                except Exception as _e:
                    print(f"[TempAlt] stop failed (continuing): {_e}")
                self._proc_i2c.start(py, [stop_script, "stop"])
                self._dmd_sequencer_running = False
                try:
                    self._button_send_triggers.setText("Start Projector Trigger")
                except Exception:
                    pass
                return

            # Run exact script and capture output/errors in console
            work_dir = str(Path(__file__).resolve().parents[2])
            # Use absolute path explicitly to avoid any ambiguity
            script_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "ZMQ_sender_mask", "i2c_test_send_commands.py")
            py = "/usr/bin/python3"

            self._proc_i2c = QProcess(self)
            self._proc_i2c.setWorkingDirectory(work_dir)
            self._attach_proc_signals(self._proc_i2c, 'i2c')
            self._proc_i2c.finished.connect(lambda *_: self._on_proc_finished('i2c'))
            self._proc_i2c.errorOccurred.connect(lambda *_: self._on_proc_finished('i2c'))

            try:
                from PyQt5.QtCore import QProcessEnvironment
                env = QProcessEnvironment.systemEnvironment()
                env.insert("PYTHONUNBUFFERED", "1")
                # Keep a clean PATH so /usr/bin/python3 resolves stable libs
                if not env.contains("PATH"):
                    env.insert("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
                self._proc_i2c.setProcessEnvironment(env)
            except Exception:
                pass

            stim_mode_sel = self._stim_mode_dropdown.currentText() if hasattr(self, "_stim_mode_dropdown") else ""

            # Helper: parse the dropdowns into i2c CLI args. Used by every branch
            # that needs to honor the user's Sequence Type + LED Color choices.
            def _resolve_seq_and_illum():
                _sel = self._seq_type_dropdown.currentText() if hasattr(self, '_seq_type_dropdown') else ""
                if "0x03" in _sel or _sel.startswith("8-bit RGB"):
                    _seq = "3"
                elif "0x02" in _sel or _sel.startswith("8-bit Mono"):
                    _seq = "2"
                elif "0x01" in _sel or _sel.startswith("1-bit RGB"):
                    _seq = "1"
                else:
                    _seq = "0"
                _led_sel = self._led_color_dropdown.currentText() if hasattr(self, "_led_color_dropdown") else "Red (0x01)"
                if "0x01" in _led_sel:
                    _il = "0x01"
                elif "0x04" in _led_sel:
                    _il = "0x04"
                elif "0x05" in _led_sel:
                    _il = "0x05"
                elif "0x07" in _led_sel:
                    _il = "0x07"
                elif "0x02" in _led_sel:
                    _il = "0x02"
                elif "0x03" in _led_sel:
                    _il = "0x03"
                else:
                    _il = "0x01"
                return _seq, _il, _sel, _led_sel

            if "Simultaneous" in stim_mode_sel:
                # Mode B by design: both R+B LEDs full PWM in 8-bit RGB sub-frame
                # cycling. The streamer composes R+B into one frame and the DMD
                # multiplexes them perceptually-simultaneously. --rgb-cycle is
                # correct ONLY for this mode.
                print(f"[I2C] Start Projector Trigger: {stim_mode_sel} (Mode B — composite R+B sub-frame multiplexing)")
                print(f"[I2C] Launch: {py} {script_path} boot --rgb-cycle")
                self._proc_i2c.start(py, [script_path, "boot", "--rgb-cycle"])
                self._trig_delay_enabled = True
                self._trig_delay_us = 11000.0
                self._trig_exp_enabled = True
                self._trig_exp_us = 5000.0
                self._trig_activation = "RisingEdge"
                print("[CAM] Blue sub-frame preset stored (delay=11000 µs, exposure=5000 µs).")
            elif "Temporal" in stim_mode_sel:
                # Temporal: boot in 8-bit MONO + RED initial, then a small
                # standalone worker thread alternates the LED RED↔BLUE per
                # phase via dlpc_i2c.fast_phase_switch (the driver the deleted
                # CS trial loop used to provide). Phase duration default 1 s,
                # tunable via STIM_TEMPORAL_PHASE_MS env var.
                _illum = "0x01"        # initial: RED only
                _seq_type = "2"        # 8-bit MONO
                print(f"[I2C] Start Projector Trigger: {stim_mode_sel} → "
                      f"booting 8-bit MONO + RED initial. Temporal alternator "
                      f"will then drive RED↔BLUE per phase.")
                print(f"[I2C] Launch: {py} {script_path} boot --illum {_illum} --seq-type {_seq_type}")
                self._proc_i2c.start(py, [script_path, "boot", "--illum", _illum, "--seq-type", _seq_type])
                # Start the alternator AFTER the boot subprocess is launched;
                # the thread sleeps a couple seconds before the first switch so
                # the boot has time to put the DLPC in External Pattern
                # Streaming (the mode fast_phase_switch needs).
                try:
                    self._start_temporal_alt_thread()
                except Exception as _e:
                    print(f"[TempAlt] could not start alternator: {_e}")
                self._trig_delay_enabled = True
                self._trig_delay_us = 11000.0
                self._trig_exp_enabled = True
                self._trig_exp_us = 5000.0
                self._trig_activation = "RisingEdge"
                print("[CAM] Blue sub-frame preset stored (delay=11000 µs, exposure=5000 µs).")
            else:
                sel = self._seq_type_dropdown.currentText()
                if "0x03" in sel or sel.startswith("8-bit RGB"):
                    seq_type = "3"
                elif "0x02" in sel or sel.startswith("8-bit Mono"):
                    seq_type = "2"
                elif "0x01" in sel or sel.startswith("1-bit RGB"):
                    seq_type = "1"
                else:
                    seq_type = "0"
                led_sel = self._led_color_dropdown.currentText() if hasattr(self, "_led_color_dropdown") else "R (0x01)"
                if "0x01" in led_sel:
                    illum = "0x01"
                elif "0x02" in led_sel:
                    illum = "0x02"
                elif "0x04" in led_sel:
                    illum = "0x04"
                elif "0x07" in led_sel:
                    illum = "0x07"
                elif "0x03" in led_sel:
                    illum = "0x03"
                else:
                    illum = "0x01"
                print(f"[I2C] Start Projector Trigger: seq_type={seq_type} ({sel}) | illum={illum} ({led_sel})")
                print(f"[I2C] Launch: {py} {script_path} boot --illum {illum} --seq-type {seq_type}")
                self._proc_i2c.start(
                    py,
                    [script_path, "boot", "--illum", illum, "--seq-type", seq_type],
                )
                # Store full-frame preset for Set Trig Params dialog.
                # User can apply manually via the dialog if needed.
                self._trig_delay_enabled = False
                self._trig_delay_us = 0.0
                self._trig_exp_enabled = False
                self._trig_exp_us = None
                self._trig_activation = "RisingEdge"
            # Track sequencer state for next toggle click
            self._dmd_sequencer_running = True
            try:
                self._button_send_triggers.setText("Stop Projector Trigger")
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to start I2C trigger script: {e}")
            self._on_proc_finished('i2c')

    def _toggle_start_projector(self):
        QProcess = self._ensure_qprocess()
        try:
            # Guard against double-launch: check if process is alive
            if self._proc_projector is not None:
                try:
                    state = self._proc_projector.state()
                    if state != QProcess.NotRunning:
                        # Process still running — kill it (toggle off)
                        self._proc_projector.kill()
                        return
                except Exception:
                    pass
                # Process object exists but not running — clean up stale ref
                try:
                    self._proc_projector.deleteLater()
                except Exception:
                    pass
                self._proc_projector = None

            if self._proc_projector is None:
                # Reopen the dedicated live engine/mask log window on each engine
                # start (even if the user closed it before), so its output is
                # visible without flooding the terminal.
                self._engine_log_user_hidden = False
                self._proc_projector = QProcess(self)
                self._proc_projector.finished.connect(lambda *_: self._on_proc_finished('projector'))
                self._proc_projector.errorOccurred.connect(lambda *_: self._on_proc_finished('projector'))
                self._attach_proc_signals(self._proc_projector, 'projector')
                try:
                    from PyQt5.QtCore import QProcessEnvironment
                    env = QProcessEnvironment.systemEnvironment()
                    env.insert("PYTHONUNBUFFERED", "1")
                    self._proc_projector.setProcessEnvironment(env)
                except Exception:
                    pass

                # Launch projector from exact local folder with your args
                proj_dir = str(Path(__file__).resolve().parent.parent.parent / "ZMQ_sender_mask")
                # Ensure latest binary is built before launch
                if not self._maybe_build_projector(proj_dir):
                    print("Failed to build projector; aborting launch")
                    self._on_proc_finished('projector')
                    return
                self._proc_projector.setWorkingDirectory(proj_dir)
                exe = f"{proj_dir}/projector"
                args = [
                    "--bind=tcp://127.0.0.1:5558",
                    "--swap-interval=0",
                    f"--visible-id={'1' if self._button_toggle_overlay.isChecked() else '0'}",
                    "--overlay-style=digits",
                    # Use projector defaults for size/position (compile-time or runtime)
                    "--overlay-bg=1",
                    "--overlay-bottom=mask",
                    "--overlay-top=proj",
                    # GPIO defaults are Jetson Orin (/dev/gpiochip1, lines 8/9).
                    # Other carrier boards differ — override via env vars.
                    f"--cam-chip={os.environ.get('STIM_GPIO_CHIP', '/dev/gpiochip1')}",
                    f"--cam-line={os.environ.get('STIM_CAM_LINE', '8')}",
                    "--cam-edge=rising",
                    f"--proj-chip={os.environ.get('STIM_GPIO_CHIP', '/dev/gpiochip1')}",
                    f"--proj-line={os.environ.get('STIM_PROJ_LINE', '9')}",
                    "--proj-edge=rising",
                    "--horiz-flip=1",
                    "--force-immediate=1"
                ]
                print(f"[PROJ] Launch: {exe} {' '.join(args)}")
                self._button_start_projector.setText("Stop Projection Engine")
                self._proc_projector.start(exe, args)
            else:
                self._proc_projector.kill()
        except Exception as e:
            print(f"Failed to toggle projector: {e}")
            self._on_proc_finished('projector')

    # ─── Temporal R/B alternator ────────────────────────────────────────────
    # Recreates what the deleted CS trial loop used to do: drive the DMD's
    # LED to alternate RED↔BLUE per phase via dlpc_i2c.fast_phase_switch.
    # The MASK alternation (R-only / B-only frames) is handled by
    # zmq_mask_sender --temporal-alternate; this thread is what makes the
    # LED actually follow along so the operator sees alternation.
    #
    # Phase duration: STIM_TEMPORAL_PHASE_MS env var (default 1000 ms = 1 s
    # per color, slow enough to be visible and well within fast_phase_switch
    # latency (~20-40 ms)). Daemon thread so a forgotten stop still dies
    # with the process.
    def _start_temporal_alt_thread(self):
        # No-op if already running.
        if getattr(self, "_temporal_alt_thread", None) is not None and \
                self._temporal_alt_thread.is_alive():
            print("[TempAlt] alternator already running; not starting again")
            return
        import os, threading
        self._temporal_alt_stop_event = threading.Event()
        print("[TempAlt] thread starting — will wait 2 s for DLPC boot then alternate")

        def _loop():
            import time as _t
            # Let the boot subprocess put the DLPC in External Pattern
            # Streaming before any switch — switching before that fails.
            _t.sleep(2.0)
            try:
                from dlpc_i2c import fast_phase_switch
                print("[TempAlt] dlpc_i2c.fast_phase_switch imported (direct path)")
            except Exception:
                try:
                    import sys as _sys
                    from pathlib import Path as _P
                    _sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "ZMQ_sender_mask"))
                    from dlpc_i2c import fast_phase_switch
                    print("[TempAlt] dlpc_i2c.fast_phase_switch imported (via sys.path insert)")
                except Exception as _e:
                    print(f"[TempAlt] dlpc_i2c import failed: {_e}; alternator OFF (no LED switching)")
                    return
            try:
                phase_ms = int(os.environ.get("STIM_TEMPORAL_PHASE_MS", "500"))
            except Exception:
                phase_ms = 500
            phase_s = max(0.05, phase_ms / 1000.0)
            print(f"[TempAlt] alternator running — phase {phase_ms} ms per color "
                  f"(STIM_TEMPORAL_PHASE_MS to tune; demo uses 0.5–1.5 s)")
            # I²C bus: Jetson Orin default is 1; other carrier boards differ.
            try:
                i2c_bus = int(os.environ.get("STIM_I2C_BUS", "1"))
            except Exception:
                i2c_bus = 1
            color = "red"  # boot left it RED; first switch flips to BLUE
            stop_event = self._temporal_alt_stop_event
            switch_n = 0
            i2c_warned = False
            while not stop_event.wait(phase_s):
                color = "blue" if color == "red" else "red"
                switch_n += 1
                try:
                    fast_phase_switch(bus=i2c_bus, color=color)
                    print(f"[TempAlt] #{switch_n} switched to {color.upper()}")
                except Exception as _e:
                    # Match the demo's "warn once, keep going" behavior
                    if not i2c_warned:
                        print(f"[TempAlt] fast_phase_switch({color}) FAILED: {_e}. "
                              f"Continuing — DMD will stay in its current LED color "
                              f"(no R/B alternation). Check: DLPC ACKing on i2c-{i2c_bus} "
                              f"(sudo i2cdetect -y {i2c_bus}, expect 1b), STIM_I2C_BUS "
                              f"env var if different, container --device=/dev/i2c-{i2c_bus} "
                              f"or --privileged.")
                        i2c_warned = True
            print(f"[TempAlt] alternator stopped after {switch_n} switches")

        self._temporal_alt_thread = threading.Thread(
            target=_loop, daemon=True, name="TempAlternator")
        self._temporal_alt_thread.start()

    def _stop_temporal_alt_thread(self):
        ev = getattr(self, "_temporal_alt_stop_event", None)
        th = getattr(self, "_temporal_alt_thread", None)
        if ev is not None:
            try:
                ev.set()
            except Exception:
                pass
        if th is not None and th.is_alive():
            try:
                th.join(timeout=2.0)
            except Exception:
                pass
        self._temporal_alt_thread = None
        self._temporal_alt_stop_event = None

