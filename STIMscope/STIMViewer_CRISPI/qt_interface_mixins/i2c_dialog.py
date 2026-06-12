"""I2CDialogMixin — extracted from qt_interface.py.

Bundles the four I²C / DLPC helper methods that work together to
launch the I²C-burst dialog and route the subprocess output back
to the GUI:

* ``_helper_python_path_for_i2c`` — selects the Python interpreter
  with smbus2 available (system python by preference).
* ``_attach_proc_signals`` — wires QProcess stdout/stderr to
  ``_on_proc_output``.
* ``_on_proc_output`` — appends DLPC subprocess output to the
  troubleshoot log + status messages.
* ``_open_i2c_custom_dialog`` — the dialog factory itself (364 LOC).

Method bodies are byte-identical to the pre-extraction code at
``qt_interface.py:403-793`` (commit ``6c49e89``); only the
surrounding module-level frame changed.

Mixin contract (Interface attributes the method reads/writes):
  * ``self._proc_dlpc`` — QProcess ref for the I²C-burst subprocess
  * ``self.warning`` — error-surfacing helper
  * ``self._helper_python_path_for_i2c`` — used by the dialog

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

class I2CDialogMixin:
    """Cluster 13 — I²C / DLPC subprocess helpers + burst-sender dialog."""

    def _helper_python_path_for_i2c(self) -> str:
        """Pick Python for I2C (prefer system where smbus2 is typically available)."""
        for cand in ("/usr/bin/python3", "/usr/local/bin/python3", sys.executable):
            try:
                if os.path.exists(cand):
                    return cand
            except Exception:
                continue
        return sys.executable

    def _attach_proc_signals(self, proc, which: str):
        try:
            from PyQt5.QtCore import QProcess
            proc.setProcessChannelMode(QProcess.MergedChannels)
            proc.readyReadStandardOutput.connect(lambda: self._on_proc_output(proc, which))
        except Exception:
            pass

    def _on_proc_output(self, proc, which: str):
        try:
            data = bytes(proc.readAllStandardOutput()).decode(errors='ignore')
            if not data:
                return
            text = data.rstrip()
            # The projector engine and mask-sending subprocesses emit per-frame
            # output that floods the terminal and buries the important
            # diagnostics (arming / measured FPS / sequence_abort), which print
            # directly to the terminal. Route those two noisy streams to a
            # dedicated LIVE log window instead; keep I²C (boot/stop/status) and
            # everything else on the terminal.
            if which in ('projector', 'masks'):
                prefix = "[MASK]" if which == 'masks' else "[PROJ]"
                self._append_engine_log(prefix, text)
            else:
                prefix = "[I2C]" if which == 'i2c' else f"[{which}]"
                print(f"{prefix} {text}")
        except Exception:
            pass

    def _ensure_engine_log_window(self):
        """Lazily build the dedicated live log window for the projector-engine
        and mask-sending subprocess output. Returns its QPlainTextEdit.

        Separate top-level window (non-modal) so the high-frequency engine/mask
        output stays out of the terminal where arming / FPS / DMD-status logs
        live. maxBlockCount caps memory under the per-frame flood.
        """
        edit = getattr(self, "_engine_log_edit", None)
        if edit is not None:
            return edit
        parent = self if isinstance(self, QtWidgets.QWidget) else None
        dlg = QtWidgets.QDialog(parent)
        dlg.setWindowTitle("Projector Engine / Mask Log (live)")
        dlg.setWindowFlags(dlg.windowFlags() | Qt.Window)
        dlg.resize(900, 420)
        v = QtWidgets.QVBoxLayout(dlg)
        edit = QtWidgets.QPlainTextEdit(dlg)
        edit.setReadOnly(True)
        edit.setMaximumBlockCount(5000)  # cap memory under the per-frame flood
        edit.setFont(QtGui.QFont("Monospace", 9))
        v.addWidget(edit)
        row = QtWidgets.QHBoxLayout()
        btn_clear = QtWidgets.QPushButton("Clear", dlg)
        btn_clear.clicked.connect(edit.clear)
        btn_close = QtWidgets.QPushButton("Close", dlg)
        btn_close.clicked.connect(self._hide_engine_log)
        row.addStretch(1)
        row.addWidget(btn_clear)
        row.addWidget(btn_close)
        v.addLayout(row)
        self._engine_log_dialog = dlg
        self._engine_log_edit = edit
        return edit

    def _hide_engine_log(self):
        """Close button: hide the window and remember the user closed it so it
        doesn't auto-pop on the next line (re-opens on next Start Projection
        Engine)."""
        self._engine_log_user_hidden = True
        dlg = getattr(self, "_engine_log_dialog", None)
        if dlg is not None:
            dlg.hide()

    def _append_engine_log(self, prefix, text):
        """Append projector/mask output to the live log window (auto-shows once,
        unless the user closed it). Never lets logging break the subprocess
        pipeline — falls back to stdout on any error."""
        try:
            edit = self._ensure_engine_log_window()
            dlg = getattr(self, "_engine_log_dialog", None)
            if (dlg is not None and not dlg.isVisible()
                    and not getattr(self, "_engine_log_user_hidden", False)):
                dlg.show()
            for line in text.splitlines():
                edit.appendPlainText(f"{prefix} {line}")
        except Exception:
            try:
                print(f"{prefix} {text}")
            except Exception:
                pass

    def _open_i2c_custom_dialog(self):
        """Multi-line I²C burst editor — type commands manually, send all at once.

        Replaces the legacy one-command-at-a-time dialog. The DLPC3479 firmware
        has a safety state machine that enters a shutdown / safe-default state
        on malformed sequences; reliable multi-step transitions (boot, color
        switch, mode change) require the writes to land as an atomic burst with
        no human-scale delay between them.

        This dialog parses one I²C write per line and fires them all in tight
        succession via in-process dlpc_i2c.raw_write — no QProcess, no subprocess
        overhead, no inter-write sleep.

        Line syntax:
            <opcode> [data_byte...]    # write opcode with given data
            # comment                    # ignored
            (blank line)                 # ignored
        Hex (0x96) and decimal both accepted; commas treated as whitespace.
        """
        try:
            from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                         QLineEdit, QPushButton, QPlainTextEdit, QComboBox)
            dlg = QDialog(self)
            dlg.setWindowTitle("I²C Burst Sender")
            dlg.setModal(False)
            dlg.resize(720, 620)

            v = QVBoxLayout(dlg)

            # Bus + address row
            top = QHBoxLayout()
            edt_bus = QLineEdit("1"); edt_bus.setFixedWidth(50)
            edt_bus.setToolTip("I²C bus number. DMD is on bus 1 on Jetson AGX Orin.")
            edt_addr = QLineEdit("0x1B"); edt_addr.setFixedWidth(70)
            edt_addr.setToolTip("7-bit I²C address. DLPC3479 = 0x1B.")
            top.addWidget(QLabel("Bus:")); top.addWidget(edt_bus)
            top.addSpacing(12)
            top.addWidget(QLabel("Address:")); top.addWidget(edt_addr)
            top.addStretch(1)
            v.addLayout(top)

            # Templates dropdown — populates the burst editor with known-good sequences
            tmpl_row = QHBoxLayout()
            tmpl_row.addWidget(QLabel("Template:"))
            cmb = QComboBox()
            templates = {
                "(blank — type your own)": "",
                # ---- MONO presets (recommended — single LED active, R/G or B physically gated) ----
                "MONO+RED, full PWM, mode 0x03  ★recommended": (
                    "# Boot DLPC into Light Ext Pattern Streaming, MONO + RED only.\n"
                    "# 4 writes land as atomic burst — DLPC enters safety shutdown\n"
                    "# if sequence is interrupted by human-scale delays.\n"
                    "0x92 0x03 0x00 0x00 0x00 0x00\n"
                    "0x96 0x02 0x01 0x01 0xF8 0x2A 0x00 0x00 0x98 0x08 0x00 0x00 0x88 0x13 0x00 0x00\n"
                    "0x54 0xFF 0x03 0x00 0x00 0x00 0x00\n"
                    "0x05 0x03"
                ),
                "MONO+BLUE, full PWM, mode 0x03  ★recommended": (
                    "# Boot DLPC into Light Ext Pattern Streaming, MONO + BLUE only,\n"
                    "# full PWM. Cleanest single-color blue config.\n"
                    "0x92 0x03 0x00 0x00 0x00 0x00\n"
                    "0x96 0x02 0x01 0x04 0xF8 0x2A 0x00 0x00 0x98 0x08 0x00 0x00 0x88 0x13 0x00 0x00\n"
                    "0x54 0x00 0x00 0x00 0x00 0xFF 0x03\n"
                    "0x05 0x03"
                ),
                "MONO+GREEN, full PWM, mode 0x03": (
                    "0x92 0x03 0x00 0x00 0x00 0x00\n"
                    "0x96 0x02 0x01 0x02 0xF8 0x2A 0x00 0x00 0x98 0x08 0x00 0x00 0x88 0x13 0x00 0x00\n"
                    "0x54 0x00 0x00 0xFF 0x03 0x00 0x00\n"
                    "0x05 0x03"
                ),
                # ---- Hard-clamp MONO presets — gate Max PWM (0x56) on inactive channels.
                # Hypothesis: ~9% R bias-current floor seen in 0x55 readback can be
                # suppressed by capping R/G max PWM to zero. Untested on our setup;
                # use to diagnose suspected hardware bias-current leakage.
                "MONO+BLUE w/ R+G max-PWM hard-clamp (bias-current diag)": (
                    "# Cap R+G max PWM to 0 via 0x56 BEFORE setting current PWM.\n"
                    "# If you still see red, leakage is mechanical (tray/dichroic),\n"
                    "# not electrical (LED bias).\n"
                    "0x56 0x00 0x00 0x00 0x00 0xFF 0x03\n"
                    "0x92 0x03 0x00 0x00 0x00 0x00\n"
                    "0x96 0x02 0x01 0x04 0xF8 0x2A 0x00 0x00 0x98 0x08 0x00 0x00 0x88 0x13 0x00 0x00\n"
                    "0x54 0x00 0x00 0x00 0x00 0xFF 0x03\n"
                    "0x05 0x03"
                ),
                "MONO+RED w/ B+G max-PWM hard-clamp (bias-current diag)": (
                    "# Cap B+G max PWM to 0; full R only.\n"
                    "0x56 0xFF 0x03 0x00 0x00 0x00 0x00\n"
                    "0x92 0x03 0x00 0x00 0x00 0x00\n"
                    "0x96 0x02 0x01 0x01 0xF8 0x2A 0x00 0x00 0x98 0x08 0x00 0x00 0x88 0x13 0x00 0x00\n"
                    "0x54 0xFF 0x03 0x00 0x00 0x00 0x00\n"
                    "0x05 0x03"
                ),
                # ---- No-Standby switch presets (3 writes, ~5ms) — for live phase change ----
                "Switch to RED — no-Standby 3-write burst": (
                    "# Atomic R-switch: no Standby, no pause. Bench-tested 4.7-5.1 ms.\n"
                    "0x96 0x02 0x01 0x01 0xF8 0x2A 0x00 0x00 0x98 0x08 0x00 0x00 0x88 0x13 0x00 0x00\n"
                    "0x54 0xFF 0x03 0x00 0x00 0x00 0x00\n"
                    "0x05 0x03"
                ),
                "Switch to BLUE — no-Standby 3-write burst": (
                    "0x96 0x02 0x01 0x04 0xF8 0x2A 0x00 0x00 0x98 0x08 0x00 0x00 0x88 0x13 0x00 0x00\n"
                    "0x54 0x00 0x00 0x00 0x00 0xFF 0x03\n"
                    "0x05 0x03"
                ),
                "Switch to GREEN — no-Standby 3-write burst": (
                    "0x96 0x02 0x01 0x02 0xF8 0x2A 0x00 0x00 0x98 0x08 0x00 0x00 0x88 0x13 0x00 0x00\n"
                    "0x54 0x00 0x00 0xFF 0x03 0x00 0x00\n"
                    "0x05 0x03"
                ),
                # ---- RGB sub-frame multiplex (Mode B / always-RGB) — TIER 1 audit recommendation ----
                "Boot RGB sub-frame R+B (Mode B / always-RGB)": (
                    "# 8-bit RGB, illum_select=0x05 (R+B), full PWM both. DMD\n"
                    "# sub-frame multiplexes R/B autonomously per HDMI frame.\n"
                    "0x92 0x03 0x00 0x00 0x00 0x00\n"
                    "0x96 0x03 0x01 0x05 0xF8 0x2A 0x00 0x00 0x98 0x08 0x00 0x00 0x88 0x13 0x00 0x00\n"
                    "0x54 0xFF 0x03 0x00 0x00 0xFF 0x03\n"
                    "0x05 0x03"
                ),
                # ---- Single-line ops ----
                "Standby (true LED off, mode 0xFF)": (
                    "# Drops out of Light Control. Kills TRIG_OUT_2.\n"
                    "# Use this to test 'is residual red from the tray?' — if you\n"
                    "# still see red here with DMD off, the leakage is optical/ambient,\n"
                    "# not the DLPC.\n"
                    "0x05 0xFF"
                ),
                "Mode → Ext Stream re-select (no reconfig)": (
                    "# Re-asserts mode 0x03; if 0x96 was queued earlier, this latches it.\n"
                    "0x05 0x03"
                ),
            }
            for name in templates:
                cmb.addItem(name)
            btn_load = QPushButton("Load")
            btn_load.setToolTip("Replace burst editor contents with the selected template.")
            tmpl_row.addWidget(cmb, 1); tmpl_row.addWidget(btn_load)
            v.addLayout(tmpl_row)

            # Help text
            help_lbl = QLabel(
                "<b>One I²C write per line.</b> Format: <tt>OPCODE [data_byte...]</tt><br/>"
                "Hex (<tt>0x96</tt>) or decimal accepted. Lines starting with <tt>#</tt> are comments.<br/>"
                "All non-empty, non-comment lines are sent as one <b>atomic burst</b> via in-process raw_write — "
                "no subprocess, no sleep between writes.<br/>"
                "<i>The DLPC firmware enters safety-shutdown on malformed sequences. Burst-send is the only reliable "
                "way to drive multi-step state-machine transitions.</i>"
            )
            help_lbl.setWordWrap(True)
            help_lbl.setStyleSheet("color: #555; font-size: 11px; padding: 4px;")
            v.addWidget(help_lbl)

            # Multi-line burst editor
            edt_burst = QPlainTextEdit()
            edt_burst.setStyleSheet("font-family: monospace;")
            edt_burst.setPlaceholderText(
                "# Type one I²C write per line — opcode then data bytes.\n"
                "# Example (boot MONO+RED, atomic 4-write burst):\n"
                "0x92 0x03 0x00 0x00 0x00 0x00\n"
                "0x96 0x02 0x01 0x01 0xF8 0x2A 0x00 0x00 0x98 0x08 0x00 0x00 0x88 0x13 0x00 0x00\n"
                "0x54 0xFF 0x03 0x00 0x00 0x00 0x00\n"
                "0x05 0x03\n"
                "\n"
                "# Or load a template above."
            )
            v.addWidget(edt_burst, 2)

            # Read-back row (single-shot reads, separate from the write burst)
            read_row = QHBoxLayout()
            read_row.addWidget(QLabel("Read-back:"))
            edt_read_op = QLineEdit("0x06"); edt_read_op.setFixedWidth(70)
            edt_read_op.setToolTip(
                "Read-opcode. Common: 0x06=op_mode, 0x0C=ctrl_id (expect 0x0C), "
                "0x97=pattern_cfg (16 bytes), 0x55=led_pwm (6 bytes), 0xD0=short_status, "
                "0xD3=comm_status (6 bytes), 0xD4=ctrl_id alt.")
            edt_read_n = QLineEdit("1"); edt_read_n.setFixedWidth(50)
            edt_read_n.setToolTip("Bytes to read.")
            btn_read = QPushButton("Read Once")
            btn_read.setToolTip("Read N bytes from the given opcode and append result to the log.")
            read_row.addWidget(QLabel("opcode")); read_row.addWidget(edt_read_op)
            read_row.addWidget(QLabel("× bytes")); read_row.addWidget(edt_read_n)
            read_row.addWidget(btn_read)
            read_row.addStretch(1)
            v.addLayout(read_row)

            # Output log
            log = QPlainTextEdit()
            log.setReadOnly(True)
            log.setMinimumHeight(140)
            log.setStyleSheet("font-family: monospace; font-size: 11px;")
            log.setPlaceholderText("Burst output and read results appear here.")
            v.addWidget(log, 1)

            # Bottom buttons
            btns = QHBoxLayout()
            btn_send_all = QPushButton("Send All  (atomic burst)")
            btn_send_all.setStyleSheet("font-weight: bold; padding: 6px 12px;")
            btn_send_all.setToolTip(
                "Parse every non-comment line and fire them all sequentially "
                "via in-process raw_write. Latency typically 5-15 ms total.")
            btn_clear_log = QPushButton("Clear Log")
            btn_close = QPushButton("Close")
            btns.addStretch(1); btns.addWidget(btn_send_all); btns.addWidget(btn_clear_log); btns.addWidget(btn_close)
            v.addLayout(btns)

            # ---- helpers ----
            def _parse_line(line):
                """Strip comments + tokenize. Returns list of int bytes, or None for skip."""
                s = line.split('#', 1)[0].strip()
                if not s:
                    return None
                toks = [t for t in s.replace(',', ' ').split() if t]
                if not toks:
                    return None
                vals = []
                for t in toks:
                    v = int(t, 0)
                    if not (0 <= v <= 0xFF):
                        raise ValueError(f"value {t!r} out of byte range (0..255)")
                    vals.append(v)
                return vals

            def _kill_bg_proc():
                """Kill any background QProcess holding the I²C bus."""
                try:
                    if getattr(self, "_proc_i2c", None) is not None:
                        if self._proc_i2c.state() != QtCore.QProcess.NotRunning:
                            log.appendPlainText("[mutex] stopping background I²C QProcess")
                            self._proc_i2c.kill()
                            self._proc_i2c.waitForFinished(1000)
                        try:
                            self._proc_i2c.deleteLater()
                        except Exception:
                            pass
                        self._proc_i2c = None
                except Exception:
                    pass

            def _ensure_dlpc_imports():
                """Make /app/ZMQ_sender_mask importable; return (raw_write, raw_read)."""
                import sys as _sys
                import os as _os
                zmq_path = '/app/ZMQ_sender_mask'
                host_path = str(Path(__file__).resolve().parent.parent.parent / 'ZMQ_sender_mask')
                for p in (zmq_path, host_path):
                    if _os.path.isdir(p) and p not in _sys.path:
                        _sys.path.insert(0, p)
                from dlpc_i2c import raw_write, raw_read
                return raw_write, raw_read

            # ---- handlers ----
            def _do_load():
                body = templates.get(cmb.currentText(), '')
                edt_burst.setPlainText(body)

            def _do_send_burst():
                log.appendPlainText("─" * 64)
                try:
                    bus = int(edt_bus.text().strip(), 0)
                    addr = int(edt_addr.text().strip(), 0)
                except Exception as e:
                    log.appendPlainText(f"[ERROR] bad bus/addr: {e}")
                    return

                text = edt_burst.toPlainText()
                try:
                    commands = []
                    for ln_no, ln in enumerate(text.splitlines(), 1):
                        try:
                            parsed = _parse_line(ln)
                        except ValueError as ve:
                            raise ValueError(f"line {ln_no}: {ve}")
                        if parsed is None:
                            continue
                        if len(parsed) < 1:
                            continue
                        commands.append((parsed[0], parsed[1:]))
                except ValueError as e:
                    log.appendPlainText(f"[PARSE ERROR] {e}")
                    return

                if not commands:
                    log.appendPlainText("[ERROR] no commands to send (text empty or all comments)")
                    return

                log.appendPlainText(f"[BURST] bus={bus} addr=0x{addr:02X} — {len(commands)} writes queued")

                _kill_bg_proc()

                try:
                    raw_write, _ = _ensure_dlpc_imports()
                except Exception as e:
                    log.appendPlainText(f"[ERROR] could not import dlpc_i2c: {e}")
                    return

                import time as _time
                t0 = _time.monotonic()
                for i, (op, data) in enumerate(commands):
                    hexdata = ' '.join(f'0x{b:02X}' for b in data) if data else '(no data)'
                    try:
                        raw_write(bus, addr, op, data)
                        log.appendPlainText(f"  [{i+1}/{len(commands)}] 0x{op:02X} {hexdata} → OK")
                    except Exception as e:
                        log.appendPlainText(f"  [{i+1}/{len(commands)}] 0x{op:02X} {hexdata} → FAILED: {e}")
                        log.appendPlainText("[BURST ABORTED] subsequent writes skipped")
                        return
                dt_ms = (_time.monotonic() - t0) * 1000
                log.appendPlainText(f"[BURST DONE] {len(commands)} writes in {dt_ms:.1f} ms")

            def _do_read():
                try:
                    bus = int(edt_bus.text().strip(), 0)
                    addr = int(edt_addr.text().strip(), 0)
                    op = int(edt_read_op.text().strip(), 0)
                    n = int(edt_read_n.text().strip(), 0)
                    if not (0 <= op <= 0xFF):
                        raise ValueError(f"opcode 0x{op:02X} out of byte range")
                    if n <= 0 or n > 256:
                        raise ValueError(f"read length {n} out of range (1..256)")
                except Exception as e:
                    log.appendPlainText(f"[READ ERROR] bad params: {e}")
                    return

                _kill_bg_proc()
                try:
                    _, raw_read = _ensure_dlpc_imports()
                except Exception as e:
                    log.appendPlainText(f"[READ ERROR] could not import dlpc_i2c: {e}")
                    return

                try:
                    r = raw_read(bus, addr, op, [], n)
                    hexr = ' '.join(f'0x{b:02X}' for b in r)
                    log.appendPlainText(f"[READ 0x{op:02X} ×{n}] {hexr}")
                except Exception as e:
                    log.appendPlainText(f"[READ ERROR] {e}")

            btn_load.clicked.connect(_do_load)
            btn_send_all.clicked.connect(_do_send_burst)
            btn_read.clicked.connect(_do_read)
            btn_clear_log.clicked.connect(lambda: log.clear())
            btn_close.clicked.connect(dlg.close)
            dlg.show()
        except Exception as e:
            self.warning(f"I²C Burst Sender dialog failed: {e}")

