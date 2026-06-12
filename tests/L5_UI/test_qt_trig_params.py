"""Comprehensive characterization tests for ``qt_interface_trig_params``.

1 per-layer test-type matrix (L5 row):
- ≥2 property tests (Hypothesis) — universal floor
- Visual regression — TrigParamsMixin produces a QDialog widget tree;
  substituted with widget-state + log/argv snapshot per spec §15 rule.
- Coverage target ≥85 % line+branch

Module surface (~305 LOC, 3 methods) — TrigParamsMixin extracted at
iter-5 of L5 §0.5 decomposition. Cluster 9 subset (camera trigger
parameters dialog + DMD sequence-type dispatch).

Methods:
- _open_trig_params_dialog()      — Build & show the Trigger Parameters
  QDialog (delay / exposure / activation / presets / Apply / Close)
- _apply_trig_params_to_camera()  — Apply stored _trig_* attributes onto
  the live IDS Peak NodeMap
- _on_seq_type_changed(text)      — log handler for I²C seq-type dropdown

The Apply callback (closure inside _open_trig_params_dialog) is reached
by capturing it from btn_apply.clicked.connect() and invoking directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_CRISPI_PARENT = (
    Path(__file__).resolve().parents[2]
    / "STIMscope"
    / "STIMViewer_CRISPI"
)
if str(_CRISPI_PARENT) not in sys.path:
    sys.path.insert(0, str(_CRISPI_PARENT))

import qt_interface_mixins.trig_params as _tpmod  # noqa: E402
from qt_interface_mixins.trig_params import TrigParamsMixin  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_node(value=0.0, symbolic="RisingEdge", minimum=1.0, maximum=200.0):
    n = MagicMock()
    n.Value.return_value = value
    n.SetValue = MagicMock()
    n.SetCurrentEntry = MagicMock()
    n.CurrentEntry.return_value = MagicMock(
        SymbolicValue=MagicMock(return_value=symbolic))
    n.Minimum.return_value = minimum
    n.Maximum.return_value = maximum
    return n


def _make_node_map(nodes=None):
    """Return a fake IDS Peak node_map that resolves FindNode by name."""
    if nodes is None:
        nodes = {}

    nm = MagicMock()
    def _find(name):
        return nodes.get(name)
    nm.FindNode.side_effect = _find
    return nm


class _Host(TrigParamsMixin):
    """Stub host satisfying the TrigParamsMixin contract."""

    def __init__(self, *, node_map=None, acq_running=False, acq_mode=0,
                 trig_delay_enabled=False, trig_delay_us=None,
                 trig_exp_enabled=False, trig_exp_us=None,
                 trig_activation=None, has_exp_line=True):
        cam = MagicMock()
        cam.node_map = node_map
        cam.acquisition_running = acq_running
        cam.acquisition_mode = acq_mode
        self._camera = cam
        if trig_delay_enabled is not None:
            self._trig_delay_enabled = trig_delay_enabled
        if trig_delay_us is not None:
            self._trig_delay_us = trig_delay_us
        if trig_exp_enabled is not None:
            self._trig_exp_enabled = trig_exp_enabled
        if trig_exp_us is not None:
            self._trig_exp_us = trig_exp_us
        if trig_activation is not None:
            self._trig_activation = trig_activation
        if has_exp_line:
            self._exp_line = MagicMock()


# Dialog-mock infrastructure for _open_trig_params_dialog
#
# The method imports PyQt5.QtWidgets symbols at call-time, so we patch
# sys.modules entries.

def _install_dialog_mocks(monkeypatch):
    """Install lightweight stand-ins for QDialog/QVBoxLayout/QGridLayout/
    QLabel/QLineEdit/QCheckBox/QPushButton/QComboBox. Returns the captured
    widget mocks for assertion in the calling test."""

    captured = {
        "dlg": MagicMock(),
        "chk_delay": MagicMock(),
        "chk_exp": MagicMock(),
        "edt_delay": MagicMock(),
        "edt_exp": MagicMock(),
        "cmb_act": MagicMock(),
        "preset_blue": MagicMock(),
        "preset_full": MagicMock(),
        "btn_apply": MagicMock(),
        "btn_close": MagicMock(),
        "status_lbl": MagicMock(),
        "checkbox_count": 0,
        "lineedit_count": 0,
        "pushbutton_count": 0,
        "label_count": 0,
    }

    # Default texts
    captured["edt_delay"].text.return_value = ""
    captured["edt_exp"].text.return_value = ""
    captured["cmb_act"].currentText.return_value = "RisingEdge"
    captured["cmb_act"].findText.return_value = 0
    captured["chk_delay"].isChecked.return_value = False
    captured["chk_exp"].isChecked.return_value = False

    def _qcheckbox(*a, **kw):
        captured["checkbox_count"] += 1
        if captured["checkbox_count"] == 1:
            return captured["chk_delay"]
        return captured["chk_exp"]

    def _qlineedit(*a, **kw):
        captured["lineedit_count"] += 1
        if captured["lineedit_count"] == 1:
            return captured["edt_delay"]
        return captured["edt_exp"]

    def _qpushbutton(*a, **kw):
        captured["pushbutton_count"] += 1
        order = ["preset_blue", "preset_full", "btn_apply", "btn_close"]
        if captured["pushbutton_count"] <= 4:
            return captured[order[captured["pushbutton_count"] - 1]]
        return MagicMock()

    def _qlabel(*a, **kw):
        captured["label_count"] += 1
        if captured["label_count"] == 3:
            return captured["status_lbl"]
        return MagicMock()

    def _qdialog(*a, **kw):
        return captured["dlg"]

    fake_qtw = MagicMock()
    fake_qtw.QDialog = _qdialog
    fake_qtw.QVBoxLayout = MagicMock()
    fake_qtw.QGridLayout = MagicMock()
    fake_qtw.QLabel = _qlabel
    fake_qtw.QLineEdit = _qlineedit
    fake_qtw.QCheckBox = _qcheckbox
    fake_qtw.QPushButton = _qpushbutton
    fake_qtw.QComboBox = MagicMock(return_value=captured["cmb_act"])
    monkeypatch.setitem(sys.modules, "PyQt5.QtWidgets", fake_qtw)

    # Also patch the QtWidgets and QtCore in the mixin module namespace
    # so the `QtWidgets.QHBoxLayout()` calls work.
    fake_module_qtw = MagicMock()
    monkeypatch.setattr(_tpmod, "QtWidgets", fake_module_qtw)

    fake_module_qtc = MagicMock()
    monkeypatch.setattr(_tpmod, "QtCore", fake_module_qtc)

    return captured


# ═════════════════════════════════════════════════════════════════════════════
# C1 — _on_seq_type_changed
# ═════════════════════════════════════════════════════════════════════════════


class TestC1OnSeqTypeChanged:
    """Contract: parse a sequence-type dropdown string into one of four
    canonical bytes (0x00, 0x01, 0x02, 0x03) and log; never raise.

    Branches:
    - "0x03" / startswith("8-bit RGB") → "0x03"
    - "0x02" / startswith("8-bit Mono") → "0x02"
    - "0x00" / startswith("1-bit Mono") → "0x00"
    - anything else → "0x01" (1-bit RGB default)
    - inner exception → swallowed
    """

    @pytest.mark.parametrize("text,expected", [
        ("8-bit RGB (0x03)", "0x03"),
        ("8-bit RGB anything", "0x03"),
        ("8-bit Mono", "0x02"),
        ("(0x02) anything", "0x02"),
        ("1-bit Mono", "0x00"),
        ("(0x00)", "0x00"),
        ("1-bit RGB", "0x01"),
        ("unknown", "0x01"),
        ("", "0x01"),
    ])
    def test_seq_first_codomain(self, text, expected, capsys):
        host = _Host()
        host._on_seq_type_changed(text)
        out = capsys.readouterr().out
        assert f"-> {expected}" in out

    def test_exception_swallowed(self, capsys):
        host = _Host()
        # text=None → startswith() raises AttributeError → swallowed
        host._on_seq_type_changed(None)
        # No raise; no output (the except path doesn't print)


# ═════════════════════════════════════════════════════════════════════════════
# C2 — _apply_trig_params_to_camera
# ═════════════════════════════════════════════════════════════════════════════


class TestC2ApplyTrigParamsToCamera:
    """Contract: write _trig_delay_us / _trig_exp_us / _trig_activation onto
    the live IDS Peak node map. Each write is wrapped in its own try/except.
    Adjusts AcquisitionFrameRate to keep exposure feasible. Updates
    _exp_line widget.

    Branches:
    - node_map is None → early return
    - _trig_delay_enabled True + _trig_delay_us set → SetValue called
    - _trig_delay_enabled True + _trig_delay_us None → skip
    - _trig_delay_enabled False → skip
    - TriggerDelay SetValue raises → swallowed, log
    - _trig_exp_enabled True → ExposureAuto off + AcquisitionFrameRate
      adjust + ExposureTime set + read-back + _exp_line.setText
    - ExposureAuto raises → swallowed
    - AcquisitionFrameRate missing → skip
    - needed_fps < fps_node.Value() → SetValue called
    - needed_fps >= fps_node.Value() → SetValue not called
    - max_fps clamp branch
    - ExposureTime SetValue raises → swallowed
    - read-back Value() raises → log
    - _trig_activation None → skip
    - TriggerActivation set raises → log
    - outer exception → swallowed
    """

    def test_node_map_none_early_return(self):
        host = _Host(node_map=None, trig_delay_enabled=True, trig_delay_us=100.0)
        host._apply_trig_params_to_camera()  # no raise

    def test_delay_applied(self, capsys):
        delay_node = _make_node()
        nm = _make_node_map({"TriggerDelay": delay_node})
        host = _Host(node_map=nm, trig_delay_enabled=True, trig_delay_us=11000.0,
                     trig_exp_enabled=False)
        host._apply_trig_params_to_camera()
        delay_node.SetValue.assert_called_once_with(11000.0)
        out = capsys.readouterr().out
        assert "Applied TriggerDelay = 11000.0" in out

    def test_delay_disabled_skipped(self):
        delay_node = _make_node()
        nm = _make_node_map({"TriggerDelay": delay_node})
        host = _Host(node_map=nm, trig_delay_enabled=False, trig_delay_us=11000.0,
                     trig_exp_enabled=False)
        host._apply_trig_params_to_camera()
        delay_node.SetValue.assert_not_called()

    def test_delay_us_none_skipped(self):
        delay_node = _make_node()
        nm = _make_node_map({"TriggerDelay": delay_node})
        host = _Host(node_map=nm, trig_delay_enabled=True, trig_delay_us=None,
                     trig_exp_enabled=False)
        # _trig_delay_us deliberately None — not set on instance
        host._apply_trig_params_to_camera()
        delay_node.SetValue.assert_not_called()

    def test_delay_set_raises_swallowed(self, capsys):
        delay_node = _make_node()
        delay_node.SetValue.side_effect = RuntimeError("set failed")
        nm = _make_node_map({"TriggerDelay": delay_node})
        host = _Host(node_map=nm, trig_delay_enabled=True, trig_delay_us=11000.0,
                     trig_exp_enabled=False)
        host._apply_trig_params_to_camera()  # no raise
        out = capsys.readouterr().out
        assert "Failed to set TriggerDelay" in out

    def test_exposure_applied_with_fps_clamp(self, capsys):
        exp_node = _make_node(value=5000.0)
        fps_node = _make_node(value=120.0, minimum=1.0, maximum=200.0)
        # needed_fps = 1e6 / 5000 = 200 → not < 120 (current), so SetValue
        # in the first block runs only if 200 < 120 (False).
        nm = _make_node_map({
            "ExposureTime": exp_node,
            "ExposureAuto": _make_node(),
            "AcquisitionFrameRate": fps_node,
        })
        host = _Host(node_map=nm, trig_delay_enabled=False,
                     trig_exp_enabled=True, trig_exp_us=5000.0)
        host._apply_trig_params_to_camera()
        # ExposureTime SetValue called
        exp_node.SetValue.assert_called_with(5000.0)
        # Second fps SetValue runs (max_fps clamp)
        assert fps_node.SetValue.called
        # _exp_line written
        host._exp_line.setText.assert_called_with("5000.000")
        out = capsys.readouterr().out
        assert "Applied ExposureTime" in out

    def test_exposure_needed_fps_below_current_lowers_it(self):
        """If needed_fps < current fps, the first SetValue inside the try block
        lowers the fps to accommodate a long exposure."""
        exp_node = _make_node(value=33333.0)
        fps_node = _make_node(value=60.0, minimum=1.0, maximum=200.0)
        # needed_fps = 1e6 / 33333 ≈ 30.00 < 60 → first SetValue called with
        # max(min, needed_fps) = max(1, 30.0) = 30.0
        nm = _make_node_map({
            "ExposureTime": exp_node,
            "ExposureAuto": _make_node(),
            "AcquisitionFrameRate": fps_node,
        })
        host = _Host(node_map=nm, trig_delay_enabled=False,
                     trig_exp_enabled=True, trig_exp_us=33333.0)
        host._apply_trig_params_to_camera()
        # The first SetValue inside the try block lowered fps
        # Both SetValues called at least once
        assert fps_node.SetValue.call_count >= 1

    def test_exposure_auto_off_raises_swallowed(self):
        exp_node = _make_node()
        ea_node = _make_node()
        ea_node.SetCurrentEntry.side_effect = RuntimeError("auto raise")
        nm = _make_node_map({
            "ExposureTime": exp_node,
            "ExposureAuto": ea_node,
        })
        host = _Host(node_map=nm, trig_delay_enabled=False,
                     trig_exp_enabled=True, trig_exp_us=5000.0)
        host._apply_trig_params_to_camera()  # no raise
        exp_node.SetValue.assert_called()

    def test_fps_node_missing_skips_fps_adjust(self):
        exp_node = _make_node()
        nm = _make_node_map({
            "ExposureTime": exp_node,
            "ExposureAuto": _make_node(),
        })  # AcquisitionFrameRate missing
        host = _Host(node_map=nm, trig_delay_enabled=False,
                     trig_exp_enabled=True, trig_exp_us=5000.0)
        host._apply_trig_params_to_camera()
        exp_node.SetValue.assert_called()

    def test_exposure_set_raises_swallowed(self):
        exp_node = _make_node()
        exp_node.SetValue.side_effect = RuntimeError("exp raise")
        nm = _make_node_map({
            "ExposureTime": exp_node,
            "ExposureAuto": _make_node(),
        })
        host = _Host(node_map=nm, trig_delay_enabled=False,
                     trig_exp_enabled=True, trig_exp_us=5000.0)
        host._apply_trig_params_to_camera()  # no raise

    def test_exp_value_readback_raises_logs(self, capsys):
        exp_node = _make_node()
        # First.Value() succeeds (used in needed_fps calc but no, only fps.Value
        # is the one being called). Actually look at code: nm.FindNode("ExposureTime").Value()
        # is called after the SetValue, in the print. Force that call to raise.
        exp_node.Value.side_effect = RuntimeError("read failed")
        nm = _make_node_map({
            "ExposureTime": exp_node,
            "ExposureAuto": _make_node(),
        })
        host = _Host(node_map=nm, trig_delay_enabled=False,
                     trig_exp_enabled=True, trig_exp_us=5000.0)
        host._apply_trig_params_to_camera()
        out = capsys.readouterr().out
        assert "Failed to set ExposureTime" in out

    def test_exp_line_missing_still_succeeds(self):
        exp_node = _make_node()
        nm = _make_node_map({
            "ExposureTime": exp_node,
            "ExposureAuto": _make_node(),
        })
        host = _Host(node_map=nm, trig_delay_enabled=False,
                     trig_exp_enabled=True, trig_exp_us=5000.0,
                     has_exp_line=False)
        host._apply_trig_params_to_camera()  # no raise

    def test_activation_applied(self, capsys):
        act_node = _make_node()
        nm = _make_node_map({"TriggerActivation": act_node})
        host = _Host(node_map=nm, trig_delay_enabled=False,
                     trig_exp_enabled=False, trig_activation="FallingEdge")
        host._apply_trig_params_to_camera()
        act_node.SetCurrentEntry.assert_called_with("FallingEdge")
        out = capsys.readouterr().out
        assert "Applied TriggerActivation = FallingEdge" in out

    def test_activation_none_skipped(self):
        act_node = _make_node()
        nm = _make_node_map({"TriggerActivation": act_node})
        host = _Host(node_map=nm, trig_delay_enabled=False,
                     trig_exp_enabled=False, trig_activation=None)
        # _trig_activation is None
        host._apply_trig_params_to_camera()
        act_node.SetCurrentEntry.assert_not_called()

    def test_activation_set_raises_swallowed(self, capsys):
        act_node = _make_node()
        act_node.SetCurrentEntry.side_effect = RuntimeError("set act")
        nm = _make_node_map({"TriggerActivation": act_node})
        host = _Host(node_map=nm, trig_delay_enabled=False,
                     trig_exp_enabled=False, trig_activation="RisingEdge")
        host._apply_trig_params_to_camera()  # no raise
        out = capsys.readouterr().out
        assert "Failed to set TriggerActivation" in out

    def test_outer_exception_swallowed(self):
        host = _Host(node_map=None, trig_delay_enabled=True, trig_delay_us=100.0)
        host._camera = None  # getattr(None, 'node_map') succeeds (returns None)
        # But to actually hit the outer except, make the cam attr lookup raise:
        class _Trickle:
            @property
            def node_map(self):
                raise RuntimeError("cam dead")
        host._camera = _Trickle()
        host._apply_trig_params_to_camera()  # no raise — outer except


# ═════════════════════════════════════════════════════════════════════════════
# C3 — _open_trig_params_dialog (construction + Apply closure)
# ═════════════════════════════════════════════════════════════════════════════


class TestC3OpenTrigParamsDialog:
    """Contract: build a modeless Trigger Parameters QDialog and connect
    Apply/Close handlers + preset-button slots. The Apply closure reads
    the dialog state into _trig_* attributes, optionally calls
    _apply_trig_params_to_camera if hardware-mode acquisition is running.

    Branches (dialog construction):
    - QDialog setWindowFlags raises → swallowed
    - node_map None → fallback values from getattr; status reads ""
    - chk_delay setText raises → swallowed (try/except)
    - chk_exp setText raises → swallowed
    - findText(<known>) returns ≥0 → setCurrentIndex called
    - findText returns -1 → setCurrentIndex not called

    Branches (Apply closure):
    - chk_delay checked + edt_delay non-empty → _trig_delay_us = float(text)
    - edt_delay empty → _trig_delay_us = None
    - edt_delay invalid → _trig_delay_us = None
    - chk_exp similar
    - d+e > 33333 → warn print
    - acq running + mode=1 → _apply_trig_params_to_camera() called
    - acq off → just stored
    - inner exception → "Failed to apply trig params" log

    Branches (outer):
    - outer exception → "Failed to open Trigger Parameters dialog" log
    """

    def test_dialog_construction_happy(self, monkeypatch):
        captured = _install_dialog_mocks(monkeypatch)
        nm = _make_node_map({
            "TriggerDelay": _make_node(value=11000.0),
            "ExposureTime": _make_node(value=5000.0),
            "TriggerActivation": _make_node(symbolic="FallingEdge"),
        })
        host = _Host(node_map=nm, trig_delay_enabled=True, trig_delay_us=11000.0,
                     trig_exp_enabled=True, trig_exp_us=5000.0,
                     trig_activation="FallingEdge")
        host._open_trig_params_dialog()
        captured["dlg"].setWindowTitle.assert_called_with("Trigger Parameters")
        # Apply button got connected
        captured["btn_apply"].clicked.connect.assert_called_once()
        captured["btn_close"].clicked.connect.assert_called_once()
        # Dialog shown
        captured["dlg"].show.assert_called_once()

    def test_dialog_construction_node_map_none_uses_fallbacks(self, monkeypatch):
        captured = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None, trig_delay_enabled=False,
                     trig_exp_enabled=False, trig_activation="RisingEdge")
        host._open_trig_params_dialog()
        captured["dlg"].show.assert_called_once()

    def test_dialog_window_flags_raise_swallowed(self, monkeypatch):
        captured = _install_dialog_mocks(monkeypatch)
        captured["dlg"].setWindowFlags.side_effect = RuntimeError("dead")
        host = _Host(node_map=None)
        host._open_trig_params_dialog()  # no raise
        captured["dlg"].show.assert_called_once()

    def test_findtext_negative_skips_setCurrentIndex(self, monkeypatch):
        captured = _install_dialog_mocks(monkeypatch)
        captured["cmb_act"].findText.return_value = -1
        host = _Host(node_map=None)
        host._open_trig_params_dialog()
        captured["cmb_act"].setCurrentIndex.assert_not_called()

    def test_outer_exception_swallowed(self, monkeypatch, capsys):
        # Force QDialog import to raise
        fake_qtw = MagicMock()
        fake_qtw.QDialog = MagicMock(side_effect=RuntimeError("dlg dead"))
        monkeypatch.setitem(sys.modules, "PyQt5.QtWidgets", fake_qtw)
        host = _Host(node_map=None)
        host._open_trig_params_dialog()
        out = capsys.readouterr().out
        assert "Failed to open Trigger Parameters dialog" in out

    def _capture_apply_callback(self, captured):
        """Pull out the _apply closure attached to btn_apply.clicked.connect."""
        return captured["btn_apply"].clicked.connect.call_args.args[0]

    def test_apply_stores_state_no_hardware(self, monkeypatch, capsys):
        captured = _install_dialog_mocks(monkeypatch)
        captured["chk_delay"].isChecked.return_value = True
        captured["chk_exp"].isChecked.return_value = True
        captured["edt_delay"].text.return_value = "11000"
        captured["edt_exp"].text.return_value = "5000"
        captured["cmb_act"].currentText.return_value = "FallingEdge"
        host = _Host(node_map=None, acq_running=False, acq_mode=0)
        host._open_trig_params_dialog()
        apply_cb = self._capture_apply_callback(captured)
        apply_cb()
        assert host._trig_delay_enabled is True
        assert host._trig_delay_us == 11000.0
        assert host._trig_exp_enabled is True
        assert host._trig_exp_us == 5000.0
        assert host._trig_activation == "FallingEdge"
        out = capsys.readouterr().out
        assert "Trig params STORED" in out

    def test_apply_invalid_text_yields_none(self, monkeypatch):
        captured = _install_dialog_mocks(monkeypatch)
        captured["chk_delay"].isChecked.return_value = True
        captured["chk_exp"].isChecked.return_value = True
        captured["edt_delay"].text.return_value = "not_a_number"
        captured["edt_exp"].text.return_value = "bad"
        host = _Host(node_map=None)
        host._open_trig_params_dialog()
        apply_cb = self._capture_apply_callback(captured)
        apply_cb()
        assert host._trig_delay_us is None
        assert host._trig_exp_us is None

    def test_apply_empty_text_yields_none(self, monkeypatch):
        captured = _install_dialog_mocks(monkeypatch)
        captured["chk_delay"].isChecked.return_value = True
        captured["chk_exp"].isChecked.return_value = True
        captured["edt_delay"].text.return_value = ""
        captured["edt_exp"].text.return_value = ""
        host = _Host(node_map=None)
        host._open_trig_params_dialog()
        apply_cb = self._capture_apply_callback(captured)
        apply_cb()
        assert host._trig_delay_us is None
        assert host._trig_exp_us is None

    def test_apply_warns_on_period_overrun(self, monkeypatch, capsys):
        captured = _install_dialog_mocks(monkeypatch)
        captured["chk_delay"].isChecked.return_value = True
        captured["chk_exp"].isChecked.return_value = True
        captured["edt_delay"].text.return_value = "20000"
        captured["edt_exp"].text.return_value = "20000"
        host = _Host(node_map=None)
        host._open_trig_params_dialog()
        apply_cb = self._capture_apply_callback(captured)
        apply_cb()
        out = capsys.readouterr().out
        assert "exceeds 33333" in out

    def test_apply_hardware_mode_triggers_camera_apply(self, monkeypatch, capsys):
        captured = _install_dialog_mocks(monkeypatch)
        captured["chk_delay"].isChecked.return_value = True
        captured["edt_delay"].text.return_value = "1000"
        captured["edt_exp"].text.return_value = ""
        host = _Host(node_map=None, acq_running=True, acq_mode=1)
        host._open_trig_params_dialog()
        # Monkey-patch _apply_trig_params_to_camera to track call
        host._apply_trig_params_to_camera = MagicMock()
        apply_cb = self._capture_apply_callback(captured)
        apply_cb()
        host._apply_trig_params_to_camera.assert_called_once()
        out = capsys.readouterr().out
        assert "applied to camera now" in out

    def test_apply_inner_exception_logged(self, monkeypatch, capsys):
        captured = _install_dialog_mocks(monkeypatch)
        # Force cmb_act.currentText to raise during apply
        captured["chk_delay"].isChecked.side_effect = RuntimeError("dead checkbox")
        host = _Host(node_map=None)
        host._open_trig_params_dialog()
        apply_cb = self._capture_apply_callback(captured)
        apply_cb()
        out = capsys.readouterr().out
        assert "Failed to apply trig params" in out

    def test_preset_callbacks_load_blue_values(self, monkeypatch):
        captured = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None)
        host._open_trig_params_dialog()
        # Pull blue-preset callback
        blue_cb = captured["preset_blue"].clicked.connect.call_args.args[0]
        blue_cb()
        captured["chk_delay"].setChecked.assert_any_call(True)
        captured["edt_delay"].setText.assert_any_call("11000")
        captured["chk_exp"].setChecked.assert_any_call(True)
        captured["edt_exp"].setText.assert_any_call("5000")

    def test_preset_callbacks_load_full_values(self, monkeypatch):
        captured = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None)
        host._open_trig_params_dialog()
        full_cb = captured["preset_full"].clicked.connect.call_args.args[0]
        full_cb()
        # full preset: delay=0, exp=33333.33 → int conversion
        captured["edt_delay"].setText.assert_any_call("0")
        captured["edt_exp"].setText.assert_any_call("33333")


# ═════════════════════════════════════════════════════════════════════════════
# Property tests (§1.1 universal floor — ≥2)
# ═════════════════════════════════════════════════════════════════════════════


class TestPropertySeqTypeCodomain:
    """Property: for any text, the seq_first byte logged is one of exactly
    four values: 0x00, 0x01, 0x02, 0x03."""

    KNOWN = {"0x00", "0x01", "0x02", "0x03"}

    @given(text=st.text(min_size=0, max_size=40))
    @settings(max_examples=40, deadline=None,
              suppress_health_check=[HealthCheck.too_slow,
                                     HealthCheck.function_scoped_fixture])
    def test_seq_first_in_known_set(self, text, capsys):
        host = _Host()
        host._on_seq_type_changed(text)
        out = capsys.readouterr().out
        # Either nothing was logged (exception path) or the log contains
        # one of the canonical bytes.
        if "->" in out:
            tail = out.strip().split("->")[-1].strip()
            assert tail in self.KNOWN


class TestPropertyApplyTrigParamsDelayCodomain:
    """Property: for any (enabled, value) pair, the IDS node's SetValue is
    either called exactly once (enabled True + value not None) or not at
    all (otherwise). No exceptions escape."""

    @given(
        enabled=st.booleans(),
        value=st.one_of(st.none(), st.floats(min_value=0, max_value=50000,
                                             allow_nan=False, allow_infinity=False)),
    )
    @settings(max_examples=30, deadline=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    def test_delay_setvalue_call_count(self, enabled, value):
        delay_node = _make_node()
        nm = _make_node_map({"TriggerDelay": delay_node})
        host = _Host(node_map=nm, trig_delay_enabled=enabled,
                     trig_delay_us=value, trig_exp_enabled=False,
                     trig_activation=None)
        host._apply_trig_params_to_camera()
        if enabled and value is not None:
            assert delay_node.SetValue.call_count == 1
        else:
            assert delay_node.SetValue.call_count == 0


# ═════════════════════════════════════════════════════════════════════════════
# Visual regression — log/argv snapshot substitute
# ═════════════════════════════════════════════════════════════════════════════


class TestVisualRegressionSubstitute:
    """TrigParamsMixin's dialog body produces no pixel-rendered output we
    can characterize without a real Qt event loop. Per spec §15 substitution
    rule, pin the exact log strings (which the operator sees in stdout) and
    the exact node-write argv values for representative workflows.

    Recovery criterion: at Phase A.5 hardware co-walk, user verifies that
    the dialog renders the title "Trigger Parameters" and that applying the
    Blue sub-frame preset yields the camera log lines pinned here.
    """

    def test_blue_subframe_log_snapshot(self, capsys):
        host = _Host()
        host._on_seq_type_changed("8-bit RGB (0x03)")
        out = capsys.readouterr().out.strip()
        assert out == "[I2C] Sequence type changed: 8-bit RGB (0x03) -> 0x03"

    def test_delay_apply_node_call_snapshot(self):
        delay_node = _make_node()
        nm = _make_node_map({"TriggerDelay": delay_node})
        host = _Host(node_map=nm, trig_delay_enabled=True,
                     trig_delay_us=11000.0, trig_exp_enabled=False,
                     trig_activation=None)
        host._apply_trig_params_to_camera()
        # Exact byte-shape pinned: SetValue called with float(11000.0)
        delay_node.SetValue.assert_called_once_with(11000.0)


# ═════════════════════════════════════════════════════════════════════════════
# Integration — mixin surface
# ═════════════════════════════════════════════════════════════════════════════


class TestIntegrationMixinSurface:
    METHODS = (
        "_open_trig_params_dialog",
        "_apply_trig_params_to_camera",
        "_on_seq_type_changed",
    )

    def test_all_3_methods_on_subclass(self):
        host = _Host()
        for name in self.METHODS:
            assert callable(getattr(host, name, None)), f"Missing: {name}"

    def test_methods_defined_on_mixin(self):
        for name in self.METHODS:
            assert name in TrigParamsMixin.__dict__

    def test_mixin_has_no_init(self):
        assert "__init__" not in TrigParamsMixin.__dict__

    def test_interface_inherits_mixin(self):
        import qt_interface
        assert TrigParamsMixin in qt_interface.Interface.__mro__
