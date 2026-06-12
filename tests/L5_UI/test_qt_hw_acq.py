"""Comprehensive characterization tests for ``qt_interface_hw_acq``.

1 per-layer test-type matrix (L5 row):
- ≥2 property-based tests (Hypothesis) — universal floor
- Visual regression — Required per sub-module; for non-image-producing
  mixins (HardwareAcqMixin produces NO pixels) we substitute with
  widget-state snapshot tests on the recording-button label codomain
  per the spec §15 substitution rule.
- Coverage target ≥85% line+branch

Module surface (~217 LOC, 7 methods) — HardwareAcqMixin extracted at
iter-2 of L5 §0.5 decomposition. Cluster 6 (recording / snapshot) +
cluster 7 (hardware acquisition mode).
"""

from __future__ import annotations

import os
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

from qt_interface_mixins.hw_acq import HardwareAcqMixin  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Stub host class
# ─────────────────────────────────────────────────────────────────────────────


def _make_camera(*, is_recording=False, is_armed=False,
                 save_dir=None, has_snapshot=True, has_save_image=False,
                 has_software_trigger=False, has_node_map=True,
                 arm_recording_return=True, snapshot_return=True):
    cam = MagicMock(spec=[])  # plain mock with no auto-attrs
    cam.is_recording = is_recording
    cam.is_armed = is_armed
    if save_dir is not None:
        cam.save_dir = save_dir
    if has_snapshot:
        cam.snapshot = MagicMock(return_value=snapshot_return)
    if has_save_image:
        cam.save_image = False
    if has_software_trigger:
        cam.software_trigger = MagicMock()
    cam.start_recording = MagicMock()
    cam.stop_recording = MagicMock()
    cam.disarm_recording = MagicMock()
    cam.arm_recording = MagicMock(return_value=arm_recording_return)
    cam.start_realtime_acquisition = MagicMock()
    cam.stop_realtime_acquisition = MagicMock()
    cam.start_hardware_acquisition = MagicMock()
    cam.stop_hardware_acquisition = MagicMock()
    if has_node_map:
        exp_node = MagicMock()
        exp_node.Value.return_value = 16667.0
        mode_entry = MagicMock(); mode_entry.SymbolicValue.return_value = "On"
        src_entry = MagicMock(); src_entry.SymbolicValue.return_value = "Line0"
        act_entry = MagicMock(); act_entry.SymbolicValue.return_value = "RisingEdge"
        mode_node = MagicMock(); mode_node.CurrentEntry.return_value = mode_entry
        src_node = MagicMock(); src_node.CurrentEntry.return_value = src_entry
        act_node = MagicMock(); act_node.CurrentEntry.return_value = act_entry

        def _find(name):
            return {
                "ExposureTime": exp_node,
                "TriggerMode": mode_node,
                "TriggerSource": src_node,
                "TriggerActivation": act_node,
            }.get(name)
        nm = MagicMock(); nm.FindNode.side_effect = _find
        cam.node_map = nm
    return cam


class _Host(HardwareAcqMixin):
    def __init__(self, *, camera=None, hardware=False, recording=False):
        self._camera = camera if camera is not None else _make_camera()
        self._button_start_recording = MagicMock()
        self._button_start_hardware_acquisition = MagicMock()
        self._dropdown_trigger_line = MagicMock()
        self._exp_line = MagicMock()
        self.acq_label = MagicMock()
        self._hardware_status = hardware
        self._recording_status = recording
        # `self.warning(msg)` is provided by Interface — stub it here
        self.warning = MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
# C1 — _update_recording_button_text
# ─────────────────────────────────────────────────────────────────────────────


class TestC1UpdateRecordingButtonText:
    """Contract: button text reflects camera.is_recording / is_armed precedence.

    Branches:
    - is_recording=True → "Stop Recording" (highest precedence)
    - is_armed=True, is_recording=False → "Disarm Recording"
    - both False → "Start Recording"
    - getattr defaults: missing attrs → both False → "Start Recording"
    """

    def test_recording_priority(self):
        cam = _make_camera(is_recording=True, is_armed=True)
        host = _Host(camera=cam)
        host._update_recording_button_text()
        host._button_start_recording.setText.assert_called_with("Stop Recording")

    def test_armed_path(self):
        cam = _make_camera(is_recording=False, is_armed=True)
        host = _Host(camera=cam)
        host._update_recording_button_text()
        host._button_start_recording.setText.assert_called_with("Disarm Recording")

    def test_idle_path(self):
        cam = _make_camera(is_recording=False, is_armed=False)
        host = _Host(camera=cam)
        host._update_recording_button_text()
        host._button_start_recording.setText.assert_called_with("Start Recording")

    def test_missing_attrs_default_to_idle(self):
        cam = MagicMock(spec=[])
        host = _Host(camera=cam)
        host._update_recording_button_text()
        host._button_start_recording.setText.assert_called_with("Start Recording")


# ─────────────────────────────────────────────────────────────────────────────
# C2 — _on_recording_started
# ─────────────────────────────────────────────────────────────────────────────


class TestC2OnRecordingStarted:
    """Contract: set status flag, force button to Stop, disable HW button +
    trigger-line dropdown."""

    def test_full_state_transition(self):
        host = _Host()
        host._on_recording_started()
        assert host._recording_status is True
        host._button_start_recording.setText.assert_called_with("Stop Recording")
        host._button_start_hardware_acquisition.setEnabled.assert_called_with(False)
        host._dropdown_trigger_line.setEnabled.assert_called_with(False)


# ─────────────────────────────────────────────────────────────────────────────
# C3 — _on_recording_stopped
# ─────────────────────────────────────────────────────────────────────────────


class TestC3OnRecordingStopped:
    """Contract: clear status flag, refresh button text, re-enable HW button;
    trigger-line dropdown re-enabled iff NOT in hardware mode.

    Branches:
    - hardware=False → trigger-line re-enabled
    - hardware=True  → trigger-line NOT re-enabled
    """

    def test_realtime_mode_reenables_trigger_dropdown(self):
        host = _Host(hardware=False, recording=True)
        host._on_recording_stopped()
        assert host._recording_status is False
        host._button_start_hardware_acquisition.setEnabled.assert_called_with(True)
        host._dropdown_trigger_line.setEnabled.assert_called_with(True)

    def test_hardware_mode_does_not_reenable_dropdown(self):
        host = _Host(hardware=True, recording=True)
        host._on_recording_stopped()
        assert host._recording_status is False
        host._button_start_hardware_acquisition.setEnabled.assert_called_with(True)
        host._dropdown_trigger_line.setEnabled.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# C4 — _on_auto_start_recording
# ─────────────────────────────────────────────────────────────────────────────


class TestC4OnAutoStartRecording:
    """Contract: call camera.start_recording(); swallow exceptions, print msg.

    Branches:
    - happy path → camera.start_recording invoked once
    - exception path → print "Auto-start recording failed", no re-raise
    """

    def test_happy_path(self):
        host = _Host()
        host._on_auto_start_recording()
        host._camera.start_recording.assert_called_once()

    def test_exception_swallowed(self, capsys):
        host = _Host()
        host._camera.start_recording.side_effect = RuntimeError("usb gone")
        host._on_auto_start_recording()  # no raise
        out = capsys.readouterr().out
        assert "Auto-start recording failed" in out


# ─────────────────────────────────────────────────────────────────────────────
# C5 — _trigger_sw_trigger
# ─────────────────────────────────────────────────────────────────────────────


class TestC5TriggerSwTrigger:
    """Contract: pick a snapshot path on the camera and invoke it; create
    save_dir if needed.

    Branches:
    - camera None → warning "No camera available for snapshot"
    - has_snapshot=True, snapshot_return=True → success (no warning)
    - has_snapshot=True, snapshot_return=False → warning "Snapshot failed"
    - has_save_image only → legacy path (sets camera.save_image=True)
    - has_software_trigger only → software_trigger() called
    - none of the above → warning "No snapshot method available"
    - outer exception → warning "Snapshot error:"
    - save_dir created if absent
    """

    def test_no_camera(self, tmp_path):
        host = _Host(camera=None)
        host._camera = None
        host._trigger_sw_trigger()
        host.warning.assert_called_with("No camera available for snapshot")

    def test_snapshot_success(self, tmp_path):
        cam = _make_camera(save_dir=str(tmp_path), has_snapshot=True,
                           snapshot_return=True)
        host = _Host(camera=cam)
        host._trigger_sw_trigger()
        cam.snapshot.assert_called_once()
        host.warning.assert_not_called()

    def test_snapshot_failure_warns(self, tmp_path):
        cam = _make_camera(save_dir=str(tmp_path), has_snapshot=True,
                           snapshot_return=False)
        host = _Host(camera=cam)
        host._trigger_sw_trigger()
        host.warning.assert_called_with("Snapshot failed - check camera status")

    def test_save_image_legacy_path(self, tmp_path, capsys):
        cam = _make_camera(save_dir=str(tmp_path), has_snapshot=False,
                           has_save_image=True)
        host = _Host(camera=cam)
        host._trigger_sw_trigger()
        assert cam.save_image is True
        assert "Legacy snapshot triggered" in capsys.readouterr().out

    def test_software_trigger_path(self, tmp_path, capsys):
        cam = _make_camera(save_dir=str(tmp_path), has_snapshot=False,
                           has_software_trigger=True)
        host = _Host(camera=cam)
        host._trigger_sw_trigger()
        cam.software_trigger.assert_called_once()
        assert "Software trigger sent" in capsys.readouterr().out

    def test_no_snapshot_method_at_all(self, tmp_path):
        cam = _make_camera(save_dir=str(tmp_path), has_snapshot=False)
        host = _Host(camera=cam)
        host._trigger_sw_trigger()
        host.warning.assert_called_with("No snapshot method available")

    def test_outer_exception_swallowed(self, monkeypatch, tmp_path):
        cam = _make_camera(save_dir=str(tmp_path))
        host = _Host(camera=cam)
        monkeypatch.setattr(os, "makedirs",
                            MagicMock(side_effect=OSError("disk full")))
        host._trigger_sw_trigger()
        # The warning is called with the error string
        assert host.warning.call_args is not None
        msg = host.warning.call_args.args[0]
        assert "Snapshot error" in msg

    def test_default_save_dir(self, tmp_path, monkeypatch):
        # No save_dir attribute on camera → defaults to './Saved_Media'
        cam = _make_camera(save_dir=None, has_snapshot=True)
        host = _Host(camera=cam)
        # Redirect makedirs so we don't pollute cwd
        calls = []

        def _fake_mkdirs(path, exist_ok=False):
            calls.append((path, exist_ok))
        monkeypatch.setattr(os, "makedirs", _fake_mkdirs)
        host._trigger_sw_trigger()
        assert calls and calls[0][0] == "./Saved_Media"
        assert calls[0][1] is True


# ─────────────────────────────────────────────────────────────────────────────
# C6 — _start_hardware_acquisition
# ─────────────────────────────────────────────────────────────────────────────


class TestC6StartHardwareAcquisition:
    """Contract: toggle between real-time and hardware acquisition modes.

    Branches:
    - hardware=False → enter HW mode (stop_realtime, start_hardware,
      read exposure, log trigger nodes, disable trigger-line dropdown,
      set acq_label to Hardware, set button text to "Stop Hardware",
      clear is_armed, refresh recording button text, toggle status to True)
    - hardware=False, exp readback raises → swallowed
    - hardware=False, trigger-node log raises → swallowed
    - hardware=False, no node_map attr → exposure readback skipped
    - hardware=True, is_armed=True → disarm called before stop_hardware
    - hardware=True, is_armed=False → no disarm
    - hardware=True, recording=False → trigger-line dropdown re-enabled
    - hardware=True, recording=True → trigger-line dropdown NOT re-enabled
    """

    def test_enter_hardware_mode(self):
        host = _Host(hardware=False)
        host._start_hardware_acquisition()
        host._camera.stop_realtime_acquisition.assert_called_once()
        host._camera.start_hardware_acquisition.assert_called_once()
        host._dropdown_trigger_line.setEnabled.assert_any_call(False)
        host.acq_label.setText.assert_called_with("Acquisition Mode: Hardware")
        host._button_start_hardware_acquisition.setText.assert_called_with(
            "Stop Hardware Acquisition")
        assert host._hardware_status is True
        assert host._camera.is_armed is False

    def test_enter_hw_exposure_readback_failure_swallowed(self, capsys):
        host = _Host(hardware=False)
        host._camera.node_map.FindNode.side_effect = RuntimeError("nm dead")
        host._start_hardware_acquisition()
        assert host._hardware_status is True
        out = capsys.readouterr().out
        assert (
            "HW mode exposure readback failed" in out
            or "Failed to read trigger nodes" in out
        )

    def test_leave_hardware_mode_with_armed(self):
        host = _Host(hardware=True)
        host._camera.is_armed = True
        host._start_hardware_acquisition()
        host._camera.disarm_recording.assert_called_once()
        host._camera.stop_hardware_acquisition.assert_called_once()
        host._camera.start_realtime_acquisition.assert_called_once()
        host.acq_label.setText.assert_called_with("Acquisition Mode: RealTime")
        host._button_start_hardware_acquisition.setText.assert_called_with(
            "Start Hardware Acquisition")
        assert host._hardware_status is False

    def test_leave_hardware_mode_not_armed(self):
        host = _Host(hardware=True)
        host._camera.is_armed = False
        host._start_hardware_acquisition()
        host._camera.disarm_recording.assert_not_called()
        assert host._hardware_status is False

    def test_leave_hw_reenables_trigger_when_not_recording(self):
        host = _Host(hardware=True, recording=False)
        host._start_hardware_acquisition()
        # setEnabled(True) was called for the trigger dropdown
        host._dropdown_trigger_line.setEnabled.assert_any_call(True)

    def test_leave_hw_does_not_reenable_trigger_when_recording(self):
        host = _Host(hardware=True, recording=True)
        host._start_hardware_acquisition()
        # No setEnabled(True) call
        for call in host._dropdown_trigger_line.setEnabled.call_args_list:
            assert call.args != (True,), \
                "trigger dropdown re-enabled despite active recording"

    def test_leave_hw_exposure_readback_swallowed_when_nm_present_but_fails(
            self):
        host = _Host(hardware=True)
        host._camera.node_map.FindNode.side_effect = RuntimeError("dead")
        host._start_hardware_acquisition()
        # Still completed
        assert host._hardware_status is False


# ─────────────────────────────────────────────────────────────────────────────
# C7 — _start_recording
# ─────────────────────────────────────────────────────────────────────────────


class TestC7StartRecording:
    """Contract: 4-way state machine on camera.is_recording / is_armed +
    hardware mode.

    Branches:
    - is_recording=True → stop_recording()
    - is_armed=True, not recording → disarm_recording() + button refresh
    - idle + hardware=True + arm_recording=True → button refresh
    - idle + hardware=True + arm_recording=False → no button refresh
    - idle + hardware=False → start_recording() (realtime)
    - exception → swallowed; print "Recording toggle failed"
    """

    def test_active_recording_stops(self):
        cam = _make_camera(is_recording=True)
        host = _Host(camera=cam)
        host._start_recording()
        cam.stop_recording.assert_called_once()

    def test_armed_disarms_and_refreshes(self):
        cam = _make_camera(is_armed=True)
        host = _Host(camera=cam)
        host._start_recording()
        cam.disarm_recording.assert_called_once()
        host._button_start_recording.setText.assert_called()

    def test_idle_hw_arm_success(self):
        cam = _make_camera(arm_recording_return=True)
        host = _Host(camera=cam, hardware=True)
        host._start_recording()
        cam.arm_recording.assert_called_once()
        host._button_start_recording.setText.assert_called()

    def test_idle_hw_arm_failure_no_refresh(self):
        cam = _make_camera(arm_recording_return=False)
        host = _Host(camera=cam, hardware=True)
        host._start_recording()
        cam.arm_recording.assert_called_once()
        host._button_start_recording.setText.assert_not_called()

    def test_idle_realtime_starts_recording(self):
        cam = _make_camera()
        host = _Host(camera=cam, hardware=False)
        host._start_recording()
        cam.start_recording.assert_called_once()

    def test_exception_swallowed(self, capsys):
        cam = _make_camera()
        cam.stop_recording.side_effect = RuntimeError("hw dead")
        cam.is_recording = True
        host = _Host(camera=cam)
        host._start_recording()  # no raise
        assert "Recording toggle failed" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────────
# Property tests (§1.1 universal floor — ≥2 per sub-module)
# ─────────────────────────────────────────────────────────────────────────────


class TestPropertyUpdateRecordingButtonTextCodomain:
    """The button label is always one of exactly three literals across
    every combination of (is_recording, is_armed).

    Pins:
    - is_recording=True dominates is_armed (precedence invariant)
    - label codomain has size 3 (no stray default branch)
    """

    @given(rec=st.booleans(), arm=st.booleans())
    @settings(max_examples=10, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_label_in_fixed_codomain(self, rec, arm):
        cam = _make_camera(is_recording=rec, is_armed=arm)
        host = _Host(camera=cam)
        host._update_recording_button_text()
        label = host._button_start_recording.setText.call_args.args[0]
        assert label in {"Start Recording", "Stop Recording",
                         "Disarm Recording"}

    @given(arm=st.booleans())
    @settings(max_examples=4, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_recording_dominates_armed(self, arm):
        cam = _make_camera(is_recording=True, is_armed=arm)
        host = _Host(camera=cam)
        host._update_recording_button_text()
        label = host._button_start_recording.setText.call_args.args[0]
        assert label == "Stop Recording"


class TestPropertyStartHardwareAcquisitionToggleParity:
    """Two consecutive _start_hardware_acquisition() calls restore
    _hardware_status to its starting value (XOR-toggle invariant).

    Pins: the function is an involution on the boolean state — any
    regression that, for example, only set _hardware_status=True
    unconditionally would fail this.
    """

    @given(start=st.booleans())
    @settings(max_examples=4, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_two_toggles_restore_state(self, start):
        host = _Host(hardware=start)
        host._start_hardware_acquisition()
        host._start_hardware_acquisition()
        assert host._hardware_status is start


# ─────────────────────────────────────────────────────────────────────────────
# Visual regression — substituted with widget-state snapshot tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVisualRegressionSubstitute:
    """HardwareAcqMixin paints no pixels. Per spec §15 substitution rule,
    we pin the widget-state snapshot: the EXACT sequence of setText / setEnabled
    calls produced by each user-visible state transition.

    Recovery criterion: when GUI verification fires on hardware (Phase A.5
    co-walk, ~1 PM daily), confirm the operator sees these exact strings;
    a regression would land as a string-substitution test failure here.
    """

    def test_snapshot_recording_started_calls(self):
        host = _Host()
        host._on_recording_started()
        # Snapshot the exact widget-mutation sequence
        rec_calls = [c.args[0] for c in
                     host._button_start_recording.setText.call_args_list]
        hw_calls = host._button_start_hardware_acquisition.setEnabled.call_args_list
        trig_calls = host._dropdown_trigger_line.setEnabled.call_args_list
        assert rec_calls == ["Stop Recording"]
        assert [c.args for c in hw_calls] == [(False,)]
        assert [c.args for c in trig_calls] == [(False,)]

    def test_snapshot_enter_hw_mode_calls(self):
        host = _Host(hardware=False)
        host._start_hardware_acquisition()
        acq_calls = [c.args[0] for c in host.acq_label.setText.call_args_list]
        hw_text_calls = [c.args[0] for c in
                         host._button_start_hardware_acquisition.setText.call_args_list]
        assert acq_calls == ["Acquisition Mode: Hardware"]
        assert hw_text_calls == ["Stop Hardware Acquisition"]

    def test_snapshot_leave_hw_mode_calls(self):
        host = _Host(hardware=True, recording=False)
        host._start_hardware_acquisition()
        acq_calls = [c.args[0] for c in host.acq_label.setText.call_args_list]
        hw_text_calls = [c.args[0] for c in
                         host._button_start_hardware_acquisition.setText.call_args_list]
        assert acq_calls == ["Acquisition Mode: RealTime"]
        assert hw_text_calls == ["Start Hardware Acquisition"]


# ─────────────────────────────────────────────────────────────────────────────
# Integration — mixin surface
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegrationMixinSurface:
    METHODS = (
        "_update_recording_button_text",
        "_on_recording_started",
        "_on_recording_stopped",
        "_on_auto_start_recording",
        "_trigger_sw_trigger",
        "_start_hardware_acquisition",
        "_start_recording",
    )

    def test_all_7_methods_on_subclass(self):
        host = _Host()
        for name in self.METHODS:
            assert callable(getattr(host, name, None)), f"Missing: {name}"

    def test_methods_defined_on_mixin(self):
        for name in self.METHODS:
            assert name in HardwareAcqMixin.__dict__

    def test_mixin_has_no_init(self):
        assert "__init__" not in HardwareAcqMixin.__dict__

    def test_interface_inherits_mixin(self):
        import qt_interface
        assert HardwareAcqMixin in qt_interface.Interface.__mro__
