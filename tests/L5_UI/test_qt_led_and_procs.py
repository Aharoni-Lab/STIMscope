"""Comprehensive characterization tests for ``qt_interface_led_and_procs``.

1 per-layer test-type matrix (L5 row):
- ≥2 property tests (Hypothesis) — universal floor
- Visual regression — required per sub-module; LEDAndProcessMixin paints
  no pixels, so we substitute with widget-state snapshot tests (button-
  label codomain + state transition sequences) per spec §15 rule.
- Coverage target ≥85% line+branch

Module surface (~260 LOC, 4 methods) — LEDAndProcessMixin extracted at
iter-3 of L5 §0.5 decomposition. Cluster 2 subset (LED live-change +
external-process lifecycle).

Methods:
- _on_led_color_changed_live(text) — debounce LED dropdown via QTimer
- _apply_led_color_live()          — spawn i2c_test_send_commands.py
- _on_proc_finished(which)         — Qt slot on QProcess finished signal
- _terminate_external_processes()  — kill all helper QProcesses on close
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

import qt_interface_mixins.led_and_procs as _ledmod  # noqa: E402
from qt_interface_mixins.led_and_procs import LEDAndProcessMixin  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Stub host class
# ─────────────────────────────────────────────────────────────────────────────


class _FakeQProcessClass:
    """Replacement for QtCore.QProcess passed through `_ensure_qprocess()`.

    Behaves like the QProcess class itself for the few static attrs the
    mixin reads. Instances are MagicMock — see _make_proc_instance().
    """

    NotRunning = 0
    Starting = 1
    Running = 2

    def __init__(self, *args, **kwargs):
        # Instantiation path (real QProcess(self) call inside the mixin).
        # We want to deliver a MagicMock instance with the same surface
        # the mixin then exercises.
        self._mock = _make_proc_instance()

    def __getattr__(self, name):
        return getattr(self._mock, name)


def _make_proc_instance(state_value=2):
    """A MagicMock standing in for a QProcess *instance*."""
    p = MagicMock()
    p.state = MagicMock(return_value=state_value)
    p.kill = MagicMock()
    p.waitForFinished = MagicMock()
    p.deleteLater = MagicMock()
    p.start = MagicMock()
    p.setWorkingDirectory = MagicMock()
    p.finished = MagicMock()
    p.errorOccurred = MagicMock()
    return p


def _ensure_qprocess_returns_fakeclass():
    """Return a callable that, when used as `self._ensure_qprocess()`,
    yields a class-like object exposing `.NotRunning` and being callable
    to produce instance mocks."""
    class _C:
        NotRunning = 0
        # Calling `_C(parent)` should produce a fresh MagicMock instance
        # (the way `proc = QProcess(self)` returns a QProcess).
        def __new__(cls, *_args, **_kwargs):
            return _make_proc_instance()
    return _C


class _Host(LEDAndProcessMixin):
    """Stub satisfying the LEDAndProcessMixin contract."""

    def __init__(self, *, dmd_running=False, cs_running=False):
        self._dmd_sequencer_running = dmd_running
        self._cs_pipeline_running = cs_running
        self._led_color_dropdown = MagicMock()
        self._seq_type_dropdown = MagicMock()
        self._proc_i2c = None
        self._proc_masks = None
        self._proc_projector = None
        self._proc_i2c_live_led = None
        self._button_send_triggers = MagicMock()
        self._button_send_masks = MagicMock()
        self._button_start_projector = MagicMock()
        # `_ensure_qprocess()` returns the QProcess CLASS in the real code
        self._ensure_qprocess = MagicMock(
            return_value=_ensure_qprocess_returns_fakeclass())
        # `_attach_proc_signals` is normally an Interface helper
        self._attach_proc_signals = MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
# C1 — _on_led_color_changed_live
# ─────────────────────────────────────────────────────────────────────────────


class TestC1OnLedColorChangedLive:
    """Contract: gated on _dmd_sequencer_running; if it's running, lazy-init
    a 250 ms single-shot debounce QTimer and restart it.

    Branches:
    - dmd_sequencer_running=False → early return, no timer
    - dmd_sequencer_running=True, no existing timer → lazy-create + start
    - dmd_sequencer_running=True, existing timer    → just restart (no reinit)
    """

    def test_dmd_off_early_return(self):
        host = _Host(dmd_running=False)
        host._on_led_color_changed_live("Blue")
        assert not hasattr(host, "_led_live_debounce_timer")

    def test_lazy_create_timer(self):
        host = _Host(dmd_running=True, cs_running=False)
        # Patch QtCore.QTimer in the mixin module
        fake_timer_cls = MagicMock()
        fake_timer = MagicMock()
        fake_timer_cls.return_value = fake_timer
        with patch.object(_ledmod, "QtCore") as fake_QtCore:
            fake_QtCore.QTimer = fake_timer_cls
            host._on_led_color_changed_live("Blue")
        # Timer was constructed once, configured, and started
        fake_timer_cls.assert_called_once_with(host)
        fake_timer.setSingleShot.assert_called_with(True)
        fake_timer.setInterval.assert_called_with(250)
        fake_timer.timeout.connect.assert_called_once()
        fake_timer.start.assert_called_once()

    def test_existing_timer_just_restarts(self):
        host = _Host(dmd_running=True, cs_running=False)
        existing = MagicMock()
        host._led_live_debounce_timer = existing
        host._on_led_color_changed_live("Red")
        # No reinit (setSingleShot not called again)
        existing.setSingleShot.assert_not_called()
        existing.setInterval.assert_not_called()
        existing.start.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# C2 — _apply_led_color_live (illum + seq_type translation + subprocess spawn)
# ─────────────────────────────────────────────────────────────────────────────


class TestC2ApplyLedColorLive:
    """Contract: translate dropdown selections to an illum bitmask + seq_type
    index; kill any prior live-change proc on the I²C bus; spawn a new
    QProcess running i2c_test_send_commands.py boot with the resolved args.

    Branches:
    - LED dropdown currentText() raises → silent early return
    - LED string contains 0x01 / 0x02 / 0x04 / 0x07 / 0x05 / 0x03 → each
      maps to its illum
    - LED string has none of those → early return
    - seq_type dropdown raises → seq_type="0"
    - seq_type contains 0x01 / 0x02 / 0x03 (each branch) + startswith
      legacy strings → each maps
    - prev live-change proc Running → kill+waitForFinished+deleteLater
    - prev live-change proc NotRunning → just deleteLater
    - outer except → swallowed
    """

    @pytest.mark.parametrize("sel,expected_illum", [
        ("UV (0x01)", "0x01"),
        ("Red (0x02)", "0x02"),
        ("Green (0x04)", "0x04"),
        ("White (0x07)", "0x07"),
        ("Magenta (0x05)", "0x05"),
        ("Yellow (0x03)", "0x03"),
    ])
    def test_illum_translation(self, sel, expected_illum):
        host = _Host()
        host._led_color_dropdown.currentText.return_value = sel
        host._seq_type_dropdown.currentText.return_value = "8-bit RGB"
        host._apply_led_color_live()
        proc = host._proc_i2c_live_led
        assert proc is not None
        # Args contain the expected illum
        call = proc.start.call_args
        args = call.args[1]
        assert "--illum" in args
        assert args[args.index("--illum") + 1] == expected_illum

    def test_unknown_color_early_return(self):
        host = _Host()
        host._led_color_dropdown.currentText.return_value = "Unrecognised"
        host._apply_led_color_live()
        # No proc launched
        assert host._proc_i2c_live_led is None

    def test_dropdown_raises_early_return(self):
        host = _Host()
        host._led_color_dropdown.currentText.side_effect = RuntimeError("ui dead")
        host._apply_led_color_live()
        assert host._proc_i2c_live_led is None

    @pytest.mark.parametrize("stxt,expected_seq", [
        ("8-bit RGB (0x03)", "3"),
        ("8-bit RGB", "3"),
        ("8-bit Mono", "2"),
        ("1-bit RGB", "1"),
        ("Unknown", "0"),
        ("(0x02) something", "2"),
        ("(0x01) something", "1"),
    ])
    def test_seq_type_translation(self, stxt, expected_seq):
        host = _Host()
        host._led_color_dropdown.currentText.return_value = "Red (0x02)"
        host._seq_type_dropdown.currentText.return_value = stxt
        host._apply_led_color_live()
        proc = host._proc_i2c_live_led
        args = proc.start.call_args.args[1]
        assert args[args.index("--seq-type") + 1] == expected_seq

    def test_seq_type_dropdown_raises_defaults_to_zero(self):
        host = _Host()
        host._led_color_dropdown.currentText.return_value = "Red (0x02)"
        host._seq_type_dropdown.currentText.side_effect = RuntimeError("dead")
        host._apply_led_color_live()
        args = host._proc_i2c_live_led.start.call_args.args[1]
        assert args[args.index("--seq-type") + 1] == "0"

    def test_prev_running_proc_killed_first(self):
        host = _Host()
        host._led_color_dropdown.currentText.return_value = "Red (0x02)"
        host._seq_type_dropdown.currentText.return_value = "8-bit RGB"
        prev = _make_proc_instance(state_value=2)  # Running
        host._proc_i2c_live_led = prev
        host._apply_led_color_live()
        prev.kill.assert_called_once()
        prev.waitForFinished.assert_called_with(500)
        prev.deleteLater.assert_called_once()
        # New proc launched
        assert host._proc_i2c_live_led is not prev

    def test_prev_not_running_just_deleted(self):
        host = _Host()
        host._led_color_dropdown.currentText.return_value = "Red (0x02)"
        host._seq_type_dropdown.currentText.return_value = "8-bit RGB"
        prev = _make_proc_instance(state_value=0)  # NotRunning
        host._proc_i2c_live_led = prev
        host._apply_led_color_live()
        prev.kill.assert_not_called()
        prev.deleteLater.assert_called_once()

    def test_no_validate_flag_in_args(self):
        host = _Host()
        host._led_color_dropdown.currentText.return_value = "Red (0x02)"
        host._seq_type_dropdown.currentText.return_value = "8-bit RGB"
        host._apply_led_color_live()
        args = host._proc_i2c_live_led.start.call_args.args[1]
        assert "--no-validate" in args
        assert "boot" in args

    def test_inner_spawn_exception_swallowed(self, capsys, monkeypatch):
        """Force the spawn block to raise; outer except prints + clears
        _proc_i2c_live_led."""
        host = _Host()
        host._led_color_dropdown.currentText.return_value = "Red (0x02)"
        host._seq_type_dropdown.currentText.return_value = "8-bit RGB"
        # Make Path(__file__).resolve().parents[1] raise via monkeypatching Path
        monkeypatch.setattr(_ledmod, "Path",
                            MagicMock(side_effect=RuntimeError("fs gone")))
        host._apply_led_color_live()  # no raise
        out = capsys.readouterr().out
        assert "LED live-change failed" in out
        assert host._proc_i2c_live_led is None

    def test_attach_signals_exception_swallowed(self):
        """An exception raised by _attach_proc_signals does NOT abort the
        spawn — the wider try block has its own swallow path."""
        host = _Host()
        host._led_color_dropdown.currentText.return_value = "Red (0x02)"
        host._seq_type_dropdown.currentText.return_value = "8-bit RGB"
        host._attach_proc_signals = MagicMock(
            side_effect=RuntimeError("signal wire dead"))
        host._apply_led_color_live()
        # Proc still launched
        assert host._proc_i2c_live_led is not None
        host._proc_i2c_live_led.start.assert_called_once()

    def test_prev_kill_swallow_then_deletelater_swallow(self):
        """Both prev.kill() AND prev.deleteLater() can raise — both are
        wrapped in independent try/except blocks and swallowed."""
        host = _Host()
        host._led_color_dropdown.currentText.return_value = "Red (0x02)"
        host._seq_type_dropdown.currentText.return_value = "8-bit RGB"
        prev = MagicMock()
        prev.state.return_value = 2  # Running
        prev.kill.side_effect = RuntimeError("kill failed")
        prev.deleteLater.side_effect = RuntimeError("delete failed")
        host._proc_i2c_live_led = prev
        host._apply_led_color_live()
        # New proc still launched
        assert host._proc_i2c_live_led is not prev

    def test_cleanup_callback_clears_self_field(self):
        """The _cleanup callback connected to finished/errorOccurred
        clears self._proc_i2c_live_led if it still points at the same
        proc instance."""
        host = _Host()
        host._led_color_dropdown.currentText.return_value = "Red (0x02)"
        host._seq_type_dropdown.currentText.return_value = "8-bit RGB"
        host._apply_led_color_live()
        proc = host._proc_i2c_live_led
        assert proc is not None
        # Pull out the cleanup connected to.finished — it was registered
        # via.connect(_cleanup). Call it directly.
        cb_finished = proc.finished.connect.call_args.args[0]
        cb_finished()
        assert host._proc_i2c_live_led is None
        proc.deleteLater.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# C3 — _on_proc_finished (i2c / masks / projector dispatch)
# ─────────────────────────────────────────────────────────────────────────────


class TestC3OnProcFinished:
    """Contract: route to the right QProcess slot + restore the right button
    label based on the `which` argument.

    Branches:
    - which='i2c', dmd running → "Stop Projector Trigger"
    - which='i2c', dmd not running → "Start Projector Trigger"
    - which='i2c', missing button → no crash
    - which='masks', proc set → "Send Masks"
    - which='projector', proc set → "Start Projection Engine"
    - which not in the 3 known → no-op
    - deleteLater raises on i2c/masks/projector → swallowed
    """

    def test_i2c_dmd_running(self):
        host = _Host(dmd_running=True)
        host._proc_i2c = MagicMock()
        host._on_proc_finished("i2c")
        assert host._proc_i2c is None
        host._button_send_triggers.setText.assert_called_with(
            "Stop Projector Trigger")

    def test_i2c_dmd_idle(self):
        host = _Host(dmd_running=False)
        host._proc_i2c = MagicMock()
        host._on_proc_finished("i2c")
        host._button_send_triggers.setText.assert_called_with(
            "Start Projector Trigger")

    def test_i2c_missing_button(self):
        host = _Host()
        host._proc_i2c = MagicMock()
        host._button_send_triggers = None
        host._on_proc_finished("i2c")
        assert host._proc_i2c is None

    def test_masks(self):
        host = _Host()
        host._proc_masks = MagicMock()
        host._on_proc_finished("masks")
        assert host._proc_masks is None
        host._button_send_masks.setText.assert_called_with("Send Masks")

    def test_projector(self):
        host = _Host()
        host._proc_projector = MagicMock()
        host._on_proc_finished("projector")
        assert host._proc_projector is None
        host._button_start_projector.setText.assert_called_with(
            "Start Projection Engine")

    def test_unknown_which_is_noop(self):
        host = _Host()
        host._proc_masks = MagicMock()
        host._proc_projector = MagicMock()
        # The 'else' branch enters the inner if/elif tree which only
        # matches masks/projector — anything else is a no-op
        host._on_proc_finished("nonsense")
        # State unchanged
        assert host._proc_masks is not None
        assert host._proc_projector is not None

    def test_i2c_deletelater_raises(self):
        host = _Host()
        host._proc_i2c = MagicMock()
        host._proc_i2c.deleteLater.side_effect = RuntimeError("dead")
        host._on_proc_finished("i2c")  # no raise
        assert host._proc_i2c is None


# ─────────────────────────────────────────────────────────────────────────────
# C4 — _terminate_external_processes
# ─────────────────────────────────────────────────────────────────────────────


class TestC4TerminateExternalProcesses:
    """Contract: kill each of (i2c, masks, projector) helper QProcesses;
    waitForFinished each with bounded timeout; restore button labels;
    swallow every exception.

    Branches:
    - all 3 procs None → no kill calls; labels restored
    - i2c.kill raises → swallowed, _proc_i2c=None still
    - masks.waitForFinished raises → swallowed
    - projector present + dmd_running → triggers button "Stop Projector
      Trigger"; not running → "Start Projector Trigger"
    - button missing → swallow inner except
    """

    def test_all_none_restores_labels(self):
        host = _Host()
        host._terminate_external_processes()
        assert host._proc_i2c is None
        assert host._proc_masks is None
        assert host._proc_projector is None
        host._button_send_masks.setText.assert_called_with("Send Masks")
        host._button_start_projector.setText.assert_called_with(
            "Start Projection Engine")
        host._button_send_triggers.setText.assert_called_with(
            "Start Projector Trigger")

    def test_dmd_running_label(self):
        host = _Host(dmd_running=True)
        host._terminate_external_processes()
        host._button_send_triggers.setText.assert_called_with(
            "Stop Projector Trigger")

    def test_all_three_killed(self):
        host = _Host()
        p_i2c, p_masks, p_proj = (MagicMock() for _ in range(3))
        host._proc_i2c = p_i2c
        host._proc_masks = p_masks
        host._proc_projector = p_proj
        host._terminate_external_processes()
        p_i2c.kill.assert_called_once()
        p_masks.kill.assert_called_once()
        p_proj.kill.assert_called_once()
        p_i2c.waitForFinished.assert_called_with(1000)
        p_masks.waitForFinished.assert_called_with(1000)
        p_proj.waitForFinished.assert_called_with(2000)
        assert host._proc_i2c is None
        assert host._proc_masks is None
        assert host._proc_projector is None

    def test_i2c_kill_raises_swallowed(self):
        host = _Host()
        p_i2c = MagicMock()
        p_i2c.kill.side_effect = RuntimeError("zombie")
        host._proc_i2c = p_i2c
        host._terminate_external_processes()
        assert host._proc_i2c is None

    def test_masks_wait_raises_swallowed(self):
        host = _Host()
        p_masks = MagicMock()
        p_masks.waitForFinished.side_effect = RuntimeError("timeout")
        host._proc_masks = p_masks
        host._terminate_external_processes()
        assert host._proc_masks is None

    def test_buttons_missing_swallowed(self):
        host = _Host()
        host._button_send_triggers = None
        host._button_send_masks = None
        host._button_start_projector = None
        host._terminate_external_processes()  # no raise

    def test_button_settext_raises_swallowed(self):
        """Each finally-block's setText() call is wrapped in its own
        try/except. Force each to raise and confirm the next finally-
        block still executes."""
        host = _Host()
        host._button_send_triggers.setText.side_effect = RuntimeError("dead")
        host._button_send_masks.setText.side_effect = RuntimeError("dead")
        host._button_start_projector.setText.side_effect = RuntimeError("dead")
        host._terminate_external_processes()  # no raise
        # All 3 setText were attempted
        host._button_send_triggers.setText.assert_called()
        host._button_send_masks.setText.assert_called()
        host._button_start_projector.setText.assert_called()

    def test_proc_kill_independent_of_neighbors(self):
        """If i2c.kill raises, masks and projector must still be killed
        (each wrapped in its own try/finally)."""
        host = _Host()
        host._proc_i2c = MagicMock()
        host._proc_i2c.kill.side_effect = RuntimeError("zombie i2c")
        host._proc_masks = MagicMock()
        host._proc_projector = MagicMock()
        host._terminate_external_processes()
        host._proc_masks  # field is now None
        # After call: each was set to None
        assert host._proc_i2c is None
        assert host._proc_masks is None
        assert host._proc_projector is None


# ─────────────────────────────────────────────────────────────────────────────
# Property tests (§1.1 universal floor — ≥2)
# ─────────────────────────────────────────────────────────────────────────────


class TestPropertyOnProcFinishedButtonCodomain:
    """For any `which` and dmd_running state, the button labels resulting
    from _on_proc_finished are drawn from a fixed codomain:

        triggers ∈ {"Stop Projector Trigger", "Start Projector Trigger"}
        masks    ∈ {"Send Masks"}
        proj     ∈ {"Start Projection Engine"}
    """

    @given(
        which=st.sampled_from(["i2c", "masks", "projector", "noop", ""]),
        dmd=st.booleans(),
    )
    @settings(max_examples=20, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_button_text_codomain(self, which, dmd):
        host = _Host(dmd_running=dmd)
        host._proc_i2c = MagicMock()
        host._proc_masks = MagicMock()
        host._proc_projector = MagicMock()
        host._on_proc_finished(which)
        # Collect any string set on a button
        for btn, allowed in (
                (host._button_send_triggers,
                 {"Start Projector Trigger", "Stop Projector Trigger"}),
                (host._button_send_masks, {"Send Masks"}),
                (host._button_start_projector,
                 {"Start Projection Engine"}),
        ):
            for call in btn.setText.call_args_list:
                assert call.args[0] in allowed, \
                    f"Unexpected text {call.args[0]} for {btn}"


class TestPropertyApplyLedColorIllumCodomain:
    """The illum string passed to the subprocess is always one of exactly
    six literal bitmasks, regardless of the rest of the dropdown text."""

    KNOWN_ILLUMS = {"0x01", "0x02", "0x03", "0x04", "0x05", "0x07"}

    @given(sel=st.sampled_from([
        "UV (0x01)", "Red (0x02)", "Yellow (0x03)", "Green (0x04)",
        "Magenta (0x05)", "White (0x07)",
        "0x01 (prefix)", "0x07 last",
    ]))
    @settings(max_examples=10, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_illum_in_known_set(self, sel):
        host = _Host()
        host._led_color_dropdown.currentText.return_value = sel
        host._seq_type_dropdown.currentText.return_value = "8-bit RGB"
        host._apply_led_color_live()
        proc = host._proc_i2c_live_led
        assert proc is not None
        args = proc.start.call_args.args[1]
        assert args[args.index("--illum") + 1] in self.KNOWN_ILLUMS


# ─────────────────────────────────────────────────────────────────────────────
# Visual regression — widget-state snapshot substitute
# ─────────────────────────────────────────────────────────────────────────────


class TestVisualRegressionSubstitute:
    """LEDAndProcessMixin paints no pixels. Per spec §15 substitution rule,
    pin the EXACT setText() argument strings for each terminal state.

    Recovery criterion: at Phase A.5 hardware co-walk, user verifies the
    exact-string labels appear after each action.
    """

    def test_proc_finished_i2c_dmd_running_snapshot(self):
        host = _Host(dmd_running=True)
        host._proc_i2c = MagicMock()
        host._on_proc_finished("i2c")
        snapshot = [c.args for c in
                    host._button_send_triggers.setText.call_args_list]
        assert snapshot == [("Stop Projector Trigger",)]

    def test_proc_finished_i2c_dmd_idle_snapshot(self):
        host = _Host(dmd_running=False)
        host._proc_i2c = MagicMock()
        host._on_proc_finished("i2c")
        snapshot = [c.args for c in
                    host._button_send_triggers.setText.call_args_list]
        assert snapshot == [("Start Projector Trigger",)]

    def test_terminate_external_processes_snapshot(self):
        host = _Host(dmd_running=False)
        host._terminate_external_processes()
        # Exact widget-mutation sequence the operator will see at close-time
        triggers = [c.args[0] for c in
                    host._button_send_triggers.setText.call_args_list]
        masks = [c.args[0] for c in
                 host._button_send_masks.setText.call_args_list]
        proj = [c.args[0] for c in
                host._button_start_projector.setText.call_args_list]
        assert triggers == ["Start Projector Trigger"]
        assert masks == ["Send Masks"]
        assert proj == ["Start Projection Engine"]


# ─────────────────────────────────────────────────────────────────────────────
# Integration — mixin surface
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegrationMixinSurface:
    METHODS = (
        "_on_led_color_changed_live",
        "_apply_led_color_live",
        "_on_proc_finished",
        "_terminate_external_processes",
    )

    def test_all_4_methods_on_subclass(self):
        host = _Host()
        for name in self.METHODS:
            assert callable(getattr(host, name, None)), f"Missing: {name}"

    def test_methods_defined_on_mixin(self):
        for name in self.METHODS:
            assert name in LEDAndProcessMixin.__dict__

    def test_mixin_has_no_init(self):
        assert "__init__" not in LEDAndProcessMixin.__dict__

    def test_interface_inherits_mixin(self):
        import qt_interface
        assert LEDAndProcessMixin in qt_interface.Interface.__mro__
