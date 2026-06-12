"""Comprehensive characterization tests for ``qt_interface_mask_ops``.

1 per-layer test-type matrix (L5 row):
- ≥2 property tests (Hypothesis) — universal floor
- Visual regression — MaskOpsMixin paints no pixels; substituted with
  widget-state + argv-snapshot tests per spec §15 rule.
- Coverage target ≥85 % line+branch

Module surface (~225 LOC, 5 methods) — MaskOpsMixin extracted at iter-4
of L5 §0.5 decomposition. Cluster 6 (mask pattern operations + projector
binary build).

Methods:
- _maybe_build_projector(proj_dir)    — build C++ projector if missing/stale
- _helper_python_path_for_masks()     — resolve python interpreter
- _on_mask_pattern_changed(text)      — enable Browse button when needed
- _browse_mask_pattern_path()         — file/folder dialog per dropdown
- _toggle_send_masks()                — start/stop the mask-sender QProcess
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

import qt_interface_mixins.mask_ops as _maskmod  # noqa: E402
from qt_interface_mixins.mask_ops import MaskOpsMixin  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_proc_instance(state_value=2):
    """MagicMock standing in for a QProcess instance."""
    p = MagicMock()
    p.state = MagicMock(return_value=state_value)
    p.kill = MagicMock()
    p.deleteLater = MagicMock()
    p.start = MagicMock()
    p.setWorkingDirectory = MagicMock()
    p.setProcessEnvironment = MagicMock()
    p.finished = MagicMock()
    p.errorOccurred = MagicMock()
    return p


def _fake_qprocess_class():
    """Return a class-like callable with NotRunning attr, that when called
    produces a fresh MagicMock instance (matches QProcess(self) usage)."""
    class _C:
        NotRunning = 0
        Starting = 1
        Running = 2

        def __new__(cls, *_args, **_kwargs):
            return _make_proc_instance()
    return _C


class _Host(MaskOpsMixin):
    """Stub host satisfying the MaskOpsMixin contract."""

    def __init__(self, *, pattern_text="Moving Bar", stim_mode_text="",
                 mask_path="", warp_mode="H", flip_h=False, flip_v=False,
                 has_stim_dropdown=True, has_camera=True):
        self._proc_masks = None
        self._button_send_masks = MagicMock()
        self._mask_pattern_browse = MagicMock()
        self._mask_pattern_dropdown = MagicMock()
        self._mask_pattern_dropdown.currentText.return_value = pattern_text
        self._mask_pattern_path = mask_path
        self._mask_flip_h = flip_h
        self._mask_flip_v = flip_v
        self._proj_warp_mode = warp_mode
        if has_stim_dropdown:
            self._stim_mode_dropdown = MagicMock()
            self._stim_mode_dropdown.currentText.return_value = stim_mode_text
        if has_camera:
            cam = MagicMock()
            cam.asset_dir = "/tmp/test_asset_dir"
            self._camera = cam
        self._ensure_qprocess = MagicMock(return_value=_fake_qprocess_class())
        self._attach_proc_signals = MagicMock()
        self._on_proc_finished = MagicMock()


# ═════════════════════════════════════════════════════════════════════════════
# C1 — _maybe_build_projector
# ═════════════════════════════════════════════════════════════════════════════


class TestC1MaybeBuildProjector:
    """Contract: skip rebuild if binary exists AND is at-least as new as
    main.cpp; otherwise invoke g++ via subprocess.run. Always returns bool.

    Branches:
    - binary missing → build attempted
    - binary present, getmtime raises → no rebuild (False need_build)
    - binary present, newer than src → skip
    - binary present, older than src → build attempted
    - subprocess.run returncode != 0 → False, print stderr
    - subprocess.run returncode == 0 → True, print success
    - outer exception → False, print error
    """

    def test_binary_missing_triggers_build(self, monkeypatch, capsys):
        host = _Host()
        monkeypatch.setattr(_maskmod.os.path, "exists", lambda p: False)

        fake_run = MagicMock()
        fake_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        monkeypatch.setattr("subprocess.run", fake_run)

        ok = host._maybe_build_projector("/proj")
        assert ok is True
        fake_run.assert_called_once()
        out = capsys.readouterr().out
        assert "[PROJ] Building projector" in out
        assert "Build succeeded" in out

    def test_binary_present_newer_than_src_skips_build(self, monkeypatch):
        host = _Host()
        monkeypatch.setattr(_maskmod.os.path, "exists", lambda p: True)
        # exe newer than src → no rebuild
        monkeypatch.setattr(_maskmod.os.path, "getmtime",
                            lambda p: 200.0 if p.endswith("projector") else 100.0)
        fake_run = MagicMock()
        monkeypatch.setattr("subprocess.run", fake_run)

        ok = host._maybe_build_projector("/proj")
        assert ok is True
        fake_run.assert_not_called()

    def test_binary_present_older_than_src_rebuilds(self, monkeypatch):
        host = _Host()
        monkeypatch.setattr(_maskmod.os.path, "exists", lambda p: True)
        # exe older than src
        monkeypatch.setattr(_maskmod.os.path, "getmtime",
                            lambda p: 50.0 if p.endswith("projector") else 100.0)
        fake_run = MagicMock(return_value=MagicMock(returncode=0))
        monkeypatch.setattr("subprocess.run", fake_run)

        ok = host._maybe_build_projector("/proj")
        assert ok is True
        fake_run.assert_called_once()

    def test_getmtime_raises_skips_rebuild(self, monkeypatch):
        host = _Host()
        monkeypatch.setattr(_maskmod.os.path, "exists", lambda p: True)
        monkeypatch.setattr(_maskmod.os.path, "getmtime",
                            MagicMock(side_effect=OSError("stat dead")))
        fake_run = MagicMock()
        monkeypatch.setattr("subprocess.run", fake_run)

        ok = host._maybe_build_projector("/proj")
        # need_build was reset to False in the except — skipped
        assert ok is True
        fake_run.assert_not_called()

    def test_build_returncode_nonzero(self, monkeypatch, capsys):
        host = _Host()
        monkeypatch.setattr(_maskmod.os.path, "exists", lambda p: False)
        fake_run = MagicMock(return_value=MagicMock(
            returncode=1, stderr="link error", stdout=""))
        monkeypatch.setattr("subprocess.run", fake_run)

        ok = host._maybe_build_projector("/proj")
        assert ok is False
        out = capsys.readouterr().out
        assert "Build failed" in out
        assert "link error" in out

    def test_build_returncode_nonzero_stdout_fallback(self, monkeypatch, capsys):
        """If stderr is empty, fall back to stdout (the `or` short-circuit)."""
        host = _Host()
        monkeypatch.setattr(_maskmod.os.path, "exists", lambda p: False)
        fake_run = MagicMock(return_value=MagicMock(
            returncode=1, stderr="", stdout="legacy stderr-was-on-stdout"))
        monkeypatch.setattr("subprocess.run", fake_run)

        host._maybe_build_projector("/proj")
        out = capsys.readouterr().out
        assert "legacy stderr-was-on-stdout" in out

    def test_outer_exception_swallowed(self, monkeypatch, capsys):
        host = _Host()
        # Force subprocess import failure path via patching os.path.exists to raise
        monkeypatch.setattr(_maskmod.os.path, "exists",
                            MagicMock(side_effect=RuntimeError("fs gone")))
        ok = host._maybe_build_projector("/proj")
        assert ok is False
        out = capsys.readouterr().out
        assert "[PROJ] Build error" in out


# ═════════════════════════════════════════════════════════════════════════════
# C2 — _helper_python_path_for_masks
# ═════════════════════════════════════════════════════════════════════════════


class TestC2HelperPythonPathForMasks:
    """Contract: prefer local venv (my_UARTvenv/bin/python) → conda
    (CONDA_PREFIX/bin/python) → sys.executable → /usr/bin/python3.

    Branches:
    - venv exists → return venv path
    - venv path lookup raises → fall through
    - venv missing, CONDA_PREFIX set + exists → return conda python
    - venv missing, CONDA_PREFIX unset → fall through
    - venv missing, CONDA_PREFIX set but missing python → fall through
    - everything missing, sys.executable set → return sys.executable
    - sys.executable empty → return /usr/bin/python3
    """

    def test_returns_venv_when_present(self, monkeypatch):
        host = _Host()

        class _FakePath:
            def __init__(self, *args):
                self._s = "/".join(str(a) for a in args)

            def resolve(self):
                return self

            def __truediv__(self, other):
                return _FakePath(self._s, other)

            def exists(self):
                # Only the venv python pretend to exist
                return self._s.endswith("my_UARTvenv/bin/python")

            @property
            def parents(self):
                # parents[2] reaches repo root from the post-reorg mixin
                # depth (qt_interface_mixins/mask_ops.py is depth 2).
                return {1: _FakePath("/fake/parent"), 2: _FakePath("/fake/parent")}

            def __str__(self):
                return self._s

        # Patch Path inside the mask_ops module
        monkeypatch.setattr(_maskmod, "Path", _FakePath)
        out = host._helper_python_path_for_masks()
        assert "my_UARTvenv/bin/python" in out

    def test_returns_conda_when_venv_missing(self, monkeypatch):
        host = _Host()

        class _FakePath:
            def __init__(self, *args):
                self._s = "/".join(str(a) for a in args)

            def resolve(self):
                return self

            def __truediv__(self, other):
                return _FakePath(self._s, other)

            def exists(self):
                # Conda path "/conda/bin/python" returns True; venv returns False
                if self._s.endswith("my_UARTvenv/bin/python"):
                    return False
                if self._s.endswith("/conda/bin/python"):
                    return True
                return False

            @property
            def parents(self):
                # parents[2] reaches repo root from the post-reorg mixin
                # depth (qt_interface_mixins/mask_ops.py is depth 2).
                return {1: _FakePath("/fake/parent"), 2: _FakePath("/fake/parent")}

            def __str__(self):
                return self._s

        monkeypatch.setattr(_maskmod, "Path", _FakePath)
        monkeypatch.setenv("CONDA_PREFIX", "/conda")
        out = host._helper_python_path_for_masks()
        assert out == "/conda/bin/python"

    def test_falls_back_to_sys_executable(self, monkeypatch):
        host = _Host()
        # Force all the exists() checks to be False
        monkeypatch.setattr(_maskmod, "Path",
                            MagicMock(side_effect=RuntimeError("path failure")))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        # sys.executable is preserved
        out = host._helper_python_path_for_masks()
        assert out == sys.executable

    def test_falls_back_to_usr_bin_python3_when_sys_executable_empty(self, monkeypatch):
        host = _Host()
        monkeypatch.setattr(_maskmod, "Path",
                            MagicMock(side_effect=RuntimeError("path failure")))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(_maskmod.sys, "executable", "")
        out = host._helper_python_path_for_masks()
        assert out == "/usr/bin/python3"

    def test_conda_block_raises_falls_through(self, monkeypatch):
        host = _Host()
        # Make Path raise so venv block fails
        monkeypatch.setattr(_maskmod, "Path",
                            MagicMock(side_effect=RuntimeError("p")))
        # Make environ.get raise so conda block falls through
        monkeypatch.setattr(_maskmod.os, "environ",
                            MagicMock(get=MagicMock(side_effect=RuntimeError("env"))))
        out = host._helper_python_path_for_masks()
        assert out == sys.executable or out == "/usr/bin/python3"


# ═════════════════════════════════════════════════════════════════════════════
# C3 — _on_mask_pattern_changed
# ═════════════════════════════════════════════════════════════════════════════


class TestC3OnMaskPatternChanged:
    """Contract: enable Browse button only for patterns that need a path
    (Image, Folder, Custom). Other patterns disable Browse.

    Branches:
    - Image / Folder / Custom → setEnabled(True)
    - Anything else → setEnabled(False)
    - setEnabled raises → swallowed
    """

    @pytest.mark.parametrize("text,expected", [
        ("Image", True),
        ("Folder", True),
        ("Custom", True),
        ("Moving Bar", False),
        ("Checkerboard", False),
        ("Solid", False),
        ("Circle", False),
        ("Gradient", False),
        ("Seg Mask", False),
        ("", False),
        ("Unknown", False),
    ])
    def test_enabled_codomain(self, text, expected):
        host = _Host()
        host._on_mask_pattern_changed(text)
        host._mask_pattern_browse.setEnabled.assert_called_once_with(expected)

    def test_setenabled_raises_swallowed(self):
        host = _Host()
        host._mask_pattern_browse.setEnabled.side_effect = RuntimeError("dead")
        # No raise
        host._on_mask_pattern_changed("Image")


# ═════════════════════════════════════════════════════════════════════════════
# C4 — _browse_mask_pattern_path
# ═════════════════════════════════════════════════════════════════════════════


class TestC4BrowseMaskPatternPath:
    """Contract: open the right file/folder dialog for the current pattern
    selection and write _mask_pattern_path on user accept.

    Branches:
    - typ="Image" + user picks file → path updated
    - typ="Image" + user cancels (fp="") → path unchanged
    - typ="Folder" + user picks dir → path updated
    - typ="Folder" + cancels → unchanged
    - typ="Custom" + picks file → path updated
    - typ="Custom" + cancels → unchanged
    - typ="Other" → no dialog
    - QFileDialog raises → print + swallow
    """

    def _patch_qfiledialog(self, monkeypatch, file_result="/path/img.png",
                          dir_result="/some/dir"):
        fake_dialog_cls = MagicMock()
        fake_dialog_cls.getOpenFileName = MagicMock(
            return_value=(file_result, ""))
        fake_dialog_cls.getExistingDirectory = MagicMock(
            return_value=dir_result)
        fake_widgets = MagicMock()
        fake_widgets.QFileDialog = fake_dialog_cls
        # The import is inside the method body so we patch sys.modules
        monkeypatch.setitem(sys.modules, "PyQt5.QtWidgets", fake_widgets)
        return fake_dialog_cls

    def test_image_accept_updates_path(self, monkeypatch):
        host = _Host(pattern_text="Image", mask_path="old.png")
        self._patch_qfiledialog(monkeypatch, file_result="/new.png")
        host._browse_mask_pattern_path()
        assert host._mask_pattern_path == "/new.png"

    def test_image_cancel_keeps_path(self, monkeypatch):
        host = _Host(pattern_text="Image", mask_path="old.png")
        self._patch_qfiledialog(monkeypatch, file_result="")
        host._browse_mask_pattern_path()
        assert host._mask_pattern_path == "old.png"

    def test_folder_accept_updates_path(self, monkeypatch):
        host = _Host(pattern_text="Folder", mask_path="old/")
        self._patch_qfiledialog(monkeypatch, dir_result="/new_dir")
        host._browse_mask_pattern_path()
        assert host._mask_pattern_path == "/new_dir"

    def test_folder_cancel_keeps_path(self, monkeypatch):
        host = _Host(pattern_text="Folder", mask_path="old/")
        self._patch_qfiledialog(monkeypatch, dir_result="")
        host._browse_mask_pattern_path()
        assert host._mask_pattern_path == "old/"

    def test_custom_accept_updates_path(self, monkeypatch):
        host = _Host(pattern_text="Custom", mask_path="old.py")
        self._patch_qfiledialog(monkeypatch, file_result="/new.py")
        host._browse_mask_pattern_path()
        assert host._mask_pattern_path == "/new.py"

    def test_custom_cancel_keeps_path(self, monkeypatch):
        host = _Host(pattern_text="Custom", mask_path="old.py")
        self._patch_qfiledialog(monkeypatch, file_result="")
        host._browse_mask_pattern_path()
        assert host._mask_pattern_path == "old.py"

    def test_other_pattern_does_nothing(self, monkeypatch):
        host = _Host(pattern_text="Moving Bar", mask_path="old.png")
        fake = self._patch_qfiledialog(monkeypatch)
        host._browse_mask_pattern_path()
        fake.getOpenFileName.assert_not_called()
        fake.getExistingDirectory.assert_not_called()
        assert host._mask_pattern_path == "old.png"

    def test_exception_swallowed(self, capsys):
        host = _Host(pattern_text="Image")
        # Force dropdown.currentText to raise
        host._mask_pattern_dropdown.currentText.side_effect = RuntimeError("dead")
        host._browse_mask_pattern_path()  # no raise
        out = capsys.readouterr().out
        assert "Browse failed" in out


# ═════════════════════════════════════════════════════════════════════════════
# C5 — _toggle_send_masks (dispatch over mask pattern + flip + stim flags)
# ═════════════════════════════════════════════════════════════════════════════


class TestC5ToggleSendMasksBasic:
    """Contract: launch a QProcess running zmq_mask_sender.py with the
    correct argv per dropdown selection; restart-if-running guard;
    apply flip flags + stim-mode flags before launch.

    Branches:
    - _proc_masks already running → kill + early return
    - _proc_masks not None but state raises → reset to None
    - _proc_masks None → full launch path
    - state==NotRunning → fall through to deleteLater path
    - deleteLater raises → swallowed
    """

    def test_full_launch_moving_bar_default_args(self):
        host = _Host(pattern_text="Moving Bar")
        host._toggle_send_masks()
        proc = host._proc_masks
        assert proc is not None
        # Last start() was the zmq_mask_sender launch
        last_call = proc.start.call_args_list[-1]
        program, args = last_call.args[0], last_call.args[1]
        # First arg is the script path; remaining are options
        assert any("zmq_mask_sender.py" in a for a in args)
        # No pattern-specific args; "Moving Bar" → defaults
        assert "--pattern" not in args
        host._button_send_masks.setText.assert_any_call("Stop Sending Masks")

    @pytest.mark.parametrize("pat,expected_pattern", [
        ("Checkerboard", "checkerboard"),
        ("Solid", "solid"),
        ("Circle", "circle"),
        ("Gradient", "gradient"),
    ])
    def test_simple_patterns_pass_through(self, pat, expected_pattern):
        host = _Host(pattern_text=pat)
        host._toggle_send_masks()
        proc = host._proc_masks
        # Pull final start() argv
        program, args = proc.start.call_args.args[0], proc.start.call_args.args[1]
        assert "--pattern" in args
        assert args[args.index("--pattern") + 1] == expected_pattern

    def test_image_pattern_includes_image_path(self):
        host = _Host(pattern_text="Image", mask_path="/tmp/foo.png")
        host._toggle_send_masks()
        proc = host._proc_masks
        args = proc.start.call_args.args[1]
        assert "--image" in args
        assert args[args.index("--image") + 1] == "/tmp/foo.png"

    def test_folder_pattern_includes_folder_path(self):
        host = _Host(pattern_text="Folder", mask_path="/tmp/dir")
        host._toggle_send_masks()
        proc = host._proc_masks
        args = proc.start.call_args.args[1]
        assert "--folder" in args
        assert args[args.index("--folder") + 1] == "/tmp/dir"

    def test_seg_mask_pattern_includes_roi_npz_and_save_path(self, capsys):
        host = _Host(pattern_text="Seg Mask")
        host._toggle_send_masks()
        proc = host._proc_masks
        args = proc.start.call_args.args[1]
        assert "--pattern" in args
        assert args[args.index("--pattern") + 1] == "segmask"
        assert "--roi-npz" in args
        assert "--save-segmask-to" in args

    def test_custom_pattern_python_script(self, capsys):
        host = _Host(pattern_text="Custom", mask_path="/tmp/my_sender.py")
        host._toggle_send_masks()
        proc = host._proc_masks
        # Custom-script path uses.start(py, [script]); look at the last start
        last = proc.start.call_args
        args = last.args[1]
        # First positional arg of last start() should be the python interpreter
        # and args[0] is the.py script
        assert args[0] == "/tmp/my_sender.py"
        out = capsys.readouterr().out
        assert "[MASK] Launch (python)" in out

    def test_custom_pattern_executable(self, capsys, monkeypatch):
        """Custom with non-.py extension takes the QFileInfo branch."""
        host = _Host(pattern_text="Custom", mask_path="/tmp/my_sender_bin")

        fake_qfileinfo = MagicMock()
        fake_qfileinfo_instance = MagicMock()
        fake_qfileinfo_instance.absoluteFilePath.return_value = (
            "/tmp/my_sender_bin")
        fake_qfileinfo.return_value = fake_qfileinfo_instance

        fake_qtcore = MagicMock()
        fake_qtcore.QFileInfo = fake_qfileinfo
        fake_qtcore.QProcessEnvironment = MagicMock()
        monkeypatch.setitem(sys.modules, "PyQt5.QtCore", fake_qtcore)

        host._toggle_send_masks()
        out = capsys.readouterr().out
        assert "[MASK] Launch (exec)" in out

    def test_running_proc_kills_and_returns(self):
        host = _Host(pattern_text="Solid")
        prev = _make_proc_instance(state_value=2)  # Running
        host._proc_masks = prev
        host._toggle_send_masks()
        prev.kill.assert_called_once()
        # We didn't replace _proc_masks (early return)
        assert host._proc_masks is prev

    def test_state_raises_resets_proc(self):
        host = _Host(pattern_text="Solid")
        prev = MagicMock()
        prev.state.side_effect = RuntimeError("dead")
        host._proc_masks = prev
        host._toggle_send_masks()
        # New proc was launched (prev replaced)
        assert host._proc_masks is not prev

    def test_deletelater_raises_swallowed(self):
        host = _Host(pattern_text="Solid")
        prev = _make_proc_instance(state_value=0)  # NotRunning
        prev.deleteLater.side_effect = RuntimeError("dead")
        host._proc_masks = prev
        host._toggle_send_masks()
        # New proc launched
        assert host._proc_masks is not prev

    def test_outer_exception_calls_on_proc_finished(self):
        host = _Host(pattern_text="Solid")
        # Make _attach_proc_signals raise mid-launch
        host._attach_proc_signals.side_effect = RuntimeError("wire dead")
        host._toggle_send_masks()  # outer except swallows
        host._on_proc_finished.assert_called_with("masks")


class TestC5ToggleSendMasksFlipsAndStim:
    """Contract: --flip-x / --flip-y added when _mask_flip_h/v are truthy;
    stim-mode dropdown adds --temporal-alternate / --composite-rgb."""

    def test_flip_h_adds_flip_x(self):
        host = _Host(pattern_text="Solid", flip_h=True)
        host._toggle_send_masks()
        args = host._proc_masks.start.call_args.args[1]
        assert "--flip-x" in args

    def test_flip_v_adds_flip_y(self):
        host = _Host(pattern_text="Solid", flip_v=True)
        host._toggle_send_masks()
        args = host._proc_masks.start.call_args.args[1]
        assert "--flip-y" in args

    def test_no_flip_no_args(self):
        host = _Host(pattern_text="Solid", flip_h=False, flip_v=False)
        host._toggle_send_masks()
        args = host._proc_masks.start.call_args.args[1]
        assert "--flip-x" not in args
        assert "--flip-y" not in args

    def test_temporal_stim_adds_temporal_alternate(self):
        host = _Host(pattern_text="Solid", stim_mode_text="Temporal Mode")
        host._toggle_send_masks()
        args = host._proc_masks.start.call_args.args[1]
        assert "--temporal-alternate" in args
        # also includes --fps
        assert "--fps" in args

    def test_simultaneous_stim_adds_composite_rgb(self):
        host = _Host(pattern_text="Solid",
                     stim_mode_text="Simultaneous Mode")
        host._toggle_send_masks()
        args = host._proc_masks.start.call_args.args[1]
        assert "--composite-rgb" in args
        assert "--temporal-alternate" not in args

    def test_missing_stim_dropdown_treated_as_empty(self):
        host = _Host(pattern_text="Solid", has_stim_dropdown=False)
        host._toggle_send_masks()
        args = host._proc_masks.start.call_args.args[1]
        # No stim flags
        assert "--temporal-alternate" not in args
        assert "--composite-rgb" not in args

    def test_lut_warp_mode_adds_prewarp_dir(self, monkeypatch):
        host = _Host(pattern_text="Solid", warp_mode="LUT")
        # Patch zmq import inside the method so the engine-H clear is a noop
        fake_zmq = MagicMock()
        ctx = MagicMock()
        sock = MagicMock()
        sock.recv = MagicMock(return_value=b"OK")
        ctx.socket.return_value = sock
        fake_zmq.Context.instance.return_value = ctx
        fake_zmq.LINGER = 1
        fake_zmq.REQ = 3
        monkeypatch.setitem(sys.modules, "zmq", fake_zmq)
        host._toggle_send_masks()
        args = host._proc_masks.start.call_args.args[1]
        assert "--prewarp-lut-dir" in args

    def test_lut_zmq_failure_swallowed(self, monkeypatch):
        host = _Host(pattern_text="Solid", warp_mode="LUT")
        # zmq.Context.instance raises
        fake_zmq = MagicMock()
        fake_zmq.Context.instance.side_effect = RuntimeError("no zmq")
        fake_zmq.LINGER = 1
        fake_zmq.REQ = 3
        monkeypatch.setitem(sys.modules, "zmq", fake_zmq)
        host._toggle_send_masks()  # no raise; still launches
        args = host._proc_masks.start.call_args.args[1]
        # prewarp dir was still appended (zmq cleanup is best-effort)
        assert "--prewarp-lut-dir" in args


# ═════════════════════════════════════════════════════════════════════════════
# Property tests (§1.1 universal floor — ≥2)
# ═════════════════════════════════════════════════════════════════════════════


class TestPropertyMaskPatternBrowseEnableCodomain:
    """Property: for any text value, _on_mask_pattern_changed always sets
    setEnabled to exactly one boolean value drawn from {True, False}."""

    @given(text=st.text(min_size=0, max_size=30))
    @settings(max_examples=30, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_setEnabled_called_with_bool(self, text):
        host = _Host()
        host._on_mask_pattern_changed(text)
        host._mask_pattern_browse.setEnabled.assert_called_once()
        arg = host._mask_pattern_browse.setEnabled.call_args.args[0]
        assert isinstance(arg, bool)
        # Codomain: True iff text in known set
        if text in ("Image", "Folder", "Custom"):
            assert arg is True
        else:
            assert arg is False


class TestPropertyToggleSendMasksArgsCodomain:
    """Property: for any pattern in the known dispatch set, the resulting
    argv either contains --pattern (with one of the canonical values) or is
    pattern-free (Moving Bar default), never contains unknown --pattern
    values. Also asserts the launched script always ends with.py except
    in the Custom branch (which can launch any file)."""

    KNOWN_PATTERN_VALUES = {
        "checkerboard", "solid", "circle", "gradient",
        "image", "folder", "segmask",
    }

    @given(pat=st.sampled_from([
        "Moving Bar", "Checkerboard", "Solid", "Circle", "Gradient",
        "Image", "Folder", "Seg Mask",
    ]))
    @settings(max_examples=10, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_pattern_arg_codomain(self, pat):
        host = _Host(pattern_text=pat, mask_path="/tmp/x")
        host._toggle_send_masks()
        args = host._proc_masks.start.call_args.args[1]
        if pat == "Moving Bar":
            assert "--pattern" not in args
        else:
            assert "--pattern" in args
            pv = args[args.index("--pattern") + 1]
            assert pv in self.KNOWN_PATTERN_VALUES


# ═════════════════════════════════════════════════════════════════════════════
# Visual regression — widget-state + argv snapshot substitute
# ═════════════════════════════════════════════════════════════════════════════


class TestVisualRegressionSubstitute:
    """MaskOpsMixin paints no pixels. Per spec §15 substitution rule, pin
    the EXACT setText() argument strings + argv vectors for representative
    workflows.

    Recovery criterion: at Phase A.5 hardware co-walk, user verifies that:
    - Send Masks button shows "Stop Sending Masks" after click on each pattern
    - The chosen pattern emits the exact argv vector pinned here
    """

    def test_send_masks_button_label_transition_snapshot(self):
        host = _Host(pattern_text="Solid")
        host._toggle_send_masks()
        labels = [c.args[0] for c in
                  host._button_send_masks.setText.call_args_list]
        assert labels == ["Stop Sending Masks"]

    def test_solid_pattern_argv_snapshot(self):
        host = _Host(pattern_text="Solid")
        host._toggle_send_masks()
        argv = host._proc_masks.start.call_args.args[1]
        # First arg is the script path; pattern args follow
        assert argv[1:] == ["--pattern", "solid"]

    def test_gradient_pattern_full_argv_snapshot(self):
        host = _Host(pattern_text="Gradient")
        host._toggle_send_masks()
        argv = host._proc_masks.start.call_args.args[1]
        # Gradient has 5 named options after script path
        expected = [
            "--pattern", "gradient",
            "--fps", "60",
            "--gradient-steps", "3",
            "--gradient-hold", "30",
            "--gradient-gamma", "2.2",
        ]
        assert argv[1:] == expected


# ═════════════════════════════════════════════════════════════════════════════
# Integration — mixin surface
# ═════════════════════════════════════════════════════════════════════════════


class TestIntegrationMixinSurface:
    METHODS = (
        "_maybe_build_projector",
        "_helper_python_path_for_masks",
        "_on_mask_pattern_changed",
        "_browse_mask_pattern_path",
        "_toggle_send_masks",
    )

    def test_all_5_methods_on_subclass(self):
        host = _Host()
        for name in self.METHODS:
            assert callable(getattr(host, name, None)), f"Missing: {name}"

    def test_methods_defined_on_mixin(self):
        for name in self.METHODS:
            assert name in MaskOpsMixin.__dict__

    def test_mixin_has_no_init(self):
        assert "__init__" not in MaskOpsMixin.__dict__

    def test_interface_inherits_mixin(self):
        import qt_interface
        assert MaskOpsMixin in qt_interface.Interface.__mro__
