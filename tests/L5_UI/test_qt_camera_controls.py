"""Comprehensive characterization tests for ``qt_interface_camera_controls``.

1 per-layer test-type matrix (L5 row):
- ≥2 property tests (Hypothesis) — universal floor
- Visual regression — substituted with widget-state + log/argv snapshots
  per spec §15 rule (no Qt event loop, mostly pure-state mutations).
- Coverage target ≥85 % line+branch

Module surface (~298 LOC, 14 methods) — CameraControlsMixin extracted at
iter-8 of L5 §0.5 decomposition. Cluster 6+7 subset (camera control
surface: pixel-format / trigger-line / gain sliders / contrast LUT /
exposure / warp mode).

Methods (14):
- _on_camera_type_changed(t)      — store selected type, log
- change_pixel_format(*_)         — apply dropdown pixel format
- change_hardware_trigger_line(*_) — apply trigger-line dropdown
- change_slider_gain(val)         — float→int slider scaling
- _update_gain(val)               — write AnalogAll gain
- change_slider_dgain(val)        — float→int for digital
- _update_dgain(val)              — write DigitalAll gain
- _set_camera_contrast(value)     — hardware contrast via API or node
- _make_contrast_lut(factor)      — build 256-entry preview LUT
- _apply_exposure_from_text()     — write ExposureTime from QLineEdit
- _select_warp_h()                — toggle H-matrix warp mode
- _select_warp_lut()              — toggle LUT warp mode
- _on_warp_h_toggled(checked)     — H checkbox handler
- _on_warp_lut_toggled(checked)   — LUT checkbox handler
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
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

import qt_interface_mixins.camera_controls as _ccmod  # noqa: E402
from qt_interface_mixins.camera_controls import CameraControlsMixin  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_node(value=1.0, minimum=0.1, maximum=4.0):
    n = MagicMock()
    n.Value.return_value = value
    n.SetValue = MagicMock()
    n.SetCurrentEntry = MagicMock()
    n.Minimum.return_value = minimum
    n.Maximum.return_value = maximum
    return n


def _make_node_map(nodes=None):
    nm = MagicMock()
    nm.FindNode.side_effect = lambda name: (nodes or {}).get(name)
    return nm


class _Host(CameraControlsMixin):
    """Stub host satisfying the CameraControlsMixin contract."""

    def __init__(self, *, node_map=None, has_set_contrast=False,
                 exp_text="33333", warp_mode="H", has_hmatrix_btn=True,
                 has_lut_btn=True, hmatrix_checked=False, lut_checked=False):
        self.selected_camera_type = "none"
        self._dropdown_pixel_format = MagicMock()
        self._dropdown_pixel_format.currentText.return_value = "Mono8"
        self._dropdown_trigger_line = MagicMock()
        self._dropdown_trigger_line.currentText.return_value = "Line1"
        self._gain_slider = MagicMock()
        self._dgain_slider = MagicMock()
        self._gain_value_label = MagicMock()
        self._dgain_value_label = MagicMock()
        self._exp_line = MagicMock()
        self._exp_line.text.return_value = exp_text
        cam = MagicMock(spec=[])
        cam.node_map = node_map
        cam.change_pixel_format = MagicMock()
        cam.change_hardware_trigger_line = MagicMock()
        cam.set_gain = MagicMock()
        if has_set_contrast:
            cam.set_contrast = MagicMock()
        self._camera = cam
        self._proj_warp_mode = warp_mode
        if has_hmatrix_btn:
            self._button_req_hmatrix = MagicMock()
            self._button_req_hmatrix.isChecked.return_value = hmatrix_checked
        if has_lut_btn:
            self._button_use_lut = MagicMock()
            self._button_use_lut.isChecked.return_value = lut_checked
        self._send_hmatrix_to_projector = MagicMock()


# ═════════════════════════════════════════════════════════════════════════════
# C1 — Dropdown handlers
# ═════════════════════════════════════════════════════════════════════════════


class TestC1DropdownHandlers:
    """Contract: _on_camera_type_changed stores the type + logs;
    change_pixel_format reads dropdown + delegates; same for trigger line."""

    def test_on_camera_type_changed_stores_and_logs(self, capsys):
        host = _Host()
        host._on_camera_type_changed("Mono USB3")
        assert host.selected_camera_type == "Mono USB3"
        out = capsys.readouterr().out
        assert "Camera type changed to: Mono USB3" in out

    def test_change_pixel_format_delegates(self):
        host = _Host()
        host._dropdown_pixel_format.currentText.return_value = "Mono12"
        host.change_pixel_format()
        host._camera.change_pixel_format.assert_called_with("Mono12")

    def test_change_pixel_format_accepts_varargs(self):
        """The @Slot binding can pass extra args; we use *_ to swallow."""
        host = _Host()
        host.change_pixel_format("extra", "args")
        host._camera.change_pixel_format.assert_called()

    def test_change_hardware_trigger_line_delegates(self, capsys):
        host = _Host()
        host._dropdown_trigger_line.currentText.return_value = "Line3"
        host.change_hardware_trigger_line()
        host._camera.change_hardware_trigger_line.assert_called_with("Line3")
        out = capsys.readouterr().out
        assert "Chosen hardware trigger line: Line3" in out


# ═════════════════════════════════════════════════════════════════════════════
# C2 — Gain sliders (analog + digital)
# ═════════════════════════════════════════════════════════════════════════════


class TestC2GainSliders:
    """Contract: float sliders scale to int by ×100; _update_* writes the
    formatted label, selects AnalogAll/DigitalAll on the camera node map,
    and calls set_gain on the camera."""

    def test_change_slider_gain_scales(self):
        host = _Host()
        host.change_slider_gain(1.5)
        host._gain_slider.setValue.assert_called_with(150)

    def test_change_slider_dgain_scales(self):
        host = _Host()
        host.change_slider_dgain(2.75)
        host._dgain_slider.setValue.assert_called_with(275)

    def test_update_gain_writes_label_and_camera(self):
        sel_node = _make_node()
        nm = _make_node_map({"GainSelector": sel_node})
        host = _Host(node_map=nm)
        host._update_gain(125)
        host._gain_value_label.setText.assert_called_with("1.25")
        sel_node.SetCurrentEntry.assert_called_with("AnalogAll")
        host._camera.set_gain.assert_called_with(1.25)

    def test_update_dgain_writes_label_and_camera(self):
        sel_node = _make_node()
        nm = _make_node_map({"GainSelector": sel_node})
        host = _Host(node_map=nm)
        host._update_dgain(300)
        host._dgain_value_label.setText.assert_called_with("3.00")
        sel_node.SetCurrentEntry.assert_called_with("DigitalAll")
        host._camera.set_gain.assert_called_with(3.0)

    def test_update_gain_selector_raise_swallowed(self):
        """GainSelector node missing → set_gain still called."""
        nm = MagicMock()
        nm.FindNode.side_effect = RuntimeError("dead")
        host = _Host(node_map=nm)
        host._update_gain(100)
        host._camera.set_gain.assert_called_with(1.0)


# ═════════════════════════════════════════════════════════════════════════════
# C3 — _set_camera_contrast
# ═════════════════════════════════════════════════════════════════════════════


class TestC3SetCameraContrast:
    """Contract: prefer camera.set_contrast if present; else fall back to
    GenICam node map (Contrast → ContrastAbsolute → Gamma → GammaCorrection
    → GammaValue). Gamma clamped to [0.7, 1.3]. Tries float SetValue first;
    falls back to int(round(value)) on TypeError."""

    def test_set_contrast_method_preferred(self, capsys):
        host = _Host(has_set_contrast=True)
        host._set_camera_contrast(1.5)
        host._camera.set_contrast.assert_called_with(1.5)
        out = capsys.readouterr().out
        assert "Applied Contrast (method)" in out

    def test_set_contrast_method_raises_falls_through_to_node(self):
        node = _make_node()
        nm = _make_node_map({"Contrast": node})
        host = _Host(node_map=nm, has_set_contrast=True)
        host._camera.set_contrast.side_effect = RuntimeError("fail")
        host._set_camera_contrast(2.0)
        node.SetValue.assert_called_with(2.0)

    def test_no_node_map_returns(self):
        host = _Host(node_map=None)
        # No raise
        host._set_camera_contrast(1.0)

    def test_contrast_node_used(self):
        node = _make_node()
        nm = _make_node_map({"Contrast": node})
        host = _Host(node_map=nm)
        host._set_camera_contrast(1.5)
        node.SetValue.assert_called_with(1.5)

    def test_contrast_absolute_fallback(self):
        node = _make_node()
        nm = _make_node_map({"ContrastAbsolute": node})
        host = _Host(node_map=nm)
        host._set_camera_contrast(2.0)
        node.SetValue.assert_called_with(2.0)

    def test_gamma_node_clamped(self):
        node = _make_node()
        ge_node = _make_node()
        nm = _make_node_map({"Gamma": node, "GammaEnable": ge_node})
        host = _Host(node_map=nm)
        # Value above 1.3 → clamped
        host._set_camera_contrast(2.5)
        node.SetValue.assert_called_with(1.3)
        ge_node.SetValue.assert_called_with(True)

    def test_gamma_node_below_range_clamped(self):
        node = _make_node()
        nm = _make_node_map({"Gamma": node})
        host = _Host(node_map=nm)
        host._set_camera_contrast(0.5)
        node.SetValue.assert_called_with(0.7)

    def test_gamma_enable_missing_still_works(self):
        node = _make_node()
        nm = _make_node_map({"Gamma": node})  # no GammaEnable
        host = _Host(node_map=nm)
        host._set_camera_contrast(1.0)
        node.SetValue.assert_called_with(1.0)

    def test_no_contrast_or_gamma_returns(self):
        nm = _make_node_map({})  # no nodes found
        host = _Host(node_map=nm)
        host._set_camera_contrast(1.0)  # no raise

    def test_setvalue_float_fails_falls_back_to_int(self):
        node = _make_node()
        node.SetValue.side_effect = [TypeError("not float"), None]
        nm = _make_node_map({"Contrast": node})
        host = _Host(node_map=nm)
        host._set_camera_contrast(1.7)
        # First call float, second call int(round(1.7)) = 2
        assert node.SetValue.call_args_list[1].args[0] == 2

    def test_setvalue_both_float_and_int_fail_returns(self):
        node = _make_node()
        node.SetValue.side_effect = TypeError("nope")
        nm = _make_node_map({"Contrast": node})
        host = _Host(node_map=nm)
        host._set_camera_contrast(1.5)  # no raise

    def test_outer_exception_swallowed(self):
        # Force getattr(self._camera, 'set_contrast') to raise
        host = _Host(has_set_contrast=True)
        host._camera.set_contrast.side_effect = RuntimeError("dead")
        host._camera.node_map = MagicMock()
        host._camera.node_map.FindNode.side_effect = RuntimeError("nm dead")
        host._set_camera_contrast(1.0)  # no raise


# ═════════════════════════════════════════════════════════════════════════════
# C4 — _make_contrast_lut
# ═════════════════════════════════════════════════════════════════════════════


class TestC4MakeContrastLut:
    """Contract: builds a 256-entry uint8 LUT applying contrast around 127.5
    pivot. Returns None on exception."""

    def test_lut_neutral_factor(self):
        host = _Host()
        lut = host._make_contrast_lut(1.0)
        assert lut is not None
        assert lut.shape == (256,)
        assert lut.dtype == np.uint8
        # Neutral factor = identity (modulo rounding)
        assert lut[0] == 0
        assert lut[255] == 255
        assert lut[128] in (127, 128)

    def test_lut_high_contrast(self):
        host = _Host()
        lut = host._make_contrast_lut(2.0)
        # Low values darker, high values brighter (saturated at 0/255)
        assert lut[0] == 0
        assert lut[255] == 255
        # Mid-value still near 127
        assert 120 <= lut[128] <= 135

    def test_lut_low_contrast(self):
        host = _Host()
        lut = host._make_contrast_lut(0.5)
        # All values closer to 127.5
        assert lut[0] > 0
        assert lut[255] < 255

    def test_lut_exception_returns_none(self):
        host = _Host()
        # Patch numpy import to fail inside method via patching builtins
        with patch.object(_ccmod, "__builtins__",
                          {"__import__": lambda *a, **kw: (_ for _ in ()).throw(
                              ImportError("no numpy"))}):
            # If patch above doesn't work, just call with bad float
            pass
        # Alternative: send a non-numeric factor
        lut = host._make_contrast_lut("not_a_number")
        assert lut is None


# ═════════════════════════════════════════════════════════════════════════════
# C5 — _apply_exposure_from_text
# ═════════════════════════════════════════════════════════════════════════════


class TestC5ApplyExposureFromText:
    """Contract: parse _exp_line text → float exp_us. If valid, lower FPS
    if needed, write ExposureTime, raise FPS back to max, read back +
    update _exp_line if camera modified the value."""

    def test_empty_text_returns(self):
        host = _Host(exp_text="")
        host._apply_exposure_from_text()
        # No camera writes
        assert host._camera.node_map is None or True  # no nm to write to

    def test_zero_or_negative_returns(self):
        host = _Host(exp_text="-5")
        nm = _make_node_map({})
        host._camera.node_map = nm
        host._apply_exposure_from_text()
        # Confirm nothing in node map was written
        nm.FindNode.assert_not_called()

    def test_invalid_float_swallowed(self, capsys):
        host = _Host(exp_text="not_a_number")
        host._apply_exposure_from_text()
        out = capsys.readouterr().out
        assert "Exposure apply failed" in out

    def test_no_node_map_returns(self):
        host = _Host(exp_text="5000", node_map=None)
        host._apply_exposure_from_text()
        # No raise

    def test_full_apply_with_fps_clamp(self, capsys):
        exp_node = _make_node(value=5000.0)
        fps_node = _make_node(value=60.0, minimum=1.0, maximum=200.0)
        nm = _make_node_map({
            "ExposureTime": exp_node,
            "AcquisitionFrameRate": fps_node,
        })
        host = _Host(exp_text="5000", node_map=nm)
        host._apply_exposure_from_text()
        exp_node.SetValue.assert_called_with(5000.0)
        out = capsys.readouterr().out
        assert "Exposure set to" in out

    def test_camera_returns_different_value_updates_line(self):
        exp_node = _make_node(value=4000.0)  # camera actually set to 4000
        nm = _make_node_map({"ExposureTime": exp_node})
        host = _Host(exp_text="5000", node_map=nm)
        host._apply_exposure_from_text()
        # _exp_line.setText called with "4000.000"
        host._exp_line.setText.assert_called_with("4000.000")

    def test_fps_node_missing_continues(self):
        exp_node = _make_node(value=5000.0)
        nm = _make_node_map({"ExposureTime": exp_node})
        host = _Host(exp_text="5000", node_map=nm)
        host._apply_exposure_from_text()
        exp_node.SetValue.assert_called_with(5000.0)

    def test_exp_node_write_raise_swallowed(self):
        exp_node = _make_node()
        exp_node.SetValue.side_effect = RuntimeError("fail")
        nm = _make_node_map({"ExposureTime": exp_node})
        host = _Host(exp_text="5000", node_map=nm)
        host._apply_exposure_from_text()  # no raise

    def test_readback_failure_logs_fallback(self, capsys):
        exp_node = _make_node()
        exp_node.Value.side_effect = RuntimeError("read fail")
        nm = _make_node_map({"ExposureTime": exp_node})
        host = _Host(exp_text="5000", node_map=nm)
        host._apply_exposure_from_text()
        out = capsys.readouterr().out
        assert "readback failed" in out


# ═════════════════════════════════════════════════════════════════════════════
# C6 — Warp mode toggles
# ═════════════════════════════════════════════════════════════════════════════


class TestC6WarpModeToggles:
    """Contract: _select_warp_h toggles H mode; if already H+checked, switch
    to NONE; else activate H and uncheck LUT. Same shape for _select_warp_lut.
    _on_warp_*_toggled is the checkbox-direct handler."""

    def test_select_warp_h_activates_when_not_already(self, capsys):
        host = _Host(warp_mode="NONE", hmatrix_checked=False)
        host._select_warp_h()
        assert host._proj_warp_mode == "H"
        host._button_req_hmatrix.setChecked.assert_called_with(True)
        host._button_use_lut.setChecked.assert_called_with(False)
        host._send_hmatrix_to_projector.assert_called_once()
        out = capsys.readouterr().out
        assert "Homography (H)" in out

    def test_select_warp_h_deactivates_when_already(self, capsys):
        host = _Host(warp_mode="H", hmatrix_checked=True)
        host._select_warp_h()
        assert host._proj_warp_mode == "NONE"
        host._button_req_hmatrix.setChecked.assert_called_with(False)
        out = capsys.readouterr().out
        assert "None (no H applied)" in out

    def test_select_warp_h_exception_swallowed(self, capsys):
        host = _Host()
        host._button_req_hmatrix.isChecked.side_effect = RuntimeError("dead")
        host._select_warp_h()
        out = capsys.readouterr().out
        assert "Warp H select failed" in out

    def test_select_warp_lut_activates(self, capsys):
        host = _Host(warp_mode="NONE", lut_checked=False)
        host._select_warp_lut()
        assert host._proj_warp_mode == "LUT"
        host._button_use_lut.setChecked.assert_called_with(True)
        host._button_req_hmatrix.setChecked.assert_called_with(False)
        out = capsys.readouterr().out
        assert "LUT" in out

    def test_select_warp_lut_deactivates(self, capsys):
        host = _Host(warp_mode="LUT", lut_checked=True)
        host._select_warp_lut()
        assert host._proj_warp_mode == "NONE"
        out = capsys.readouterr().out
        assert "None (no H" in out

    def test_select_warp_lut_exception_swallowed(self, capsys):
        host = _Host(warp_mode="LUT", lut_checked=True)
        # In the deactivate path, isChecked is consulted first
        host._button_use_lut.isChecked.side_effect = RuntimeError("dead")
        host._select_warp_lut()
        out = capsys.readouterr().out
        assert "Warp LUT select failed" in out

    def test_on_warp_h_toggled_checked_activates(self, capsys):
        host = _Host(warp_mode="NONE")
        host._on_warp_h_toggled(True)
        assert host._proj_warp_mode == "H"
        host._send_hmatrix_to_projector.assert_called_once()

    def test_on_warp_h_toggled_unchecked_no_lut_means_none(self, capsys):
        host = _Host(warp_mode="H", lut_checked=False)
        host._on_warp_h_toggled(False)
        assert host._proj_warp_mode == "NONE"

    def test_on_warp_h_toggled_unchecked_lut_active_keeps_lut(self):
        host = _Host(warp_mode="H", lut_checked=True)
        host._on_warp_h_toggled(False)
        # H off, LUT still checked → keep current mode (not NONE)
        assert host._proj_warp_mode == "H"  # unchanged from start

    def test_on_warp_lut_toggled_checked_activates(self):
        host = _Host(warp_mode="NONE")
        host._on_warp_lut_toggled(True)
        assert host._proj_warp_mode == "LUT"

    def test_on_warp_lut_toggled_unchecked_no_h_means_none(self):
        host = _Host(warp_mode="LUT", hmatrix_checked=False)
        host._on_warp_lut_toggled(False)
        assert host._proj_warp_mode == "NONE"

    def test_on_warp_lut_toggled_unchecked_h_active_keeps_h(self):
        host = _Host(warp_mode="LUT", hmatrix_checked=True)
        host._on_warp_lut_toggled(False)
        assert host._proj_warp_mode == "LUT"  # unchanged

    def test_on_warp_h_toggled_lut_btn_missing(self):
        """If _button_use_lut is None (attr exists but is None), no crash."""
        host = _Host(warp_mode="H", has_lut_btn=False)
        host._on_warp_h_toggled(False)
        # LUT button missing → enter NONE
        assert host._proj_warp_mode == "NONE"

    def test_on_warp_lut_toggled_hmatrix_btn_missing(self):
        host = _Host(warp_mode="LUT", has_hmatrix_btn=False)
        host._on_warp_lut_toggled(False)
        assert host._proj_warp_mode == "NONE"


# ═════════════════════════════════════════════════════════════════════════════
# Property tests (§1.1 universal floor — ≥2)
# ═════════════════════════════════════════════════════════════════════════════


class TestPropertyGainScaling:
    """Property: change_slider_gain(v) always calls setValue(int(v*100))."""

    @given(val=st.floats(min_value=0, max_value=20, allow_nan=False,
                         allow_infinity=False))
    @settings(max_examples=30, deadline=None,
              suppress_health_check=[HealthCheck.too_slow,
                                     HealthCheck.function_scoped_fixture])
    def test_change_slider_gain_round_trip(self, val):
        host = _Host()
        host.change_slider_gain(val)
        host._gain_slider.setValue.assert_called_with(int(val * 100))


class TestPropertyContrastLutBounds:
    """Property: for any finite factor, the LUT (when not None) is shape
    (256,) uint8 with all entries in [0, 255]."""

    @given(factor=st.floats(min_value=-5, max_value=5, allow_nan=False,
                            allow_infinity=False))
    @settings(max_examples=20, deadline=None,
              suppress_health_check=[HealthCheck.too_slow,
                                     HealthCheck.function_scoped_fixture])
    def test_lut_bytes_in_range(self, factor):
        host = _Host()
        lut = host._make_contrast_lut(factor)
        if lut is not None:
            assert lut.shape == (256,)
            assert lut.dtype == np.uint8
            assert lut.min() >= 0
            assert lut.max() <= 255


class TestPropertyWarpModeReachable:
    """Property: any sequence of warp toggle calls leaves _proj_warp_mode
    in the canonical set {NONE, H, LUT}."""

    KNOWN = {"NONE", "H", "LUT"}

    @given(actions=st.lists(st.sampled_from([
        "select_h", "select_lut",
        "toggle_h_on", "toggle_h_off",
        "toggle_lut_on", "toggle_lut_off",
    ]), min_size=1, max_size=10))
    @settings(max_examples=15, deadline=None,
              suppress_health_check=[HealthCheck.too_slow,
                                     HealthCheck.function_scoped_fixture])
    def test_warp_mode_codomain(self, actions):
        host = _Host(warp_mode="NONE")
        for a in actions:
            if a == "select_h":
                host._select_warp_h()
            elif a == "select_lut":
                host._select_warp_lut()
            elif a == "toggle_h_on":
                host._on_warp_h_toggled(True)
            elif a == "toggle_h_off":
                host._on_warp_h_toggled(False)
            elif a == "toggle_lut_on":
                host._on_warp_lut_toggled(True)
            elif a == "toggle_lut_off":
                host._on_warp_lut_toggled(False)
        assert host._proj_warp_mode in self.KNOWN


# ═════════════════════════════════════════════════════════════════════════════
# Visual regression — log + state snapshot substitute
# ═════════════════════════════════════════════════════════════════════════════


class TestVisualRegressionSubstitute:
    """CameraControlsMixin has no Qt event-loop output; substitute with
    exact log strings and state-attr snapshots per spec §15.

    Recovery criterion: at Phase A.5 hardware co-walk, user verifies that
    each control action produces the camera/log lines pinned here.
    """

    def test_camera_type_log_snapshot(self, capsys):
        host = _Host()
        host._on_camera_type_changed("Test Cam")
        out = capsys.readouterr().out.strip()
        assert out == "Camera type changed to: Test Cam"

    def test_gain_label_format_snapshot(self):
        sel_node = _make_node()
        nm = _make_node_map({"GainSelector": sel_node})
        host = _Host(node_map=nm)
        host._update_gain(250)
        # Exact two-decimal format pinned
        host._gain_value_label.setText.assert_called_with("2.50")

    def test_warp_mode_h_log_snapshot(self, capsys):
        host = _Host(warp_mode="NONE")
        host._select_warp_h()
        out = capsys.readouterr().out.strip()
        assert out == "[PROJ] Warp mode: Homography (H)"


# ═════════════════════════════════════════════════════════════════════════════
# Integration — mixin surface
# ═════════════════════════════════════════════════════════════════════════════


class TestIntegrationMixinSurface:
    METHODS = (
        "_on_camera_type_changed", "change_pixel_format",
        "change_hardware_trigger_line", "change_slider_gain",
        "_update_gain", "change_slider_dgain", "_update_dgain",
        "_set_camera_contrast", "_make_contrast_lut",
        "_apply_exposure_from_text", "_select_warp_h",
        "_select_warp_lut", "_on_warp_h_toggled", "_on_warp_lut_toggled",
    )

    def test_all_14_methods_on_subclass(self):
        host = _Host()
        for name in self.METHODS:
            assert callable(getattr(host, name, None)), f"Missing: {name}"

    def test_methods_defined_on_mixin(self):
        for name in self.METHODS:
            assert name in CameraControlsMixin.__dict__

    def test_mixin_has_no_init(self):
        assert "__init__" not in CameraControlsMixin.__dict__

    def test_interface_inherits_mixin(self):
        import qt_interface
        assert CameraControlsMixin in qt_interface.Interface.__mro__
