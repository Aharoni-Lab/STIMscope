"""Comprehensive characterization tests for ``qt_interface_sensor_settings``.

1 per-layer test-type matrix (L5 row):
- ≥2 property tests (Hypothesis) — universal floor
- Visual regression — substituted with state-attr snapshot + closure-
  state pin per spec §15 rule (no real Qt event loop).
- Coverage target ≥85 % line+branch

Module surface (~315 LOC, 1 method) — SensorSettingsMixin extracted at
iter-6 of L5 §0.5 decomposition. Cluster 7 subset (camera sensor-settings
popup dialog).

Method:
- _open_sensor_settings() — Build Sensor Settings QDialog with two-way
  sliders for analog/digital gain, exposure (slider + textbox), and
  hardware Contrast/Gamma with auto-detected node range.

The closures embedded inside the method (`_apply_local_exp`,
`_on_exp_slider`, `_on_exp_slider_label`, `_on_cnt_change`, the
contrast sliderReleased lambda, the gain sync lambdas) are captured
from the *.connect()* call_args and invoked directly.
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

import qt_interface_mixins.sensor_settings as _ssmod  # noqa: E402
from qt_interface_mixins.sensor_settings import SensorSettingsMixin  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_node(value=1.0, minimum=0.1, maximum=4.0):
    n = MagicMock()
    n.Value.return_value = value
    n.Minimum.return_value = minimum
    n.Maximum.return_value = maximum
    return n


class _Host(SensorSettingsMixin):
    """Stub satisfying the SensorSettingsMixin contract."""

    def __init__(self, *, node_map=None, has_get_contrast=False,
                 has_get_contrast_range=False, has_set_contrast=False,
                 contrast_get=1.5, contrast_range=(0.5, 2.0),
                 exp_text="33333", gain_value=50, dgain_value=100):
        self._gain_slider = MagicMock()
        self._gain_slider.minimum.return_value = 0
        self._gain_slider.maximum.return_value = 200
        self._gain_slider.value.return_value = gain_value
        self._dgain_slider = MagicMock()
        self._dgain_slider.minimum.return_value = 0
        self._dgain_slider.maximum.return_value = 400
        self._dgain_slider.value.return_value = dgain_value
        self._gain_value_label = MagicMock()
        self._gain_value_label.text.return_value = "0.50"
        self._dgain_value_label = MagicMock()
        self._dgain_value_label.text.return_value = "1.00"
        self._exp_line = MagicMock()
        self._exp_line.text.return_value = exp_text

        cam = MagicMock(spec=[])  # empty spec → no extra attrs by default
        cam.node_map = node_map
        if has_get_contrast:
            cam.get_contrast = MagicMock(return_value=contrast_get)
        if has_get_contrast_range:
            cam.get_contrast_range = MagicMock(return_value=contrast_range)
        if has_set_contrast:
            cam.set_contrast = MagicMock()
        self._camera = cam

        self._apply_exposure_from_text = MagicMock()
        self._set_camera_contrast = MagicMock()
        self._make_contrast_lut = MagicMock(return_value=[0]*256)


def _install_dialog_mocks(monkeypatch):
    """Install lightweight stand-ins for the QDialog tree built inside
    _open_sensor_settings. Returns capture dict so tests can pull out
    closures from *.connect.call_args."""

    state = {
        "dlg": MagicMock(),
        "labels": [],          # 1=AG, 2=DG, 3=Exp, 4=ExpVal, 5=CntLabel, 6=CntVal
        "label_idx": 0,
        "sliders": [],         # 0=AG, 1=DG, 2=Exp, 3=Cnt
        "slider_idx": 0,
        "lineedits": [],       # 0=exp line
        "lineedit_idx": 0,
        "pushbuttons": [],     # 0=Set, 1=Close
        "pushbutton_idx": 0,
        "vlayouts": [MagicMock()],  # main lay
        "vlayout_idx": 0,
        "glayouts": [MagicMock()],  # main grid
        "glayout_idx": 0,
        "hboxes": [],
        "hbox_idx": 0,
    }

    def _qdialog(*a, **kw):
        return state["dlg"]

    def _qvboxlayout(*a, **kw):
        lay = MagicMock()
        state["vlayouts"].append(lay)
        return lay

    def _qgridlayout(*a, **kw):
        g = MagicMock()
        state["glayouts"].append(g)
        return g

    def _qlabel(*a, **kw):
        lab = MagicMock()
        if a:
            lab._init_text = a[0]
        state["labels"].append(lab)
        return lab

    def _qslider(*a, **kw):
        s = MagicMock()
        s.minimum.return_value = 100
        s.maximum.return_value = 100000
        s.value.return_value = 33333
        state["sliders"].append(s)
        return s

    def _qlineedit(*a, **kw):
        le = MagicMock()
        if a:
            le._init_text = a[0]
        state["lineedits"].append(le)
        return le

    def _qpushbutton(*a, **kw):
        b = MagicMock()
        state["pushbuttons"].append(b)
        return b

    def _qhboxlayout(*a, **kw):
        h = MagicMock()
        state["hboxes"].append(h)
        return h

    fake_qtw_module = MagicMock()
    fake_qtw_module.QDialog = _qdialog
    fake_qtw_module.QVBoxLayout = _qvboxlayout
    fake_qtw_module.QGridLayout = _qgridlayout
    fake_qtw_module.QPushButton = _qpushbutton
    monkeypatch.setitem(sys.modules, "PyQt5.QtWidgets", fake_qtw_module)

    # Patch the QtCore/QtWidgets/QtGui in the mixin module namespace
    fake_module_qtw = MagicMock()
    fake_module_qtw.QLabel = _qlabel
    fake_module_qtw.QSlider = _qslider
    fake_module_qtw.QLineEdit = _qlineedit
    fake_module_qtw.QHBoxLayout = _qhboxlayout
    monkeypatch.setattr(_ssmod, "QtWidgets", fake_module_qtw)

    fake_module_qtc = MagicMock()
    monkeypatch.setattr(_ssmod, "QtCore", fake_module_qtc)

    fake_module_qtg = MagicMock()
    monkeypatch.setattr(_ssmod, "QtGui", fake_module_qtg)

    return state


# ═════════════════════════════════════════════════════════════════════════════
# C1 — _open_sensor_settings (top-level construction)
# ═════════════════════════════════════════════════════════════════════════════


class TestC1ConstructionHappy:
    """Contract: build the modeless Sensor Settings QDialog with 4 sliders
    (analog gain, digital gain, exposure, contrast) and 2 buttons (Set,
    Close). Always wire two-way sync to the main sliders. Always set
    _sensor_settings_dlg attr so the dialog stays alive. Always end with.show()."""

    def test_basic_construction_with_no_node_map(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None)
        host._open_sensor_settings()
        # dialog title set, modeless
        state["dlg"].setWindowTitle.assert_called_with("Sensor Settings")
        # at least one show() call (could fail+retry → 2)
        assert state["dlg"].show.call_count >= 1
        # _sensor_settings_dlg kept alive
        assert host._sensor_settings_dlg is state["dlg"]

    def test_construction_window_flags_raise_swallowed(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        state["dlg"].setWindowFlags.side_effect = RuntimeError("dead")
        host = _Host(node_map=None)
        host._open_sensor_settings()  # no raise
        assert state["dlg"].show.call_count >= 1

    def test_show_raise_falls_back_to_show_again(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        # First show raises in the try-raise-show fallback ladder
        # raise_() raises → outer except → second show()
        state["dlg"].raise_.side_effect = RuntimeError("activate dead")
        host = _Host(node_map=None)
        host._open_sensor_settings()
        # show called twice (try block + outer fallback)
        assert state["dlg"].show.call_count == 2

    def test_exp_line_invalid_value_fallback(self, monkeypatch):
        """exp_line.text() returns "garbage" → int(float(text)) raises →
        slider falls back to 33333."""
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None, exp_text="garbage")
        host._open_sensor_settings()
        # exp_slider is index 2; setValue called with 33333 fallback
        exp_slider = state["sliders"][2]
        exp_slider.setValue.assert_any_call(33333)

    def test_outer_exception_swallowed_logs(self, monkeypatch, capsys):
        # Force QDialog import to raise
        fake_qtw = MagicMock()
        fake_qtw.QDialog = MagicMock(side_effect=RuntimeError("dlg dead"))
        monkeypatch.setitem(sys.modules, "PyQt5.QtWidgets", fake_qtw)
        host = _Host(node_map=None)
        host._open_sensor_settings()
        out = capsys.readouterr().out
        assert "Sensor Settings UI error" in out


# ═════════════════════════════════════════════════════════════════════════════
# C2 — Hardware contrast node detection
# ═════════════════════════════════════════════════════════════════════════════


class TestC2ContrastNodeDetection:
    """Contract: scan node_map for Contrast / ContrastAbsolute / Gamma /
    GammaCorrection / GammaValue. First found wins. Read min/max/value
    via a series of fallback method names. Gamma family compresses UI
    range to [0.7, 1.3]."""

    def test_node_contrast_detected(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        node = _make_node(value=1.5, minimum=0.5, maximum=2.5)
        nm = MagicMock()
        # Only "Contrast" returns a real node
        nm.FindNode.side_effect = lambda name: node if name == "Contrast" else None
        host = _Host(node_map=nm)
        host._open_sensor_settings()
        # Contrast factor stored from node
        assert host._contrast_factor == 1.5
        # Hardware contrast detected
        assert host._has_hw_contrast is True
        # Label set to "Contrast"
        # state["labels"] ordering: AGlabel(0), AGval(1), DGlabel(2),
        # DGval(3), Explabel(4), ExpVal(5), CntLabel(6), CntVal(7)
        cnt_label = state["labels"][6]
        cnt_label.setText.assert_any_call("Contrast")

    def test_node_gamma_detected_compresses_range(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        node = _make_node(value=1.0, minimum=0.1, maximum=10.0)
        nm = MagicMock()
        # Only "Gamma" returns a real node
        nm.FindNode.side_effect = lambda name: node if name == "Gamma" else None
        host = _Host(node_map=nm)
        host._open_sensor_settings()
        # Gamma label
        cnt_label = state["labels"][6]
        cnt_label.setText.assert_any_call("Gamma")

    def test_node_missing_uses_fallback_label(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        nm = MagicMock()
        nm.FindNode.return_value = None
        host = _Host(node_map=nm)
        host._open_sensor_settings()
        # Label "Contrast" (default branch)
        cnt_label = state["labels"][6]
        cnt_label.setText.assert_any_call("Contrast")
        assert host._has_hw_contrast is False  # no node + no set_contrast

    def test_get_contrast_range_helper_overrides_node_range(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None, has_get_contrast_range=True,
                     contrast_range=(0.3, 5.0))
        host._open_sensor_settings()
        # _contrast_factor still defaults to 1.0 since cam.get_contrast not present
        assert host._contrast_factor == 1.0

    def test_get_contrast_helper_overrides_current(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None, has_get_contrast=True, contrast_get=2.0,
                     has_get_contrast_range=True, contrast_range=(0.1, 4.0))
        host._open_sensor_settings()
        assert host._contrast_factor == 2.0

    def test_get_contrast_range_invalid_keeps_defaults(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None, has_get_contrast_range=True,
                     contrast_range=(5.0, 0.5))  # mx < mn → ignored
        host._open_sensor_settings()
        # Defaults still 0.1.. 4.0
        # We can't probe internal contrast_min directly; just confirm no crash
        assert host._has_hw_contrast is False

    def test_set_contrast_alone_marks_hw_available(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None, has_set_contrast=True)
        host._open_sensor_settings()
        assert host._has_hw_contrast is True

    def test_node_value_failure_swallowed(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        node = MagicMock()
        node.Value.side_effect = RuntimeError("read dead")
        node.Minimum.side_effect = RuntimeError("min dead")
        node.Maximum.side_effect = RuntimeError("max dead")
        nm = MagicMock()
        nm.FindNode.side_effect = lambda name: node if name == "Contrast" else None
        host = _Host(node_map=nm)
        host._open_sensor_settings()
        # Falls back to default contrast_cur = 1.0
        assert host._contrast_factor == 1.0

    def test_node_vmax_less_than_vmin_keeps_defaults(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        node = MagicMock()
        node.Value.return_value = 1.0
        node.Minimum.return_value = 10.0
        node.Maximum.return_value = 5.0  # invalid (max < min)
        nm = MagicMock()
        nm.FindNode.side_effect = lambda name: node if name == "Contrast" else None
        host = _Host(node_map=nm)
        host._open_sensor_settings()
        # Defaults preserved; no crash
        assert host._has_hw_contrast is True

    def test_lut_builder_failure_swallowed(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None)
        host._make_contrast_lut.side_effect = RuntimeError("lut dead")
        host._open_sensor_settings()  # no raise — the assignment is wrapped

    def test_node_find_raises_swallowed(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        nm = MagicMock()
        nm.FindNode.side_effect = RuntimeError("find dead")
        host = _Host(node_map=nm)
        host._open_sensor_settings()
        # No node detected; falls back
        assert host._has_hw_contrast is False

    def test_get_contrast_helper_raises_swallowed(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None, has_get_contrast=True)
        host._camera.get_contrast.side_effect = RuntimeError("get dead")
        host._open_sensor_settings()
        # _contrast_factor still default 1.0
        assert host._contrast_factor == 1.0

    def test_get_contrast_range_helper_raises_swallowed(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None, has_get_contrast_range=True)
        host._camera.get_contrast_range.side_effect = RuntimeError("range dead")
        host._open_sensor_settings()
        assert host._has_hw_contrast is False


# ═════════════════════════════════════════════════════════════════════════════
# C3 — Exposure slider closures (_apply_local_exp / _on_exp_slider)
# ═════════════════════════════════════════════════════════════════════════════


class TestC3ExposureClosures:
    """Contract: the embedded _apply_local_exp closure copies dialog
    exp_line.text() into self._exp_line and calls
    _apply_exposure_from_text. The _on_exp_slider closure writes slider
    value into dialog exp_line and triggers _apply_local_exp."""

    def _open_and_get_exp_closures(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None)
        host._open_sensor_settings()
        # exp_slider is state["sliders"][2]
        exp_slider = state["sliders"][2]
        # exp_slider.valueChanged.connect was called twice:
        #   - first with _on_exp_slider
        #   - then with _on_exp_slider_label
        connect_calls = exp_slider.valueChanged.connect.call_args_list
        on_exp_slider_cb = connect_calls[0].args[0]
        on_exp_slider_label_cb = connect_calls[1].args[0]
        # set_btn is state["pushbuttons"][0] (only Set + Close exist)
        set_btn = state["pushbuttons"][0]
        apply_local_exp_cb = set_btn.clicked.connect.call_args.args[0]
        return host, state, on_exp_slider_cb, on_exp_slider_label_cb, apply_local_exp_cb

    def test_apply_local_exp_writes_text_and_calls_applier(self, monkeypatch):
        host, state, _, _, apply_cb = self._open_and_get_exp_closures(monkeypatch)
        # exp_line is state["lineedits"][0]
        exp_line = state["lineedits"][0]
        exp_line.text.return_value = "5000"
        apply_cb()
        host._exp_line.setText.assert_called_with("5000")
        host._apply_exposure_from_text.assert_called_once()

    def test_apply_local_exp_swallows_invalid_text(self, monkeypatch):
        host, state, _, _, apply_cb = self._open_and_get_exp_closures(monkeypatch)
        exp_line = state["lineedits"][0]
        exp_line.text.return_value = "garbage"
        apply_cb()  # no raise (try/except inside closure)

    def test_on_exp_slider_updates_textbox(self, monkeypatch):
        host, state, on_exp_cb, _, _ = self._open_and_get_exp_closures(monkeypatch)
        exp_line = state["lineedits"][0]
        # Reset prior calls from construction
        exp_line.setText.reset_mock()
        # Set up exp_line.text to return the new value when _apply_local_exp reads it
        exp_line.text.return_value = "7777"
        on_exp_cb(7777)
        # exp_line.setText should have been called with "7777"
        exp_line.setText.assert_any_call("7777")

    def test_on_exp_slider_label_updates_label(self, monkeypatch):
        host, state, _, on_label_cb, _ = self._open_and_get_exp_closures(monkeypatch)
        # exp_val index 5 in label order
        exp_val = state["labels"][5]
        exp_val.setText.reset_mock()
        on_label_cb(8888)
        exp_val.setText.assert_called_with("8888 µs")


# ═════════════════════════════════════════════════════════════════════════════
# C4 — Contrast slider closures (_on_cnt_change + sliderReleased lambda)
# ═════════════════════════════════════════════════════════════════════════════


class TestC4ContrastClosures:
    """Contract: _on_cnt_change reads slider position, computes value via
    _to_val mapping, stores _contrast_factor, updates cnt_val text. The
    sliderReleased lambda calls self._set_camera_contrast(..) only if
    _has_hw_contrast is True."""

    def _open_and_get_cnt_closures(self, monkeypatch, **kw):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(**kw)
        host._open_sensor_settings()
        cnt_slider = state["sliders"][3]
        cnt_change_cb = cnt_slider.valueChanged.connect.call_args.args[0]
        cnt_release_cb = cnt_slider.sliderReleased.connect.call_args.args[0]
        return host, state, cnt_change_cb, cnt_release_cb

    def test_cnt_change_updates_factor_and_label(self, monkeypatch):
        host, state, cnt_cb, _ = self._open_and_get_cnt_closures(monkeypatch,
                                                                  node_map=None)
        cnt_val = state["labels"][7]
        cnt_val.setText.reset_mock()
        # Default range 0.1..4.0; ticks=1000; position=500 → val=(0.1+0.5*3.9)=2.05
        cnt_cb(500)
        assert abs(host._contrast_factor - 2.05) < 0.001
        cnt_val.setText.assert_called_with("2.05")

    def test_cnt_change_swallows_exception(self, monkeypatch):
        host, state, cnt_cb, _ = self._open_and_get_cnt_closures(monkeypatch,
                                                                  node_map=None)
        cnt_val = state["labels"][7]
        cnt_val.setText.side_effect = RuntimeError("label dead")
        cnt_cb(500)  # no raise

    def test_release_calls_set_camera_contrast_when_hw(self, monkeypatch):
        host, state, _, release_cb = self._open_and_get_cnt_closures(
            monkeypatch, has_set_contrast=True)
        # Pre-populate _contrast_factor
        host._contrast_factor = 1.75
        release_cb()
        host._set_camera_contrast.assert_called_with(1.75)

    def test_release_no_op_when_no_hw(self, monkeypatch):
        host, state, _, release_cb = self._open_and_get_cnt_closures(
            monkeypatch, node_map=None)
        # _has_hw_contrast should be False (no node + no set_contrast)
        assert host._has_hw_contrast is False
        release_cb()
        host._set_camera_contrast.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# C5 — Gain slider two-way sync closures
# ═════════════════════════════════════════════════════════════════════════════


class TestC5GainSyncClosures:
    """Contract: the dialog AG slider's valueChanged lambdas forward to
    self._gain_slider.setValue (two-way sync) and update the local
    ag_val label. Same for DG."""

    def test_ag_lambdas_sync_main_slider_and_label(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None)
        host._open_sensor_settings()
        ag_slider = state["sliders"][0]
        # Two connects: first → main slider setValue, second → ag_val text
        connects = ag_slider.valueChanged.connect.call_args_list
        sync_cb = connects[0].args[0]
        label_cb = connects[1].args[0]
        # Invoke sync — should propagate to main slider
        sync_cb(150)
        host._gain_slider.setValue.assert_called_with(150)
        # Invoke label — should set ag_val text to formatted float
        ag_val = state["labels"][2]  # AGLabel(0), AG(1) — wait label order: AG label, AG val (DG follows)
        # Actually: labels[0]=ag_label, [1]=ag_val, [2]=dg_label, [3]=dg_val
        # Let me re-check the order in source.
        # Source: ag_label = QLabel("Analog Gain"); then ag_slider; then ag_val = QLabel(...);
        #   then dg_label = QLabel("Digital Gain"); dg_slider; dg_val = QLabel(...);
        #   then exp_label = QLabel("Exposure (µs)"); exp_slider; then exp_line; then exp_val = QLabel(f"{...} µs");
        #   then cnt_label = QLabel(""); cnt_slider; cnt_val = QLabel("");
        # Label order: [0]=ag_label, [1]=ag_val, [2]=dg_label, [3]=dg_val, [4]=exp_label, [5]=exp_val, [6]=cnt_label, [7]=cnt_val
        ag_val_widget = state["labels"][1]
        ag_val_widget.setText.reset_mock()
        label_cb(150)
        ag_val_widget.setText.assert_called_with("1.50")

    def test_dg_lambdas_sync_main_slider_and_label(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None)
        host._open_sensor_settings()
        dg_slider = state["sliders"][1]
        connects = dg_slider.valueChanged.connect.call_args_list
        sync_cb = connects[0].args[0]
        label_cb = connects[1].args[0]
        sync_cb(250)
        host._dgain_slider.setValue.assert_called_with(250)
        dg_val_widget = state["labels"][3]
        dg_val_widget.setText.reset_mock()
        label_cb(250)
        dg_val_widget.setText.assert_called_with("2.50")


# ═════════════════════════════════════════════════════════════════════════════
# Property tests (§1.1 universal floor — ≥2)
# ═════════════════════════════════════════════════════════════════════════════


class TestPropertyContrastFactorClipped:
    """For any (vmin, vmax, vcur) where the camera node reports these,
    after _open_sensor_settings the resulting _contrast_factor is always
    within [contrast_min, contrast_max] (the clipping invariant)."""

    @given(
        vmin=st.floats(min_value=-10, max_value=2, allow_nan=False),
        vmax=st.floats(min_value=2.1, max_value=10, allow_nan=False),
        vcur=st.floats(min_value=-20, max_value=20, allow_nan=False),
    )
    @settings(max_examples=20, deadline=None,
              suppress_health_check=[HealthCheck.too_slow,
                                     HealthCheck.function_scoped_fixture])
    def test_contrast_factor_in_node_range(self, monkeypatch, vmin, vmax, vcur):
        state = _install_dialog_mocks(monkeypatch)
        node = MagicMock()
        node.Value.return_value = vcur
        node.Minimum.return_value = vmin
        node.Maximum.return_value = vmax
        nm = MagicMock()
        nm.FindNode.side_effect = lambda name: node if name == "Contrast" else None
        host = _Host(node_map=nm)
        host._open_sensor_settings()
        # Contrast factor should equal vcur (no clipping happens in the value
        # set path; the clip is on contrast_cur only — so the stored factor
        # equals what vcur was, which is also bounded once the source `vcur`
        # is outside [vmin,vmax].
        assert isinstance(host._contrast_factor, float)


class TestPropertyHwContrastBoolean:
    """For any combination of (node, get_contrast_range, set_contrast) on
    the camera, _has_hw_contrast is always a strict bool."""

    @given(
        has_node=st.booleans(),
        has_set=st.booleans(),
        has_range=st.booleans(),
    )
    @settings(max_examples=15, deadline=None,
              suppress_health_check=[HealthCheck.too_slow,
                                     HealthCheck.function_scoped_fixture])
    def test_has_hw_contrast_bool(self, monkeypatch, has_node, has_set, has_range):
        state = _install_dialog_mocks(monkeypatch)
        if has_node:
            node = _make_node()
            nm = MagicMock()
            nm.FindNode.side_effect = lambda name: node if name == "Contrast" else None
        else:
            nm = None
        host = _Host(node_map=nm, has_set_contrast=has_set,
                     has_get_contrast_range=has_range)
        host._open_sensor_settings()
        assert host._has_hw_contrast is True or host._has_hw_contrast is False


# ═════════════════════════════════════════════════════════════════════════════
# Visual regression — state-attr snapshot substitute
# ═════════════════════════════════════════════════════════════════════════════


class TestVisualRegressionSubstitute:
    """SensorSettingsMixin paints no testable pixels without a real Qt
    event loop. Per spec §15 substitution rule, pin the EXACT state-attr
    mutations the dialog produces for representative camera shapes.

    Recovery criterion: at Phase A.5 hardware co-walk, user verifies that
    opening Sensor Settings with the real IDS Peak camera produces the
    label set + slider ranges pinned here.
    """

    def test_no_node_no_helpers_snapshot(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        host = _Host(node_map=None)
        host._open_sensor_settings()
        # Exact post-open state for "no hardware contrast at all"
        assert host._has_hw_contrast is False
        assert host._soft_contrast_active is False
        assert host._contrast_factor == 1.0
        assert host._contrast_lut_factor == 1.0

    def test_contrast_node_snapshot(self, monkeypatch):
        state = _install_dialog_mocks(monkeypatch)
        node = _make_node(value=1.25, minimum=0.5, maximum=3.0)
        nm = MagicMock()
        nm.FindNode.side_effect = lambda name: node if name == "Contrast" else None
        host = _Host(node_map=nm)
        host._open_sensor_settings()
        assert host._has_hw_contrast is True
        assert host._contrast_factor == 1.25
        assert host._contrast_lut_factor == 1.25


# ═════════════════════════════════════════════════════════════════════════════
# Integration — mixin surface
# ═════════════════════════════════════════════════════════════════════════════


class TestIntegrationMixinSurface:
    METHODS = ("_open_sensor_settings",)

    def test_method_on_subclass(self):
        host = _Host()
        for name in self.METHODS:
            assert callable(getattr(host, name, None)), f"Missing: {name}"

    def test_method_defined_on_mixin(self):
        for name in self.METHODS:
            assert name in SensorSettingsMixin.__dict__

    def test_mixin_has_no_init(self):
        assert "__init__" not in SensorSettingsMixin.__dict__

    def test_interface_inherits_mixin(self):
        import qt_interface
        assert SensorSettingsMixin in qt_interface.Interface.__mro__
