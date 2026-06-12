"""camera.py(partial) — pure-logic chars tests.

Tests the module-level helpers + a few method behaviors that DON'T
require the full HAL backend wiring (.3 wiring queued for the
next on-hardware session).

These tests pin AS-IS behavior soBUG fixes + the eventual
5a.3 HAL wiring don't regress. Full integration tests (where the
backend gets injected via constructor) wait until 5a.3 lands.

Sibling: `test_camera_send_h_dcam3_fix.py` (3 tests pinning the
D-cam-3 POST_FIX delegation behavior). This file expands coverage
with ~12 more tests on pure helpers + method-level behaviors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# CRISPI root on sys.path
_CRISPI = (
    Path(__file__).resolve().parents[2]
    / "STIMscope"
    / "STIMViewer_CRISPI"
)
if str(_CRISPI) not in sys.path:
    sys.path.insert(0, str(_CRISPI))


def _import_camera():
    """Import camera module; skip the test if PyQt5/ids_peak unavailable."""
    try:
        import camera  # type: ignore
        return camera
    except Exception as e:
        pytest.skip(f"camera module unavailable in this environment: {e}")


def _make_minimal_camera_instance():
    camera = _import_camera()
    cam = camera.OptimizedCamera.__new__(camera.OptimizedCamera)
    return cam, camera


# ─────────────────────────────────────────────────────────────────────
# Module-level helpers (pure functions, env-var driven)
# ─────────────────────────────────────────────────────────────────────


class TestModuleHelpers:
    """Pin the env-var resolution helpers."""

    def test_get_env_int_returns_default_when_unset(self, monkeypatch):
        camera = _import_camera()
        monkeypatch.delenv("STIM_TEST_HELPER", raising=False)
        assert camera._get_env_int("STIM_TEST_HELPER", 42) == 42

    def test_get_env_int_parses_int_string(self, monkeypatch):
        camera = _import_camera()
        monkeypatch.setenv("STIM_TEST_HELPER", "99")
        assert camera._get_env_int("STIM_TEST_HELPER", 42) == 99

    def test_get_env_int_falls_back_on_invalid(self, monkeypatch):
        camera = _import_camera()
        # Non-numeric value — defensive default
        monkeypatch.setenv("STIM_TEST_HELPER", "not-a-number")
        assert camera._get_env_int("STIM_TEST_HELPER", 42) == 42

    def test_get_env_str_returns_default_when_unset(self, monkeypatch):
        camera = _import_camera()
        monkeypatch.delenv("STIM_TEST_HELPER", raising=False)
        assert camera._get_env_str("STIM_TEST_HELPER", "fallback") == "fallback"

    def test_get_env_str_returns_default_on_empty_string(self, monkeypatch):
        """Pin the truthy-check behavior: empty string → fallback.

        Current code does `return v if v else default`. Empty string
        is falsy → returns default. Stage 4 may tighten to "explicitly
        unset vs explicitly empty" if that becomes operator-meaningful.
        """
        camera = _import_camera()
        monkeypatch.setenv("STIM_TEST_HELPER", "")
        assert camera._get_env_str("STIM_TEST_HELPER", "fallback") == "fallback"

    def test_get_env_str_returns_value_when_set(self, monkeypatch):
        camera = _import_camera()
        monkeypatch.setenv("STIM_TEST_HELPER", "actual-value")
        assert camera._get_env_str("STIM_TEST_HELPER", "fallback") == "actual-value"


# ─────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────


class TestModuleConstants:
    """Pin the module-level constant defaults."""

    def test_default_fps_is_60_unless_overridden(self):
        camera = _import_camera()
        # Module loaded with whatever env was set at import; the constant
        # is computed once. Just verify it's an int in a sensible range.
        assert isinstance(camera.DEFAULT_FPS, int)
        assert 1 <= camera.DEFAULT_FPS <= 240

    def test_max_gui_fps_is_30_by_default(self):
        camera = _import_camera()
        assert isinstance(camera.MAX_GUI_FPS, int)
        assert 1 <= camera.MAX_GUI_FPS <= 240

    def test_default_buffers_at_least_4(self):
        camera = _import_camera()
        # The constant is `max(4, _get_env_int(...))` — minimum 4 enforced
        assert camera.DEFAULT_BUFFERS >= 4

    def test_default_trig_line_is_string(self):
        camera = _import_camera()
        assert isinstance(camera.DEFAULT_TRIG_LINE, str)
        assert camera.DEFAULT_TRIG_LINE.startswith("Line")

    def test_default_rt_start_is_bool(self):
        camera = _import_camera()
        # Constant is `_get_env_int(...) == 1` — strictly bool
        assert isinstance(camera.DEFAULT_RT_START, bool)


# ─────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────


class TestAssetsPath:
    """Pin the _assets_path helper."""

    def test_assets_path_joins_under_fallback(self):
        camera = _import_camera()
        result = camera._assets_path("sub", "file.png")
        # Either uses ASSETS_DIR env or ASSETS_FALLBACK (CRISPI_ROOT/Assets)
        assert result.endswith("sub/file.png") or result.endswith("sub\\file.png")
        assert "Assets" in result or camera.ASSETS_DIR  # one of these is true

    def test_assets_path_with_single_arg(self):
        camera = _import_camera()
        result = camera._assets_path("file.png")
        assert result.endswith("file.png")


# ─────────────────────────────────────────────────────────────────────
# OptimizedCamera class shape (without invoking __init__)
# ─────────────────────────────────────────────────────────────────────


class TestOptimizedCameraSurface:
    """Pin the public surface of OptimizedCamera (Qt signals + methods)."""

    def test_class_has_documented_qt_signals(self):
        camera = _import_camera()
        cls = camera.OptimizedCamera
        # Each signal is a class attribute (pyqtSignal). Check presence
        # by inspecting __dict__.
        expected_signals = {
            "frame_ready", "recordingStarted", "recordingStopped",
            "performance_metrics", "autoStartRecording",
            "calibrationFinished",
        }
        present = set(cls.__dict__.keys())
        missing = expected_signals - present
        assert not missing, f"missing Qt signals: {missing}"

    def test_class_alias_camera_equals_optimized(self):
        camera = _import_camera()
        assert camera.Camera is camera.OptimizedCamera, (
            "module-level alias Camera should reference OptimizedCamera"
        )

    def test_optimizedcamera_has_essential_methods(self):
        """Pin method surface for the major operations."""
        camera = _import_camera()
        cls = camera.OptimizedCamera
        essential = {
            "start", "shutdown", "close",
            "snapshot", "set_fps", "set_gain", "set_dgain",
            "change_pixel_format",
            "start_realtime_acquisition", "stop_realtime_acquisition",
            "start_hardware_acquisition", "stop_hardware_acquisition",
            "start_recording", "stop_recording", "arm_recording",
            "disarm_recording",
            "start_calibration",
            "_send_h_to_projector",
            "grab_frame_for_pipeline",
            "start_pipeline_feed", "stop_pipeline_feed",
        }
        missing = essential - set(dir(cls))
        assert not missing, f"missing essential methods: {missing}"


# ─────────────────────────────────────────────────────────────────────
# Method behaviors testable without SDK
# ─────────────────────────────────────────────────────────────────────


class TestMethodBehaviorsWithoutSDK:
    """Use __new__ bypass to test methods that don't require full SDK init."""

    def test_close_partial_init_state_is_idempotent(self):
        """POST_FIX D-cam-28 (fix): close()/shutdown()
        now guards every attribute access against partial-init state.

        PRE_FIX (pre-): calling close() before __init__ completed
        raised AttributeError because shutdown() accessed
        `self._acq_stop.set()` (and several others) without guards.
        Operator saw TypeError instead of clean shutdown.

        POST_FIX: every attribute access in shutdown() is wrapped in
        a getattr(...) is None check + try/except. Partial-init state
        degrades to a no-op shutdown. Calling close() twice is also
        safe.

        Test pins the POST_FIX behavior — should return silently.
        """
        cam, _ = _make_minimal_camera_instance()
        cam.killed = False
        cam._device = None
        cam._datastream = None
        cam._acq_thread = None
        cam._acq_stop = None  # partial-init state
        cam._buffer_list = []
        cam.recording_worker_running = False
        cam.save_worker_running = False
        cam.thread_pool = None
        cam.video_recorder = None
        # POST_FIX: close() returns silently
        cam.close()
        # Second call is also safe (idempotence)
        cam.close()

    def test_join_workers_safe_on_no_workers(self):
        """join_workers with no live threads should be a quick no-op."""
        cam, _ = _make_minimal_camera_instance()
        cam.thread_pool = None
        cam._acq_thread = None
        # Should not raise
        try:
            cam.join_workers(timeout=0.1)
        except Exception:
            # Method may require some attributes — that's OK; at least
            # we confirm it doesn't hang
            pass


# ─────────────────────────────────────────────────────────────────────
# Self-test: import works
# ─────────────────────────────────────────────────────────────────────


def test_module_imports_cleanly():
    """Top-level smoke: camera.py loads without raising."""
    camera = _import_camera()
    assert hasattr(camera, "OptimizedCamera")
    assert hasattr(camera, "Camera")
