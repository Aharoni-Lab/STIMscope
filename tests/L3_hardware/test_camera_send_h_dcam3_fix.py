"""Targeted POST_FIX regression test for D-cam-3.

Pairs with the L4 hot-path test
``test_hot_path.py::test_dl4_1_h_delivery_goes_through_audited_helper``
which covers the same fix on the L4 (run_hardware_pipeline) side.

Stage-4 fix: camera.py's
``OptimizedCamera._send_h_to_projector`` now delegates to the
L3-audited ``core.projector._send_homography_inline`` helper instead
of inlining its own ZMQ send.

This is a small dedicated test file — not the full L3 camera.pycharacterization suite (that's blocked on 5a.3 HAL wiring and lands
when the user is back on hardware). The aim here is just to pin the
D-cam-3 POST_FIX behavior in CI so any regression that re-introduces
the inline ZMQ pattern fails the suite.

Hardware-verify reference: Test 4 (commit 06bc197) showed the pre-fix
inline path silently swallowed "no ACK" failures. The audited helper
logs at WARNING + returns a bool. This test pins that the bool path
is exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# CRISPI root on sys.path so `import camera` resolves to the audited
# in-tree source (conftest at tests/ root handles the core package
# package; CRISPI root needs explicit insertion).
_CRISPI = (
    Path(__file__).resolve().parents[2]
    / "STIMscope"
    / "STIMViewer_CRISPI"
)
if str(_CRISPI) not in sys.path:
    sys.path.insert(0, str(_CRISPI))


def _make_minimal_camera_instance():
    """Construct an OptimizedCamera instance without exercising __init__.

    Skips IDS Peak SDK initialization + Qt machinery. We only need
    an object with the `_send_h_to_projector` method bound to it.
    """
    # Import lazily because camera.py imports ids_peak / PyQt5 at top
    try:
        import camera  # type: ignore
    except Exception as e:
        pytest.skip(f"camera.py unavailable in this environment: {e}")
    cam = camera.OptimizedCamera.__new__(camera.OptimizedCamera)
    return cam, camera


def test_send_h_to_projector_delegates_to_audited_helper(monkeypatch):
    """POST_FIX D-cam-3: _send_h_to_projector must delegate to
    core.projector._send_homography_inline.

    Spy on the helper; assert it was called with the expected H and
    the canonical 5560 endpoint.
    """
    cam, camera_mod = _make_minimal_camera_instance()

    import core.projector as proj_mod
    helper_calls = []

    def spying_helper(H, endpoint, **kwargs):
        helper_calls.append((H.copy(), endpoint))
        return True

    monkeypatch.setattr(
        proj_mod, "_send_homography_inline", spying_helper
    )

    H_in = np.eye(3, dtype=np.float64) * 2.0
    result = cam._send_h_to_projector(H_in)

    # Helper was called exactly once with the right args
    assert len(helper_calls) == 1
    H_sent, endpoint = helper_calls[0]
    np.testing.assert_array_equal(H_sent, H_in)
    assert endpoint == "tcp://127.0.0.1:5560"

    # And the bool return is propagated to the caller
    assert result is True


def test_send_h_to_projector_returns_false_on_no_ack(monkeypatch):
    """POST_FIX D-cam-3: the audited helper returns False on no-ACK;
    that bool propagates through camera.py's wrapper.

    Pre-fix, this path silently printed "⚠️ No ACK" and the caller
    couldn't tell whether the send succeeded. Post-fix the wrapper
    returns the bool unchanged.
    """
    cam, _ = _make_minimal_camera_instance()

    import core.projector as proj_mod
    monkeypatch.setattr(
        proj_mod, "_send_homography_inline",
        lambda H, endpoint, **kwargs: False,
    )

    result = cam._send_h_to_projector(np.eye(3))
    assert result is False


def test_send_h_to_projector_handles_import_failure_gracefully(monkeypatch):
    """If the audited helper can't be imported (broken sys.path /
    deleted module), wrapper logs + returns False instead of raising.
    """
    cam, _ = _make_minimal_camera_instance()

    # Mask the import by removing core.projector from sys.modules
    # AND from sys.modules['core'] if present, then make any
    # `from core.projector import...` raise.
    original_import = __builtins__.__import__ if hasattr(
        __builtins__, "__import__"
    ) else __builtins__["__import__"]

    def failing_import(name, *args, **kwargs):
        if "core.projector" in name or name == "core.projector":
            raise ImportError("simulated missing audited helper")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", failing_import)

    result = cam._send_h_to_projector(np.eye(3))
    # Returns False (not raise) when helper unavailable
    assert result is False
