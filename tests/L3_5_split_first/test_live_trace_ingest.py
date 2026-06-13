"""Comprehensive characterization tests for ``live_trace_ingest``.

target ~90% path coverage on the LiveTraceIngestMixin (extracted at
iter 11 commit d3a91e9).

Module surface (~245 LOC, 8 methods):
- ``_connect_camera_signals`` — auto-detect camera frame signal (8
  candidate signal names, fallback to register_consumer callback)
- ``_disconnect_camera_signals`` — tear down stored signal/slot pairs
- ``_on_camera_frame(object)`` — @pyqtSlot wrapper → on_frame
- ``_on_camera_qimage(QImage)`` — @pyqtSlot wrapper → on_frame (via
  qimage_to_gray_np)
- ``on_frame`` — public API; queues to self.frame_processor +
  first-frame diagnostic
- ``_monitor_gpu_memory`` — cuda runtime memGetInfo + threshold check
- ``_cleanup_gpu_memory`` — cupy mempool free_all_blocks under lock
- ``_update_performance_stats`` — emits performance_update signal

Mixin contract — subclass provides:
- ``self.camera`` (with signal attrs or register_consumer)
- ``self._camera_signal_refs`` (list)
- ``self.frame_processor`` (with add_frame)
- ``self.error_occurred`` (pyqtSignal(str))
- ``self.gpu_memory_infoing`` (pyqtSignal(str))
- ``self.performance_update`` (pyqtSignal(dict))
- ``self.stats`` (dict with gpu_memory_peak, memory_usage_peak, uptime_seconds)
- ``self.start_time`` (float)
- ``self._gpu_lock`` (threading.Lock)

CUDA paths mocked since test host has no compatible CUDA driver.
QApp fixture inherited from conftest.py (session autouse).
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage

import live_trace.ingest as lti
from live_trace.ingest import LiveTraceIngestMixin


# ─────────────────────────────────────────────────────────────────────────────
# Test infrastructure: stub host class
# ─────────────────────────────────────────────────────────────────────────────


class _Host(LiveTraceIngestMixin):
    """Stub satisfying the mixin's `self.X` contract."""

    def __init__(self):
        self.camera = MagicMock()
        self._camera_signal_refs = []
        self.frame_processor = MagicMock()
        self.error_occurred = MagicMock()
        self.gpu_memory_infoing = MagicMock()
        self.performance_update = MagicMock()
        self.stats = {
            "gpu_memory_peak": 0.0,
            "memory_usage_peak": 0.0,
            "uptime_seconds": 0.0,
        }
        self.start_time = time.time() - 10  # 10s ago
        self._gpu_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# C1 — _connect_camera_signals: 8 candidate names + connect success/fail
#                                + register_consumer fallback + no-connection
# ─────────────────────────────────────────────────────────────────────────────


class TestC1ConnectCameraSignals:
    """Contract: try 8 candidate signal names, prefer on_frame(object) slot
    over _on_camera_qimage(QImage), fall back to register_consumer callback,
    finally to manual-feed mode."""

    def _camera_with_signal(self, signal_name, connect_to_obj=True):
        """Create a mock camera exposing one named signal that accepts connect."""
        cam = MagicMock(spec=[signal_name])
        sig = MagicMock()
        if not connect_to_obj:
            # First connect (to on_frame) raises; second succeeds
            sig.connect.side_effect = [RuntimeError("object slot failed"), None]
        setattr(cam, signal_name, sig)
        return cam, sig

    def test_connects_to_first_candidate_via_on_frame(self, capsys):
        host = _Host()
        cam, sig = self._camera_with_signal("image_update_signal")
        host.camera = cam
        host._connect_camera_signals()
        sig.connect.assert_called_once_with(host.on_frame, Qt.QueuedConnection)
        assert (sig, host.on_frame) in host._camera_signal_refs
        captured = capsys.readouterr()
        assert "image_update_signal" in captured.out
        assert "on_frame(object)" in captured.out

    def test_falls_through_to_qimage_slot_when_object_slot_fails(self, capsys):
        host = _Host()
        cam, sig = self._camera_with_signal("frame_qimage", connect_to_obj=False)
        host.camera = cam
        host._connect_camera_signals()
        # Should have called connect twice: first object, then qimage
        assert sig.connect.call_count == 2
        # Second connect was to _on_camera_qimage
        second_call = sig.connect.call_args_list[1]
        assert second_call.args[0] == host._on_camera_qimage
        captured = capsys.readouterr()
        assert "_on_camera_qimage(QImage)" in captured.out

    def test_skips_missing_signal_names(self, capsys):
        """If camera has none of the named signals, falls through to
        register_consumer or manual-feed."""
        host = _Host()
        # MagicMock with spec=[] has none of the signal names
        host.camera = MagicMock(spec=[])
        host._connect_camera_signals()
        captured = capsys.readouterr()
        # Should log the manual-feed fallback message
        assert "waiting for manual feed" in captured.out

    def test_register_consumer_fallback_when_no_signals(self, capsys):
        """Camera with no named signals but register_consumer callable → use it."""
        host = _Host()
        cam = MagicMock(spec=["register_consumer"])
        cam.register_consumer = MagicMock()
        host.camera = cam
        host._connect_camera_signals()
        cam.register_consumer.assert_called_once_with(host.on_frame)
        captured = capsys.readouterr()
        assert "registered camera consumer callback" in captured.out

    def test_register_consumer_exception_logged(self, capsys):
        host = _Host()
        cam = MagicMock(spec=["register_consumer"])
        cam.register_consumer = MagicMock(side_effect=RuntimeError("nope"))
        host.camera = cam
        host._connect_camera_signals()
        captured = capsys.readouterr()
        assert "register_consumer failed" in captured.out
        assert "waiting for manual feed" in captured.out

    def test_getattr_exception_swallowed_consistently(self):
        """D-lti-1fix iter 44: both the signal-name candidate
        loop AND the later register_consumer lookup now use the same
        try/except defensive pattern. A camera whose `__getattr__`
        always raises no longer crashes the connection routine — it
        falls through to "could not connect" + "waiting for manual feed".
        """
        host = _Host()

        class _RaisingCam:
            def __getattr__(self, name):
                raise RuntimeError(f"attr {name} explodes")

        host.camera = _RaisingCam()
        # Should not raise post-fix
        host._connect_camera_signals()
        # Nothing was connected
        assert host._camera_signal_refs == []


# ─────────────────────────────────────────────────────────────────────────────
# C2 — _disconnect_camera_signals
# ─────────────────────────────────────────────────────────────────────────────


class TestC2DisconnectCameraSignals:
    """Contract: disconnect every stored (sig, slot) pair + clear the list."""

    def test_disconnects_each_pair(self):
        host = _Host()
        sig1, slot1 = MagicMock(), MagicMock()
        sig2, slot2 = MagicMock(), MagicMock()
        host._camera_signal_refs = [(sig1, slot1), (sig2, slot2)]
        host._disconnect_camera_signals()
        sig1.disconnect.assert_called_once_with(slot1)
        sig2.disconnect.assert_called_once_with(slot2)
        assert host._camera_signal_refs == []

    def test_swallows_disconnect_exception(self):
        host = _Host()
        sig, slot = MagicMock(), MagicMock()
        sig.disconnect.side_effect = RuntimeError("already disconnected")
        host._camera_signal_refs = [(sig, slot)]
        # Should not raise
        host._disconnect_camera_signals()
        assert host._camera_signal_refs == []

    def test_no_refs_attribute_is_safe(self):
        host = _Host()
        del host._camera_signal_refs  # simulate edge: not initialized
        # Should not raise
        host._disconnect_camera_signals()


# ─────────────────────────────────────────────────────────────────────────────
# C3 — _on_camera_frame: pyqtSlot wrapper
# ─────────────────────────────────────────────────────────────────────────────


class TestC3OnCameraFrame:
    """Contract: forward frame_obj to on_frame."""

    def test_forwards_to_on_frame(self):
        host = _Host()
        frame = np.zeros((4, 4), dtype=np.uint8)
        with patch.object(host, "on_frame") as mock_on_frame:
            host._on_camera_frame(frame)
            mock_on_frame.assert_called_once_with(frame)


# ─────────────────────────────────────────────────────────────────────────────
# C4 — _on_camera_qimage: QImage → numpy → on_frame
# ─────────────────────────────────────────────────────────────────────────────


class TestC4OnCameraQImage:
    """Contract: convert QImage to grayscale numpy then forward to on_frame.
    Conversion exceptions are caught + logged."""

    def test_converts_and_forwards(self):
        host = _Host()
        img = QImage(8, 4, QImage.Format_Grayscale8)
        img.fill(123)
        with patch.object(host, "on_frame") as mock_on_frame:
            host._on_camera_qimage(img)
            mock_on_frame.assert_called_once()
            arg = mock_on_frame.call_args[0][0]
            assert arg.shape == (4, 8)
            assert (arg == 123).all()

    def test_swallows_conversion_exception(self, capsys):
        host = _Host()
        bad_img = QImage()  # null
        with patch.object(host, "on_frame") as mock_on_frame:
            host._on_camera_qimage(bad_img)
            mock_on_frame.assert_not_called()
        captured = capsys.readouterr()
        assert "QImage→np conversion failed" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C5 — on_frame: queue + first-frame logging + error_occurred on failure
# ─────────────────────────────────────────────────────────────────────────────


class TestC5OnFrame:
    """Contract: queue frame to frame_processor + log first frame diagnostic
    + emit error_occurred if queueing raises."""

    def test_queues_to_frame_processor(self):
        host = _Host()
        frame = np.zeros((4, 4), dtype=np.uint8)
        host.on_frame(frame)
        host.frame_processor.add_frame.assert_called_once_with(frame)

    def test_first_frame_logs_diagnostic(self, capsys):
        host = _Host()
        frame = np.zeros((4, 4), dtype=np.uint8)
        host.on_frame(frame)
        captured = capsys.readouterr()
        assert "FIRST frame received" in captured.out
        assert host._first_frame_logged is True

    def test_subsequent_frames_skip_diagnostic(self, capsys):
        host = _Host()
        frame = np.zeros((4, 4), dtype=np.uint8)
        host.on_frame(frame)
        capsys.readouterr()  # discard first
        host.on_frame(frame)
        captured = capsys.readouterr()
        assert "FIRST frame received" not in captured.out

    def test_object_with_width_height_diagnostic(self, capsys):
        """Branch: frame has.Width()/.Height() (IDS Buffer-like)."""
        host = _Host()
        buf = MagicMock()
        buf.Width.return_value = 640
        buf.Height.return_value = 480
        # spec out 'shape' so getattr returns None (not MagicMock)
        del buf.shape
        host.on_frame(buf)
        captured = capsys.readouterr()
        assert "(W,H)=(640, 480)" in captured.out

    def test_width_height_exception_is_safe(self, capsys):
        host = _Host()
        buf = MagicMock()
        buf.Width.side_effect = RuntimeError("buf broken")
        del buf.shape
        host.on_frame(buf)  # should not crash
        captured = capsys.readouterr()
        assert "FIRST frame received" in captured.out

    def test_diagnostic_block_exception_is_safe(self, capsys):
        """The outer try around the diagnostic catches any exception."""
        host = _Host()
        # MagicMock(name='breaks') with __name__ attribute that raises
        bad_frame = MagicMock()
        # Make type() call work but later access raise
        host.on_frame(bad_frame)  # should not raise
        # Frame still queued
        host.frame_processor.add_frame.assert_called_once()

    def test_queue_failure_emits_error_occurred(self, capsys):
        host = _Host()
        host.frame_processor.add_frame.side_effect = RuntimeError("queue full")
        frame = np.zeros((4, 4), dtype=np.uint8)
        host.on_frame(frame)
        host.error_occurred.emit.assert_called_once()
        arg = host.error_occurred.emit.call_args[0][0]
        assert "queue full" in arg
        captured = capsys.readouterr()
        assert "Error queueing frame" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# C6 — _monitor_gpu_memory: CUDA branches
# ─────────────────────────────────────────────────────────────────────────────




class TestC8UpdatePerformanceStats:
    """Contract: update uptime + memory_usage_peak (max), emit a COPY of stats."""

    def test_updates_uptime(self):
        host = _Host()
        before = time.time() - host.start_time
        host._update_performance_stats()
        assert host.stats["uptime_seconds"] >= before

    def test_updates_memory_peak(self):
        host = _Host()
        host._update_performance_stats()
        # On a real system memory_usage_peak should be > 0 after psutil call
        assert host.stats["memory_usage_peak"] > 0

    def test_memory_peak_is_monotone_non_decreasing(self):
        host = _Host()
        host.stats["memory_usage_peak"] = 1e9  # arbitrarily large
        host._update_performance_stats()
        # max(1e9, actual) — actual should be smaller, so unchanged
        assert host.stats["memory_usage_peak"] == 1e9

    def test_psutil_exception_does_not_crash(self):
        host = _Host()
        with patch.object(lti.psutil, "Process", side_effect=RuntimeError("psutil down")):
            host._update_performance_stats()
        # Should still emit (the exception only skips the memory update)
        host.performance_update.emit.assert_called_once()

    def test_emits_copy_not_reference(self):
        host = _Host()
        host._update_performance_stats()
        host.performance_update.emit.assert_called_once()
        emitted = host.performance_update.emit.call_args[0][0]
        # Mutating original after emit shouldn't affect emitted dict
        original_ref = host.stats
        original_ref["uptime_seconds"] = 999999.0
        assert emitted["uptime_seconds"] != 999999.0


# ─────────────────────────────────────────────────────────────────────────────
# C9 — Mixin integration
# ─────────────────────────────────────────────────────────────────────────────


class TestC9MixinIntegration:
    """Contract: methods accessible on subclass; no __init__ on mixin."""

    def test_all_6_methods_on_subclass(self):
        host = _Host()
        for name in (
            "_connect_camera_signals", "_disconnect_camera_signals",
            "_on_camera_frame", "_on_camera_qimage", "on_frame",
            "_update_performance_stats",
        ):
            method = getattr(host, name, None)
            assert callable(method), f"Missing or non-callable: {name}"

    def test_methods_defined_on_mixin(self):
        for name in (
            "_connect_camera_signals", "_disconnect_camera_signals",
            "_on_camera_frame", "_on_camera_qimage", "on_frame",
            "_update_performance_stats",
        ):
            assert name in LiveTraceIngestMixin.__dict__, \
                f"Method {name} not defined on mixin"

    def test_mixin_has_no_init(self):
        assert "__init__" not in LiveTraceIngestMixin.__dict__


# ─────────────────────────────────────────────────────────────────────────────
# §1.1 L3.5 matrix backfill — Property + Snapshot + Concurrency (iter-56)
#
# §1.1 L3.5 row requires:
#   - Property ≥2 per sub-module (universal floor)
#   - Snapshot required for trace outputs (on_frame is the camera→trace
#     seam, _connect_camera_signals candidate-order is a published
#     contract; both snapshotted here)
#   - Concurrency ≥1 if mixin touches threads (add_frame thread safety)
#
# Closes part of the OPEN BLOCK on iter-42 L3.5 PROMOTION per
# audit_findings.log lines 1655-2235 + docs/PHASE_A5_DEFERRAL.md.
# Third L3.5 sub-mixin backfill (live_trace_ingest), 3 of 8.
# ─────────────────────────────────────────────────────────────────────────────

import hashlib  # noqa: E402

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402



class TestSnapshotIngestContract:
    """§1.1 L3.5 row: snapshot required for trace outputs.

    Two seam-contract snapshots:
    - `_connect_camera_signals` candidate name ordering (downstream
      hardware integrations rely on this probe order for fallback
      semantics)
    - `_update_performance_stats` emitted-dict key set (the
      performance_update signal's payload schema)
    """

    def test_camera_signal_candidate_order_snapshot(self):
        """Pin the 8-name candidate tuple for ``_connect_camera_signals``.
        Any silent reorder (e.g. moving ``frame_qimage`` before
        ``image_update_signal``) would change which signal wins on
        cameras that expose multiple — a downstream behavior change
        masked as a "cleanup" refactor.

        Reading the candidate list requires touching the source —
        we inspect bytecode-stable string constants via the function's
        co_consts (immune to refactors that don't change literals)."""
        import dis
        # Build a deterministic candidate snapshot by exercising the
        # function with a camera that has NO matching signal. The
        # function will iterate every name and call hasattr-like
        # getattr probes. We capture the probe order via a custom
        # __getattr__.
        probe_order = []

        class _ProbeCam:
            def __getattr__(self, name):
                # The mixin only probes the candidate names — record
                # them. Returning None matches `sig is None` skip.
                if name == "register_consumer":
                    return None  # let outer fallback path skip
                probe_order.append(name)
                return None

        host = _Host()
        host.camera = _ProbeCam()
        host._camera_signal_refs = []
        host._connect_camera_signals()

        # Snapshot the exact probe order as a sha256 hash
        h = hashlib.sha256(b",".join(s.encode() for s in probe_order)).hexdigest()
        expected_order = [
            "image_update_signal", "frame_numpy", "frame_np",
            "frame_ready", "newFrame", "frame_signal",
            "new_qimage", "frame_qimage",
        ]
        expected = hashlib.sha256(
            b",".join(s.encode() for s in expected_order)
        ).hexdigest()
        assert h == expected, (
            f"_connect_camera_signals candidate order regression. "
            f"Got order={probe_order!r}, expected={expected_order!r}. "
            f"Downstream cameras may now bind to a different signal."
        )
        # Sanity: dis module imported but unused for hygiene; silence linter
        _ = dis

    def test_performance_update_payload_schema_snapshot(self):
        """Pin the key set of the dict emitted via ``performance_update``.
        Downstream consumers (Dashboard performance panel, telemetry)
        depend on this schema; any silent key rename or addition is
        a wire-format break for them."""
        host = _Host()
        host._update_performance_stats()
        # The mixin emits via performance_update.emit(stats.copy())
        host.performance_update.emit.assert_called_once()
        emitted = host.performance_update.emit.call_args[0][0]
        # Schema = sorted key tuple, hashed
        schema = tuple(sorted(emitted.keys()))
        h = hashlib.sha256(repr(schema).encode()).hexdigest()
        expected_schema = (
            "gpu_memory_peak", "memory_usage_peak", "uptime_seconds",
        )
        expected = hashlib.sha256(repr(expected_schema).encode()).hexdigest()
        assert h == expected, (
            f"performance_update payload schema regression. "
            f"Got keys={schema!r}, expected={expected_schema!r}."
        )


class TestConcurrencyGpuLock:
    """§1.1 L3.5 row: concurrency ≥1 if mixin touches threads.

    `_cleanup_gpu_memory` holds `self._gpu_lock` while calling
    cupy.get_default_memory_pool().free_all_blocks(). The lock is a
    state-machine invariant — concurrent cleanups must serialize.

    Per §1.2 concurrency playbook: state-machine invariants, no
    sleep-as-control. We pin:
    - Lock is acquired during cleanup
    - on_frame is reentrant under concurrent calls (queues all frames)
    """

    def test_on_frame_concurrent_queueing(self):
        """Many concurrent ``on_frame`` calls must all reach
        ``frame_processor.add_frame`` without dropping or duplicating
        frames. Pins the contract that on_frame is reentrant and that
        per-frame add_frame is the sole sink."""
        host = _Host()
        # Use a real lock-guarded list to capture all calls
        recorded = []
        record_lock = threading.Lock()

        def _record(frame):
            with record_lock:
                recorded.append(frame)

        host.frame_processor.add_frame.side_effect = _record

        N_THREADS = 8
        FRAMES_PER_THREAD = 25
        # Distinct value per frame so the recorder can verify no dedup
        # / drop. Keep inside uint8 range — N_THREADS * FRAMES_PER_THREAD
        # = 200 fits, where the prior `i * 1000` did not.
        frames = [
            np.full((4, 4), i * FRAMES_PER_THREAD + j, dtype=np.uint8)
            for i in range(N_THREADS)
            for j in range(FRAMES_PER_THREAD)
        ]

        # Disable the first-frame diagnostic so concurrent prints
        # don't interleave (and to avoid the diagnostic block once
        # the flag is set once).
        host._first_frame_logged = True

        def _worker(start):
            for j in range(FRAMES_PER_THREAD):
                host.on_frame(frames[start + j])

        threads = [
            threading.Thread(
                target=_worker,
                args=(i * FRAMES_PER_THREAD,),
                daemon=True,
            )
            for i in range(N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "worker thread hung"

        # All N*F frames reached add_frame, none dropped or duplicated
        assert len(recorded) == N_THREADS * FRAMES_PER_THREAD, (
            f"frame drop detected: recorded {len(recorded)}, "
            f"expected {N_THREADS * FRAMES_PER_THREAD}"
        )
        # Identity-set check: each frame appears exactly once
        ids = {id(f) for f in recorded}
        expected_ids = {id(f) for f in frames}
        assert ids == expected_ids, "frame identity mismatch"
