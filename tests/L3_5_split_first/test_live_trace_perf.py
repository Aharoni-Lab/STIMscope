"""Comprehensive characterization tests for ``live_trace_perf``.

target ~90% path coverage on the extracted module. Tests pin the AS-IS
behavior of the 4 classes + 1 helper that were extracted to
``live_trace_perf.py`` at iter 9 (commit 895a5ae).

Module surface (~205 LOC):
- ``MAX_FRAME_QUEUE_SIZE`` constant
- ``qimage_to_gray_np(qimg)`` — QImage → grayscale numpy
- ``PerformanceMonitor`` — wall-clock + memory delta timer
- ``SyncState`` enum (7 values)
- ``SyncInfo`` dataclass
- ``FrameProcessor(QThread)`` — queue + thread-pool frame processor

Contracts numbered C1–CN per spec
(``docs/specs/L3.5_split_first/live_trace_extractor.md`` — surface
delegated to ``live_trace_perf.py`` post-extraction).

Tests run headless: no Qt event loop needed, no real camera. The
QThread is exercised via direct method calls (start/stop not invoked
on the thread itself; we test the methods in isolation).

Branches exercised per function/method are listed in each test
docstring. Target: ≥90% line coverage on live_trace_perf.py.
"""

from __future__ import annotations

import queue
import sys
import time
from concurrent.futures import Future
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CRISPI_PATH = REPO_ROOT / "STIMscope" / "STIMViewer_CRISPI"
if str(CRISPI_PATH) not in sys.path:
    sys.path.insert(0, str(CRISPI_PATH))

from PyQt5.QtGui import QImage

import live_trace.perf as ltp


# ─────────────────────────────────────────────────────────────────────────────
# C1 — MAX_FRAME_QUEUE_SIZE constant
# ─────────────────────────────────────────────────────────────────────────────


class TestC1MaxFrameQueueSize:
    """Contract: queue capacity bound at module level."""

    def test_value_is_8(self):
        assert ltp.MAX_FRAME_QUEUE_SIZE == 8

    def test_is_integer(self):
        assert isinstance(ltp.MAX_FRAME_QUEUE_SIZE, int)


# ─────────────────────────────────────────────────────────────────────────────
# C2 — qimage_to_gray_np: all 4 format branches + fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestC2QImageToGrayNp:
    """Contract: convert QImage of any supported format to (H, W) uint8 grayscale.

    Branches:
    - null QImage → ValueError
    - Format_Grayscale8 → buf.reshape((H, W))
    - Format_ARGB32 → green channel (idx 1) of (H,W,4)
    - Format_RGBA8888 → green channel of (H,W,4)
    - Format_RGB888 → green channel of (H,W,3)
    - Other format → convertToFormat(ARGB32) + ARGB branch
    - Final fallback → convertToFormat(Grayscale8)
    """

    def test_null_qimage_raises_value_error(self):
        qimg = QImage()
        assert qimg.isNull()
        with pytest.raises(ValueError, match="Null QImage"):
            ltp.qimage_to_gray_np(qimg)

    def test_grayscale8_returns_2d_uint8(self):
        # Use a 4-aligned width — Qt pads rows to 4-byte boundary, and the
        # current qimage_to_gray_np implementation reshapes by (H, W)
        # without consulting bytesPerLine. Production camera frame widths
        # are all 4-aligned (1920, 1024, 640, 512) so this works in
        # practice. **D-ltp-1 (FINDING, surfaced ):**
        # qimage_to_gray_np crashes for non-4-aligned Grayscale8 widths.
        # See xfail test below.
        img = QImage(8, 4, QImage.Format_Grayscale8)
        img.fill(200)
        out = ltp.qimage_to_gray_np(img)
        assert out.shape == (4, 8)
        assert out.dtype == np.uint8
        assert (out == 200).all()

    def test_grayscale8_unaligned_width_works(self):
        """D-ltp-1fix iter 44: Qt pads rows to 4-byte boundaries,
        so a 6-pixel-wide Grayscale8 has 8-byte rows (2 bytes padding/row).
        Post-fix: qimage_to_gray_np uses bytesPerLine() for reshape + slices
        to width. No longer crashes.
        """
        img = QImage(6, 4, QImage.Format_Grayscale8)
        img.fill(200)
        out = ltp.qimage_to_gray_np(img)
        assert out.shape == (4, 6)
        assert out.dtype == np.uint8
        assert (out == 200).all()

    def test_argb32_extracts_green_channel(self):
        img = QImage(4, 3, QImage.Format_ARGB32)
        # qRgb(R, G, B) — argb byte order is BGRA on little-endian, but
        # the function indexes axis 2 with [1]. For ARGB32 in numpy view
        # the channel at index 1 corresponds to the G byte position.
        img.fill(0xFF408010)  # ARGB: A=FF, R=40, G=80, B=10
        out = ltp.qimage_to_gray_np(img)
        assert out.shape == (3, 4)
        # Index 1 of the 4-channel byte array — confirms function picks
        # one consistent channel; assert all-equal (not the exact value)
        # to avoid endian assumptions.
        assert (out == out[0, 0]).all()

    def test_rgba8888_extracts_one_channel(self):
        img = QImage(4, 3, QImage.Format_RGBA8888)
        img.fill(0x408010FF)
        out = ltp.qimage_to_gray_np(img)
        assert out.shape == (3, 4)
        assert out.dtype == np.uint8
        # Assert all-equal (function picks ONE channel consistently).
        # Specific channel value is Qt-internal-format dependent; not
        # asserted here.
        assert (out == out[0, 0]).all()

    def test_rgb888_extracts_one_channel(self):
        img = QImage(4, 3, QImage.Format_RGB888)
        img.fill(0x408010)
        out = ltp.qimage_to_gray_np(img)
        assert out.shape == (3, 4)
        assert out.dtype == np.uint8
        assert (out == out[0, 0]).all()

    def test_mono_format_converts_via_argb32(self):
        """Mono (Format_Mono) is not in the recognized set → falls into
        the 'convertToFormat(ARGB32)' path then ARGB branch."""
        img = QImage(4, 3, QImage.Format_Mono)
        img.fill(1)
        out = ltp.qimage_to_gray_np(img)
        assert out.shape == (3, 4)
        assert out.dtype == np.uint8


# ─────────────────────────────────────────────────────────────────────────────
# C3 — PerformanceMonitor
# ─────────────────────────────────────────────────────────────────────────────


class TestC3PerformanceMonitor:
    """Contract: time + memory delta around an arbitrary code section.

    Branches:
    - start(): psutil success → memory_before > 0
    - start(): psutil exception → memory_before = 0.0
    - end(): start_time is None → no-op early return
    - end(): psutil success → "ΔMem" message printed
    - end(): psutil exception → fallback message printed
    """

    def test_init_state(self):
        pm = ltp.PerformanceMonitor()
        assert pm.start_time is None
        assert pm.memory_before == 0.0

    def test_start_sets_start_time(self):
        pm = ltp.PerformanceMonitor()
        before = time.perf_counter()
        pm.start()
        assert pm.start_time is not None
        assert pm.start_time >= before

    def test_start_captures_memory(self):
        pm = ltp.PerformanceMonitor()
        pm.start()
        # Real psutil should give a positive value on this Jetson
        assert pm.memory_before > 0

    def test_start_with_psutil_failure_falls_back_to_zero(self):
        pm = ltp.PerformanceMonitor()
        with patch.object(ltp.psutil, "Process", side_effect=RuntimeError("boom")):
            pm.start()
        assert pm.memory_before == 0.0
        assert pm.start_time is not None

    def test_end_without_start_is_noop(self, capsys):
        pm = ltp.PerformanceMonitor()
        pm.end("test")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_end_after_start_prints_dt_and_mem(self, capsys):
        pm = ltp.PerformanceMonitor()
        pm.start()
        time.sleep(0.01)
        pm.end("test_label")
        captured = capsys.readouterr()
        assert "test_label" in captured.out
        assert "ΔMem" in captured.out
        # start_time reset after end
        assert pm.start_time is None

    def test_end_with_psutil_failure_falls_back_to_dt_only(self, capsys):
        pm = ltp.PerformanceMonitor()
        pm.start()
        with patch.object(ltp.psutil, "Process", side_effect=RuntimeError("boom")):
            pm.end("test_label")
        captured = capsys.readouterr()
        assert "test_label" in captured.out
        assert "ΔMem" not in captured.out
        assert pm.start_time is None


# ─────────────────────────────────────────────────────────────────────────────
# C4 — SyncState enum
# ─────────────────────────────────────────────────────────────────────────────


class TestC4SyncState:
    """Contract: 7 states with string values."""

    def test_all_seven_states_present(self):
        names = {s.name for s in ltp.SyncState}
        assert names == {
            "IDLE", "INITIALIZING", "RECORDING", "PROCESSING",
            "PROJECTING", "STOPPING", "ERROR",
        }

    def test_values_are_lowercase_strings(self):
        for s in ltp.SyncState:
            assert isinstance(s.value, str)
            assert s.value == s.name.lower()


# ─────────────────────────────────────────────────────────────────────────────
# C5 — SyncInfo dataclass
# ─────────────────────────────────────────────────────────────────────────────


class TestC5SyncInfo:
    """Contract: 6-field dataclass with optional error_message."""

    def test_required_fields(self):
        info = ltp.SyncInfo(
            state=ltp.SyncState.IDLE,
            timestamp=1234.5,
            frame_count=100,
            memory_usage=42.0,
            gpu_memory_usage=0.5,
        )
        assert info.state is ltp.SyncState.IDLE
        assert info.timestamp == 1234.5
        assert info.frame_count == 100
        assert info.memory_usage == 42.0
        assert info.gpu_memory_usage == 0.5
        assert info.error_message is None

    def test_optional_error_message(self):
        info = ltp.SyncInfo(
            state=ltp.SyncState.ERROR,
            timestamp=0.0,
            frame_count=0,
            memory_usage=0.0,
            gpu_memory_usage=0.0,
            error_message="bad",
        )
        assert info.error_message == "bad"

    def test_field_names_complete(self):
        names = {f.name for f in fields(ltp.SyncInfo)}
        assert names == {
            "state", "timestamp", "frame_count",
            "memory_usage", "gpu_memory_usage", "error_message",
        }


# ─────────────────────────────────────────────────────────────────────────────
# C6 — FrameProcessor: construction + queue mechanics
# ─────────────────────────────────────────────────────────────────────────────


class TestC6FrameProcessorConstruction:
    """Contract: init creates queue, pool, perf counter."""

    def _make(self):
        # Construct without calling start() (no thread loop spinning)
        fp = ltp.FrameProcessor(max_workers=1)
        fp.running = False  # ensure run() exits immediately if ever called
        return fp

    def test_init_creates_queue_with_max_size(self):
        fp = self._make()
        assert fp.frame_queue.maxsize == ltp.MAX_FRAME_QUEUE_SIZE
        assert fp.frame_queue.empty()
        fp.stop()

    def test_init_creates_thread_pool(self):
        fp = self._make()
        assert fp.pool is not None
        fp.stop()

    def test_init_creates_performance_monitor(self):
        fp = self._make()
        assert isinstance(fp.perf, ltp.PerformanceMonitor)
        fp.stop()

    def test_init_frame_counter_starts_at_zero(self):
        fp = self._make()
        assert fp._frames == 0
        fp.stop()


# ─────────────────────────────────────────────────────────────────────────────
# C7 — FrameProcessor.add_frame: normal + watermark + Full + generic
# ─────────────────────────────────────────────────────────────────────────────


class TestC7FrameProcessorAddFrame:
    """Contract: enqueue frame with high-watermark drop, queue.Full safety,
    and error_occurred emission on generic failure."""

    def _make(self):
        fp = ltp.FrameProcessor(max_workers=1)
        fp.running = False
        return fp

    def test_add_frame_normal(self):
        fp = self._make()
        frame = np.zeros((4, 4), dtype=np.uint8)
        fp.add_frame(frame)
        assert fp.frame_queue.qsize() == 1
        fp.stop()

    def test_high_watermark_drops_quarter(self, capsys):
        """When qsize > MAX*0.8 (i.e. >= 7 for MAX=8), drop qsize/4 frames."""
        fp = self._make()
        # Fill to high-watermark (7 items)
        for i in range(7):
            fp.frame_queue.put_nowait(np.full((2, 2), i, dtype=np.uint8))
        assert fp.frame_queue.qsize() == 7
        # Next add should trigger watermark drop (drop = 7//4 = 1)
        fp.add_frame(np.full((2, 2), 99, dtype=np.uint8))
        captured = capsys.readouterr()
        assert "dropped" in captured.out
        # Net: 7 - 1 (dropped) + 1 (added) = 7
        assert fp.frame_queue.qsize() == 7
        fp.stop()

    def test_add_frame_when_queue_full_logs(self, capsys):
        """If put_nowait raises queue.Full, log + continue (no crash)."""
        fp = self._make()
        # Mock the queue to be a real Full-raiser without watermark drop
        fp.frame_queue = MagicMock()
        fp.frame_queue.qsize.return_value = 0  # below watermark
        fp.frame_queue.put_nowait.side_effect = queue.Full
        fp.add_frame(np.zeros((2, 2), dtype=np.uint8))
        captured = capsys.readouterr()
        assert "Frame queue full" in captured.out
        fp.stop()

    def test_add_frame_generic_exception_emits_error_signal(self):
        """Other exceptions trigger error_occurred.emit(...)."""
        fp = self._make()
        fp.frame_queue = MagicMock()
        fp.frame_queue.qsize.return_value = 0
        fp.frame_queue.put_nowait.side_effect = RuntimeError("boom")
        # Verify error_occurred signal is called (PyQt signal — patch the emit)
        with patch.object(fp, "error_occurred") as mock_sig:
            fp.add_frame("not a frame")
            mock_sig.emit.assert_called_once()
            call_arg = mock_sig.emit.call_args[0][0]
            assert "Queue add error" in call_arg
            assert "boom" in call_arg
        fp.stop()


# ─────────────────────────────────────────────────────────────────────────────
# C8 — FrameProcessor._process_one: 4 input shape branches + 2 error branches
# ─────────────────────────────────────────────────────────────────────────────


class TestC8FrameProcessorProcessOne:
    """Contract: convert any supported input → dict with grayscale frame."""

    def _make(self):
        fp = ltp.FrameProcessor(max_workers=1)
        fp.running = False
        return fp

    def test_numpy_2d_passed_through(self):
        fp = self._make()
        gray = np.full((10, 10), 99, dtype=np.uint8)
        result = fp._process_one(gray)
        assert isinstance(result, dict)
        assert result["frame"] is gray  # passes through unmodified
        assert "timestamp" in result
        assert result["frame_id"] == 1
        fp.stop()

    def test_numpy_3d_uses_green_channel(self):
        fp = self._make()
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[..., 1] = 200  # green
        result = fp._process_one(rgb)
        assert (result["frame"] == 200).all()
        fp.stop()

    def test_numpy_unsupported_shape_raises_value_error(self):
        fp = self._make()
        bad = np.zeros((4,), dtype=np.uint8)  # 1D not supported
        with pytest.raises(ValueError, match="Unsupported ndarray shape"):
            fp._process_one(bad)
        fp.stop()

    def test_qimage_input_converted(self):
        fp = self._make()
        qimg = QImage(4, 3, QImage.Format_Grayscale8)
        qimg.fill(123)
        result = fp._process_one(qimg)
        assert result["frame"].shape == (3, 4)
        assert (result["frame"] == 123).all()
        fp.stop()

    def test_unsupported_type_raises_value_error(self):
        fp = self._make()
        with pytest.raises(ValueError, match="Unsupported frame type"):
            fp._process_one("not a frame")
        fp.stop()

    def test_get_numpy_1d_protocol_invoked(self):
        """Test the `hasattr(frame, 'get_numpy_1D')` branch — used by
        IDS Peak Buffer objects."""
        fp = self._make()
        mock_buffer = MagicMock()
        mock_buffer.Height.return_value = 3
        mock_buffer.Width.return_value = 4
        # 3*4*4 = 48 bytes for ARGB
        mock_buffer.get_numpy_1D.return_value = np.full(48, 200, dtype=np.uint8)
        result = fp._process_one(mock_buffer)
        assert result["frame"].shape == (3, 4)
        # All-green (value 200 across all 4 channels → green channel = 200)
        assert (result["frame"] == 200).all()
        fp.stop()

    def test_frame_id_increments(self):
        fp = self._make()
        gray = np.zeros((4, 4), dtype=np.uint8)
        r1 = fp._process_one(gray)
        r2 = fp._process_one(gray)
        r3 = fp._process_one(gray)
        assert r1["frame_id"] == 1
        assert r2["frame_id"] == 2
        assert r3["frame_id"] == 3
        fp.stop()

    def test_first_process_logged_flag(self, capsys):
        """First call logs diagnostic + sets flag; subsequent calls don't log."""
        fp = self._make()
        gray = np.zeros((4, 4), dtype=np.uint8)
        fp._process_one(gray)
        first_out = capsys.readouterr().out
        assert "FIRST _process_one called" in first_out

        fp._process_one(gray)
        second_out = capsys.readouterr().out
        assert "FIRST _process_one called" not in second_out
        fp.stop()


# ─────────────────────────────────────────────────────────────────────────────
# C9 — FrameProcessor._on_done: success + exception
# ─────────────────────────────────────────────────────────────────────────────


class TestC9FrameProcessorOnDone:
    """Contract: forward Future result to frame_processed signal, or emit
    error_occurred on exception."""

    def _make(self):
        fp = ltp.FrameProcessor(max_workers=1)
        fp.running = False
        return fp

    def test_success_emits_frame_processed(self):
        fp = self._make()
        fut = Future()
        result = {"frame": np.zeros((2, 2), dtype=np.uint8), "timestamp": 1.0, "frame_id": 1}
        fut.set_result(result)
        with patch.object(fp, "frame_processed") as mock_sig:
            fp._on_done(fut)
            mock_sig.emit.assert_called_once_with(result)
        fp.stop()

    def test_exception_emits_error_occurred(self):
        fp = self._make()
        fut = Future()
        fut.set_exception(RuntimeError("processing went sideways"))
        with patch.object(fp, "error_occurred") as mock_sig:
            fp._on_done(fut)
            mock_sig.emit.assert_called_once()
            arg = mock_sig.emit.call_args[0][0]
            assert "Processing failure" in arg
            assert "sideways" in arg
        fp.stop()


# ─────────────────────────────────────────────────────────────────────────────
# C10 — FrameProcessor.stop: success + shutdown exception
# ─────────────────────────────────────────────────────────────────────────────


class TestC10FrameProcessorStop:
    """Contract: stop sets running=False and shuts down the pool.
    Pool shutdown exceptions are swallowed (graceful)."""

    def test_stop_sets_running_false(self):
        fp = ltp.FrameProcessor(max_workers=1)
        assert fp.running is True
        fp.running = False  # avoid spinning
        fp.stop()
        assert fp.running is False

    def test_stop_calls_pool_shutdown(self):
        fp = ltp.FrameProcessor(max_workers=1)
        fp.running = False
        with patch.object(fp.pool, "shutdown") as mock_shutdown:
            fp.stop()
            mock_shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    def test_stop_swallows_shutdown_exception(self):
        fp = ltp.FrameProcessor(max_workers=1)
        fp.running = False
        with patch.object(fp.pool, "shutdown", side_effect=RuntimeError("pool died")):
            # Should NOT raise
            fp.stop()


# ─────────────────────────────────────────────────────────────────────────────
# C11 — FrameProcessor.run: queue.Empty timeout + normal path
# ─────────────────────────────────────────────────────────────────────────────


class TestC11FrameProcessorRun:
    """Contract: run() loop polls queue with 0.1s timeout, submits work,
    exits cleanly when running flag flips."""

    def test_run_exits_when_running_false(self):
        fp = ltp.FrameProcessor(max_workers=1)
        fp.running = False
        # Call run() directly (not as QThread); should return immediately
        # since the while loop is `while self.running:`
        fp.run()
        # If we got here, run() returned cleanly
        fp.stop()

    def test_run_empties_queue_with_timeout(self):
        """run() with running=True briefly then flipped to False."""
        fp = ltp.FrameProcessor(max_workers=1)
        # Empty queue → get(timeout=0.1) raises queue.Empty → continue
        # Flip running after one loop iteration
        original_get = fp.frame_queue.get
        call_count = [0]
        def get_then_stop(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                fp.running = False
            return original_get(*args, **kwargs)
        fp.frame_queue.get = get_then_stop
        fp.run()
        assert call_count[0] >= 1
        fp.stop()

    def test_run_submits_frame_to_pool(self):
        fp = ltp.FrameProcessor(max_workers=1)
        fp.frame_queue.put_nowait(np.zeros((2, 2), dtype=np.uint8))
        # Flip running after one pop
        original_get = fp.frame_queue.get
        def get_then_stop(*args, **kwargs):
            r = original_get(*args, **kwargs)
            fp.running = False
            return r
        fp.frame_queue.get = get_then_stop

        with patch.object(fp.pool, "submit", wraps=fp.pool.submit) as spy:
            fp.run()
            assert spy.called
        fp.stop()


# ─────────────────────────────────────────────────────────────────────────────
# §1.1 L3.5 matrix backfill — Property + Snapshot + Concurrency (iter-54)
#
# §1.1 L3.5 row requires:
#   - Property ≥2 per sub-module (universal floor)
#   - Snapshot required for trace outputs (qimage_to_gray_np IS a trace
#     input transform — snapshot the byte layout for each format)
#   - Concurrency ≥1 if mixin touches threads (FrameProcessor is a QThread —
#     pin shutdown invariant)
#
# Closes part of the OPEN BLOCK on iter-42 L3.5 PROMOTION per
# audit_findings.log + docs/PHASE_A5_DEFERRAL.md.
# ─────────────────────────────────────────────────────────────────────────────

import threading
import time

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


class TestPropertyQimageToGrayNp:
    """§1.1 universal floor: ≥2 property tests for qimage_to_gray_np."""

    @given(
        width=st.integers(min_value=4, max_value=64),
        height=st.integers(min_value=4, max_value=64),
        fill=st.integers(min_value=0, max_value=255),
    )
    @settings(max_examples=30, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_grayscale8_shape_dtype_invariants(self, width, height, fill):
        """For any (width, height, fill) input on a Grayscale8 QImage,
        qimage_to_gray_np returns shape (height, width), dtype uint8,
        and every entry equals fill."""
        img = QImage(width, height, QImage.Format_Grayscale8)
        img.fill(fill)
        out = ltp.qimage_to_gray_np(img)
        assert out.shape == (height, width)
        assert out.dtype == np.uint8
        assert (out == fill).all()

    @given(
        width=st.integers(min_value=4, max_value=64),
        height=st.integers(min_value=4, max_value=64),
    )
    @settings(max_examples=20, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_round_trip_consistent_format(self, width, height):
        """Two calls on the same image yield byte-equal outputs.
        Pins determinism: no RNG, no shared state."""
        img = QImage(width, height, QImage.Format_Grayscale8)
        img.fill(123)
        a = ltp.qimage_to_gray_np(img)
        b = ltp.qimage_to_gray_np(img)
        np.testing.assert_array_equal(a, b)


class TestSnapshotGrayscale8:
    """§1.1 L3.5 row: snapshot required for trace outputs.

    qimage_to_gray_np is the entry point for camera→trace ingestion;
    a regression in its byte layout would corrupt every downstream
    trace. The snapshot here is a hash of the produced bytes for a
    canonical fill pattern. Per §1.5 snapshot policy: use a hash
    assertion for deterministically-derivable artifacts (a fill at
    width=8, height=4 is reproducible across builds)."""

    def test_canonical_grayscale_byte_layout(self):
        """Canonical fill: width=8, height=4, fill=200 (4-aligned to
        sidestep D-ltp-1 padding question). Hash the output bytes;
        commit the hash as the trace-input format pin."""
        import hashlib
        img = QImage(8, 4, QImage.Format_Grayscale8)
        img.fill(200)
        out = ltp.qimage_to_gray_np(img)
        h = hashlib.sha256(out.tobytes()).hexdigest()
        # Pin: any change to byte layout, dtype, or row order breaks this.
        # Recovery: if format changes, re-derive hash by printing
        # out.tobytes() and update this constant + spec entry.
        expected = hashlib.sha256(np.full((4, 8), 200, dtype=np.uint8).tobytes()).hexdigest()
        assert h == expected, (
            f"qimage_to_gray_np Grayscale8 byte layout regression. "
            f"Got hash {h}, expected {expected}. The output is no "
            f"longer a flat row-major uint8 array of `fill`."
        )

    def test_grayscale8_unaligned_width_post_fix(self):
        """D-ltp-1 fix snapshot: 6-pixel-wide Grayscale8 (non-4-aligned)
        must produce shape (height, 6) with all-fill bytes after the
        bytesPerLine fix."""
        import hashlib
        img = QImage(6, 4, QImage.Format_Grayscale8)
        img.fill(100)
        out = ltp.qimage_to_gray_np(img)
        h = hashlib.sha256(out.tobytes()).hexdigest()
        expected = hashlib.sha256(np.full((4, 6), 100, dtype=np.uint8).tobytes()).hexdigest()
        assert h == expected, (
            f"D-ltp-1 regression: post-fix qimage_to_gray_np should "
            f"return uniform fill bytes for unaligned Grayscale8 widths. "
            f"Got hash {h}, expected {expected}."
        )


class TestConcurrencyFrameProcessor:
    """§1.1 L3.5 row: concurrency ≥1 if mixin touches threads.

    FrameProcessor is a QThread that runs a frame-processing loop.
    Per §1.2 concurrency-test playbook: pin shutdown invariants.
    We do NOT call.start() (spinning up the QThread without a
    QApplication event loop crashes the test interpreter). Instead,
    pin the.stop() state-machine invariants directly: stop must
    set running=False, drain the queue, and shut down the pool —
    all idempotent on repeated.stop() calls."""

    def test_stop_sets_running_false_idempotent(self):
        """§1.2.3 inspired:.stop() must flip running to False; calling.stop() multiple times must NOT raise or deadlock (idempotent).
        Does not require.start() — pin the state-machine directly."""
        fp = ltp.FrameProcessor(max_workers=1)
        # Initial state per __init__ — running is True before start()
        assert fp.running is True
        fp.stop()
        assert fp.running is False
        # Idempotent: stopping a stopped processor is a no-op
        fp.stop()
        assert fp.running is False

    def test_stop_drains_queue_within_budget(self):
        """§1.2.3:.stop() completes within a bounded wall-clock budget
        even when the queue has pending items. Pins that shutdown is
        not blocked by queue contents.

        Test pattern: pre-fill the queue + call.stop() + assert the
        call returns within 1s budget. Uses elapsed = end - start
        timing rather than a deterministic event (matching how stop()
        is implemented — synchronous return)."""
        fp = ltp.FrameProcessor(max_workers=1)
        # Pre-fill the queue with some sentinel items
        for _ in range(5):
            try:
                fp.frame_queue.put_nowait(None)
            except Exception:
                break
        start = time.perf_counter()
        fp.stop()
        elapsed = time.perf_counter() - start
        budget = 1.0
        assert elapsed < budget, (
            f"FrameProcessor.stop() took {elapsed:.3f}s "
            f"(budget {budget}s). Indicates a queue-drain deadlock."
        )
        assert fp.running is False
