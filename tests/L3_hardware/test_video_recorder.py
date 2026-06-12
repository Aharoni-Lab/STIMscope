"""LIGHT-tier audit pins for `STIMViewer_CRISPI/video_recorder.py`.

Focused on the candidate segfault fix in `_to_numpy`. See
`docs/specs/L3_hardware/video_recorder.md` §1 for the analysis.

The full module is NOT under audit-grade test coverage — see the
spec's LIGHT-tier rationale. These tests pin the one invariant the
candidate fix introduces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def stimviewer_path():
    return (
        Path(__file__).resolve().parent.parent.parent
        / "STIMscope"
        / "STIMViewer_CRISPI"
    )


@pytest.fixture
def video_recorder_module(monkeypatch, stimviewer_path):
    """Import video_recorder fresh."""
    monkeypatch.syspath_prepend(str(stimviewer_path))
    # video_recorder imports `cv2` at module level; cv2 is real in docker.
    # PyQt5 not needed (recorder doesn't import it).
    sys.modules.pop("video_recorder", None)
    import importlib
    return importlib.import_module("video_recorder")


class _FakeVendorFrame:
    """IDS-Peak-like buffer wrapper for testing _to_numpy.

    Holds a mutable numpy buffer that simulates the SDK recycling
    memory mid-write. After ``recycle()`` is called the buffer's
    bytes are zeroed — if the caller held a VIEW (not a copy), they'd
    see zeros and a real-world segfault could happen during async
    writes.
    """

    def __init__(self, h: int, w: int, fill_value: int = 200):
        self._w = w
        self._h = h
        self._buf = np.full((h, w), fill_value, dtype=np.uint8)

    def Width(self):
        return self._w

    def Height(self):
        return self._h

    def get_numpy_2D(self):
        return self._buf  # returns the BACKING array, not a copy

    def get_numpy_1D(self):
        return self._buf.ravel()

    def recycle(self):
        """Simulate the SDK overwriting its buffer after a frame is
        published. Zeroes the backing memory in place — if our writer
        kept a view, the next read sees zeros."""
        self._buf.fill(0)


# ─────────────────────────────────────────────────────────────────────────────
# Segfault candidate-fix regression (Hypothesis #1: buffer aliasing)
# ─────────────────────────────────────────────────────────────────────────────


class TestSegfaultFixHypothesis1BufferAliasing:
    """Pin that `_to_numpy` returns a copy independent of the source
    buffer. PRE-FIX it returned a view; POST-FIX it copies.

    Repro of the segfault scenario:
      1. Camera thread publishes a buffer (FakeVendorFrame with fill=200)
      2. Recorder's _to_numpy is called (writer thread side)
      3. Camera-side SDK recycles the buffer (fill=0)
      4. Writer holds a copy → still sees 200; original is gone but
         our memory is safe → no segfault.

    If _to_numpy ever regresses to `copy=False` semantics, this test
    fails and the segfault risk returns.
    """

    def test_to_numpy_shaped_getter_returns_independent_copy(
        self, video_recorder_module
    ):
        VideoRecorder = video_recorder_module.VideoRecorder
        frame = _FakeVendorFrame(h=480, w=640, fill_value=200)
        arr = VideoRecorder._to_numpy(frame)
        assert arr is not None
        assert arr.shape == (480, 640)
        # Verify the array is a COPY by mutating the source
        frame.recycle()  # zeros the SDK buffer
        # The recorder's copy must be unchanged
        assert arr[0, 0] == 200, (
            "REGRESSION: _to_numpy returned a view, not a copy. "
            "Buffer aliasing risk returns — see "
            "docs/specs/L3_hardware/video_recorder.md §1 Hypothesis #1."
        )
        assert arr.mean() == 200.0

    def test_to_numpy_1d_getter_returns_independent_copy(
        self, video_recorder_module
    ):
        """Same invariant for the 1D-getter fallback path."""
        VideoRecorder = video_recorder_module.VideoRecorder

        # Build a frame that only has get_numpy_1D (no shaped getter).
        class _Vendor1DOnly:
            def __init__(self, h, w, fill):
                self._h = h
                self._w = w
                self._buf = np.full((h * w,), fill, dtype=np.uint8)

            def Width(self):
                return self._w

            def Height(self):
                return self._h

            def get_numpy_1D(self):
                return self._buf

            def recycle(self):
                self._buf.fill(0)

        frame = _Vendor1DOnly(h=240, w=320, fill=128)
        arr = VideoRecorder._to_numpy(frame)
        assert arr is not None
        assert arr.shape == (240, 320)
        frame.recycle()
        assert arr[0, 0] == 128, (
            "REGRESSION: 1D-getter fallback path returned a view, not "
            "a copy. See video_recorder.md §1 Hypothesis #1."
        )

    def test_to_numpy_numpy_array_passthrough_unchanged(
        self, video_recorder_module
    ):
        """When the input is already a numpy array, _to_numpy returns
        it as-is (line 215: `if isinstance(frame, np.ndarray): return frame`).
        This is the simulation path; not affected by the copy fix
        because no SDK buffer is involved.
        """
        VideoRecorder = video_recorder_module.VideoRecorder
        src = np.full((100, 100), 77, dtype=np.uint8)
        arr = VideoRecorder._to_numpy(src)
        assert arr is src  # documented passthrough


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis #2 mitigation: video_writer is None'd after writer-loop close
# ─────────────────────────────────────────────────────────────────────────────


class TestHypothesis2WriterNulledAfterClose:
    """The writer-loop's finally block must set `self.video_writer = None`
    immediately after closing, so cleanup() can't double-close the same
    TiffWriter.
    """

    def test_video_writer_none_after_loop_exit(self, video_recorder_module):
        """We can't easily run the whole writer loop in a unit test (it
        needs a TiffWriter + queue + threading). Instead, source-pin
        that the finally block contains `self.video_writer = None`
        AFTER the close() call.
        """
        import inspect
        VideoRecorder = video_recorder_module.VideoRecorder
        src = inspect.getsource(VideoRecorder._writer_loop)
        # Look for the finally pattern: close then None
        # (the exact line ordering matters for the mitigation)
        finally_block = src.split("finally:")[-1]
        close_pos = finally_block.find("self.video_writer.close()")
        none_pos = finally_block.find("self.video_writer = None")
        assert close_pos != -1, "writer-loop finally must call close()"
        assert none_pos != -1, "writer-loop finally must null out video_writer"
        assert none_pos > close_pos, (
            "Hypothesis #2 mitigation: video_writer must be None'd AFTER "
            "close() so cleanup()'s redundant close is a no-op."
        )
