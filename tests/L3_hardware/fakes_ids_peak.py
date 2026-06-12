"""In-memory test double for ``IDSPeakBackend``.

Stage 5a.2 of L3 camera.py audit. Pairs with the
``IDSPeakBackend`` Protocol in
``STIMscope/STIMViewer_CRISPI/ids_peak_backend.py``.

``FakeIDSPeakBackend`` exposes the same surface as ``IDSPeakSDKBackend``
but holds:

  - a dict-backed in-memory NodeMap
  - a deterministic synthetic frame generator (seeded RNG)
  - an internal queue of "buffered" frames
  - lifecycle flags + telemetry the tests assert on

It emits NO real I/O — every method is pure in-memory state mutation
plus numpy/path operations. Tests can construct it in microseconds
and run thousands of cases without a real camera.

Scripted-behavior hooks:

  - ``force_timeout_next`` — next ``wait_for_frame`` returns None
  - ``force_node_access_error`` set — those node names raise
    IDSPeakNodeError on get_node_value, return False on
    set_node_value / execute_node / node_access_writable
  - ``force_not_writable`` set — those node names return False on
    set_node_value but still succeed on get_node_value

Telemetry (tests assert on these):

  - ``calls: List[Tuple[str, tuple, dict]]`` — every method invocation
    with positional + keyword args, in call order
  - ``requeue_count: int`` — how many times ``requeue_frame`` was
    invoked
  - ``write_png_calls: List[Tuple[str, Tuple[int, int]]]`` — (path,
    (H, W)) for every ``write_frame_png`` call

Thread safety: the production backend is single-acquisition-thread +
multi-reader. The fake guards mutable state with an RLock so
concurrent tests don't race the telemetry lists.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

# Add the production module to sys.path so the Protocol + types
# are importable from the test directory.
_CRISPI = Path(__file__).resolve().parents[2] / "STIMscope" / "STIMViewer_CRISPI"
if str(_CRISPI) not in sys.path:
    sys.path.insert(0, str(_CRISPI))

from ids_peak_backend import (  # type: ignore  # noqa: E402
    FrameHandle,
    IDSPeakBackend,
    IDSPeakNodeError,
    PixelFormat,
)


# ─────────────────────────────────────────────────────────────────────
# Default NodeMap — covers nodes camera.py asks for in normal operation
# ─────────────────────────────────────────────────────────────────────


_DEFAULT_FAKE_NODEMAP: Dict[str, Any] = {
    # Acquisition
    "AcquisitionFrameRate": 30.0,
    "AcquisitionFrameRateMax": 60.0,
    "AcquisitionMode": "Continuous",
    "AcquisitionStart": None,  # command nodes
    "AcquisitionStop": None,
    # Frame size
    "Width": 1936,
    "Height": 1096,
    "PayloadSize": 1936 * 1096 * 1,  # MONO8
    # Gain
    "Gain": 1.0,
    "GainMax": 4.0,
    "DigitalGainAll": 1.0,
    # Trigger
    "TriggerMode": "Off",
    "TriggerSource": "Line0",
    "TriggerActivation": "RisingEdge",
    "TriggerDelay": 0.0,
    "ExposureTime": 33333.333,
    "LineSelector": "Line0",
    "LineMode": "Input",
    # Pixel format
    "PixelFormat": "Mono8",
}


# ─────────────────────────────────────────────────────────────────────
# Frame container — wraps a numpy array as an opaque FrameHandle
# ─────────────────────────────────────────────────────────────────────


class _FakeFrame:
    """Opaque container the fake uses as FrameHandle.

    Holds the synthesized ndarray + a sentinel ``requeued`` flag so
    double-requeue can be detected by tests. Production callers
    treat this as an opaque object (never introspect).
    """

    __slots__ = ("_array", "requeued")

    def __init__(self, array: np.ndarray) -> None:
        self._array = array
        self.requeued = False

    def as_array(self) -> np.ndarray:
        return self._array

    def __repr__(self) -> str:
        h, w = self._array.shape[:2]
        return f"_FakeFrame({h}x{w}, requeued={self.requeued})"


# ─────────────────────────────────────────────────────────────────────
# FakeIDSPeakBackend
# ─────────────────────────────────────────────────────────────────────


class FakeIDSPeakBackend:
    """In-memory implementation of ``IDSPeakBackend`` for L3 camera tests.

    Construct with default settings::

        backend = FakeIDSPeakBackend()
        backend.open()
        # backend is now open + ready to serve frames

    Override the nodemap::

        backend = FakeIDSPeakBackend(
            nodemap_defaults={"Width": 320, "Height": 240,
                              "PayloadSize": 320 * 240},
        )

    Force scripted failure::

        backend = FakeIDSPeakBackend()
        backend.force_timeout_next = True
        assert backend.wait_for_frame(timeout_ms=100) is None

    The backend's behavior is otherwise identical to the production
    one: same Protocol surface, same idempotence rules, same error
    contract.
    """

    def __init__(
        self,
        frame_shape: Tuple[int, int] = (1096, 1936),
        nodemap_defaults: Optional[Mapping[str, Any]] = None,
        frame_seed: int = 42,
    ) -> None:
        self._frame_shape = frame_shape
        self._nodemap: Dict[str, Any] = dict(_DEFAULT_FAKE_NODEMAP)
        if nodemap_defaults is not None:
            self._nodemap.update(nodemap_defaults)
        # Default nodemap is 1936x1096 — let constructor frame_shape win.
        self._nodemap["Width"] = frame_shape[1]
        self._nodemap["Height"] = frame_shape[0]
        self._nodemap["PayloadSize"] = frame_shape[0] * frame_shape[1]

        self._rng = np.random.default_rng(frame_seed)
        self._open = False
        self._acquiring = False
        self._current_format: PixelFormat = PixelFormat.MONO8
        self._supported_formats: Tuple[PixelFormat,...] = tuple(PixelFormat)

        # Scripted-behavior hooks
        self.force_timeout_next: bool = False
        self.force_node_access_error: Set[str] = set()
        self.force_not_writable: Set[str] = set()

        # Telemetry
        self.calls: List[Tuple[str, tuple, dict]] = []
        self.requeue_count: int = 0
        self.write_png_calls: List[Tuple[str, Tuple[int, int]]] = []

        # Thread safety
        self._lock = threading.RLock()

    def _record(self, method: str, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self.calls.append((method, args, kwargs))

    # ─── Lifecycle ────────────────────────────────────────────────

    def open(self) -> None:
        self._record("open")
        with self._lock:
            self._open = True

    def close(self) -> None:
        self._record("close")
        with self._lock:
            self._open = False
            self._acquiring = False

    @property
    def is_open(self) -> bool:
        return self._open

    # ─── NodeMap ──────────────────────────────────────────────────

    def get_node_value(self, name: str) -> Any:
        self._record("get_node_value", name)
        if name in self.force_node_access_error:
            raise IDSPeakNodeError(f"forced access error on {name!r}")
        with self._lock:
            if name not in self._nodemap:
                raise IDSPeakNodeError(f"node {name!r} not found in fake")
            return self._nodemap[name]

    def set_node_value(self, name: str, value: Any) -> bool:
        self._record("set_node_value", name, value)
        if name in self.force_node_access_error:
            return False
        if name in self.force_not_writable:
            return False
        with self._lock:
            if name not in self._nodemap:
                raise IDSPeakNodeError(f"node {name!r} not found in fake")
            self._nodemap[name] = value
            # PayloadSize stays consistent with Width × Height
            if name in ("Width", "Height"):
                w = self._nodemap.get("Width", 0)
                h = self._nodemap.get("Height", 0)
                self._nodemap["PayloadSize"] = w * h
                self._frame_shape = (h, w)
            return True

    def execute_node(self, name: str) -> bool:
        self._record("execute_node", name)
        if name in self.force_node_access_error:
            return False
        with self._lock:
            if name not in self._nodemap:
                raise IDSPeakNodeError(f"command node {name!r} not found in fake")
            return True

    def node_access_writable(self, name: str) -> bool:
        self._record("node_access_writable", name)
        if name in self.force_node_access_error:
            return False
        if name in self.force_not_writable:
            return False
        with self._lock:
            return name in self._nodemap

    # ─── Acquisition ──────────────────────────────────────────────

    def start_acquisition(self) -> None:
        self._record("start_acquisition")
        with self._lock:
            self._acquiring = True

    def stop_acquisition(self) -> None:
        self._record("stop_acquisition")
        with self._lock:
            self._acquiring = False

    def flush_discard_all(self) -> None:
        self._record("flush_discard_all")
        # No-op for the fake — there's no persistent queue to drain.

    @property
    def is_acquiring(self) -> bool:
        return self._acquiring

    # ─── Frame I/O ────────────────────────────────────────────────

    def wait_for_frame(self, timeout_ms: int) -> Optional[FrameHandle]:
        self._record("wait_for_frame", timeout_ms)
        if self.force_timeout_next:
            self.force_timeout_next = False
            return None
        if not self._open or not self._acquiring:
            return None
        with self._lock:
            h, w = self._frame_shape
            # Deterministic synthetic frame: low-frequency gradient + noise
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            base = ((x + y) // 8) & 0xFF
            noise = self._rng.integers(0, 16, size=(h, w))
            frame_data = ((base + noise) & 0xFF).astype(np.uint8)
            return FrameHandle(_FakeFrame(frame_data))

    def requeue_frame(self, frame: FrameHandle) -> None:
        self._record("requeue_frame")
        if frame is None:
            return
        # Cast back from FrameHandle to _FakeFrame; in tests we know
        # the concrete type
        if isinstance(frame, _FakeFrame):
            frame.requeued = True
        with self._lock:
            self.requeue_count += 1

    def frame_to_ndarray(
        self,
        frame: FrameHandle,
        dest_format: PixelFormat,
    ) -> np.ndarray:
        self._record("frame_to_ndarray", dest_format)
        if not isinstance(frame, _FakeFrame):
            raise TypeError(f"expected _FakeFrame, got {type(frame).__name__}")
        arr = frame.as_array().copy()
        # Map to dest_format dimensions
        if dest_format == PixelFormat.MONO8:
            return arr  # 2D uint8
        elif dest_format in (PixelFormat.BGR8, PixelFormat.RGB8):
            return np.stack([arr, arr, arr], axis=-1)  # (H, W, 3)
        elif dest_format in (PixelFormat.BGRA8, PixelFormat.RGBA8):
            alpha = np.full_like(arr, 255)
            return np.stack([arr, arr, arr, alpha], axis=-1)  # (H, W, 4)
        else:  # pragma: no cover
            raise ValueError(f"unsupported dest_format {dest_format!r}")

    def write_frame_png(self, path: str, frame: FrameHandle) -> bool:
        self._record("write_frame_png", path)
        if not isinstance(frame, _FakeFrame):
            return False
        h, w = frame.as_array().shape[:2]
        with self._lock:
            self.write_png_calls.append((path, (h, w)))
        # Don't actually write the file — telemetry is enough for
        # tests. If a test wants a real file on disk it can mock
        # cv2.imwrite at the call site instead.
        return True

    # ─── Pixel format ─────────────────────────────────────────────

    def supported_dest_formats(self) -> Sequence[PixelFormat]:
        self._record("supported_dest_formats")
        return self._supported_formats

    def set_dest_format(self, fmt: PixelFormat) -> None:
        self._record("set_dest_format", fmt)
        with self._lock:
            self._current_format = fmt

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._frame_shape

    @property
    def current_format(self) -> PixelFormat:
        return self._current_format


# ─────────────────────────────────────────────────────────────────────
# Self-tests on the fake (run via pytest tests/L3_hardware/fakes_ids_peak.py)
# ─────────────────────────────────────────────────────────────────────


def _verify_protocol_conformance() -> bool:
    """Static check that FakeIDSPeakBackend conforms to IDSPeakBackend."""
    fake = FakeIDSPeakBackend()
    return isinstance(fake, IDSPeakBackend)


if __name__ == "__main__":
    assert _verify_protocol_conformance(), "FakeIDSPeakBackend doesn't conform!"
    print("FakeIDSPeakBackend ✓ conforms to IDSPeakBackend Protocol")

    # Quick smoke
    fake = FakeIDSPeakBackend(frame_shape=(240, 320))
    fake.open()
    assert fake.is_open
    fake.start_acquisition()
    assert fake.is_acquiring
    h = fake.wait_for_frame(100)
    assert h is not None
    arr = fake.frame_to_ndarray(h, PixelFormat.MONO8)
    assert arr.shape == (240, 320)
    fake.requeue_frame(h)
    assert fake.requeue_count == 1
    fake.stop_acquisition()
    fake.close()
    print("Smoke test ✓")
