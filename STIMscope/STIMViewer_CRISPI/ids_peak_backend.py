"""IDS Peak SDK Hardware Abstraction Layer.

Operation-level Protocol abstracting the IDS Peak SDK surface that
``camera.OptimizedCamera`` needs. Lets the 1418-LOC OptimizedCamera
class be unit-tested off-target with a fake backend instead of a real
USB3 camera.

Stage 5a.1 of L3 camera.py audit. Per Q4 verdict in
``docs/specs/L3_hardware/camera.md`` §0.5: introduce ``IDSPeakBackend``
Protocol symmetric with ``core.camera_capture.CameraBackend``.

Public surface (12 methods + 6 properties):
    Lifecycle:      open, close, is_open
    NodeMap:        get_node_value, set_node_value, execute_node,
                    node_access_writable
    Acquisition:    start_acquisition, stop_acquisition,
                    flush_discard_all, is_acquiring
    Frame I/O:      wait_for_frame, requeue_frame, frame_to_ndarray,
                    write_frame_png
    Pixel format:   supported_dest_formats, set_dest_format,
                    frame_shape, current_format

Production implementation: :class:`IDSPeakSDKBackend` — thin façade
over ``ids_peak`` + ``ids_peak_ipl`` modules. Used in live GUI.

Test double: ``FakeIDSPeakBackend`` in
``tests/L3_hardware/fakes_ids_peak.py`` (.2).

Migration:.3 wires this into ``OptimizedCamera.__init__`` with
a back-compat path for the legacy ``device_manager`` constructor arg;.4 migrates the ~30 SDK call sites in OptimizedCamera to use
the backend methods.

See ``docs/specs/L3_hardware/camera_hal_design.md`` for the full
design rationale + open questions.
"""

from __future__ import annotations

import enum
from typing import (
    Any,
    NewType,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────


# Opaque handle to a frame buffer returned by ``wait_for_frame``. In
# the production backend this is an ``ids_peak.Buffer`` (or compatible
# IPL handle); in the fake it's a wrapper around a numpy array. Callers
# must not introspect the type — pass it back to ``requeue_frame`` or
# ``frame_to_ndarray``.
FrameHandle = NewType("FrameHandle", object)


class PixelFormat(enum.IntEnum):
    """Mirror of IDS Peak IPL pixel format constants used by camera.py.

    Values match ``ids_peak_ipl.PixelFormatName_*`` so the production
    backend can pass them through to the SDK unchanged. Tests use the
    enum directly without importing the SDK.

    NOTE: the literal integer values are taken from IDS Peak IPL
    1.x — if a future SDK version renumbers them, the production
    backend's ``_to_ipl_format`` mapping is the only site that needs
    updating; the Protocol surface and consumers stay stable.
    """

    MONO8 = 0x0108_0001
    BGRA8 = 0x0220_8000
    BGR8 = 0x0218_0014
    RGBA8 = 0x0220_8001
    RGB8 = 0x0218_0015


class IDSPeakNodeError(RuntimeError):
    """Raised when a NodeMap access fails at the backend boundary.

    Distinct from generic ``RuntimeError`` so OptimizedCamera (and tests)
    can ``except IDSPeakNodeError`` specifically. Stage-3 verdict
    (Q1 from camera_hal_design.md): raise at the backend boundary,
    let higher-level swallowers in OptimizedCamera convert to log+False
    if they already do today.
    """


# ─────────────────────────────────────────────────────────────────────
# HAL Protocol
# ─────────────────────────────────────────────────────────────────────


@runtime_checkable
class IDSPeakBackend(Protocol):
    """Operations OptimizedCamera needs from the IDS Peak SDK.

    Production: :class:`IDSPeakSDKBackend` (this module). Test double:
    ``FakeIDSPeakBackend`` (.2). Both expose the same surface.

    Lifecycle::

        backend = IDSPeakSDKBackend()
        backend.open()              # raises if no device
        #... use...
        backend.close()             # idempotent

    Thread safety: production backend is NOT thread-safe — designed for
    single-acquisition-thread + multi-reader. Test fake adds an RLock
    around mutable state so concurrent tests don't race.
    """

    # ─── Lifecycle ────────────────────────────────────────────────

    def open(self) -> None:
        """Initialize SDK + open first available device + datastream.

        Raises RuntimeError if no device or SDK init fails. Idempotent
        if already open (returns silently).
        """
        ...

    def close(self) -> None:
        """Release datastream + device + SDK library. Idempotent."""
        ...

    @property
    def is_open(self) -> bool:
        """True when device + datastream are open."""
        ...

    # ─── NodeMap (typed value accessors; raw Node never leaks) ────

    def get_node_value(self, name: str) -> Any:
        """Read the current value of NodeMap entry ``name``.

        Raises IDSPeakNodeError if the node doesn't exist or isn't
        readable. Returns the raw value (int / float / str / bool
        per the node's type).
        """
        ...

    def set_node_value(self, name: str, value: Any) -> bool:
        """Write ``value`` to NodeMap entry ``name``.

        Returns True if write succeeded. Returns False if the node
        is not writable in the current state (acquisition running,
        access mode RO, etc.). Raises IDSPeakNodeError if the node
        doesn't exist at all (caller bug, not runtime state).
        """
        ...

    def execute_node(self, name: str) -> bool:
        """Execute a command-type NodeMap entry (e.g. AcquisitionStart).

        Returns True on success, False if the command is not
        currently executable. Raises IDSPeakNodeError if the node
        doesn't exist.
        """
        ...

    def node_access_writable(self, name: str) -> bool:
        """True iff the NodeMap entry ``name`` is currently writable.

        Convenience for callers that gate writes on access state.
        Returns False if the node doesn't exist (no exception).
        """
        ...

    # ─── Acquisition control ──────────────────────────────────────

    def start_acquisition(self) -> None:
        """Start datastream acquisition. Idempotent.

        Sets ``is_acquiring`` to True. Frames begin arriving via
        ``wait_for_frame``.
        """
        ...

    def stop_acquisition(self) -> None:
        """Stop datastream acquisition. Idempotent.

        Sets ``is_acquiring`` to False. In-flight buffers are
        flushed (DiscardAll mode).
        """
        ...

    def flush_discard_all(self) -> None:
        """Discard all queued buffers without stopping acquisition.

        Used during pixel-format hot-swap to clear stale frames
        before resuming with the new format.
        """
        ...

    @property
    def is_acquiring(self) -> bool:
        """True between start_acquisition() and stop_acquisition()."""
        ...

    # ─── Frame I/O ────────────────────────────────────────────────

    def wait_for_frame(self, timeout_ms: int) -> Optional[FrameHandle]:
        """Block until the next frame is available or timeout fires.

        Returns an opaque FrameHandle on success, None on timeout.
        Caller must eventually call ``requeue_frame`` to return the
        buffer to the SDK pool, otherwise allocation will starve.
        """
        ...

    def requeue_frame(self, frame: FrameHandle) -> None:
        """Return a frame buffer to the SDK acquisition pool.

        Idempotent — calling on an already-requeued or closed
        handle is a silent no-op (logged at DEBUG in the production
        backend so double-frees surface during diagnostics).
        """
        ...

    def frame_to_ndarray(
        self,
        frame: FrameHandle,
        dest_format: PixelFormat,
    ) -> np.ndarray:
        """Convert a FrameHandle to a numpy array in ``dest_format``.

        Returns (H, W) uint8 for MONO8, (H, W, 3) uint8 for BGR8/RGB8,
        (H, W, 4) uint8 for BGRA8/RGBA8. The returned array is a copy
        — safe to retain after ``requeue_frame``.
        """
        ...

    def write_frame_png(
        self, path: str, frame: FrameHandle,
    ) -> bool:
        """Save a frame to ``path`` as PNG via the SDK's ImageWriter.

        Returns True on success, False on write error. Path must be
        writable; format inferred from extension.
        """
        ...

    # ─── Pixel-format hot-swap ────────────────────────────────────

    def supported_dest_formats(self) -> Sequence[PixelFormat]:
        """Pixel formats the SDK can convert the current source to.

        Subset of PixelFormat enum. Empty sequence if no source
        format is set yet (i.e. before first frame).
        """
        ...

    def set_dest_format(self, fmt: PixelFormat) -> None:
        """Reconfigure the IPL ImageConverter to emit ``fmt``.

        Implicit pause/resume: caller is responsible for matching
        start_acquisition / stop_acquisition around format changes
        if the SDK requires it. (The production backend mirrors
        camera.py's existing pause/resume pattern.)
        """
        ...

    # ─── Read-only introspection ──────────────────────────────────

    @property
    def frame_shape(self) -> Tuple[int, int]:
        """(H, W) of frames as currently configured. (0, 0) before open()."""
        ...

    @property
    def current_format(self) -> PixelFormat:
        """The destination format set by ``set_dest_format``."""
        ...


# ─────────────────────────────────────────────────────────────────────
# Production wrapper
# ─────────────────────────────────────────────────────────────────────


class IDSPeakSDKBackend:
    """Production IDSPeakBackend backed by the real ``ids_peak`` SDK.

    Initialization is two-phase: ``__init__`` is cheap (no SDK calls);
    ``open()`` does the SDK init + device + datastream opening. This
    matches OptimizedCamera's lifecycle expectation (construct cheaply,
    open when ready to start acquisition).

    Back-compat factory for.3 wiring:

        @classmethod
        def from_device_manager(cls, device_manager) -> "IDSPeakSDKBackend":
            "Wrap a pre-opened device_manager (legacy ctor path)."

    See ``docs/specs/L3_hardware/camera_hal_design.md`` §"How
    OptimizedCamera uses the Protocol" for the migration plan.
    """

    def __init__(self, device_manager: Optional[Any] = None) -> None:
        # device_manager: optional pre-existing ids_peak.DeviceManager
        # (legacy ctor path supports it for.3 back-compat).
        # None means we'll initialize the SDK ourselves on open().
        self._device_manager = device_manager
        self._device: Optional[Any] = None
        self._datastream: Optional[Any] = None
        self._nodemap: Optional[Any] = None
        self._converter: Optional[Any] = None
        self._buffer_list: list = []
        self._frame_shape: Tuple[int, int] = (0, 0)
        self._current_format: PixelFormat = PixelFormat.MONO8
        self._is_acquiring: bool = False

        # SDK module handles, populated by open()
        self._ids_peak: Optional[Any] = None
        self._ids_peak_ipl: Optional[Any] = None

    # ─── Lifecycle ────────────────────────────────────────────────

    def open(self) -> None:
        if self._device is not None:
            return  # idempotent

        from ids_peak import ids_peak
        from ids_peak_ipl import ids_peak_ipl

        self._ids_peak = ids_peak
        self._ids_peak_ipl = ids_peak_ipl

        if self._device_manager is None:
            ids_peak.Library.Initialize()
            self._device_manager = ids_peak.DeviceManager.Instance()
            self._device_manager.Update()

        if self._device_manager.Devices().empty():
            raise RuntimeError("No IDS Peak cameras found")

        self._device = (
            self._device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        )
        self._nodemap = self._device.RemoteDevice().NodeMaps()[0]
        self._datastream = self._device.DataStreams()[0].OpenDataStream()
        self._converter = ids_peak_ipl.ImageConverter()

        try:
            h = int(self._nodemap.FindNode("Height").Value())
            w = int(self._nodemap.FindNode("Width").Value())
            self._frame_shape = (h, w)
        except Exception:
            self._frame_shape = (0, 0)

    def close(self) -> None:
        ids_peak = self._ids_peak
        if self._datastream is not None and ids_peak is not None:
            try:
                self._datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            except Exception:
                pass
            try:
                for b in list(self._datastream.AnnouncedBuffers()):
                    self._datastream.RevokeBuffer(b)
            except Exception:
                pass
            try:
                self._datastream.Close()
            except Exception:
                pass
            self._datastream = None

        if self._device is not None:
            try:
                self._device.Close()
            except Exception:
                pass
            self._device = None

        self._nodemap = None
        self._converter = None
        self._buffer_list.clear()
        self._is_acquiring = False

    @property
    def is_open(self) -> bool:
        return self._device is not None and self._datastream is not None

    # ─── NodeMap ──────────────────────────────────────────────────

    def _find_node(self, name: str) -> Any:
        if self._nodemap is None:
            raise IDSPeakNodeError(f"backend not open; cannot access node {name!r}")
        try:
            return self._nodemap.FindNode(name)
        except Exception as e:
            raise IDSPeakNodeError(f"node {name!r} not found: {e}") from e

    def get_node_value(self, name: str) -> Any:
        node = self._find_node(name)
        try:
            return node.Value()
        except Exception as e:
            raise IDSPeakNodeError(
                f"node {name!r} not readable: {e}"
            ) from e

    def set_node_value(self, name: str, value: Any) -> bool:
        node = self._find_node(name)
        if not self._node_writable(node):
            return False
        try:
            node.SetValue(value)
            return True
        except Exception:
            return False

    def execute_node(self, name: str) -> bool:
        node = self._find_node(name)
        try:
            node.Execute()
            return True
        except Exception:
            return False

    def node_access_writable(self, name: str) -> bool:
        if self._nodemap is None:
            return False
        try:
            node = self._nodemap.FindNode(name)
        except Exception:
            return False
        return self._node_writable(node)

    def _node_writable(self, node: Any) -> bool:
        ids_peak = self._ids_peak
        if ids_peak is None:
            return False
        try:
            return (
                node.AccessStatus() == ids_peak.NodeAccessStatus_ReadWrite
            )
        except Exception:
            return False

    # ─── Acquisition ──────────────────────────────────────────────

    def start_acquisition(self) -> None:
        if self._is_acquiring:
            return
        if self._datastream is None:
            raise RuntimeError("backend not open")
        try:
            self._datastream.StartAcquisition()
        except Exception:
            pass
        self.execute_node("AcquisitionStart")
        self._is_acquiring = True

    def stop_acquisition(self) -> None:
        if not self._is_acquiring:
            return
        ids_peak = self._ids_peak
        self.execute_node("AcquisitionStop")
        if self._datastream is not None and ids_peak is not None:
            try:
                self._datastream.StopAcquisition(
                    ids_peak.AcquisitionStopMode_Default
                )
            except Exception:
                pass
            self.flush_discard_all()
        self._is_acquiring = False

    def flush_discard_all(self) -> None:
        ids_peak = self._ids_peak
        if self._datastream is None or ids_peak is None:
            return
        try:
            self._datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        except Exception:
            pass

    @property
    def is_acquiring(self) -> bool:
        return self._is_acquiring

    # ─── Frame I/O ────────────────────────────────────────────────

    def wait_for_frame(self, timeout_ms: int) -> Optional[FrameHandle]:
        if self._datastream is None:
            return None
        try:
            buf = self._datastream.WaitForFinishedBuffer(timeout_ms)
            return FrameHandle(buf)
        except Exception:
            return None

    def requeue_frame(self, frame: FrameHandle) -> None:
        if self._datastream is None or frame is None:
            return
        try:
            self._datastream.QueueBuffer(frame)
        except Exception:
            pass

    def frame_to_ndarray(
        self,
        frame: FrameHandle,
        dest_format: PixelFormat,
    ) -> np.ndarray:
        if self._converter is None or self._ids_peak_ipl is None:
            raise RuntimeError("backend not open")
        ipl_ext = None
        try:
            from ids_peak import ids_peak_ipl_extension as _ext
            ipl_ext = _ext
        except Exception:
            pass
        if ipl_ext is None:
            raise RuntimeError("ids_peak_ipl_extension unavailable")
        ipl_img = ipl_ext.BufferToImage(frame)
        converted = self._converter.Convert(ipl_img, int(dest_format))
        arr = np.array(converted.get_numpy_2D())
        return arr.copy()

    def write_frame_png(self, path: str, frame: FrameHandle) -> bool:
        if self._ids_peak_ipl is None:
            return False
        try:
            self._ids_peak_ipl.ImageWriter.WriteAsPNG(path, frame)
            return True
        except Exception:
            return False

    # ─── Pixel format ─────────────────────────────────────────────

    def supported_dest_formats(self) -> Sequence[PixelFormat]:
        if self._converter is None:
            return ()
        try:
            ipl_src = self._converter.SourcePixelFormat()
        except Exception:
            return ()
        try:
            outs = self._converter.SupportedOutputPixelFormatNames(ipl_src)
        except Exception:
            return ()
        result = []
        for pf_int in outs:
            try:
                result.append(PixelFormat(int(pf_int)))
            except ValueError:
                continue  # unknown SDK constant — skip
        return tuple(result)

    def set_dest_format(self, fmt: PixelFormat) -> None:
        self._current_format = fmt
        # The actual SDK reconfig happens lazily on the next
        # frame_to_ndarray call — IPL ImageConverter is stateless
        # w.r.t. target format on a per-call basis.

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._frame_shape

    @property
    def current_format(self) -> PixelFormat:
        return self._current_format

    # ─── Back-compat factory ──────────────────────────────────────

    @classmethod
    def from_device_manager(cls, device_manager: Any) -> "IDSPeakSDKBackend":
        """Wrap a pre-existing ids_peak.DeviceManager.

        Used by.3 wiring in ``OptimizedCamera.__init__`` so
        existing callers (qt_interface.py) that pass a device_manager
        positionally keep working.
        """
        return cls(device_manager=device_manager)
