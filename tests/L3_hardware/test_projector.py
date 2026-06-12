"""Stage-2 characterization tests for `core.projector`.

Pins the as-is behavior described in
`docs/specs/L3_hardware/projector.md` §1 (contract) and §3 (divergence
ledger). Stage 4 will mutate D-prj-1 from "documents the bug" to
"verifies the fix".

Tests are NUMBERED by the contract clause they pin (C1, C2,...) and
by the divergence they pre-stage (D-prj-N).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# HAL Protocol stand-in — formalized atin production module
# ─────────────────────────────────────────────────────────────────────────────


class InMemoryProjectorBackend:
    """Test double for the ProjectorBackend Protocol (target).

    Records every send_mask + send_homography call.
    """

    def __init__(self) -> None:
        self.masks_sent: List[np.ndarray] = []
        self.homographies_sent: List[np.ndarray] = []
        self.endpoints: List[str] = []

    def send_mask(self, mask: np.ndarray, immediate: bool = True) -> int:
        self.masks_sent.append(mask.copy())
        return len(self.masks_sent)

    def send_homography(self, H: np.ndarray,
                        endpoint: str = "tcp://127.0.0.1:5560") -> None:
        self.homographies_sent.append(H.copy())
        self.endpoints.append(endpoint)


# ─────────────────────────────────────────────────────────────────────────────
# Fakes that simulate ProjectorClient and zmq.Context
# ─────────────────────────────────────────────────────────────────────────────


class FakeProjectorClient:
    """In-memory stand-in for STIMViewer_CRISPI/projector_client.ProjectorClient."""

    def __init__(self, endpoint: str, width: int, height: int) -> None:
        self.endpoint = endpoint
        self.width = width
        self.height = height
        self.gray_calls: List[tuple] = []
        self.rgb_calls: List[tuple] = []
        self.closed = False

    def send_gray(self, mask, frame_id, immediate):
        self.gray_calls.append((mask.copy(), frame_id, immediate))

    def send_rgb(self, rgb, frame_id, immediate):
        self.rgb_calls.append((rgb.copy(), frame_id, immediate))

    def close(self):
        self.closed = True


class FakeZMQSocket:
    """Captures all socket operations for verification."""

    def __init__(self, socket_type: int) -> None:
        self.socket_type = socket_type
        self.options: dict = {}
        self.connected_endpoint: Optional[str] = None
        self.bound_endpoint: Optional[str] = None
        self.multipart_messages: List[List[bytes]] = []
        self.recv_called = 0
        self.recv_response = b"OK"
        self.closed = False
        # If set, recv() raises this exception
        self.recv_raises: Optional[BaseException] = None

    def setsockopt(self, opt: int, value):
        self.options[opt] = value

    def connect(self, endpoint: str):
        self.connected_endpoint = endpoint

    def bind(self, endpoint: str):
        self.bound_endpoint = endpoint

    def send_multipart(self, parts, copy: bool = True, **_):
        self.multipart_messages.append(list(parts))

    def recv(self, *args, **kwargs):
        self.recv_called += 1
        # Stash kwargs for inspection (D-prj-1 sniffs `timeout=` kwarg)
        self._last_recv_kwargs = dict(kwargs)
        self._last_recv_args = tuple(args)
        if self.recv_raises is not None:
            raise self.recv_raises
        return self.recv_response

    def close(self, linger: int = 0):
        self.closed = True


class FakeZMQContext:
    """Per-test ZMQ Context capturing every socket created."""

    def __init__(self) -> None:
        self.sockets: List[FakeZMQSocket] = []

    def socket(self, socket_type: int) -> FakeZMQSocket:
        s = FakeZMQSocket(socket_type)
        self.sockets.append(s)
        return s


class FakeZMQModule:
    """Stand-in for `import zmq`. Exposes the constants the production
    code touches plus `Context.instance()` returning a FakeZMQContext.
    """

    # zmq socket-type constants
    PUSH = 8
    REQ = 3
    # zmq option constants
    LINGER = 17
    RCVTIMEO = 27  # used by post-fix
    SNDTIMEO = 28

    def __init__(self) -> None:
        self._ctx = FakeZMQContext()
        self.Context = types.SimpleNamespace(instance=lambda: self._ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Module loader fixture — installs fakes BEFORE projector.py is imported
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def cs_path():
    return (
        Path(__file__).resolve().parent.parent.parent
        / "STIMscope"
        / "STIMViewer_CRISPI"
        / "CS"
    )


def _force_reimport_projector():
    """Force core.projector to re-execute its module body.

    sys.modules.pop() alone is insufficient because ``from core import
    projector`` consults the ``core`` package's ``projector`` attribute
    via Python's import-fromlist semantics. Popping only the sys.modules
    cache leaves a stale attribute that the next ``from`` import returns
    unchanged. Need to delete the attribute AND the sys.modules entry
    AND use importlib.import_module so the from-import bytecode path
    is bypassed entirely.
    """
    import importlib
    sys.modules.pop("core.projector", None)
    try:
        import core
        if hasattr(core, "projector"):
            delattr(core, "projector")
    except ImportError:
        pass
    return importlib.import_module("core.projector")


@pytest.fixture
def projector_module_no_client_no_zmq(monkeypatch, cs_path):
    """Total failure path: no projector_client, no zmq. Construction
    succeeds; all send_* are no-ops returning incrementing IDs.
    """
    monkeypatch.syspath_prepend(str(cs_path))
    # Block projector_client import
    monkeypatch.setitem(sys.modules, "projector_client", None)
    # Block zmq
    monkeypatch.setitem(sys.modules, "zmq", None)
    return _force_reimport_projector()


@pytest.fixture
def projector_module_with_zmq(monkeypatch, cs_path):
    """Inline-ZMQ fallback: no projector_client, but zmq is available."""
    monkeypatch.syspath_prepend(str(cs_path))
    monkeypatch.setitem(sys.modules, "projector_client", None)
    fake_zmq = FakeZMQModule()
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)
    mod = _force_reimport_projector()
    # Stash the fake on the module so tests can introspect
    mod._test_fake_zmq = fake_zmq
    return mod


@pytest.fixture
def projector_module_with_client(monkeypatch, cs_path):
    """Preferred path: projector_client wraps the connection."""
    monkeypatch.syspath_prepend(str(cs_path))
    fake_client_mod = types.ModuleType("projector_client")
    fake_client_mod.ProjectorClient = FakeProjectorClient
    monkeypatch.setitem(sys.modules, "projector_client", fake_client_mod)
    return _force_reimport_projector()


# ─────────────────────────────────────────────────────────────────────────────
# C1 — Construction graceful degradation
# ─────────────────────────────────────────────────────────────────────────────


class TestC1ConstructionGracefulDegradation:
    """Construction MUST succeed regardless of available dependencies."""

    def test_construction_with_no_dependencies_succeeds(
        self, projector_module_no_client_no_zmq
    ):
        mp = projector_module_no_client_no_zmq.MaskProjector()
        assert mp._client is None
        assert mp._sock is None
        assert mp.proj_width == 1920
        assert mp.proj_height == 1080
        assert mp._mask_id == 0

    def test_construction_inline_zmq_path(self, projector_module_with_zmq):
        mp = projector_module_with_zmq.MaskProjector()
        # Client unavailable → falls through to _init_zmq
        assert mp._client is None
        assert mp._sock is not None
        fake_zmq = projector_module_with_zmq._test_fake_zmq
        assert len(fake_zmq._ctx.sockets) == 1
        assert fake_zmq._ctx.sockets[0].socket_type == FakeZMQModule.PUSH
        assert fake_zmq._ctx.sockets[0].connected_endpoint == "tcp://127.0.0.1:5558"
        # LINGER=0 means "don't block on close"
        assert fake_zmq._ctx.sockets[0].options.get(FakeZMQModule.LINGER) == 0

    def test_construction_wraps_projector_client_when_available(
        self, projector_module_with_client
    ):
        mp = projector_module_with_client.MaskProjector()
        assert isinstance(mp._client, FakeProjectorClient)
        assert mp._client.endpoint == "tcp://127.0.0.1:5558"
        assert mp._client.width == 1920
        assert mp._client.height == 1080

    def test_custom_resolution_propagated_to_client(
        self, projector_module_with_client
    ):
        mp = projector_module_with_client.MaskProjector(
            endpoint="tcp://127.0.0.1:9999", proj_width=640, proj_height=480
        )
        assert mp._client.endpoint == "tcp://127.0.0.1:9999"
        assert mp._client.width == 640
        assert mp._client.height == 480

    def test_close_idempotent_when_no_resources(
        self, projector_module_no_client_no_zmq
    ):
        mp = projector_module_no_client_no_zmq.MaskProjector()
        mp.close()  # must not raise
        mp.close()  # must not raise

    def test_close_propagates_to_client(self, projector_module_with_client):
        mp = projector_module_with_client.MaskProjector()
        mp.close()
        assert mp._client.closed is True

    def test_close_propagates_to_inline_socket(self, projector_module_with_zmq):
        mp = projector_module_with_zmq.MaskProjector()
        sock = projector_module_with_zmq._test_fake_zmq._ctx.sockets[0]
        mp.close()
        assert sock.closed is True


# ─────────────────────────────────────────────────────────────────────────────
# C2 — send_mask monotonic ID + shape coercion + no-op path
# ─────────────────────────────────────────────────────────────────────────────


class TestC2SendMask:
    """send_mask returns monotonically-incrementing IDs even when no
    downstream is available; coerces shape silently (see D-prj-2)."""

    def test_returns_monotonic_ids_with_no_downstream(
        self, projector_module_no_client_no_zmq
    ):
        mp = projector_module_no_client_no_zmq.MaskProjector()
        ids = [mp.send_mask(np.zeros((1080, 1920), dtype=np.uint8)) for _ in range(5)]
        assert ids == [1, 2, 3, 4, 5]

    def test_dispatches_to_client_when_available(
        self, projector_module_with_client
    ):
        mp = projector_module_with_client.MaskProjector()
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        mid = mp.send_mask(mask, immediate=False)
        assert mid == 1
        assert len(mp._client.gray_calls) == 1
        _, frame_id, immediate = mp._client.gray_calls[0]
        assert frame_id == 1
        assert immediate is False

    def test_inline_zmq_sends_multipart_with_json_header(
        self, projector_module_with_zmq
    ):
        mp = projector_module_with_zmq.MaskProjector()
        sock = projector_module_with_zmq._test_fake_zmq._ctx.sockets[0]
        mask = np.full((1080, 1920), 200, dtype=np.uint8)
        mid = mp.send_mask(mask, immediate=True)
        assert mid == 1
        assert len(sock.multipart_messages) == 1
        meta_bytes, _ = sock.multipart_messages[0]
        meta = json.loads(meta_bytes.decode("utf-8"))
        assert meta == {"id": 1, "immediate": True}

    def test_inline_zmq_resizes_mask_to_projector_resolution(
        self, projector_module_with_zmq
    ):
        mp = projector_module_with_zmq.MaskProjector(
            proj_width=320, proj_height=240
        )
        sock = projector_module_with_zmq._test_fake_zmq._ctx.sockets[0]
        mp.send_mask(np.zeros((480, 640), dtype=np.uint8))
        _, payload = sock.multipart_messages[0]
        # payload size matches 320*240 (resized)
        assert len(bytes(payload)) == 320 * 240


# ─────────────────────────────────────────────────────────────────────────────
# C3 — send_mask_rgb shape strictness (no silent coerce)
# ─────────────────────────────────────────────────────────────────────────────


class TestC3SendMaskRGB:
    """send_mask_rgb requires (H,W,3) shape and raises ValueError otherwise."""

    def test_raises_on_grayscale_input(self, projector_module_with_zmq):
        mp = projector_module_with_zmq.MaskProjector()
        with pytest.raises(ValueError, match="H, W, 3"):
            mp.send_mask_rgb(np.zeros((1080, 1920), dtype=np.uint8))

    def test_raises_on_wrong_channels(self, projector_module_with_zmq):
        mp = projector_module_with_zmq.MaskProjector()
        with pytest.raises(ValueError, match="H, W, 3"):
            mp.send_mask_rgb(np.zeros((1080, 1920, 4), dtype=np.uint8))

    def test_inline_zmq_sends_rgb_payload(self, projector_module_with_zmq):
        mp = projector_module_with_zmq.MaskProjector(
            proj_width=320, proj_height=240
        )
        sock = projector_module_with_zmq._test_fake_zmq._ctx.sockets[0]
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[..., 0] = 200  # Red channel
        mid = mp.send_mask_rgb(rgb, immediate=True)
        assert mid == 1
        meta_bytes, payload = sock.multipart_messages[0]
        assert json.loads(meta_bytes.decode("utf-8"))["id"] == 1
        # Resized to 320×240×3
        assert len(bytes(payload)) == 320 * 240 * 3

    def test_returns_monotonic_id_with_no_downstream(
        self, projector_module_no_client_no_zmq
    ):
        mp = projector_module_no_client_no_zmq.MaskProjector()
        rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ids = [mp.send_mask_rgb(rgb) for _ in range(3)]
        assert ids == [1, 2, 3]

    def test_grayscale_via_send_mask_with_3channel_input_silently_coerces(
        self, projector_module_with_zmq
    ):
        """D-prj-2: send_mask (NOT send_mask_rgb) silently auto-converts
        3-channel input to grayscale via cv2.cvtColor. This pins the
        as-is behavior;may tighten to a warning log.
        """
        mp = projector_module_with_zmq.MaskProjector(
            proj_width=320, proj_height=240
        )
        sock = projector_module_with_zmq._test_fake_zmq._ctx.sockets[0]
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        mid = mp.send_mask(rgb)  # would be a bug-pattern call site
        assert mid == 1
        _, payload = sock.multipart_messages[0]
        # Payload is grayscale (1 byte/pixel) at projector resolution
        assert len(bytes(payload)) == 320 * 240


# ─────────────────────────────────────────────────────────────────────────────
# C4 / D-prj-1 — send_homography: REQ/REP + timeout + socket cleanup
# ─────────────────────────────────────────────────────────────────────────────


class TestC4D1SendHomography:
    """send_homography uses a one-shot REQ/REP. D-prj-1 documents the
    pre-fix bug (sock.recv accepts timeout=KW);mutates to the
    post-fix expectation (RCVTIMEO + try/finally cleanup).
    """

    def test_sends_3x3_float64_homography(self, projector_module_with_zmq):
        mp = projector_module_with_zmq.MaskProjector()
        fake_zmq = projector_module_with_zmq._test_fake_zmq
        H = np.eye(3, dtype=np.float64)
        mp.send_homography(H)
        # Two sockets total: the PUSH from __init__, and the REQ from this call
        req_socks = [s for s in fake_zmq._ctx.sockets
                     if s.socket_type == FakeZMQModule.REQ]
        assert len(req_socks) == 1
        req = req_socks[0]
        assert req.connected_endpoint == "tcp://127.0.0.1:5560"
        # Multipart: [b"H", H.tobytes()]
        assert len(req.multipart_messages) == 1
        topic, payload = req.multipart_messages[0]
        assert topic == b"H"
        assert payload == H.astype(np.float64).tobytes()

    def test_recv_called_after_send(self, projector_module_with_zmq):
        mp = projector_module_with_zmq.MaskProjector()
        fake_zmq = projector_module_with_zmq._test_fake_zmq
        mp.send_homography(np.eye(3))
        req = [s for s in fake_zmq._ctx.sockets
               if s.socket_type == FakeZMQModule.REQ][0]
        assert req.recv_called == 1

    def test_d_prj_1_POST_FIX_uses_rcvtimeo_socket_option(
        self, projector_module_with_zmq
    ):
        """D-prj-1 POST-FIX (commit landing this assertion): code uses
        ``setsockopt(RCVTIMEO, 2000)`` to bound the recv blocking, NOT
        the invalid ``recv(timeout=2000)`` keyword. Replaces thePRE-FIX pin (which proved the bug existed).
        """
        mp = projector_module_with_zmq.MaskProjector()
        fake_zmq = projector_module_with_zmq._test_fake_zmq
        mp.send_homography(np.eye(3))
        req = [s for s in fake_zmq._ctx.sockets
               if s.socket_type == FakeZMQModule.REQ][0]
        # POST-FIX: RCVTIMEO option set, recv called WITHOUT timeout kwarg
        assert req.options.get(FakeZMQModule.RCVTIMEO) == 2000
        assert "timeout" not in req._last_recv_kwargs

    def test_d_prj_1_POST_FIX_socket_closed_on_recv_exception(
        self, projector_module_with_zmq
    ):
        """D-prj-1 POST-FIX (commit landing this assertion): if recv
        raises, the socket is still closed via try/finally — no leak.
        Replaces thePRE-FIX pin.
        """
        mp = projector_module_with_zmq.MaskProjector()
        fake_zmq = projector_module_with_zmq._test_fake_zmq
        original_socket = fake_zmq._ctx.socket

        def make_socket(socket_type):
            s = original_socket(socket_type)
            if socket_type == FakeZMQModule.REQ:
                s.recv_raises = RuntimeError("simulated timeout")
            return s

        fake_zmq._ctx.socket = make_socket
        mp.send_homography(np.eye(3))
        req_socks = [s for s in fake_zmq._ctx.sockets
                     if s.socket_type == FakeZMQModule.REQ]
        assert len(req_socks) == 1
        # POST-FIX: socket closed despite the exception in recv
        assert req_socks[0].closed is True


# ─────────────────────────────────────────────────────────────────────────────
# C5 — Protocol stand-in works for the as-is duck-typed interface
# ─────────────────────────────────────────────────────────────────────────────


class TestC5ProtocolStandIn:
    """The InMemoryProjectorBackend test double satisfies the duck-typed
    interface that MaskProjector exposes. Stage 5a will formalize the
    Protocol relationship; this test makes the contract explicit.
    """

    def test_in_memory_backend_records_masks(self):
        backend = InMemoryProjectorBackend()
        mask = np.zeros((100, 100), dtype=np.uint8)
        mid = backend.send_mask(mask)
        assert mid == 1
        assert len(backend.masks_sent) == 1
        assert backend.masks_sent[0] is not mask  # defensive copy

    def test_in_memory_backend_records_homography(self):
        backend = InMemoryProjectorBackend()
        H = np.eye(3)
        backend.send_homography(H, endpoint="tcp://test:1234")
        assert len(backend.homographies_sent) == 1
        assert backend.endpoints == ["tcp://test:1234"]

    def test_in_memory_backend_has_same_method_signatures_as_maskprojector(
        self, projector_module_no_client_no_zmq
    ):
        """Duck-typing check: backend exposes send_mask + send_homography
        with the same arity as MaskProjector. Oncelands, both
        will be `isinstance(_, ProjectorBackend)`.
        """
        mp = projector_module_no_client_no_zmq.MaskProjector()
        backend = InMemoryProjectorBackend()
        # Both expose send_mask(mask, immediate=True) and
        # send_homography(H, endpoint="...").
        assert hasattr(mp, "send_mask") and hasattr(backend, "send_mask")
        assert hasattr(mp, "send_homography") and hasattr(backend, "send_homography")


# ─────────────────────────────────────────────────────────────────────────────
# C6 — Stage 5a: ProjectorBackend Protocol relocated to projector.py
# ─────────────────────────────────────────────────────────────────────────────


class TestC6ProtocolRelocation:
    """Stage 5a — ProjectorBackend Protocol now lives in core.projector.
    calibration_service re-exports it for backward compatibility.
    """

    def test_protocol_lives_in_projector_module(
        self, projector_module_no_client_no_zmq
    ):
        assert hasattr(projector_module_no_client_no_zmq, "ProjectorBackend")
        from typing import _ProtocolMeta  # type: ignore[attr-defined]
        assert isinstance(
            projector_module_no_client_no_zmq.ProjectorBackend, _ProtocolMeta
        )

    def test_maskprojector_is_runtime_checkable_protocol_conformant(
        self, projector_module_no_client_no_zmq
    ):
        """isinstance(MaskProjector(...), ProjectorBackend) holds via
        structural typing — the canonical conformance evidence for.
        """
        mp = projector_module_no_client_no_zmq.MaskProjector()
        assert isinstance(mp, projector_module_no_client_no_zmq.ProjectorBackend)

    def test_in_memory_backend_is_runtime_checkable_protocol_conformant(
        self, projector_module_no_client_no_zmq
    ):
        backend = InMemoryProjectorBackend()
        assert isinstance(backend, projector_module_no_client_no_zmq.ProjectorBackend)
