"""ZMQ-based projector client for the CRISPI pipeline.

Sends grayscale + packed-RGB masks to the C++ projection engine over ZMQ
PUSH (mask channel) and homography matrices over ZMQ REQ (sideband). The
C++ engine handles homography warp, horizontal flip, overlay, and GPIO
trigger output on projector refresh.

Canonical home of the ``ProjectorBackend`` Protocol (of L3
). The Protocol was previously defined in
``core.calibration_service`` (module 2'srefactor) as a
forward placeholder; module 3a is the canonical implementation, so the
Protocol relocates here. ``core.calibration_service`` now imports
``ProjectorBackend`` from this module for its type annotations — the
producer-side hosts the contract, consumers depend on it.

This module is the canonical implementation of the projector half of
the L3 hardware HAL. Previously lived in ``core.hardware_bridge`` as
the ``MaskProjector`` class; split out as the.5 pre-stage of
L3 module 3 audit.

module 3).
"""

import json
import os
import sys
from typing import Protocol, runtime_checkable

import numpy as np

from.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5e hoisted imports (D-prj-3 + D-prj-4)
# ─────────────────────────────────────────────────────────────────────────────
#
# cv2 used to be lazy-imported inside send_mask/send_mask_rgb on every
# call (branch-predict miss on the hot path). Pulled up to module load
# time. cv2 is a hard dependency of the whole platform — if it's
# missing the rest of the pipeline is broken anyway, so a top-level
# import failure is acceptable. Module still degrades gracefully on
# ZMQ-or-ProjectorClient missing; cv2 missing is a real environment bug.

try:
    import cv2 as _cv2  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover — cv2 is a hard dep
    _cv2 = None  # type: ignore[assignment]
    logger.warning(
        "cv2 unavailable at projector module load — send_mask and "
        "send_mask_rgb will no-op when downstream is inline ZMQ. "
        "Install opencv-python to enable resize/convert paths."
    )


# projector_client used to be re-imported every __init__ with a fresh
# sys.path mutation. Cache the result at module load (one mutation, one
# import) and re-use across all MaskProjector instances.

_HERE = os.path.dirname(os.path.abspath(__file__))
_STIMVIEWER_DIR = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..'))
if _STIMVIEWER_DIR not in sys.path:
    sys.path.insert(0, _STIMVIEWER_DIR)

try:
    from projector_client import ProjectorClient as _ProjectorClient  # type: ignore[import-not-found]
except ImportError:
    _ProjectorClient = None


# ─────────────────────────────────────────────────────────────────────────────
# Named constants ( — D-prj-5)
# ─────────────────────────────────────────────────────────────────────────────
#
# All magic literals in this module are pulled up here with docstring
# rationale so callers and reviewers see the design intent. Defaults
# match the historical hardcoded values; production callers that need
# different values pass them via constructor / method kwargs.

#: ZMQ PUSH endpoint that the C++ projection engine binds for mask data.
#: Matches the C++ engine's `--mask-endpoint` default and the legacy
#: pre-split MaskProjector default.
DEFAULT_MASK_ENDPOINT: str = "tcp://127.0.0.1:5558"

#: ZMQ REQ/REP sideband for one-shot homography updates. Matches the
#: C++ engine's `--homography-endpoint` default.
DEFAULT_HOMOGRAPHY_ENDPOINT: str = "tcp://127.0.0.1:5560"

#: 1920x1080 = DMD native resolution. Callers driving smaller test
#: rigs (e.g. desktop demos) pass smaller values; the resize is
#: handled inside send_mask / send_mask_rgb.
DEFAULT_PROJECTOR_WIDTH: int = 1920
DEFAULT_PROJECTOR_HEIGHT: int = 1080

#: PUSH socket LINGER (ms). 0 = drop pending messages on close; the
#: projector engine treats mid-flight masks as best-effort, so we
#: don't want close() to block.
PUSH_LINGER_MS: int = 0

#: REQ socket LINGER (ms). 1000 = give the homography reply a chance
#: to drain; homography is one-shot per calibration so a short wait is fine.
REQ_LINGER_MS: int = 1000

#: REQ socket RCVTIMEO (ms). D-prj-1 fix uses this to bound how long
#: we block waiting for the C++ engine to ack a homography send. 2 s
#: matches the original "give the engine time but don't hang forever"
#: intent from the buggy `recv(timeout=2000)` call.
REQ_RCVTIMEO_MS: int = 2000


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5f: shared one-shot REQ/REP homography send helper (D-prj-9)
# ─────────────────────────────────────────────────────────────────────────────
#
# Both ``MaskProjector.send_homography`` (this module) and
# ``core.calibration_service.CalibrationService.send_to_projector``
# historically held the same inline-ZMQ REQ/REP pattern. Q2=A verdict
# fromrecon: extract into one helper, both modules call it.
# The helper is private (leading underscore) because callers should
# prefer the Protocol surface (``send_homography``) — this function is
# the one-line fallback for the "no projector backend wired up" path.


def _send_homography_inline(H: np.ndarray, endpoint: str,
                            linger_ms: int = REQ_LINGER_MS,
                            rcvtimeo_ms: int = REQ_RCVTIMEO_MS,
                            log=None) -> bool:
    """Send one homography over a fresh ZMQ REQ socket; close on exit.

    Used by:
      - :meth:`MaskProjector.send_homography`
      - :meth:`core.calibration_service.CalibrationService.send_to_projector`
        (only when no projector dependency is injected)

    Protocol on the wire:
      Two-frame multipart: ``[b"H", H.astype(float64).tobytes()]``.
      Expects a single reply frame (content unused, logged at INFO).

    Returns ``True`` on successful send+ACK, ``False`` on timeout or
    any error. Errors are caught and logged at WARNING; the function
    never raises.

    Parameters
    ----------
    H : (3, 3) float64
        Camera→projector homography matrix.
    endpoint : str
        ZMQ REQ endpoint (e.g. ``"tcp://127.0.0.1:5560"``).
    linger_ms : int
        Socket LINGER on close. Default: :data:`REQ_LINGER_MS`.
    rcvtimeo_ms : int
        recv() timeout (D-prj-1 fix). Default: :data:`REQ_RCVTIMEO_MS`.
    log : logging.Logger or None
        Logger to use for success/failure messages. Falls back to the
        module logger when ``None``.

    Notes
    -----
    Stage 4 fix for D-prj-1 / D-cs-3 (both module 2 + module 3a
    audits): ``zmq.Socket.recv`` has no ``timeout=`` kwarg; use the
    ``RCVTIMEO`` socket option BEFORE recv. Socket close lives in a
    ``try/finally`` so cleanup is guaranteed on exception paths.
    """
    if log is None:
        log = logger
    sock = None
    try:
        import zmq
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, linger_ms)
        sock.setsockopt(zmq.RCVTIMEO, rcvtimeo_ms)
        sock.connect(endpoint)
        sock.send_multipart([b"H", H.astype(np.float64).tobytes()])
        try:
            reply = sock.recv()
        except Exception as recv_e:
            # zmq.Again is the canonical "no ACK within RCVTIMEO" signal;
            # we catch all Exception so test fakes that raise generic
            # RuntimeError also exercise the close-on-exception path.
            log.warning(
                "send_homography: no ACK within %dms (endpoint=%s): %s",
                rcvtimeo_ms, endpoint, recv_e,
            )
            return False
        log.info("Homography sent to %s, reply: %r", endpoint, reply)
        return True
    except Exception as e:
        log.warning("send_homography to %s failed: %s", endpoint, e)
        return False
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# HAL: ProjectorBackend Protocol
# ─────────────────────────────────────────────────────────────────────────────
#
# Originally declared in core.calibration_service (module 2's).
# Relocated here in module 3a'sbecause the
# producer (this module) is the natural home of the contract. Consumers
# (``core.calibration_service`` and any future L3 service that takes a
# projector dependency) import from here.


@runtime_checkable
class ProjectorBackend(Protocol):
    """Sends mask images and homography matrices to the projection engine.

    Production implementation: :class:`MaskProjector` in this module.
    Test doubles: ``tests.L3_hardware.test_projector.InMemoryProjectorBackend``
    and ``tests.L3_hardware.test_calibration_service.InMemoryProjectorBackend``.
    """

    def send_mask(self, mask: np.ndarray, immediate: bool = True) -> int:
        """Send a mask image to the projector. Returns a mask ID."""
        ...

    def send_homography(self, H: np.ndarray,
                        endpoint: str = DEFAULT_HOMOGRAPHY_ENDPOINT) -> None:
        """Send a 3x3 homography matrix over a sideband ZMQ socket."""
        ...


class MaskProjector:
    """
    Sends 1920x1080 grayscale + packed-RGB masks to the STIMscope C++
    projection engine via ZMQ PUSH. Wraps the ProjectorClient from the
    STIMscope codebase when available; falls back to inline ZMQ.

    The C++ engine handles:
      - Homography warp (send H via port 5560, engine precomputes LUT)
      - Horizontal flip (engine --horiz-flip flag)
      - Overlay digits/barcodes
      - GPIO trigger output on projector refresh

    Parameters
    ----------
    endpoint : str
        ZMQ PUSH endpoint for mask data (default: tcp://127.0.0.1:5558)
    proj_width : int
        Projector resolution width (default: 1920)
    proj_height : int
        Projector resolution height (default: 1080)
    """

    def __init__(self, endpoint: str = DEFAULT_MASK_ENDPOINT,
                 proj_width: int = DEFAULT_PROJECTOR_WIDTH,
                 proj_height: int = DEFAULT_PROJECTOR_HEIGHT):
        self.proj_width = proj_width
        self.proj_height = proj_height
        self._mask_id = 0
        self._client = None
        self._sock = None
        self._zmq = None
        self._json = None

        try:
            # Stage 5e: ProjectorClient + sys.path manipulation hoisted to
            # module load (see _ProjectorClient binding above). Caches the
            # import result and only touches sys.path once per process.
            if _ProjectorClient is not None:
                self._client = _ProjectorClient(
                    endpoint=endpoint,
                    width=proj_width,
                    height=proj_height,
                )
                logger.info("Connected to %s via ProjectorClient", endpoint)
            else:
                logger.info(
                    "ProjectorClient not available; using inline ZMQ to %s",
                    endpoint,
                )
                self._init_zmq(endpoint)
        except Exception as e:
            logger.warning(
                "Could not connect to projection engine at %s: %s — "
                "masks will not be projected (simulation-only mode)",
                endpoint, e,
            )

    def _init_zmq(self, endpoint):
        """Minimal ZMQ PUSH socket as fallback."""
        try:
            import zmq
            self._zmq = zmq
            self._json = json
            ctx = zmq.Context.instance()
            self._sock = ctx.socket(zmq.PUSH)
            self._sock.setsockopt(zmq.LINGER, PUSH_LINGER_MS)
            self._sock.connect(endpoint)
            logger.info("ZMQ PUSH connected to %s", endpoint)
        except Exception as e:
            self._sock = None
            # D-prj-10: include endpoint so failures are debuggable
            logger.warning("ZMQ init failed for endpoint %s: %s", endpoint, e)

    def send_mask(self, mask: np.ndarray, immediate: bool = True) -> int:
        """
        Send a mask to the projection engine.

        Parameters
        ----------
        mask : (H, W) uint8
            Binary or grayscale mask. Will be resized to projector resolution.
        immediate : bool
            If True, bypass LATENCY_FRAMES aging (display ASAP).

        Returns
        -------
        mask_id : int  — ID assigned to this mask
        """
        self._mask_id += 1
        mid = self._mask_id

        if self._client is not None:
            self._client.send_gray(mask, frame_id=mid, immediate=immediate)
            return mid

        if self._sock is not None and _cv2 is not None:
            cv2 = _cv2
            if mask.ndim == 3:
                # D-prj-2: silent auto-coerce of 3-channel input. Log a
                # debug warning so callers can find accidentally-RGB inputs
                # in tests/logs. Behavior preserved; once L4 audit confirms
                # no live RGB-as-mask call sites, this can tighten to raise.
                logger.debug(
                    "send_mask received 3-channel array %s — auto-converting "
                    "to grayscale; if caller meant RGB use send_mask_rgb()",
                    mask.shape,
                )
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            if mask.shape != (self.proj_height, self.proj_width):
                mask = cv2.resize(mask, (self.proj_width, self.proj_height),
                                  interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.uint8)
            meta = self._json.dumps({"id": mid, "immediate": immediate}).encode("utf-8")
            self._sock.send_multipart([meta, memoryview(mask)], copy=False)
            return mid

        return mid  # no-op if no connection

    def send_mask_rgb(self, rgb: np.ndarray, immediate: bool = True) -> int:
        """Send a packed-RGB frame (H, W, 3) uint8 to the projection engine.

        Used for Mode A (Temporal) / Mode B (Simultaneous) / Mode C (Selective)
        where stim and observe patterns live in separate RGB channels (R=stim,
        B=observe) and the DMD sub-frame multiplexes them.
        """
        self._mask_id += 1
        mid = self._mask_id

        if self._client is not None and hasattr(self._client, 'send_rgb'):
            self._client.send_rgb(rgb, frame_id=mid, immediate=immediate)
            return mid

        if self._sock is not None and _cv2 is not None:
            cv2 = _cv2
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("send_mask_rgb requires shape (H, W, 3)")
            if rgb.shape[:2] != (self.proj_height, self.proj_width):
                rgb = cv2.resize(rgb, (self.proj_width, self.proj_height),
                                 interpolation=cv2.INTER_NEAREST)
            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)
            if not rgb.flags['C_CONTIGUOUS']:
                rgb = np.ascontiguousarray(rgb)
            meta = self._json.dumps({"id": mid, "immediate": immediate}).encode("utf-8")
            self._sock.send_multipart([meta, memoryview(rgb)], copy=False)
            return mid

        return mid  # no-op if no connection

    def send_homography(self, H: np.ndarray,
                        endpoint: str = DEFAULT_HOMOGRAPHY_ENDPOINT) -> None:
        """Send calibration homography to the C++ engine.

        Delegates to the module-level :func:`_send_homography_inline`
        helper (D-prj-9). The helper is shared with
        :meth:`core.calibration_service.CalibrationService.send_to_projector`
        — both call sites historically had the same inline-REQ-REP
        pattern duplicated. See helper docstring for protocol details.

        Parameters
        ----------
        H : (3, 3) float64  — camera-to-projector homography
        endpoint : str  — ZMQ REQ endpoint (default 5560).
        """
        _send_homography_inline(H, endpoint, log=logger)

    def close(self):
        if self._client is not None:
            self._client.close()
        if self._sock is not None:
            self._sock.close(0)
