"""Centralised logging configuration for the CRISPI pipeline.

Usage:
    from core.logging_config import get_logger
    log = get_logger(__name__)
    log.info("frame %d acquired in %.1f ms", i, dt_ms)

Design notes
------------
- One process-wide root configuration, applied on first ``get_logger`` call.
- Verbosity controlled by the ``STIM_LOG_LEVEL`` env var
  (default: INFO; valid: DEBUG, INFO, WARNING, ERROR, CRITICAL).
- Output goes to stderr so stdout stays clean for the GUI's
  machine-readable progress lines.
- Subprocess code (projector binary stdout capture) is *not* reconfigured
  here — it has its own handlers.

Why not f-strings in log calls?
    The stdlib logger defers formatting until the level filter passes.
    ``log.debug("x=%s", x)`` skips the format step when the level is INFO.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

_FMT: Final[str] = "%(asctime)s %(levelname)-7s %(name)s — %(message)s"
_DATEFMT: Final[str] = "%H:%M:%S"
_ENV_VAR: Final[str] = "STIM_LOG_LEVEL"
_DEFAULT_LEVEL: Final[str] = "INFO"

_configured = False


def _resolve_level() -> int:
    raw = os.environ.get(_ENV_VAR, _DEFAULT_LEVEL).upper().strip()
    return getattr(logging, raw, logging.INFO)


_OUR_HANDLER_TAG: Final[str] = "_cics_default_handler"


def _has_our_handler(root: logging.Logger) -> bool:
    return any(getattr(h, _OUR_HANDLER_TAG, False) for h in root.handlers)


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    if not _has_our_handler(root):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        setattr(handler, _OUR_HANDLER_TAG, True)
        root.addHandler(handler)
    root.setLevel(_resolve_level())
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name`` (use ``__name__``)."""
    _configure_root()
    return logging.getLogger(name)
