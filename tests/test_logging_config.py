"""Sentinel tests for core/logging_config.py.

These tests verify the contract the rest of the codebase will lean on as the
L5 print->log conversion progresses:

1. ``get_logger(__name__)`` returns a stdlib ``logging.Logger``.
2. The root logger respects the ``STIM_LOG_LEVEL`` env var.
3. Calling ``get_logger`` more than once does not double-add handlers.
4. Output goes to stderr, not stdout (so the GUI's machine-readable
   stdout stays clean).
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def fresh_logging_module(monkeypatch):
    """Force a clean ``core.logging_config`` import and a clean root logger."""
    CS = (
        Path(__file__).resolve().parent.parent
        / "STIMscope"
        / "STIMViewer_CRISPI"
        / "CS"
    )
    monkeypatch.syspath_prepend(str(CS))

    sys.modules.pop("core.logging_config", None)

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    for h in saved_handlers:
        root.removeHandler(h)

    yield

    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)
    sys.modules.pop("core.logging_config", None)


def test_get_logger_returns_stdlib_logger(fresh_logging_module):
    from core.logging_config import get_logger

    log = get_logger("test.module")
    assert isinstance(log, logging.Logger)
    assert log.name == "test.module"


def test_default_level_is_info(fresh_logging_module, monkeypatch):
    monkeypatch.delenv("STIM_LOG_LEVEL", raising=False)
    from core.logging_config import get_logger

    get_logger("test.default")
    assert logging.getLogger().level == logging.INFO


def test_env_var_overrides_level(fresh_logging_module, monkeypatch):
    monkeypatch.setenv("STIM_LOG_LEVEL", "DEBUG")
    from core.logging_config import get_logger

    get_logger("test.debug")
    assert logging.getLogger().level == logging.DEBUG


def test_invalid_level_falls_back_to_info(fresh_logging_module, monkeypatch):
    monkeypatch.setenv("STIM_LOG_LEVEL", "NONSENSE")
    from core.logging_config import get_logger

    get_logger("test.invalid")
    assert logging.getLogger().level == logging.INFO


_OUR_TAG = "_cics_default_handler"


def _our_handlers():
    return [h for h in logging.getLogger().handlers if getattr(h, _OUR_TAG, False)]


def test_double_call_does_not_duplicate_handlers(fresh_logging_module):
    from core.logging_config import get_logger

    get_logger("test.first")
    get_logger("test.second")
    assert len(_our_handlers()) == 1


def test_handler_writes_to_stderr(fresh_logging_module):
    """The configured handler must target stderr, not stdout."""
    from core.logging_config import get_logger

    get_logger("test.stream")
    ours = _our_handlers()
    assert len(ours) == 1
    assert ours[0].stream is sys.stderr or ours[0].stream is sys.__stderr__
