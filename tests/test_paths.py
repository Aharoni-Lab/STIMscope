"""Sentinel tests for `core.paths`.

The path helper is the single source of truth for where the platform reads
and writes data. These tests pin the contracts
that downstream L3/L4 audits will lean on as they migrate hardcoded paths.

Two contract families:
  A. Path *shape* — every helper returns the documented subdirectory of
     DATA_ROOT.
  B. Env var override — `STIM_DATA_ROOT` flips the root for every
     helper consistently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def paths_module(monkeypatch):
    """Force a clean import so module-level `_resolve_root()` reads the
    monkeypatched env var, not whatever was set when pytest first loaded."""
    CS = (
        Path(__file__).resolve().parent.parent
        / "STIMscope"
        / "STIMViewer_CRISPI"
        / "CS"
    )
    monkeypatch.syspath_prepend(str(CS))
    sys.modules.pop("core.paths", None)
    import core.paths as paths
    yield paths
    sys.modules.pop("core.paths", None)


def test_default_root_is_relative_data(monkeypatch, paths_module):
    monkeypatch.delenv("STIM_DATA_ROOT", raising=False)
    # Functions re-read env var on call; constants frozen at import time.
    assert paths_module.data_root() == Path("data")


def test_env_var_overrides_root(monkeypatch, paths_module, tmp_path):
    monkeypatch.setenv("STIM_DATA_ROOT", str(tmp_path))
    assert paths_module.data_root() == tmp_path


def test_subdir_shape_matches_design(monkeypatch, paths_module, tmp_path):
    monkeypatch.setenv("STIM_DATA_ROOT", str(tmp_path))
    assert paths_module.config_dir() == tmp_path / "config"
    assert paths_module.assets_dir() == tmp_path / "assets"
    assert paths_module.homography_dir() == tmp_path / "assets" / "homography"
    assert paths_module.sl_patterns_dir() == tmp_path / "assets" / "sl_patterns"
    assert paths_module.diagnostic_dir() == tmp_path / "assets" / "diagnostic"
    assert paths_module.runs_dir() == tmp_path / "runs"
    assert paths_module.recordings_dir() == tmp_path / "recordings"
    assert paths_module.cache_dir() == tmp_path / "cache"


def test_run_dir_creates_with_explicit_timestamp(monkeypatch, paths_module, tmp_path):
    monkeypatch.setenv("STIM_DATA_ROOT", str(tmp_path))
    p = paths_module.run_dir(timestamp="20260513_120000")
    assert p == tmp_path / "runs" / "20260513_120000"
    assert p.is_dir()


def test_run_dir_default_timestamp_is_now_format(monkeypatch, paths_module, tmp_path):
    monkeypatch.setenv("STIM_DATA_ROOT", str(tmp_path))
    p = paths_module.run_dir()
    # Format: YYYYMMDD_HHMMSS — 8 digits underscore 6 digits
    name = p.name
    assert len(name) == 15
    assert name[8] == "_"
    assert name[:8].isdigit()
    assert name[9:].isdigit()
    assert p.is_dir()


def test_recording_dir_parallel_to_run_dir(monkeypatch, paths_module, tmp_path):
    monkeypatch.setenv("STIM_DATA_ROOT", str(tmp_path))
    p = paths_module.recording_dir(timestamp="20260513_120000")
    assert p == tmp_path / "recordings" / "20260513_120000"
    assert p.is_dir()


def test_ensure_layout_creates_all_subdirs(monkeypatch, paths_module, tmp_path):
    monkeypatch.setenv("STIM_DATA_ROOT", str(tmp_path))
    paths_module.ensure_layout()
    for sub in ("config", "assets", "assets/homography", "assets/sl_patterns",
                "assets/diagnostic", "runs", "recordings", "cache"):
        assert (tmp_path / sub).is_dir(), f"ensure_layout did not create {sub}"


def test_ensure_layout_is_idempotent(monkeypatch, paths_module, tmp_path):
    monkeypatch.setenv("STIM_DATA_ROOT", str(tmp_path))
    paths_module.ensure_layout()
    # Second call must not raise even though all dirs already exist.
    paths_module.ensure_layout()


def test_run_dir_with_create_false_does_not_make_dir(monkeypatch, paths_module, tmp_path):
    monkeypatch.setenv("STIM_DATA_ROOT", str(tmp_path))
    p = paths_module.run_dir(timestamp="never_create", create=False)
    assert p == tmp_path / "runs" / "never_create"
    assert not p.exists()


def test_module_level_constants_freeze_at_import(monkeypatch, paths_module):
    """The CONSTANTS (uppercase) freeze at import. The FUNCTIONS re-read.
    This test documents the deliberate asymmetry so future contributors
    don't expect both to behave the same."""
    monkeypatch.delenv("STIM_DATA_ROOT", raising=False)
    # Module already imported in fixture WITHOUT the env var.
    assert paths_module.DATA_ROOT == Path("data")
    # Set env var AFTER import — function picks it up, constant doesn't.
    monkeypatch.setenv("STIM_DATA_ROOT", "/elsewhere")
    assert paths_module.data_root() == Path("/elsewhere")
    assert paths_module.DATA_ROOT == Path("data")  # frozen
