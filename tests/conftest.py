"""Shared pytest fixtures for the CRISPI Phase A audit test suite.

This file is auto-discovered by pytest at the top of `tests/`. Fixtures
defined here are available to every test below it without explicit import.

Layer conventions:
  - L1_algorithms  — pure functions, deterministic with seeded RNG
  - L2_orchestration — config/dispatch, no hardware
  - L3_io          — single-threaded I/O, may use mock_camera / mock_projector
  - L4_concurrency — multi-threaded, may need fake clock / thread harness
  - L5_ui          — Qt, usually marked @pytest.mark.skipif headless

Golden-data fixtures (`golden_dir`) point at `tests/fixtures/golden/<layer>/`
where committed reference outputs live as.npz /.json.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Path setup — make the STIMscope core package importable without docker.
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
CS_PIPELINE_ROOT = (
    REPO_ROOT / "STIMscope" / "STIMViewer_CRISPI" / "CS"
)

# Insert at index 0 so `from core.projector import...` resolves to the
# in-tree source under audit, not whatever happens to be on the host path.
if str(CS_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(CS_PIPELINE_ROOT))

# REPO_ROOT on sys.path so `from tests.<layer>.<helper> import...` resolves
# for cross-layer test helpers. Inserted AFTER the CS root so that `core.*`
# resolves to the audited copy first.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Determinism — seeded RNG for every test that asks for one.
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_SEED = 42  # reference seed for deterministic tests


@pytest.fixture
def rng() -> np.random.Generator:
    """Fresh seeded numpy Generator. Use this in every algorithm test."""
    return np.random.default_rng(CANONICAL_SEED)


@pytest.fixture
def seed() -> int:
    """The canonical seed for cross-test reproducibility."""
    return CANONICAL_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Golden-data paths — where committed reference outputs live.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the CRISPI repo root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def cs_pipeline_root() -> Path:
    """Absolute path to the STIMscope core source root (where core/ lives)."""
    return CS_PIPELINE_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def golden_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "golden"


@pytest.fixture(scope="session")
def canonical_seed() -> int:
    """The audit-wide canonical seed (matches `cs_paper_fidelity_audit.md`)."""
    return CANONICAL_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Capability gates — skip tests that need hardware, GPU, or Qt when absent.
# ─────────────────────────────────────────────────────────────────────────────


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def _has_qt() -> bool:
    if "QT_QPA_PLATFORM" not in os.environ and not os.environ.get("DISPLAY"):
        return False
    try:
        from PyQt5 import QtWidgets  # noqa: F401

        return True
    except Exception:
        return False


HAS_CUPY = _has_cupy()
HAS_QT = _has_qt()
HAS_HARDWARE = os.environ.get("STIM_HARDWARE_PRESENT") == "1"


needs_cupy = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
needs_qt = pytest.mark.skipif(not HAS_QT, reason="Qt/X11 not available")
needs_hardware = pytest.mark.skipif(
    not HAS_HARDWARE,
    reason="Set STIM_HARDWARE_PRESENT=1 to run hardware tests",
)
