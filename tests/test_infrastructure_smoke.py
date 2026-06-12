"""Smoke test for the Phase A audit infrastructure itself.

Confirms that the pytest setup, conftest fixtures, and core-package import
path all work before we add a single module-specific test. If this file
fails, no other test in the suite can be trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def test_repo_root_resolves(repo_root: Path):
    """conftest's repo_root fixture points at a directory containing the README."""
    assert repo_root.is_dir()
    assert (repo_root / "README.md").exists()


def test_cs_pipeline_root_on_sys_path(cs_pipeline_root: Path):
    """The STIMscope core source is importable as `core.<module>`."""
    assert cs_pipeline_root.is_dir()
    assert (cs_pipeline_root / "core" / "__init__.py").exists()
    assert str(cs_pipeline_root) in sys.path


def test_can_import_a_pure_core_module():
    """core.paths is pure-stdlib shared infra; importing it on the host must work."""
    import importlib

    mod = importlib.import_module("core.paths")
    assert mod is not None


def test_rng_fixture_is_seeded(rng):
    """The rng fixture produces deterministic output across test runs."""
    sample = rng.standard_normal(5)
    # If the canonical seed ever drifts, this changes.
    # Values below pinned for seed=42 with numpy.random.default_rng (PCG64).
    expected = np.array([0.30471708, -1.03998411, 0.7504512, 0.94056472, -1.95103519])
    np.testing.assert_allclose(sample, expected, rtol=1e-7)


def test_seed_fixture_matches_canonical(seed: int, canonical_seed: int):
    """The seed and canonical_seed fixtures agree."""
    assert seed == canonical_seed == 42


def test_golden_dir_exists(golden_dir: Path):
    """tests/fixtures/golden/ exists for committed reference outputs."""
    assert golden_dir.is_dir()


@pytest.mark.L1_algorithms
def test_marker_registration():
    """The L1_algorithms marker is registered (strict-markers would fail otherwise)."""
