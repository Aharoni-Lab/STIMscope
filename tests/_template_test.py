"""Characterization-test template — copy to tests/<layer>/test_<module>.py.

Phase A discipline:
1. Spec the module.
2. Characterize current behavior with this test file.
3. Audit: compare spec vs reality, mark divergences.
4. Fix bugs surgically; each fix extends a test.
5. Refactor toward Phase B target architecture; tests stay green.

A characterization test pins what the code *currently does*. Once pinned,
we can refactor freely. The test exists to detect change, not to validate
correctness — that's(audit). After a bug is identified, the
test is updated to assert the *new* (correct) behavior, and the bug fix
makes it pass.

Naming convention:
    test_<contract_id>_<short_description>
e.g. test_C1_mu_shape, test_C2_deterministic_with_seed.

Each test maps to at least one contract (C1, C2,...) in the module spec.
"""

from __future__ import annotations

import numpy as np
import pytest

# Layer marker — pytest will pick this up via the markers registered in
# pyproject.toml. Skip-by-default for other layers happens via -m flags.
pytestmark = pytest.mark.L1_algorithms

# Import the module under audit. The conftest puts the STIMscope core root on
# sys.path so `core.<module>` resolves to the in-tree source.
# from core import <module>  # noqa: F401  -- uncomment + replace


# ─────────────────────────────────────────────────────────────────────────────
# Contract C1 — describes what we promise about return shape / dtype.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="template — fill in for the real module")
def test_C1_returns_expected_shape(rng):
    """C1: <module>.foo(x) returns ndarray of shape (N,)."""
    # arrange
    x = rng.standard_normal((10, 20))

    # act
    # result = module.foo(x)

    # assert
    # assert result.shape == (10,)
    # assert result.dtype == np.float64


# ─────────────────────────────────────────────────────────────────────────────
# Contract C2 — determinism with seeded RNG.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="template — fill in for the real module")
def test_C2_deterministic_with_seed(seed):
    """C2: same seed yields identical output across two independent calls."""
    # rng1 = np.random.default_rng(seed)
    # rng2 = np.random.default_rng(seed)
    # out1 = module.foo(x, rng=rng1)
    # out2 = module.foo(x, rng=rng2)
    # np.testing.assert_array_equal(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# Contract C3 — input immutability.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="template — fill in for the real module")
def test_C3_does_not_mutate_inputs(rng):
    """C3: foo() does not mutate its inputs."""
    # x = rng.standard_normal((10, 20))
    # x_before = x.copy()
    # _ = module.foo(x)
    # np.testing.assert_array_equal(x, x_before)


# ─────────────────────────────────────────────────────────────────────────────
# Golden-data characterization — pins exact numerical output against a
# committed reference. Use sparingly: only for the canonical algorithm path.
# Regenerate intentionally with: pytest --golden-regenerate (custom flag, TBD).
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="template — fill in for the real module")
@pytest.mark.golden
def test_golden_canonical_output(rng, golden_dir):
    """Pins exact output for the canonical seed=42, N=20, K=40 scenario."""
    # x = build_canonical_input(rng)
    # result = module.foo(x)
    # ref = np.load(golden_dir / "L1_algorithms" / "<module>_canonical.npz")
    # np.testing.assert_allclose(result, ref["expected"], rtol=1e-7, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Invariant violations — fail-fast on bad inputs.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="template — fill in for the real module")
def test_I1_rejects_empty_input():
    """I1: empty input raises ValueError (not silent garbage)."""
    # with pytest.raises(ValueError):
    #     module.foo(np.empty((0, 0)))


# ─────────────────────────────────────────────────────────────────────────────
# Divergences from spec () — one test per BUG / MISSING.
# Initially marked xfail with a tracking note; turns green when fix lands.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="template — fill in for the real module")
@pytest.mark.xfail(reason="D1: known bug per spec §8 — fix planned in")
def test_D1_known_bug_placeholder():
    """D1: current code does X, spec says Y. Will turn green when fixed."""
    # assert observed_behavior == spec_promised_behavior
