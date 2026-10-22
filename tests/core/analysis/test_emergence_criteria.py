"""Test the integrated information criteria of emergence and the closure of coarse-grainings."""

import numpy as np
import pytest

from skcomplex.core.analysis import closure, emergence_criteria
from skcomplex.core.base import AnalysisError

LUMPABLE = np.array(
    [
        [0.5, 0.3, 0.1, 0.1],
        [0.2, 0.6, 0.15, 0.05],
        [0.1, 0.1, 0.4, 0.4],
        [0.1, 0.1, 0.2, 0.6],
    ],
)
NOT_LUMPABLE = np.array(
    [
        [0.5, 0.3, 0.1, 0.1],
        [0.2, 0.3, 0.45, 0.05],
        [0.1, 0.1, 0.4, 0.4],
        [0.1, 0.1, 0.2, 0.6],
    ],
)


def _persistent_parity(steps: int, bits: int, flip: float, seed: int) -> np.ndarray:
    """Return random bits whose parity persists from step to step except for rare flips."""
    rng = np.random.default_rng(seed)
    states = np.zeros((steps, bits), dtype=np.int64)
    parity = 0
    for t in range(steps):
        parity ^= int(rng.random() < flip)
        states[t, :-1] = rng.integers(0, 2, bits - 1)
        states[t, -1] = (states[t, :-1].sum() + parity) % 2
    return states


def _chain(matrix: np.ndarray, steps: int, seed: int) -> np.ndarray:
    """Return a path of a Markov chain with a transition matrix."""
    rng = np.random.default_rng(seed)
    cumulative = np.cumsum(matrix, axis=1)
    states = np.zeros(steps, dtype=np.int64)
    draws = rng.random(steps)
    for t in range(1, steps):
        states[t] = min(int(np.searchsorted(cumulative[states[t - 1]], draws[t])), matrix.shape[0] - 1)
    return states


def test_emergence_criteria_parity_is_emergent():
    """Test that the persistent parity of noisy bits has a positive Ψ and no downward causation."""
    micro = _persistent_parity(20_000, 4, 0.05, 0)
    criteria = emergence_criteria(micro, micro.sum(axis=1) % 2)
    expected = 1 + 0.05 * np.log2(0.05) + 0.95 * np.log2(0.95)
    assert criteria.psi == pytest.approx(expected, abs=0.02)
    assert criteria.psi > 0
    assert criteria.gamma == pytest.approx(0.0, abs=0.01)
    assert criteria.delta <= 0.01


def test_emergence_criteria_micro_variable_is_not_emergent():
    """Test that one micro bit of the persistent parity system has no positive Ψ."""
    micro = _persistent_parity(20_000, 4, 0.05, 1)
    assert emergence_criteria(micro, micro[:, 0]).psi <= 0.01


def test_emergence_criteria_gaussian_difference_hides_common_noise():
    """Test that the difference of two variables sharing a strong noise reveals a slow process with a positive Ψ."""
    rng = np.random.default_rng(2)
    slow = np.zeros(10_000)
    for t in range(1, slow.size):
        slow[t] = 0.95 * slow[t - 1] + rng.normal()
    noise = 10.0 * rng.normal(size=slow.size)
    micro = np.column_stack([slow + noise, noise])
    assert emergence_criteria(micro, micro[:, 0] - micro[:, 1], estimator='gaussian').psi > 1.0


def test_emergence_criteria_unknown_estimator():
    """Test that an unknown estimator raises an error naming the known ones."""
    with pytest.raises(AnalysisError, match='plugin'):
        emergence_criteria(np.zeros((10, 2)), np.zeros(10), estimator='binned')


def test_closure_lumpable_chain_is_closed():
    """Test that the block coarse-graining of a lumpable chain is closed within the estimator error."""
    micro = _chain(LUMPABLE, 50_000, 3)
    result = closure(micro, [0, 0, 1, 1])
    assert result.gap == pytest.approx(0.0, abs=0.002)
    assert result.macro_uncertainty == pytest.approx(-0.8 * np.log2(0.8) - 0.2 * np.log2(0.2), abs=0.01)


def test_closure_not_lumpable_chain_is_not_closed():
    """Test that the block coarse-graining of a chain that is not lumpable leaves a gap."""
    micro = _chain(NOT_LUMPABLE, 50_000, 4)
    assert closure(micro, [0, 0, 1, 1]).gap > 0.02


def test_closure_partition_as_function():
    """Test that a partition given as a function of the micro series gives the same closure as its labels."""
    micro = _chain(NOT_LUMPABLE, 5000, 5)
    by_labels = closure(micro, [0, 0, 1, 1], history=2)
    by_function = closure(micro, lambda states: states // 2, history=2)
    assert by_function == by_labels
