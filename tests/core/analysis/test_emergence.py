"""Test partition coarse-graining, effective information, and causal emergence of transition matrices."""

import numpy as np
import pytest

from skcomplex.core.analysis import (
    causal_emergence,
    effective_information,
    partition_coarse_grain,
    transition_matrix,
)
from skcomplex.core.base import AnalysisError, UnobservedStatesWarning

BLOCKS = np.asarray([[0.3, 0.7], [0.6, 0.4]])
NOISY_STATES = 7


def _lumpable() -> np.ndarray:
    """Return a four-state chain whose blocks {0, 1} and {2, 3} move with the block matrix, unequal within blocks."""
    within = np.asarray([[0.2, 0.8], [0.5, 0.5]])
    return np.block(
        [
            [BLOCKS[0, 0] * within, BLOCKS[0, 1] * within[::-1]],
            [BLOCKS[1, 0] * within[::-1], BLOCKS[1, 1] * within],
        ],
    )


def _noisy_chain() -> np.ndarray:
    """Return the eight-state chain whose first seven states move uniformly among themselves and whose last absorbs."""
    matrix = np.zeros((NOISY_STATES + 1, NOISY_STATES + 1))
    matrix[:NOISY_STATES, :NOISY_STATES] = 1 / NOISY_STATES
    matrix[NOISY_STATES, NOISY_STATES] = 1.0
    return matrix


def _chain(matrix: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
    """Return a path of a discrete-time chain with a transition matrix, started in state 0."""
    cumulative = np.cumsum(matrix, axis=1)
    draws = rng.random(steps)
    states = np.zeros(steps, dtype=int)
    for t in range(1, steps):
        states[t] = min(int(np.searchsorted(cumulative[states[t - 1]], draws[t])), matrix.shape[0] - 1)
    return states


@pytest.mark.parametrize('weighting', ['uniform', 'stationary'])
def test_partition_coarse_grain_lumpable_chain(weighting):
    """Test that a lumpable four-state chain coarse-grains to its block matrix with either weighting."""
    macro = partition_coarse_grain(_lumpable(), [0, 0, 1, 1], matrix=True, weighting=weighting)
    np.testing.assert_allclose(macro, BLOCKS, atol=1e-12)
    np.testing.assert_allclose(macro.sum(axis=1), 1.0, atol=1e-12)


def test_partition_coarse_grain_trajectory_by_function():
    """Test that a labelling function maps every frame and that the macro estimate matches the coarse-grained matrix."""
    micro = _lumpable()
    path = _chain(micro, 40_000, np.random.default_rng(0))
    labels = partition_coarse_grain(path, lambda state: state // 2)
    np.testing.assert_array_equal(labels, path // 2)
    expected = partition_coarse_grain(micro, lambda state: state // 2, matrix=True, weighting='stationary')
    np.testing.assert_allclose(transition_matrix(labels), expected, atol=0.02)


def test_partition_coarse_grain_rejects_missing_labels():
    """Test that a partition with an empty block or the wrong length is rejected."""
    with pytest.raises(AnalysisError, match='label 1'):
        partition_coarse_grain(np.eye(3), [0, 0, 2], matrix=True)
    with pytest.raises(AnalysisError, match='one label per state'):
        partition_coarse_grain(np.eye(3), [0, 1], matrix=True)


@pytest.mark.parametrize('n_states', [2, 5, 16])
def test_effective_information_permutation(n_states):
    """Test that a permutation of n states has effective information and determinism log2 n and no degeneracy."""
    permutation = np.eye(n_states)[np.random.default_rng(n_states).permutation(n_states)]
    found = effective_information(permutation)
    assert found.information == pytest.approx(np.log2(n_states), abs=1e-12)
    assert found.determinism == pytest.approx(np.log2(n_states), abs=1e-12)
    assert found.degeneracy == pytest.approx(0.0, abs=1e-12)


def test_effective_information_fully_noisy_chain():
    """Test that uniform rows carry no effective information, with equal determinism and degeneracy."""
    found = effective_information(np.full((6, 6), 1 / 6))
    assert found.information == pytest.approx(0.0, abs=1e-12)
    assert found.determinism == pytest.approx(found.degeneracy, abs=1e-12)


def test_effective_information_warns_on_unobserved_states():
    """Test that rows without transitions are dropped with a warning that names them."""
    counts = np.asarray([[0, 4, 0], [0, 0, 0], [3, 0, 0]])
    with pytest.warns(UnobservedStatesWarning, match=r'\[1\]'):
        found = effective_information(counts)
    assert found.information == pytest.approx(1.0, abs=1e-12)


def test_causal_emergence_noisy_micro_states():
    """Test the emergence of Hoel, Albantakis, and Tononi's chain and that the greedy search finds its partition."""
    matrix = _noisy_chain()
    merged = [0] * NOISY_STATES + [1]
    given = causal_emergence(matrix, merged)
    assert given.micro.information == pytest.approx(3 - 7 / 8 * np.log2(7), abs=1e-12)
    assert given.macro.information == pytest.approx(1.0, abs=1e-12)
    assert given.gain == pytest.approx(1 - 3 + 7 / 8 * np.log2(7), abs=1e-12)
    found = causal_emergence(matrix)
    assert found.partition.tolist() == merged
    assert found.gain == pytest.approx(given.gain, abs=1e-12)


def test_causal_emergence_exhaustive_agrees_on_small_chain():
    """Test that the exhaustive search finds the merged partition and is refused above its state count."""
    found = causal_emergence(_noisy_chain(), search='exhaustive')
    assert found.partition.tolist() == [0] * NOISY_STATES + [1]
    with pytest.raises(AnalysisError, match='exhaustive'):
        causal_emergence(np.full((12, 12), 1 / 12), search='exhaustive')


def test_causal_emergence_permutation_has_no_gain():
    """Test that the greedy search on a permutation returns the identity partition with zero gain."""
    permutation = np.eye(6)[[1, 2, 3, 4, 5, 0]]
    found = causal_emergence(permutation)
    assert found.partition.tolist() == list(range(6))
    assert found.gain == pytest.approx(0.0, abs=1e-12)
