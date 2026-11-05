"""Test effective information, causal emergence, the integrated information criteria, and closure."""

import numpy as np
import pytest

from skcomplex.analysis import EmergenceAnalysis, MarkovAnalysis
from skcomplex.errors import AnalysisError, UnobservedStatesWarning

BLOCKS = np.asarray([[0.3, 0.7], [0.6, 0.4]])
NOISY_STATES = 7
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


def _emergence(kind: str, source: object, **settings: object) -> EmergenceAnalysis:
    """Return the run emergence analysis of a source for one kind."""
    return EmergenceAnalysis(kind, **settings).prepare(source).run()


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


def _parity(micro: np.ndarray) -> np.ndarray:
    """Return the parity of the micro bits at every time."""
    return micro.sum(axis=1) % 2


@pytest.mark.parametrize('weighting', ['uniform', 'stationary'])
def test_partition_coarse_grain_lumpable_chain(weighting):
    """Test that a lumpable four-state chain coarse-grains to its block matrix with either weighting."""
    macro = _emergence('causal', _lumpable(), partition=[0, 0, 1, 1], weighting=weighting).macro_matrix_
    np.testing.assert_allclose(macro, BLOCKS, atol=1e-12)
    np.testing.assert_allclose(macro.sum(axis=1), 1.0, atol=1e-12)


def test_partition_coarse_grain_by_function_matches_the_macro_path():
    """Test that a labelling function gives the macro matrix that the macro path of the chain estimates."""
    micro = _lumpable()
    path = _chain(micro, 40_000, np.random.default_rng(0))
    found = _emergence('causal', micro, partition=lambda states: states // 2, weighting='stationary')
    assert found.partition_.tolist() == [0, 0, 1, 1]
    estimated = MarkovAnalysis().prepare(path // 2).run().transition_matrix_
    np.testing.assert_allclose(estimated, found.macro_matrix_, atol=0.02)


def test_partition_coarse_grain_rejects_missing_labels():
    """Test that a partition with an empty block or the wrong length is rejected."""
    with pytest.raises(AnalysisError, match='label 1'):
        _emergence('causal', np.eye(3), partition=[0, 0, 2])
    with pytest.raises(AnalysisError, match='one label per state'):
        _emergence('causal', np.eye(3), partition=[0, 1])


@pytest.mark.parametrize('n_states', [2, 5, 16])
def test_effective_information_permutation(n_states):
    """Test that a permutation of n states has effective information and determinism log2 n and no degeneracy."""
    permutation = np.eye(n_states)[np.random.default_rng(n_states).permutation(n_states)]
    found = _emergence('effective-information', permutation)
    assert found.information_ == pytest.approx(np.log2(n_states), abs=1e-12)
    assert found.determinism_ == pytest.approx(np.log2(n_states), abs=1e-12)
    assert found.degeneracy_ == pytest.approx(0.0, abs=1e-12)


def test_effective_information_fully_noisy_chain():
    """Test that uniform rows carry no effective information, with equal determinism and degeneracy."""
    found = _emergence('effective-information', np.full((6, 6), 1 / 6))
    assert found.information_ == pytest.approx(0.0, abs=1e-12)
    assert found.determinism_ == pytest.approx(found.degeneracy_, abs=1e-12)


def test_effective_information_warns_on_unobserved_states():
    """Test that rows without transitions are dropped with a warning that names them."""
    counts = np.asarray([[0, 4, 0], [0, 0, 0], [3, 0, 0]])
    with pytest.warns(UnobservedStatesWarning, match=r'\[1\]'):
        found = _emergence('effective-information', counts)
    assert found.information_ == pytest.approx(1.0, abs=1e-12)
    assert found.states_.tolist() == [0, 2]


def test_effective_information_ignores_a_partition():
    """Test that the effective information rejects a partition, which only the other kinds read."""
    with pytest.raises(AnalysisError, match='does not read partition'):
        EmergenceAnalysis('effective-information', partition=[0, 1]).prepare()


def test_causal_emergence_noisy_micro_states():
    """Test the emergence of Hoel, Albantakis, and Tononi's chain and that the greedy search finds its partition."""
    matrix = _noisy_chain()
    merged = [0] * NOISY_STATES + [1]
    given = _emergence('causal', matrix, partition=merged)
    assert given.micro_information_ == pytest.approx(3 - 7 / 8 * np.log2(7), abs=1e-12)
    assert given.macro_information_ == pytest.approx(1.0, abs=1e-12)
    assert given.gain_ == pytest.approx(1 - 3 + 7 / 8 * np.log2(7), abs=1e-12)
    found = _emergence('causal', matrix)
    assert found.partition_.tolist() == merged
    assert found.gain_ == pytest.approx(given.gain_, abs=1e-12)


def test_causal_emergence_exhaustive_agrees_on_small_chain():
    """Test that the exhaustive search finds the merged partition and is refused above its state count."""
    found = _emergence('causal', _noisy_chain(), search='exhaustive')
    assert found.partition_.tolist() == [0] * NOISY_STATES + [1]
    with pytest.raises(AnalysisError, match='exhaustive'):
        _emergence('causal', np.full((12, 12), 1 / 12), search='exhaustive')


def test_causal_emergence_permutation_has_no_gain():
    """Test that the greedy search on a permutation returns the identity partition with zero gain."""
    permutation = np.eye(6)[[1, 2, 3, 4, 5, 0]]
    found = _emergence('causal', permutation)
    assert found.partition_.tolist() == list(range(6))
    assert found.gain_ == pytest.approx(0.0, abs=1e-12)


def test_emergence_criteria_parity_is_emergent():
    """Test that the persistent parity of noisy bits has a positive Ψ and no downward causation."""
    micro = _persistent_parity(20_000, 4, 0.05, 0)
    criteria = _emergence('criteria', micro, partition=_parity)
    expected = 1 + 0.05 * np.log2(0.05) + 0.95 * np.log2(0.95)
    assert criteria.psi_ == pytest.approx(expected, abs=0.02)
    assert criteria.psi_ > 0
    assert criteria.gamma_ == pytest.approx(0.0, abs=0.01)
    assert criteria.delta_ <= 0.01


def test_emergence_criteria_micro_variable_is_not_emergent():
    """Test that one micro bit of the persistent parity system has no positive Ψ."""
    micro = _persistent_parity(20_000, 4, 0.05, 1)
    assert _emergence('criteria', micro, partition=lambda values: values[:, 0]).psi_ <= 0.01


def test_emergence_criteria_gaussian_difference_hides_common_noise():
    """Test that the difference of two variables sharing a strong noise reveals a slow process with a positive Ψ."""
    rng = np.random.default_rng(2)
    slow = np.zeros(10_000)
    for t in range(1, slow.size):
        slow[t] = 0.95 * slow[t - 1] + rng.normal()
    noise = 10.0 * rng.normal(size=slow.size)
    micro = np.column_stack([slow + noise, noise])
    criteria = _emergence('criteria', micro, partition=lambda values: values[:, 0] - values[:, 1], estimator='gaussian')
    assert criteria.psi_ > 1.0


def test_emergence_criteria_unknown_estimator():
    """Test that an unknown estimator raises an error naming the known ones."""
    with pytest.raises(AnalysisError, match='plugin'):
        EmergenceAnalysis('criteria', partition=_parity, estimator='binned').prepare()


def test_emergence_criteria_needs_a_partition():
    """Test that the criteria without a macro feature raise an error asking for the partition."""
    with pytest.raises(AnalysisError, match='partition'):
        EmergenceAnalysis('criteria').prepare()


def test_closure_lumpable_chain_is_closed():
    """Test that the block coarse-graining of a lumpable chain is closed within the estimator error."""
    micro = _chain(LUMPABLE, 50_000, np.random.default_rng(3))
    result = _emergence('closure', micro, partition=[0, 0, 1, 1])
    assert result.gap_ == pytest.approx(0.0, abs=0.002)
    assert result.macro_uncertainty_ == pytest.approx(-0.8 * np.log2(0.8) - 0.2 * np.log2(0.2), abs=0.01)


def test_closure_not_lumpable_chain_is_not_closed():
    """Test that the block coarse-graining of a chain that is not lumpable leaves a gap."""
    micro = _chain(NOT_LUMPABLE, 50_000, np.random.default_rng(4))
    assert _emergence('closure', micro, partition=[0, 0, 1, 1]).gap_ > 0.02


def test_closure_partition_as_function():
    """Test that a partition given as a function of the micro series gives the same closure as its labels."""
    micro = _chain(NOT_LUMPABLE, 5000, np.random.default_rng(5))
    by_labels = _emergence('closure', micro, partition=[0, 0, 1, 1], history=2)
    by_function = _emergence('closure', micro, partition=lambda states: states // 2, history=2)
    assert by_function.macro_uncertainty_ == by_labels.macro_uncertainty_
    assert by_function.micro_uncertainty_ == by_labels.micro_uncertainty_


def test_emergence_criteria_knn_agrees_with_gaussian():
    """Test that the nearest-neighbour criteria of a linear Gaussian system agree in sign with the Gaussian ones."""
    rng = np.random.default_rng(6)
    slow = np.zeros(2000)
    for t in range(1, slow.size):
        slow[t] = 0.95 * slow[t - 1] + rng.normal()
    noise = 10.0 * rng.normal(size=slow.size)
    micro = np.column_stack([slow + noise, noise])
    criteria = _emergence(
        'criteria',
        micro,
        partition=lambda values: np.column_stack([values[:, 0] - values[:, 1], values[:, 1]]),
        estimator='knn',
    )
    assert criteria.psi_ > 0.5
