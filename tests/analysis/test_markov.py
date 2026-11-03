"""Test the Markov analysis of transition matrices, stationary laws, spectral gaps, and Markov state models."""

import math

import networkx as nx
import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.analysis import MarkovAnalysis
from skcomplex.errors import AnalysisError
from skcomplex.models import run


def _markov(kind: str, source: object, **settings: object) -> MarkovAnalysis:
    """Return the Markov analysis of a source for one kind, run."""
    return MarkovAnalysis(kind, **settings).check().prepare(source).run()


THREE_STATES = np.asarray([[0.90, 0.08, 0.02], [0.04 / 0.3, 1 - 0.04 / 0.3 - 0.12, 0.12], [0.05, 0.18, 0.77]])
FOUR_STATES = np.asarray(
    [[0.90, 0.09, 0.01, 0.0], [0.08, 0.90, 0.02, 0.0], [0.0, 0.02, 0.90, 0.08], [0.01, 0.0, 0.09, 0.90]],
)
DOUBLE_WELL = np.asarray(
    [
        [0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.5, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.45, 0.5, 0.05, 0.0, 0.0],
        [0.0, 0.0, 0.05, 0.5, 0.45, 0.0],
        [0.0, 0.0, 0.0, 0.2, 0.5, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.2, 0.8],
    ],
)


def _chain(matrix: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
    """Return a path of a discrete-time chain with a transition matrix, started in state 0."""
    cumulative = np.cumsum(matrix, axis=1)
    draws = rng.random(steps)
    states = np.zeros(steps, dtype=int)
    for t in range(1, steps):
        states[t] = int(np.searchsorted(cumulative[states[t - 1]], draws[t]))
    return states


def _double_well(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Return the bin of an overdamped particle in the well `(x² - 1)²` at temperature 0.25, every 0.1, and the step."""
    dt, record, steps = 0.005, 20, 4_000_000
    noise = rng.standard_normal(steps) * np.sqrt(2 * 0.25 * dt)
    x = np.empty(steps // record)
    position = -1.0
    for index in range(steps):
        position += -4 * position * (position**2 - 1) * dt + noise[index]
        if index % record == 0:
            x[index // record] = position
    bins = np.clip(((x + 1.8) / 3.6 * 50).astype(int), 0, 49)
    return bins, dt * record


def _schlogl(rng: np.random.Generator, volume: float, duration: float) -> tuple[np.ndarray, float]:
    """Return the copy number of a bistable Schlögl network every 0.05, with stable states at 1 and 5 per volume."""
    copies, time, next_record, recorded, interval = int(volume), 0.0, 0.0, [], 0.05
    while time < duration:
        density = copies / volume
        birth = volume * (9 * density**2 + 15)
        death = volume * (density**3 + 23 * density)
        step = rng.exponential(1 / (birth + death))
        while next_record < time + step and next_record < duration:
            recorded.append(copies)
            next_record += interval
        time += step
        copies += 1 if rng.random() < birth / (birth + death) else -1
    return np.asarray(recorded), interval


def _dwell_times(path: np.ndarray, low: float, high: float, dt: float) -> tuple[float, float]:
    """Return the mean times the path stays in each well, from entering its core at or below `low` or above `high`."""
    dwells: dict[int, list[float]] = {0: [], 1: []}
    side, entered = -1, 0
    for index, value in enumerate(path.tolist()):
        new = 0 if value <= low else 1 if value >= high else side
        if new != side:
            if side >= 0:
                dwells[side].append((index - entered) * dt)
            side, entered = new, index
    return float(np.mean(dwells[0])), float(np.mean(dwells[1]))


def test_markov_state_model_of_a_known_three_state_chain():
    """Test that 10^5 steps recover every transition probability and flat implied timescales."""
    path = _chain(THREE_STATES, 100_000, np.random.default_rng(0))
    model = _markov('state-model', path)
    np.testing.assert_allclose(model.transition_matrix_, THREE_STATES, atol=0.02)
    timescales = _markov('timescales', path, lags=[1, 2, 4], n_timescales=2).timescales_
    np.testing.assert_allclose(timescales, np.broadcast_to(timescales[0], timescales.shape), rtol=0.05)


def test_markov_state_model_double_well_on_two_domains():
    """Test that a double well and a bistable Schlögl network give two sets and a timescale set by the passages."""
    rng = np.random.default_rng(1)
    well, well_dt = _double_well(rng)
    network, network_dt = _schlogl(rng, 4.0, 4000.0)
    for states, dt, cores in ((well, well_dt, (18, 31)), (network, network_dt, (6, 18))):
        model = _markov('state-model', states, lag=20, n_metastable=2, dt=dt)
        assert len(model.metastable_sets_) == 2
        there, back = _dwell_times(states, *cores, dt)
        assert model.timescales_[0] == pytest.approx(1 / (1 / there + 1 / back), rel=0.2)


@pytest.mark.parametrize(
    ('settings', 'matrix', 'stationary', 'timescales'),
    [
        (
            {'reversible': True},
            [
                [0.904124, 0.086266, 0.005074, 0.004535],
                [0.081714, 0.896502, 0.021784, 0.0],
                [0.003981, 0.018042, 0.898018, 0.079959],
                [0.004589, 0.0, 0.103147, 0.892264],
            ],
            [0.231562, 0.244462, 0.295165, 0.228811],
            [34.048475, 5.093107, 4.434706],
        ),
        (
            {'reversible': True, 'lag': 3, 'dt': 0.5},
            [
                [0.755455, 0.214517, 0.017822, 0.012206],
                [0.203109, 0.737574, 0.054408, 0.004909],
                [0.013976, 0.045062, 0.750127, 0.190835],
                [0.012347, 0.005245, 0.246176, 0.736233],
            ],
            [0.231485, 0.244487, 0.295194, 0.228834],
            [17.354038, 2.529955, 2.224246],
        ),
        (
            {'reversible': False},
            [
                [0.904124, 0.085727, 0.010149, 0.0],
                [0.082225, 0.896502, 0.021272, 0.0],
                [0.0, 0.018465, 0.898018, 0.083517],
                [0.009178, 0.0, 0.098558, 0.892264],
            ],
            [0.231562, 0.244462, 0.295165, 0.228811],
            [33.785728, 5.065865, 4.461684],
        ),
        (
            {'reversible': False, 'lag': 3, 'dt': 0.5},
            [
                [0.755455, 0.213437, 0.028516, 0.002592],
                [0.204132, 0.737574, 0.055226, 0.003068],
                [0.00559, 0.044384, 0.750127, 0.199898],
                [0.022072, 0.007212, 0.234484, 0.736233],
            ],
            [0.231485, 0.244487, 0.295194, 0.228834],
            [17.223849, 2.520568, 2.236057],
        ),
    ],
)
def test_state_model_keeps_its_reference_values(
    settings: dict[str, object],
    matrix: list[list[float]],
    stationary: list[float],
    timescales: list[float],
):
    """Test the fits of a fixed path against the values recorded from the previous library."""
    model = _markov('state-model', _chain(FOUR_STATES, 20_000, np.random.default_rng(0)), **settings)
    assert model.states_.tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(model.transition_matrix_, matrix, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(model.stationary_, stationary, rtol=1e-3)
    np.testing.assert_allclose(model.timescales_, timescales, rtol=1e-3)


def test_state_model_of_several_paths_keeps_its_reference_values():
    """Test that the rows of two paths count together, against the recorded values."""
    path = _chain(FOUR_STATES, 20_000, np.random.default_rng(0))
    model = _markov('state-model', np.vstack([path[:10_000], path[10_000:]]))
    np.testing.assert_allclose(model.stationary_, [0.231573, 0.244474, 0.29513, 0.228823], rtol=1e-3)
    np.testing.assert_allclose(model.timescales_, [34.047193, 5.09301, 4.434383], rtol=1e-3)


def test_state_model_keeps_the_largest_connected_set():
    """Test that a transient state leaves the model and that a tie gives the later set, as recorded."""
    transient = _markov('state-model', [3, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
    assert transient.states_.tolist() == [0, 1]
    np.testing.assert_allclose(transient.transition_matrix_, [[0.2, 0.8], [0.75, 0.25]], rtol=1e-3)
    np.testing.assert_allclose(transient.stationary_, [0.483871, 0.516129], rtol=1e-3)
    np.testing.assert_allclose(transient.timescales_, [1.672697], rtol=1e-3)
    tie = _markov('state-model', [0, 1, 0, 1, 0, 2, 3, 2, 3, 2], reversible=False)
    assert tie.states_.tolist() == [2, 3]
    assert tie.timescales_.tolist() == [np.inf]


def test_state_model_finds_the_recorded_metastable_sets():
    """Test the partition of a double-well chain and its timescales against the recorded values."""
    model = _markov('state-model', _chain(DOUBLE_WELL, 20_000, np.random.default_rng(1)), n_metastable=2)
    assert sorted(found.tolist() for found in model.metastable_sets_) == [[0, 1, 2], [3, 4, 5]]
    np.testing.assert_allclose(model.timescales_, [82.644678, 2.623531, 2.207486, 0.557418, 0.467036], rtol=1e-3)


def test_timescales_keep_their_reference_values():
    """Test the timescales at every lag of both fits, and that a large number stops at the modes there are."""
    path = _chain(FOUR_STATES, 20_000, np.random.default_rng(0))
    reversible = _markov('timescales', path, lags=[1, 3], n_timescales=2, dt=0.5).timescales_
    np.testing.assert_allclose(reversible, [[17.024238, 2.546553], [17.354038, 2.529955]], rtol=1e-3)
    irreversible = _markov('timescales', path, lags=[1, 3], n_timescales=2, reversible=False).timescales_
    np.testing.assert_allclose(irreversible, [[33.785728, 5.065865], [34.447698, 5.041137]], rtol=1e-3)
    every = _markov('timescales', path, n_timescales=9).timescales_
    np.testing.assert_allclose(every, [[34.048475, 5.093107, 4.434706]], rtol=1e-3)


def test_state_model_needs_two_connected_states():
    """Test that a path with no two connected states raises an error."""
    with pytest.raises(AnalysisError, match='connected'):
        _markov('state-model', [2, 2, 2])
    with pytest.raises(AnalysisError, match='connected'):
        _markov('state-model', [0, 1], reversible=False)


def test_metastable_sets_of_every_number_of_sets():
    """Test one set, one set per state, and too many sets, as recorded from the previous library."""
    path = _chain(DOUBLE_WELL, 20_000, np.random.default_rng(1))
    whole = _markov('state-model', path, n_metastable=1).metastable_sets_
    assert [found.tolist() for found in whole] == [[0, 1, 2, 3, 4, 5]]
    apart = _markov('state-model', path, n_metastable=6).metastable_sets_
    assert [found.tolist() for found in apart] == [[0], [1], [2], [3], [4], [5]]
    with pytest.raises(AnalysisError, match='sets'):
        _markov('state-model', path, n_metastable=7)


def test_metastable_sets_need_a_reversible_model():
    """Test that metastable sets of a model without detailed balance raise an error."""
    with pytest.raises(AnalysisError, match='reversible'):
        MarkovAnalysis('state-model', n_metastable=2, reversible=False).check()


def test_stationary_distribution_ehrenfest_urn_and_ring_walk():
    """Test the binomial law of the Ehrenfest urn and the spectral gap of the lazy walk on a cycle of 11 nodes."""
    balls = 20
    rates = np.zeros((balls + 1, balls + 1))
    for count in range(balls + 1):
        if count > 0:
            rates[count, count - 1] = count
        if count < balls:
            rates[count, count + 1] = balls - count
    binomial = np.asarray([math.comb(balls, count) for count in range(balls + 1)]) / 2**balls
    np.testing.assert_allclose(_markov('stationary', rates, matrix='rate').stationary_, binomial, atol=1e-10)
    adjacency = nx.to_numpy_array(nx.cycle_graph(11))
    walk = 0.5 * np.eye(11) + 0.5 * adjacency / 2
    gap = _markov('spectral-gap', walk, matrix='transition')
    assert gap.spectral_gap_ == pytest.approx((1 - math.cos(2 * math.pi / 11)) / 2, abs=1e-10)
    assert gap.relaxation_times_[0] == pytest.approx(-1 / math.log(1 - gap.spectral_gap_), rel=1e-9)


def test_spectral_gap_of_rates():
    """Test the gap and relaxation time of a two-state rate matrix with rates 1 and 2."""
    gap = _markov('spectral-gap', [[0.0, 1.0], [2.0, 0.0]], matrix='rate')
    assert gap.spectral_gap_ == pytest.approx(3)
    assert gap.relaxation_times_.tolist() == pytest.approx([1 / 3])


def test_transition_matrix_estimates_a_known_chain():
    """Test that a long path recovers its transition matrix and counts transitions at a lag."""
    matrix = np.asarray([[0.9, 0.1], [0.3, 0.7]])
    path = _chain(matrix, 50_000, np.random.default_rng(0))
    np.testing.assert_allclose(_markov('transition-matrix', path).transition_matrix_, matrix, atol=0.02)
    assert _markov('transition-matrix', path, lag=2).counts_.sum() == path.size - 2
    assert _markov('transition-matrix', [0, 0], n_states=2).transition_matrix_[1].tolist() == [0.0, 1.0]


def test_stationary_distribution_of_a_path():
    """Test that the stationary law of a path is that of its estimated transition matrix."""
    path = _chain(np.asarray([[0.5, 0.5], [0.25, 0.75]]), 50_000, np.random.default_rng(3))
    np.testing.assert_allclose(_markov('stationary', path).stationary_, [1 / 3, 2 / 3], atol=0.02)


def test_markov_analysis_reads_the_jump_path_of_a_trajectory():
    """Test that a trajectory of a jump process gives the transition matrix of its jump path."""
    trajectory = run('poisson_process', n_processes=1, settings={'until': 30.0, 'random_state': 0}).trajectory_
    matrix = _markov('transition-matrix', trajectory).transition_matrix_
    np.testing.assert_allclose(matrix.sum(axis=1), 1)


def test_markov_analysis_rejects_bad_settings():
    """Test that an unknown kind, an ignored setting, a bad matrix, and a non-square matrix raise errors."""
    with pytest.raises(AnalysisError, match='kind'):
        MarkovAnalysis('unknown').check()
    with pytest.raises(AnalysisError, match='does not read n_metastable'):
        MarkovAnalysis('stationary', n_metastable=2).check()
    with pytest.raises(AnalysisError, match='matrix'):
        MarkovAnalysis('stationary', matrix='unknown').check()
    with pytest.raises(AnalysisError, match='square'):
        _markov('stationary', [[1.0, 0.0]], matrix='transition')


def test_markov_analysis_clones_without_results():
    """Test that a clone of a run analysis has its settings and no results."""
    analysis = _markov('transition-matrix', [0, 1, 0, 1], lag=1)
    cloned = clone(analysis)
    assert cloned.get_params() == analysis.get_params()
    assert not hasattr(cloned, 'transition_matrix_')
