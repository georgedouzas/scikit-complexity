"""Test the Markov analysis of transition matrices, stationary laws, spectral gaps, and Markov state models."""

import math

import networkx as nx
import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.analysis import MarkovAnalysis
from skcomplex.core.errors import AnalysisError
from skcomplex.models import run


def _markov(kind: str, source: object, **settings: object) -> MarkovAnalysis:
    """Return the Markov analysis of a source for one kind, run."""
    return MarkovAnalysis(kind, **settings).check().prepare(source).run()


THREE_STATES = np.asarray([[0.90, 0.08, 0.02], [0.04 / 0.3, 1 - 0.04 / 0.3 - 0.12, 0.12], [0.05, 0.18, 0.77]])


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
