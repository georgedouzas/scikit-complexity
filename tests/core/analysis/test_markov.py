"""Test Markov state models, implied timescales, committors, and mean first-passage times."""

import numpy as np
import pytest

from skcomplex.core.analysis import committor, implied_timescales, markov_state_model, mean_first_passage_time

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
    side, entered, dwells = -1, 0, {0: [], 1: []}
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
    model = markov_state_model(path)
    np.testing.assert_allclose(model.transition_matrix, THREE_STATES, atol=0.02)
    timescales = implied_timescales(path, [1, 2, 4], k=2)
    np.testing.assert_allclose(timescales, np.broadcast_to(timescales[0], timescales.shape), rtol=0.05)


def test_markov_state_model_double_well_on_two_domains():
    """Test that a double well and a bistable Schlögl network give two sets and a timescale set by the passages."""
    rng = np.random.default_rng(1)
    well, well_dt = _double_well(rng)
    network, network_dt = _schlogl(rng, 4.0, 4000.0)
    for states, dt, cores in ((well, well_dt, (18, 31)), (network, network_dt, (6, 18))):
        model = markov_state_model(states, lag=20, n_metastable=2, dt=dt)
        assert len(model.metastable_sets) == 2
        there, back = _dwell_times(states, *cores, dt)
        assert model.timescales[0] == pytest.approx(1 / (1 / there + 1 / back), rel=0.2)


def test_committor_and_mean_first_passage_time_of_a_walk():
    """Test the committor of a lazy walk on a path of 5 states and its mean first-passage time."""
    walk = np.zeros((5, 5))
    for state in range(5):
        walk[state, max(state - 1, 0)] += 0.25
        walk[state, min(state + 1, 4)] += 0.25
        walk[state, state] += 0.5
    np.testing.assert_allclose(committor(walk, [0], [4]), np.linspace(0, 1, 5), atol=1e-12)
    assert mean_first_passage_time(walk, [0], [4]) > mean_first_passage_time(walk, [0], [1])
