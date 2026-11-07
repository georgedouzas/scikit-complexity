"""Test the passage analysis of first passages, absorption, returns, and committors."""

import numpy as np
import pytest

from skcomplex.core.analysis import PassageResult, fit_law, passage
from skcomplex.core.errors import AnalysisError

RUNS = 2000


def _passage(kind: str, source: object, **settings: object) -> PassageResult:
    """Return the passage analysis of a source for one kind, run."""
    return passage(source, kind=kind, **settings)


def _absorbed_walks(rng: np.random.Generator, start: int, size: int, steps: int, *, moran: bool) -> np.ndarray:
    """Return walks from a start absorbed at 0 and `size`, by fair steps, or by neutral Moran steps kept every 20."""
    runs = 5000 if moran else RUNS
    states = np.full(runs, start)
    stride = 20 if moran else 1
    path = [states.copy()]
    for step in range(steps):
        inside = (states > 0) & (states < size)
        if moran:
            rate = states * (size - states) / size**2
            draw = rng.random(runs)
            change = np.where(draw < rate, 1, np.where(draw < 2 * rate, -1, 0))
        else:
            change = rng.choice([-1, 1], runs)
        states = np.where(inside, states + change, states)
        if (step + 1) % stride == 0:
            path.append(states.copy())
    return np.asarray(path).T


def test_first_passage_times_drifted_diffusion_and_gamblers_ruin():
    """Test the mean passage of drifted Brownian paths to 5 and the mean duration of fair ruin from 3 in 0 to 10."""
    rng = np.random.default_rng(0)
    dt = 0.01
    steps = rng.normal(0.5 * dt, np.sqrt(dt), (RUNS, 6000))
    paths = np.concatenate([np.zeros((RUNS, 1)), np.cumsum(steps, axis=1)], axis=1)
    times = _passage('first-passage', paths, threshold=5.0, dt=dt).passage_times
    assert np.nanmean(times) == pytest.approx(10, rel=0.05)
    walks = _absorbed_walks(rng, 3, 10, 600, moran=False)
    durations = _passage('first-passage', walks, target=[0, 10]).passage_times
    assert np.nanmean(durations) == pytest.approx(3 * (10 - 3), rel=0.05)


def test_first_passage_times_downward_and_never():
    """Test passages from above and paths that never pass."""
    down = _passage('first-passage', [3.0, 2.0, 1.0], threshold=1.5, direction='down')
    assert down.passage_times.tolist() == [2.0]
    assert np.isnan(_passage('first-passage', [0.0, 0.0], threshold=1.0).passage_times[0])
    with pytest.raises(AnalysisError, match='needs a threshold'):
        passage([0, 1, 0, 1], kind='first-passage')
    with pytest.raises(AnalysisError, match='direction'):
        passage([0, 1, 0, 1], kind='first-passage', threshold=1.0, direction='sideways')
    with pytest.raises(AnalysisError, match='does not read level'):
        passage([0, 1, 0, 1], kind='first-passage', threshold=1.0, level=2.0)


def test_return_times_of_a_random_walk():
    """Test that returns of a random walk have a tail of exponent 3/2."""
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.choice([-1, 1], 2_000_000))
    returns = _passage('return-times', walk).return_times / 2
    fit = fit_law(returns, minimum=10, discrete=True, bootstrap=0)
    assert fit.exponent == pytest.approx(1.5, abs=0.1)


def test_hitting_probabilities_fixation_of_a_mutant():
    """Test that one neutral mutant in Moran populations of 20 fixes with probability 1/20 within 0.01."""
    paths = _absorbed_walks(np.random.default_rng(2), 1, 20, 20_000, moran=True)
    absorbed, fixed = _passage('hitting', paths, target=[0, 20]).hitting_probabilities
    assert fixed == pytest.approx(1 / 20, abs=0.01)
    assert absorbed + fixed == pytest.approx(1, abs=0.01)


def test_committor_and_mean_first_passage_time_keep_their_reference_values():
    """Test a chain of four states, with sets, a lag, and a time step, against the recorded values."""
    chain = np.asarray(
        [[0.90, 0.09, 0.01, 0.0], [0.08, 0.90, 0.02, 0.0], [0.0, 0.02, 0.90, 0.08], [0.01, 0.0, 0.09, 0.90]],
    )
    one = _passage('committor', chain, origin=[0], target=[3]).committor
    np.testing.assert_allclose(one, [0.0, 0.166667, 0.833333, 1.0], rtol=1e-3, atol=1e-9)
    two = _passage('committor', chain, origin=[0, 1], target=[3]).committor
    np.testing.assert_allclose(two, [0.0, 0.0, 0.8, 1.0], rtol=1e-3, atol=1e-9)
    time = _passage('mean-first-passage', chain, origin=[0], target=[3]).mean_first_passage_time
    assert time == pytest.approx(96.428571, rel=1e-3)
    both = _passage('mean-first-passage', chain, origin=[0, 1], target=[2, 3], lag=2, dt=0.5)
    assert both.mean_first_passage_time == pytest.approx(65.959821, rel=1e-3)
    overlap = _passage('mean-first-passage', chain, origin=[0, 3], target=[3]).mean_first_passage_time
    assert overlap == pytest.approx(49.876847, rel=1e-3)


def test_committor_and_mean_first_passage_time_of_periodic_chains():
    """Test a ring of four states, which never settles, and a chain of two states that swap."""
    ring = np.roll(np.eye(4), 1, axis=1)
    np.testing.assert_allclose(_passage('committor', ring, origin=[0], target=[2]).committor, [0, 1, 1, 0], atol=1e-9)
    assert _passage('mean-first-passage', ring, origin=[0], target=[2]).mean_first_passage_time == pytest.approx(2.0)
    swap = [[0.0, 1.0], [1.0, 0.0]]
    assert _passage('mean-first-passage', swap, origin=[0], target=[1]).mean_first_passage_time == pytest.approx(1.0)


def test_committor_of_an_absorbing_walk():
    """Test the committor of a walk absorbed at both ends, which is linear in the state."""
    walk = [[1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]]
    committor = _passage('committor', walk, origin=[0], target=[3]).committor
    np.testing.assert_allclose(committor, [0.0, 1 / 3, 2 / 3, 1.0], atol=1e-9)


def test_mean_first_passage_time_of_a_reducible_chain_raises_an_error():
    """Test that a chain with no single stationary law raises an error rather than give a time."""
    walk = [[1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]]
    with pytest.raises(AnalysisError, match='reducible'):
        _passage('mean-first-passage', walk, origin=[0], target=[3])


def test_committor_and_mean_first_passage_time_of_a_walk():
    """Test the committor of a lazy walk on a path of 5 states and its mean first-passage time."""
    walk = np.zeros((5, 5))
    for state in range(5):
        walk[state, max(state - 1, 0)] += 0.25
        walk[state, min(state + 1, 4)] += 0.25
        walk[state, state] += 0.5
    committor = _passage('committor', walk, origin=[0], target=[4]).committor
    np.testing.assert_allclose(committor, np.linspace(0, 1, 5), atol=1e-12)
    far = _passage('mean-first-passage', walk, origin=[0], target=[4]).mean_first_passage_time
    near = _passage('mean-first-passage', walk, origin=[0], target=[1]).mean_first_passage_time
    assert far > near
    assert (
        _passage('mean-first-passage', [[0.5, 0.5], [0.5, 0.5]], origin=[0], target=[1]).mean_first_passage_time == 2.0
    )
