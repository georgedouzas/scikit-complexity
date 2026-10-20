"""Test first passages, returns, laminar phases, and absorption."""

import numpy as np
import pytest

from skcomplex.core.analysis import (
    first_passage_times,
    fit_power_law,
    hitting_probabilities,
    laminar_durations,
    return_times,
)
from skcomplex.core.base import AnalysisError

RUNS = 2000


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
    times = first_passage_times(paths, 5.0, times=np.arange(paths.shape[1]) * dt)
    assert np.nanmean(times) == pytest.approx(10, rel=0.05)
    walks = _absorbed_walks(rng, 3, 10, 600, moran=False)
    durations = first_passage_times(walks, states=[0, 10])
    assert np.nanmean(durations) == pytest.approx(3 * (10 - 3), rel=0.05)


def test_first_passage_times_downward_and_never():
    """Test passages from above and paths that never pass."""
    assert first_passage_times([3.0, 2.0, 1.0], 1.5, direction='down').tolist() == [2.0]
    assert np.isnan(first_passage_times([0.0, 0.0], 1.0)[0])
    with pytest.raises(AnalysisError, match='Give a threshold'):
        first_passage_times([0.0, 1.0])
    with pytest.raises(AnalysisError, match='Unknown direction'):
        first_passage_times([0.0, 1.0], 1.0, direction='sideways')


def test_return_times_on_two_domains():
    """Test that returns of a random walk and laminar phases of on-off intermittency have tails of exponent 3/2."""
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.choice([-1, 1], 2_000_000))
    returns = return_times(walk) / 2
    assert fit_power_law(returns, xmin=10, discrete=True, bootstrap=0).alpha == pytest.approx(1.5, abs=0.1)
    size = 2_000_000
    values = np.empty(size)
    values[0] = 0.5
    factors = np.e * rng.random(size)
    for t in range(1, size):
        values[t] = max(factors[t] * values[t - 1] * (1 - values[t - 1]), 1e-12)
    durations = laminar_durations(values, 1e-3)
    assert fit_power_law(durations, xmin=10, discrete=True, bootstrap=0).alpha == pytest.approx(1.5, abs=0.1)


def test_hitting_probabilities_fixation_of_a_mutant():
    """Test that one neutral mutant in Moran populations of 20 fixes with probability 1/20 within 0.01."""
    paths = _absorbed_walks(np.random.default_rng(2), 1, 20, 20_000, moran=True)
    probabilities = hitting_probabilities(paths, [0, 20])
    assert probabilities[20] == pytest.approx(1 / 20, abs=0.01)
    assert probabilities[0] + probabilities[20] == pytest.approx(1, abs=0.01)
