"""Test the complexity-entropy plane and the entropy of ensembles."""

import math

import numpy as np
import pytest

from skcomplex.analysis import (
    complexity_entropy,
    ensemble_entropy,
    stationary_distribution,
)
from skcomplex.core.errors import AnalysisError

RUNS = 20_000


def _logistic(r: float, size: int, start: float = 0.3) -> np.ndarray:
    """Return iterates of the logistic map after a transient of 500."""
    values = np.empty(size + 500)
    values[0] = start
    for t in range(1, values.size):
        values[t] = r * values[t - 1] * (1 - values[t - 1])
    return values[500:]


def _ehrenfest(rng: np.random.Generator, balls: int, steps: int) -> np.ndarray:
    """Return an ensemble of lazy Ehrenfest urns of `balls` balls started all in the first urn, runs by steps."""
    states = np.full(RUNS, balls)
    frames = [states.copy()]
    for _ in range(steps):
        active = rng.random(RUNS) < 0.5
        moved = rng.random(RUNS) < states / balls
        states = states + np.where(active, np.where(moved, -1, 1), 0)
        frames.append(states.copy())
    return np.asarray(frames).T


def _birth_death(rng: np.random.Generator, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an ensemble of a birth-death chain on 0 to 20 started at 20, and its transition matrix."""
    size = 21
    matrix = np.zeros((size, size))
    for state in range(size):
        up, down = (0.3 if state < size - 1 else 0.0), (0.4 if state > 0 else 0.0)
        if state < size - 1:
            matrix[state, state + 1] = up
        if state > 0:
            matrix[state, state - 1] = down
        matrix[state, state] = 1 - up - down
    states = np.full(RUNS, size - 1)
    frames = [states.copy()]
    cumulative = matrix.cumsum(axis=1)
    for _ in range(steps):
        states = (rng.random(RUNS)[:, None] > cumulative[states]).sum(axis=1)
        frames.append(states.copy())
    return np.asarray(frames).T, matrix


def test_ensemble_entropy_h_theorem_of_a_relaxing_ensemble():
    """Test that the divergence to the stationary law of two relaxing chains falls and ends below 0.01 bits."""
    rng = np.random.default_rng(1)
    balls = 20
    urn = _ehrenfest(rng, balls, 150)
    binomial = np.asarray([math.comb(balls, count) for count in range(balls + 1)]) / 2**balls
    chain, matrix = _birth_death(rng, 600)
    for samples, law in ((urn, binomial), (chain, stationary_distribution(matrix))):
        relaxation = ensemble_entropy(samples, law)
        noise = 5 * relaxation.error
        assert np.all(np.diff(relaxation.divergence) <= noise)
        assert relaxation.divergence[-1] < 0.01


def test_ensemble_entropy_rejects_outcomes_outside_the_law():
    """Test that outcomes beyond the reference law raise an error."""
    with pytest.raises(AnalysisError, match='Outcomes must lie'):
        ensemble_entropy([[0, 3]], [0.5, 0.5])


def test_complexity_entropy_plane():
    """Test that noise, the logistic map at r = 4, and a sine take their places between the bounds of the plane."""
    noise = complexity_entropy(np.random.default_rng(5).standard_normal(200_000), 6)
    chaos = complexity_entropy(_logistic(4.0, 200_000), 6)
    tone = complexity_entropy(np.sin(np.arange(200_000) * 0.05), 6)
    assert noise.entropy > 0.99
    assert noise.complexity < 0.02
    assert chaos.complexity > 0.3
    for point in (noise, chaos, tone):
        low = np.interp(point.entropy, *point.minimum[np.argsort(point.minimum[:, 0])].T)
        high = np.interp(point.entropy, *point.maximum[np.argsort(point.maximum[:, 0])].T)
        assert low - 1e-6 <= point.complexity <= high + 1e-6
