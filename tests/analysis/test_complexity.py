"""Test generalized and regularity entropies, ordinal patterns of fields, and information storage and transfer."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from skcomplex.analysis import (
    MarkovAnalysis,
    active_information_storage,
    complexity_entropy,
    ensemble_entropy,
    symbolize,
    transfer_entropy,
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


def _coupled_rossler(coupling: float, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the first coordinates of a Rössler driver and a detuned response coupled one way, sampled every 1."""

    def drift(_, state):
        x1, y1, z1, x2, y2, z2 = state
        return [
            -y1 - z1,
            x1 + 0.2 * y1,
            0.2 + z1 * (x1 - 5.7),
            -0.95 * y2 - z2 + coupling * (x1 - x2),
            0.95 * x2 + 0.2 * y2,
            0.2 + z2 * (x2 - 5.7),
        ]

    times = np.arange(100, 100 + samples)
    solution = solve_ivp(drift, (0, times[-1]), [1, 1, 0, -2, 0.5, 0], t_eval=times, rtol=1e-8)
    return solution.y[0], solution.y[3]


def test_ensemble_entropy_h_theorem_of_a_relaxing_ensemble():
    """Test that the divergence to the stationary law of two relaxing chains falls and ends below 0.01 bits."""
    rng = np.random.default_rng(1)
    balls = 20
    urn = _ehrenfest(rng, balls, 150)
    binomial = np.asarray([math.comb(balls, count) for count in range(balls + 1)]) / 2**balls
    chain, matrix = _birth_death(rng, 600)
    stationary = MarkovAnalysis('stationary', matrix='transition').check().prepare(matrix).run().stationary_
    for samples, law in ((urn, binomial), (chain, stationary)):
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


def test_active_information_storage_of_a_periodic_sequence():
    """Test that the sequence 0011 stores 1 bit with history 2 and a fair coin stores almost nothing."""
    assert active_information_storage([0, 0, 1, 1] * 500, 2) == pytest.approx(1, abs=1e-5)
    assert active_information_storage(np.random.default_rng(6).integers(0, 2, 100_000), 2) < 0.01


def test_transfer_entropy_linear_gaussian():
    """Test that the coupled autoregression gives the Granger value in nats and no reverse transfer."""
    rng = np.random.default_rng(7)
    size = 20_000
    x, noise = rng.standard_normal(size), rng.standard_normal(size)
    y = np.zeros(size)
    for t in range(1, size):
        y[t] = 0.5 * y[t - 1] + 0.4 * x[t - 1] + noise[t]
    forward = transfer_entropy(x, y, estimator='knn', base=math.e)
    assert forward == pytest.approx(0.5 * math.log(1.16), abs=0.02)
    assert transfer_entropy(y, x, estimator='knn', base=math.e) < 0.01


def test_transfer_entropy_directed_coupling_of_two_domains():
    """Test that symbolized logistic maps and continuous Rössler oscillators coupled one way show the direction."""
    size = 20_000
    driver = _logistic(4.0, size)
    response = np.empty(size)
    response[0] = 0.6
    for t in range(1, size):
        response[t] = 0.8 * 4 * response[t - 1] * (1 - response[t - 1]) + 0.2 * 4 * driver[t - 1] * (1 - driver[t - 1])
    first, second = symbolize(driver, 'bins', 2), symbolize(response, 'bins', 2)
    assert transfer_entropy(first, second) > 5 * transfer_entropy(second, first)
    source, target = _coupled_rossler(0.2, 12_000)
    forward = transfer_entropy(source, target, estimator='knn', history=2, k=8)
    backward = transfer_entropy(target, source, estimator='knn', history=2, k=8)
    assert forward > 5 * max(backward, 0.0)
    assert forward > 0.1


def test_transfer_entropy_rejects_unknown_estimators():
    """Test that an unknown estimator raises an error."""
    with pytest.raises(AnalysisError, match='Unknown estimator'):
        transfer_entropy([0, 1], [1, 0], estimator='kernel')
