"""Test generalized and regularity entropies, ordinal patterns of fields, and information storage and transfer."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from skcomplex.core.analysis import (
    active_information_storage,
    approximate_entropy,
    break_ties,
    complexity_entropy,
    ensemble_entropy,
    fields_of,
    multiscale_entropy,
    permutation_entropy,
    renyi_entropy,
    sample_entropy,
    shannon_entropy,
    snapshots_of,
    spectral_entropy,
    stationary_distribution,
    symbolize,
    transfer_entropy,
    tsallis_entropy,
)
from skcomplex.core.base import AnalysisError
from skcomplex.core.runtime import Simulation
from skcomplex.models import create, run

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


def _pink_noise(rng: np.random.Generator, size: int) -> np.ndarray:
    """Return 1/f noise shaped in Fourier space."""
    frequencies = np.fft.rfftfreq(size)
    amplitudes = np.zeros_like(frequencies)
    amplitudes[1:] = 1 / np.sqrt(frequencies[1:])
    return np.fft.irfft(amplitudes * np.exp(2j * np.pi * rng.random(frequencies.size)), n=size)


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


def test_renyi_entropy_of_a_uniform_distribution():
    """Test that every order of the Rényi entropy of a uniform law on 8 outcomes gives 3 bits."""
    uniform = np.full(8, 1 / 8)
    for order in (0, 0.5, 1, 2, math.inf):
        assert renyi_entropy(uniform, order, probabilities=True) == pytest.approx(3, abs=1e-12)


def test_tsallis_entropy_limit():
    """Test that the Tsallis entropy near index 1 equals the Shannon entropy in nats on two domains."""
    occupation = np.random.default_rng(0).poisson(3.0, 10_000)
    wealth = snapshots_of(run('boltzmann_wealth', settings={'until': 50}).trajectory_).states['wealth'][-1]
    for samples in (occupation, wealth):
        _, counts = np.unique(samples, return_counts=True)
        probabilities = counts / counts.sum()
        expected = shannon_entropy(probabilities, base=math.e, probabilities=True)
        assert tsallis_entropy(probabilities, 1 + 1e-9, probabilities=True) == pytest.approx(expected, abs=1e-6)
    assert tsallis_entropy([0.5, 0.5], 1, probabilities=True) == pytest.approx(math.log(2))


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


def test_sample_entropy_of_white_noise():
    """Test that 10,000 values of white noise give a sample entropy within 0.1 of 2.2 nats."""
    noise = np.random.default_rng(2).standard_normal(10_000)
    assert sample_entropy(noise) == pytest.approx(2.2, abs=0.1)
    assert approximate_entropy(noise) > 1.5


def test_sample_entropy_of_periodic_orbits_on_two_domains():
    """Test that the Van der Pol limit cycle and the logistic map at r = 3.2 give sample entropies below 0.05."""
    trajectory = Simulation(create('van_der_pol'), dt=0.01).run(until=100.0).trajectory_
    cycle = trajectory['oscillator.x'][len(trajectory) // 2 :]
    assert sample_entropy(cycle) < 0.05
    assert sample_entropy(_logistic(3.2, 3000)) < 0.05


def test_multiscale_entropy_signature():
    """Test that white noise loses entropy with the scale while 1/f noise keeps it within 15 percent."""
    rng = np.random.default_rng(3)
    white = multiscale_entropy(rng.standard_normal(20_000)).entropies
    pink = multiscale_entropy(_pink_noise(rng, 20_000)).entropies
    assert np.all(np.diff(white) < 0)
    assert np.all(np.abs(pink / pink[0] - 1) < 0.15)


def test_spectral_entropy_of_a_tone_and_of_noise():
    """Test that a pure sine gives below 0.1 and white noise above 0.9."""
    tone = np.sin(2 * np.pi * 64 * np.arange(4096) / 4096)
    assert spectral_entropy(tone) < 0.1
    assert spectral_entropy(np.random.default_rng(4).standard_normal(4096)) > 0.9


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


def test_permutation_entropy_of_fields_of_two_domains():
    """Test that a hot Ising lattice gives above 0.98 and a Gray-Scott pattern at least 0.1 less."""
    spins = fields_of(run('ising', size=64, temperature=10.0, settings={'until': 20}).trajectory_)[-1]
    spots = fields_of(run('gray_scott').trajectory_)[-1]
    hot = permutation_entropy(spins, (2, 2), ties='random', random_state=0)
    pattern = permutation_entropy(spots, (2, 2), ties='random', random_state=0)
    assert hot > 0.98
    assert pattern < hot - 0.1


def test_break_ties_rejects_unknown_rules():
    """Test that an unknown tie rule raises an error and the order rule keeps the values."""
    np.testing.assert_array_equal(break_ties([1.0, 1.0], 'order'), [1.0, 1.0])
    with pytest.raises(AnalysisError, match='Unknown tie rule'):
        break_ties([1.0], 'first')


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
