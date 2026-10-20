"""Test Fano factors, event counts, relative fluctuations, and Kramers-Moyal coefficients."""

import numpy as np
import pytest

from skcomplex.core.analysis import event_counts, fano_factor, kramers_moyal, relative_fluctuations
from skcomplex.core.base import AnalysisError
from skcomplex.models import run


def _bursty_expression(rng: np.random.Generator, samples: int) -> np.ndarray:
    """Return mRNA copy numbers sampled once per unit time from bursts of geometric size with mean 4 and decay 1."""
    burst_rate, decay, mean_burst = 5.0, 1.0, 4.0
    copies, time, next_sample, recorded = 0, 0.0, 50.0, []
    while len(recorded) < samples:
        total = burst_rate + decay * copies
        step = rng.exponential(1 / total)
        while next_sample < time + step and len(recorded) < samples:
            recorded.append(copies)
            next_sample += 1.0
        time += step
        if rng.random() < burst_rate / total:
            copies += rng.geometric(1 / (1 + mean_burst)) - 1
        else:
            copies -= 1
    return np.asarray(recorded, dtype=float)


def test_fano_factor_poisson_and_bursty_production():
    """Test that Gillespie production counts give a Fano factor near 1 and bursty mRNA copies near 5."""
    trajectory = run('poisson_process', n_processes=1, settings={'until': 50_000.0, 'random_state': 0}).trajectory_
    counts = np.diff(trajectory['counters.count'][:, 0][::10])
    assert fano_factor(counts) == pytest.approx(1, abs=0.05)
    assert fano_factor(_bursty_expression(np.random.default_rng(0), 20_000)) == pytest.approx(5, abs=0.3)


def test_event_counts_and_fano_of_event_times():
    """Test that the event times of a Poisson process give counts per window with a Fano factor near 1."""
    times = np.cumsum(np.random.default_rng(1).exponential(1.0, 100_000))
    counts = event_counts(times, 10.0)
    assert counts.sum() <= times.size
    assert fano_factor(counts) == pytest.approx(1, abs=0.1)
    with pytest.raises(AnalysisError, match='mean 0'):
        fano_factor([0, 0])


def test_kramers_moyal_ornstein_uhlenbeck_coefficients():
    """Test that 10^6 steps of an Ornstein-Uhlenbeck process give drift -x within 0.1 and diffusion 1 within 0.05."""
    rng = np.random.default_rng(2)
    dt = 0.02
    noise = rng.standard_normal(1_000_000) * np.sqrt(2 * dt)
    values = np.empty(noise.size)
    values[0] = 0.0
    for t in range(1, values.size):
        values[t] = values[t - 1] - values[t - 1] * dt + noise[t]
    coefficients = kramers_moyal(values, dt, bins=41)
    inside = np.abs(coefficients.positions) < 1.5
    assert np.max(np.abs(coefficients.drift[inside] + coefficients.positions[inside])) < 0.1
    assert np.max(np.abs(coefficients.diffusion[inside] - 1)) < 0.05


def test_relative_fluctuations_of_an_ideal_lattice_gas():
    """Test that the particle number in subvolumes of a lattice gas fluctuates as the inverse root of the volume."""
    occupied = np.random.default_rng(3).random((200, 256, 256)) < 0.3
    samples = {side**2: occupied[:, :side, :side].sum(axis=(1, 2)) for side in (4, 8, 16, 32, 64)}
    assert relative_fluctuations(samples).exponent == pytest.approx(-0.5, abs=0.05)
