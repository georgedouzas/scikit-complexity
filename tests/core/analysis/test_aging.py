"""Test two-time correlations, aging collapses, responses, and fluctuation-dissipation ratios."""

import numpy as np
import pytest

from skcomplex.core.analysis import (
    aging_collapse,
    collapse_distributions,
    fields_of,
    fluctuation_dissipation_ratio,
    response,
    two_time_correlation,
)
from skcomplex.core.base import AnalysisError
from skcomplex.core.runtime import Simulation
from skcomplex.models import create

RUNS = 20_000
STEP = 0.1


def _ornstein_uhlenbeck(rng: np.random.Generator, frames: int, field: float, switch: int) -> np.ndarray:
    """Return an equilibrium ensemble of `dx = (h - x) dt + sqrt(2) dW` updated exactly, the field on from a frame."""
    decay = np.exp(-STEP)
    spread = np.sqrt(1 - decay**2)
    states = rng.standard_normal(RUNS)
    frames_list = [states.copy()]
    noise_rng = np.random.default_rng(99)
    for frame in range(1, frames):
        forcing = field if frame > switch else 0.0
        states = states * decay + forcing * (1 - decay) + spread * noise_rng.standard_normal(RUNS)
        frames_list.append(states.copy())
    return np.asarray(frames_list).T


def _trap_renewals(rng: np.random.Generator, walkers: int, horizon: float) -> list[np.ndarray]:
    """Return the jump times of walkers in Bouchaud's trap model with T / T_g = 0.5, waiting times of exponent 1/2."""
    jumps = []
    for _ in range(walkers):
        times = np.cumsum(rng.random(64) ** (-1 / 0.5))
        while times[-1] < horizon:
            times = np.concatenate([times, times[-1] + np.cumsum(rng.random(64) ** (-1 / 0.5))])
        jumps.append(times)
    return jumps


def test_two_time_correlation_equilibrium_has_no_aging():
    """Test that an equilibrium Ornstein-Uhlenbeck ensemble decays as exp(-tau) at both waiting times, with ratio 1."""
    rng = np.random.default_rng(0)
    lags = np.arange(31)
    unperturbed = _ornstein_uhlenbeck(rng, 1031, 0.0, 1000)
    correlation = two_time_correlation(unperturbed, [100, 1000], lags)
    np.testing.assert_allclose(correlation, np.exp(-lags * STEP)[None, :].repeat(2, axis=0), atol=0.03)
    perturbed = _ornstein_uhlenbeck(np.random.default_rng(0), 1031, 0.1, 1000)
    susceptibility = response(perturbed[:, 1000:], unperturbed[:, 1000:], 0.1)
    assert fluctuation_dissipation_ratio(correlation[1], susceptibility, 1.0) == pytest.approx(1, abs=0.1)


def test_two_time_correlation_aging_on_two_domains():
    """Test that an Ising quench and the trap model age: correlations grow with the age and collapse on tau / t_w."""
    system = create('ising', size=256, temperature=1.0, sampler='glauber')
    frames = fields_of(Simulation(system, dt=1).run(until=400).trajectory_)
    waiting, lags = [50, 100, 200], np.arange(0, 200)
    ising = two_time_correlation(frames[None], waiting, lags)
    horizon = 1000.0
    renewals = _trap_renewals(np.random.default_rng(1), 20_000, horizon)
    grid = np.arange(0, 1001, 5.0)
    counts = np.asarray([np.searchsorted(times, grid, side='right') for times in renewals])
    trap_waiting = [20, 40, 80]
    trap_lags = np.arange(0, 21)
    trap_curves = np.asarray(
        [[np.mean(counts[:, start] == counts[:, start + lag]) for lag in trap_lags] for start in trap_waiting],
    )
    for curves, starts, steps, scale in ((ising, waiting, lags, 1.0), (trap_curves, trap_waiting, trap_lags, 5.0)):
        assert np.all(np.diff(curves[:, 10]) > 0)
        collapse = aging_collapse(np.asarray(starts) * scale, steps * scale, curves)
        assert collapse.ratio_spread < 0.05
        assert collapse.time_spread > collapse.ratio_spread


def test_collapse_distributions_trap_model():
    """Test that the trapping times of the traps occupied at ages 10^3, 10^4, and 10^5 collapse within 10 percent."""
    renewals = _trap_renewals(np.random.default_rng(2), 20_000, 1e5)
    samples = {}
    for age in (1e3, 1e4, 1e5):
        durations = []
        for times in renewals:
            index = np.searchsorted(times, age)
            start = times[index - 1] if index > 0 else 0.0
            durations.append(times[index] - start)
        samples[age] = np.asarray(durations)
    collapse = collapse_distributions(samples, np.geomspace(0.3, 3, 5))
    assert collapse.spread < 0.1


def test_two_time_correlation_rejects_late_frames():
    """Test that a waiting time plus a lag beyond the last frame raises an error."""
    with pytest.raises(AnalysisError, match='beyond'):
        two_time_correlation(np.zeros((2, 5)), [3], [2])
