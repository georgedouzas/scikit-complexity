"""Test number, relative, and velocity fluctuations, and Fano factors."""

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from skcomplex.analysis import FluctuationAnalysis, coarse_grain, snapshots_of
from skcomplex.core.errors import AnalysisError
from skcomplex.models import run


def _fluctuations(kind: str, source: object, **settings: object) -> FluctuationAnalysis:
    """Return the fluctuation analysis of a source for one kind, run."""
    return FluctuationAnalysis(kind, **settings).check().prepare(source).run()


def _bursty_expression(rng: np.random.Generator, samples: int) -> np.ndarray:
    """Return mRNA copy numbers sampled once per unit time from bursts of geometric size with mean 4 and decay 1."""
    burst_rate, decay, mean_burst = 5.0, 1.0, 4.0
    recorded: list[int] = []
    copies, time, next_sample = 0, 0.0, 50.0
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


def _vicsek_densities(rng: np.random.Generator, agents: int, side: float, noise: float, steps: int) -> np.ndarray:
    """Return coarse-grained densities of Vicsek agents of speed 0.5 aligning within distance 1 in a periodic box."""
    positions = rng.random((agents, 2)) * side
    headings = rng.uniform(-np.pi, np.pi, agents)
    densities = []
    for step in range(steps):
        pairs = cKDTree(positions, boxsize=side).query_pairs(1.0, output_type='ndarray')
        sines, cosines = np.sin(headings), np.cos(headings)
        total_sines, total_cosines = sines.copy(), cosines.copy()
        for first, second in ((0, 1), (1, 0)):
            np.add.at(total_sines, pairs[:, first], sines[pairs[:, second]])
            np.add.at(total_cosines, pairs[:, first], cosines[pairs[:, second]])
        headings = np.arctan2(total_sines, total_cosines) + noise * rng.uniform(-np.pi, np.pi, agents)
        positions = np.mod(positions + 0.5 * np.column_stack([np.cos(headings), np.sin(headings)]), side)
        if step >= steps // 3 and step % 50 == 0:
            densities.append(coarse_grain(positions, [side, side], [64, 64]))
    return np.asarray(densities)


def _langevin_velocities(rng: np.random.Generator, particles: int, temperature: float) -> np.ndarray:
    """Return velocities of underdamped Langevin particles with friction 1 after relaxing from rest."""
    decay = math.exp(-0.01)
    velocities = np.zeros((particles, 2))
    for _ in range(1000):
        velocities = velocities * decay + math.sqrt(temperature * (1 - decay**2)) * rng.standard_normal(
            velocities.shape,
        )
    return velocities


def test_fano_factor_poisson_and_bursty_production():
    """Test that Gillespie production counts give a Fano factor near 1 and bursty mRNA copies near 5."""
    trajectory = run('poisson_process', n_processes=1, settings={'until': 50_000.0, 'random_state': 0}).trajectory_
    counts = np.diff(trajectory.values('counters.count')[:, 0][::10])
    assert _fluctuations('fano', counts).factor_[0] == pytest.approx(1, abs=0.05)
    bursty = _bursty_expression(np.random.default_rng(0), 20_000)
    assert _fluctuations('fano', bursty).factor_[0] == pytest.approx(5, abs=0.3)
    with pytest.raises(AnalysisError, match='mean 0'):
        _fluctuations('fano', [0, 0])


def test_relative_fluctuations_of_an_ideal_lattice_gas():
    """Test that the particle number in subvolumes of a lattice gas fluctuates as the inverse root of the volume."""
    occupied = np.random.default_rng(3).random((200, 256, 256)) < 0.3
    samples = {side**2: occupied[:, :side, :side].sum(axis=(1, 2)) for side in (4, 8, 16, 32, 64)}
    assert _fluctuations('relative', samples).exponent_ == pytest.approx(-0.5, abs=0.05)
    with pytest.raises(AnalysisError, match='mapping from size to samples'):
        _fluctuations('relative', np.ones(3))


def test_number_fluctuations_giant_in_ordered_flocks():
    """Test that an ideal gas has normal number fluctuations and ordered Vicsek agents giant ones."""
    rng = np.random.default_rng(2)
    gas = np.asarray([coarse_grain(rng.random((4000, 2)) * 32, [32, 32], [64, 64]) for _ in range(40)])
    assert _fluctuations('number', gas).exponent_ == pytest.approx(0.5, abs=0.05)
    assert _fluctuations('number', _vicsek_densities(rng, 2000, 22.6, 0.3, 2000)).exponent_ > 0.6


def test_maxwell_boltzmann_test_in_two_and_three_dimensions():
    """Test that ideal gases in two and three dimensions follow the law of their constant kinetic temperature."""
    for dimension in (2, 3):
        trajectory = run('ideal_gas', n_dim=dimension, random_state=0, settings={'until': 1.0}).trajectory_
        temperatures = _fluctuations('kinetic-temperature', trajectory).temperature_
        assert np.ptp(temperatures) < 1e-10
        velocities = snapshots_of(trajectory).velocities
        assert _fluctuations('maxwell-boltzmann', velocities[-1]).p_value_[0] > 0.05
        assert _fluctuations('maxwell-boltzmann', trajectory).p_value_.shape == temperatures.shape


def test_kinetic_temperature_of_langevin_particles():
    """Test that 2000 underdamped Langevin particles reach the bath temperature 1.5 and its speed law."""
    velocities = _langevin_velocities(np.random.default_rng(3), 2000, 1.5)
    assert _fluctuations('kinetic-temperature', velocities).temperature_[0] == pytest.approx(1.5, rel=0.03)
    assert _fluctuations('maxwell-boltzmann', velocities).p_value_[0] > 0.05


def test_fluctuation_analysis_checks_its_settings():
    """Test the errors of an unknown kind, an ignored setting, and a negative mass."""
    with pytest.raises(AnalysisError, match='Use one of number'):
        FluctuationAnalysis('shot').check()
    with pytest.raises(AnalysisError, match='does not read mass'):
        FluctuationAnalysis('fano', mass=2.0).check()
    with pytest.raises(AnalysisError, match='positive values'):
        FluctuationAnalysis('kinetic-temperature', mass=-1.0).check()
