"""Test the `_strings` module of `skcomplex.systems.physics.waves`."""

import math

import numpy as np
import pytest

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import MassPointElement
from skcomplex.systems.physics.waves import (
    HarmonicPotentialInteraction,
    PotentialEnergyInteraction,
    SpringChainSystem,
    SpringInteraction,
    StandingWavesSystem,
)

from ..conftest import _run, _series


def test_plucked_spring_chain_keeps_its_fundamental_mode():
    """Test that the fundamental transverse mode of the chain oscillates at `2 sqrt(k / m) sin(pi / (2 (N + 1)))`."""
    n_masses = 15
    trajectory = SpringChainSystem(n_masses).prepare().simulate(until=4.0, dt=0.1)
    heights = np.stack([_series(trajectory, f'mass_{index}.x')[0][:, 1] for index in range(n_masses)], axis=1)
    mode = np.sin(np.pi * np.arange(1, n_masses + 1) / (n_masses + 1))
    amplitude = heights @ mode / (mode @ mode)
    frequency = 2 * math.sqrt(1.0 / 0.01) * math.sin(math.pi / (2 * (n_masses + 1)))
    assert amplitude == pytest.approx(amplitude[0] * np.cos(frequency * trajectory.times), abs=2e-4)


def test_standing_wave_returns_after_its_period():
    """Test that a standing wave on a string with fixed ends returns to its shape after `2 L / (n c)`."""
    n_sites, length, mode = 49, 2.0, 2
    spacing = length / (n_sites + 1)
    frequency = 2 / spacing * math.sin(mode * math.pi / (2 * (n_sites + 1)))
    period = 2 * math.pi / frequency
    trajectory = _run(StandingWavesSystem(n_sites, length=length, mode=mode), period, period / 20)
    shape = _series(trajectory, 'string.u')[0]
    assert period == pytest.approx(2 * length / mode, rel=1e-3)
    assert np.allclose(shape[-1], shape[0], atol=1e-7)
    assert np.allclose(shape[10], -shape[0], atol=1e-7)


def test_potential_energy_changes_the_velocity_by_the_force_over_the_mass():
    """Test that the energy `k x**2 / 2` makes a particle of mass `m` oscillate with the frequency `sqrt(k / m)`."""
    trap = PotentialEnergyInteraction(
        'trap',
        target='ball',
        potential=lambda s: 2.0 * s.target.x.dot_product(s.target.x),
    )
    system = System(
        elements=[MassPointElement('ball', x=[1.0], m=4.0)],
        interactions=[trap],
        space=ContinuousSpace(n_dim=1),
        solver=ScipyODESolver(rtol=1e-10, atol=1e-12),
    )
    trajectory = system.prepare().simulate(until=4.0, dt=0.1)
    np.testing.assert_allclose(trajectory.get_values('ball.x')[:, 0], np.cos(trajectory.times), atol=1e-7)


def test_harmonic_potential_oscillates_about_its_center():
    """Test that a particle in a harmonic trap oscillates about the center at the angular frequency `sqrt(k / m)`."""
    system = System(
        elements=[MassPointElement('ball', x=[2.0], m=2.0)],
        interactions=[HarmonicPotentialInteraction('trap', target='ball', k=8.0, center=1.0)],
        space=ContinuousSpace(n_dim=1),
        solver=ScipyODESolver(rtol=1e-10, atol=1e-12),
    )
    trajectory = system.prepare().simulate(until=3, dt=0.1)
    np.testing.assert_allclose(trajectory.get_values('ball.x')[:, 0], 1.0 + np.cos(2.0 * trajectory.times), atol=1e-6)


def test_spring_conserves_energy():
    """Test that two particles joined by a spring keep their total energy."""
    system = System(
        elements=[
            MassPointElement('a', x=[0.0, 0.0], values={"x'": [0.0, 1.0]}),
            MassPointElement('b', x=[2.0, 0.0], m=2.0),
        ],
        interactions=[SpringInteraction('spring', first='a', second='b', k=3.0, length=1.0)],
        space=ContinuousSpace(n_dim=2),
        solver=ScipyODESolver(rtol=1e-10, atol=1e-12),
    )
    trajectory = system.prepare().simulate(until=5, dt=0.1)
    speeds = [(trajectory.get_values(f"{name}.x'") ** 2).sum(axis=1) for name in 'ab']
    kinetic = 0.5 * speeds[0] + 0.5 * 2.0 * speeds[1]
    distance = np.linalg.norm(trajectory.get_values('a.x') - trajectory.get_values('b.x'), axis=1)
    energy = kinetic + 0.5 * 3.0 * (distance - 1.0) ** 2
    assert np.allclose(energy, energy[0], rtol=1e-6)
