"""Test the `_kinetics` module of `skcomplex.systems.physics.fluids`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import SSPRKSolver
from skcomplex.systems.physics.fluids import discrete_velocity_boltzmann, wave_breaking


def test_wave_breaking_steepens_after_the_breaking_time():
    """Test that the wave number stays smooth before the breaking time, steepens after it, and is conserved."""
    trajectory = Simulation(wave_breaking(), dt=0.5, solver=SSPRKSolver()).prepare().run(until=12).trajectory_
    numbers = trajectory.values('train.k')
    jumps = np.abs(np.diff(numbers, axis=1)).max(axis=1)
    centers = (np.arange(200) + 0.5) * 2 * np.pi / 200
    initial = 1 + 0.5 * np.sin(centers)
    breaking = 1 / np.max(0.5 * np.cos(centers) / (4 * initial**1.5))
    assert jumps[trajectory.times < 0.7 * breaking].max() < 0.06
    assert jumps[-1] > 0.2
    assert numbers.sum(axis=1) == pytest.approx(np.full(len(numbers), numbers[0].sum()))


def test_discrete_velocity_boltzmann_conserves_mass_and_relaxes():
    """Test that the Carleman gas conserves its mass and relaxes to equal densities of both velocities."""
    trajectory = (
        Simulation(discrete_velocity_boltzmann(), dt=0.1, solver=SSPRKSolver()).prepare().run(until=5).trajectory_
    )
    right, left = trajectory.values('gas.f_plus'), trajectory.values('gas.f_minus')
    assert (right + left).sum(axis=1) == pytest.approx(np.full(len(right), (right + left)[0].sum()))
    assert np.abs(right - left).max(axis=1)[-1] < 0.05 * np.abs(right - left).max(axis=1)[0]
