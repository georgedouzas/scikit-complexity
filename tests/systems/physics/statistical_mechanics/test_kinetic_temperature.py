"""Test the `_kinetic_temperature` module of `skcomplex.systems.physics.statistical_mechanics`."""

import math

import numpy as np
import pytest

from skcomplex.core.analysis import distribution
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem, kinetic_temperature


def _langevin_velocities(rng: np.random.Generator, particles: int, temperature: float) -> np.ndarray:
    """Return velocities of underdamped Langevin particles with friction 1 after relaxing from rest."""
    decay = math.exp(-0.01)
    velocities = np.zeros((particles, 2))
    for _ in range(1000):
        velocities = velocities * decay + math.sqrt(temperature * (1 - decay**2)) * rng.standard_normal(
            velocities.shape,
        )
    return velocities


def test_kinetic_temperature_of_an_ideal_gas_is_the_second_moment_of_its_speeds():
    """Test that an ideal gas keeps its kinetic temperature, the mass times the second moment of speeds over d."""
    for dimension in (2, 3):
        trajectory = simulate(IdealGasSystem(n_dim=dimension, random_state=0), until=1.0, random_state=0)
        temperatures = kinetic_temperature(trajectory, mass=2.0).temperature
        assert np.ptp(temperatures) < 1e-10
        speeds = distribution(trajectory, kind='moments')
        np.testing.assert_allclose(temperatures, 2.0 * (speeds.variance + speeds.mean**2) / dimension)


def test_kinetic_temperature_of_langevin_particles():
    """Test that 2000 underdamped Langevin particles reach the bath temperature 1.5."""
    velocities = _langevin_velocities(np.random.default_rng(3), 2000, 1.5)
    temperature = kinetic_temperature(velocities).temperature
    assert temperature[0] == pytest.approx(1.5, rel=0.03)
    with pytest.raises(AnalysisError, match='positive mass'):
        kinetic_temperature(velocities, mass=-1.0)
