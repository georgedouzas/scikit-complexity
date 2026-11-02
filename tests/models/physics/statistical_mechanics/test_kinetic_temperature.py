"""Test the `_kinetic_temperature` module of `skcomplex.models.physics.statistical_mechanics`."""

import math

import numpy as np
import pytest

from skcomplex.analysis import DistributionAnalysis
from skcomplex.errors import AnalysisError
from skcomplex.models import run
from skcomplex.models.physics.statistical_mechanics import KineticTemperatureAnalysis


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
        trajectory = run('ideal_gas', n_dim=dimension, random_state=0, settings={'until': 1.0}).trajectory_
        temperatures = KineticTemperatureAnalysis(mass=2.0).check().prepare(trajectory).run().temperature_
        assert np.ptp(temperatures) < 1e-10
        speeds = DistributionAnalysis('moments').check().prepare(trajectory).run()
        np.testing.assert_allclose(temperatures, 2.0 * (speeds.variance_ + speeds.mean_**2) / dimension)


def test_kinetic_temperature_of_langevin_particles():
    """Test that 2000 underdamped Langevin particles reach the bath temperature 1.5."""
    velocities = _langevin_velocities(np.random.default_rng(3), 2000, 1.5)
    temperature = KineticTemperatureAnalysis().check().prepare(velocities).run().temperature_
    assert temperature[0] == pytest.approx(1.5, rel=0.03)
    with pytest.raises(AnalysisError, match='positive mass'):
        KineticTemperatureAnalysis(mass=-1.0).check()
