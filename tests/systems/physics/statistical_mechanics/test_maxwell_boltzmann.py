"""Test the `_maxwell_boltzmann` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np
import pytest

from skcomplex.core.analysis import fit_law
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem, kinetic_temperature, maxwell_boltzmann


def test_maxwell_boltzmann_test_in_two_and_three_dimensions():
    """Test that ideal gases in two and three dimensions follow the law of their constant kinetic temperature."""
    for dimension in (2, 3):
        trajectory = simulate(IdealGasSystem(n_dim=dimension, random_state=0), until=1.0, random_state=0)
        test = maxwell_boltzmann(trajectory)
        temperatures = kinetic_temperature(trajectory).temperature
        np.testing.assert_array_equal(test.temperature, temperatures)
        velocities = np.asarray(trajectory.get_values('gas.x_d1'), dtype=float)
        assert maxwell_boltzmann(velocities[-1]).p_value[0] > 0.05
        assert test.p_value.shape == temperatures.shape
    with pytest.raises(AnalysisError, match='positive values'):
        maxwell_boltzmann(velocities, temperature=0.0)


def test_maxwell_boltzmann_test_agrees_with_the_fitted_maxwell_law():
    """Test that three-dimensional Gaussian velocities pass both the test and the fit of the Maxwell law."""
    velocities = np.random.default_rng(0).standard_normal((2000, 3)) * 1.5
    test = maxwell_boltzmann(velocities)
    fit = fit_law(np.linalg.norm(velocities, axis=-1), law='maxwell', fixed={'loc': 0})
    assert test.p_value[0] > 0.05
    assert fit.p_value > 0.05
    assert fit.parameters['scale'] ** 2 == pytest.approx(test.temperature[0], rel=0.05)
