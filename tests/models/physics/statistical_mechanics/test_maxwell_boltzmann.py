"""Test the `_maxwell_boltzmann` module of `skcomplex.models.physics.statistical_mechanics`."""

import numpy as np
import pytest

from skcomplex.analysis import FitAnalysis
from skcomplex.errors import AnalysisError
from skcomplex.models import run
from skcomplex.models.physics.statistical_mechanics import KineticTemperatureAnalysis, MaxwellBoltzmannAnalysis


def test_maxwell_boltzmann_test_in_two_and_three_dimensions():
    """Test that ideal gases in two and three dimensions follow the law of their constant kinetic temperature."""
    for dimension in (2, 3):
        trajectory = run('ideal_gas', n_dim=dimension, random_state=0, settings={'until': 1.0}).trajectory_
        test = MaxwellBoltzmannAnalysis().check().prepare(trajectory).run()
        temperatures = KineticTemperatureAnalysis().check().prepare(trajectory).run().temperature_
        np.testing.assert_array_equal(test.temperature_, temperatures)
        velocities = np.asarray(trajectory.values('gas.x_d1'), dtype=float)
        assert MaxwellBoltzmannAnalysis().check().prepare(velocities[-1]).run().p_value_[0] > 0.05
        assert test.p_value_.shape == temperatures.shape
    with pytest.raises(AnalysisError, match='positive values'):
        MaxwellBoltzmannAnalysis(temperature=0.0).check()


def test_maxwell_boltzmann_test_agrees_with_the_fitted_maxwell_law():
    """Test that three-dimensional Gaussian velocities pass both the test and the fit of the Maxwell law."""
    velocities = np.random.default_rng(0).standard_normal((2000, 3)) * 1.5
    test = MaxwellBoltzmannAnalysis().check().prepare(velocities).run()
    fit = FitAnalysis('maxwell', fixed={'loc': 0}).check().prepare(np.linalg.norm(velocities, axis=-1)).run()
    assert test.p_value_[0] > 0.05
    assert fit.p_value_ > 0.05
    assert fit.parameters_['scale'] ** 2 == pytest.approx(test.temperature_[0], rel=0.05)
