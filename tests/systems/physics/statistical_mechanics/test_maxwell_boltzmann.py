"""Test the `_maxwell_boltzmann` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np
import pytest

from skcomplex.core.analysis import FitAnalysis
from skcomplex.core.errors import AnalysisError
from skcomplex.systems import create_simulation
from skcomplex.systems.physics.statistical_mechanics import KineticTemperatureAnalysis, MaxwellBoltzmannAnalysis


def test_maxwell_boltzmann_test_in_two_and_three_dimensions():
    """Test that ideal gases in two and three dimensions follow the law of their constant kinetic temperature."""
    for dimension in (2, 3):
        trajectory = (
            create_simulation('ideal_gas', n_dim=dimension, random_state=0).prepare().run(until=1.0).trajectory_
        )
        test = MaxwellBoltzmannAnalysis().prepare(trajectory).run()
        temperatures = KineticTemperatureAnalysis().prepare(trajectory).run().temperature_
        np.testing.assert_array_equal(test.temperature_, temperatures)
        velocities = np.asarray(trajectory.values('gas.x_d1'), dtype=float)
        assert MaxwellBoltzmannAnalysis().prepare(velocities[-1]).run().p_value_[0] > 0.05
        assert test.p_value_.shape == temperatures.shape
    with pytest.raises(AnalysisError, match='positive values'):
        MaxwellBoltzmannAnalysis(temperature=0.0).prepare()


def test_maxwell_boltzmann_test_agrees_with_the_fitted_maxwell_law():
    """Test that three-dimensional Gaussian velocities pass both the test and the fit of the Maxwell law."""
    velocities = np.random.default_rng(0).standard_normal((2000, 3)) * 1.5
    test = MaxwellBoltzmannAnalysis().prepare(velocities).run()
    fit = FitAnalysis('maxwell', fixed={'loc': 0}).prepare(np.linalg.norm(velocities, axis=-1)).run()
    assert test.p_value_[0] > 0.05
    assert fit.p_value_ > 0.05
    assert fit.parameters_['scale'] ** 2 == pytest.approx(test.temperature_[0], rel=0.05)
