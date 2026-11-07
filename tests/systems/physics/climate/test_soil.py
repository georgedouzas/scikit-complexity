"""Test the `_soil` module of `skcomplex.systems.physics.climate`."""

import numpy as np

from skcomplex.systems.physics.climate import SoilTemperatureSystem

from ..conftest import _run_default


def test_soil_temperature_reaches_its_fixed_point():
    """Test that the coupled soil maps settle on their nontrivial fixed point for a moderate logistic parameter."""
    trajectory = _run_default(SoilTemperatureSystem(r=2.5, epsilon=0.2), until=200, dt=1)
    x = (2.5 + 0.2**2 / 0.8 - 1) / 2.5
    assert np.isclose(trajectory.values('soil.x')[-1], x)
    assert np.isclose(trajectory.values('soil.y')[-1], 0.2 * x / 0.8)
