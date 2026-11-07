"""Test the `_cooling` module of `skcomplex.systems.physics.heat`."""

import numpy as np
import pytest

from skcomplex.systems.physics.heat import CoolingCoffeeSystem

from ..conftest import _run, _series


def test_coffee_cools_exponentially_to_the_room():
    """Test that the coffee follows `T_room + (T0 - T_room) exp(-k t)`."""
    trajectory = _run(CoolingCoffeeSystem(90.0, room=20.0, k=0.1), 30.0, 1.0)
    expected = 20.0 + 70.0 * np.exp(-0.1 * trajectory.times)
    assert _series(trajectory, 'coffee.T')[0] == pytest.approx(expected, rel=1e-8)
