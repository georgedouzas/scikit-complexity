"""Test the `_optimal_harvesting` module of `skcomplex.systems.economics.resources`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.resources import OptimalHarvestingSystem


def test_optimal_harvesting_holds_the_singular_level():
    """Test that the optimal effort takes its bounds at both ends and holds the stock steady in between."""
    trajectory = simulate(OptimalHarvestingSystem())
    effort, stock = trajectory.get_values('resource.effort'), trajectory.get_values('resource.x')
    assert effort[0] == pytest.approx(1.0, abs=1e-6)
    assert effort[-1] == pytest.approx(1.0, abs=1e-6)
    middle = (trajectory.times > 5) & (trajectory.times < 15)
    assert np.ptp(stock[middle]) < 1e-2
    np.testing.assert_allclose(trajectory.get_values('catch'), effort * stock)
