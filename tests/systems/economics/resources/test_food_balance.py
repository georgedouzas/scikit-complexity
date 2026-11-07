"""Test the `_food_balance` module of `skcomplex.systems.economics.resources`."""

import numpy as np
import pytest

from skcomplex.systems import describe
from skcomplex.systems.economics.resources import rice_food_balance

from ..conftest import _run


def test_rice_food_balance_rises_then_falls_as_the_land_fills():
    """Test that the area approaches its balance with the land lost and that the food security peaks and then falls."""
    system = rice_food_balance()
    trajectory = _run(system, 100.0, 0.25, observables=describe('rice_food_balance').observables_of(system))
    area, security, times = (
        trajectory.values('region.area_cultivated'),
        trajectory.values('region.food_security'),
        np.asarray(trajectory.times),
    )
    assert area[-1] == pytest.approx(120000.0 / (1 + 15.0 * 0.00085), rel=0.01)
    assert security[0] == pytest.approx(68000.0 * 3.2 / (2800000.0 * 0.18))
    assert 10 < times[np.argmax(security)] < 35
    assert security[-1] < security[0]
    assert trajectory.values('region.population')[-1] == pytest.approx(
        2800000.0 * np.exp((0.025 - 1 / 65.0) * 100.0), rel=0.01,
    )
