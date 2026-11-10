"""Test the `_kolkata` module of `skcomplex.systems.economics.econophysics`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.econophysics import (
    KolkataRestaurantSystem,
)


def test_kolkata_utilization():
    """Random choice uses `1 - 1/e` of the restaurants and stochastic crowd avoidance about 0.8."""
    utilization = {}
    for strategy in ('random', 'crowd_avoiding'):
        system = KolkataRestaurantSystem(200, strategy=strategy, random_state=0).prepare().system_
        values = simulate(system, until=60, random_state=0).get_values('restaurants.utilization')
        utilization[strategy] = np.mean(values[20:])
    assert utilization['random'] == pytest.approx(1 - 1 / np.e, abs=0.02)
    assert utilization['crowd_avoiding'] == pytest.approx(0.8, abs=0.04)
