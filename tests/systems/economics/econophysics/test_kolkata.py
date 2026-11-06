"""Test the `_kolkata` module of `skcomplex.systems.economics.econophysics`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.economics.econophysics import (
    kolkata_restaurant,
)


def test_kolkata_utilization():
    """Random choice uses `1 - 1/e` of the restaurants and stochastic crowd avoidance about 0.8."""
    utilization = {}
    for strategy in ('random', 'crowd_avoiding'):
        system = kolkata_restaurant(200, strategy=strategy, random_state=0)
        values = (
            Simulation(system, random_state=0).prepare().run(until=60).trajectory_.values('restaurants.utilization')
        )
        utilization[strategy] = np.mean(values[20:])
    assert utilization['random'] == pytest.approx(1 - 1 / np.e, abs=0.02)
    assert utilization['crowd_avoiding'] == pytest.approx(0.8, abs=0.04)
