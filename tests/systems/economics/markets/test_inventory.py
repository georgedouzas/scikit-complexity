"""Test the `_inventory` module of `skcomplex.systems.economics.markets`."""

import numpy as np
import pytest

from skcomplex.systems.economics.markets import (
    car_dealer,
)

from ...conftest import _run_seeded
from ..conftest import _run


def test_car_dealer_oscillates_with_a_short_response_delay():
    """Test that the inventory stays at 200 cars until demand steps up, then oscillates unless the response is slow."""
    fast = _run(car_dealer(), 100.0, dt=1.0).values('dealer.inventory')
    slow = _run(car_dealer(response_delay=6.0), 100.0, dt=1.0).values('dealer.inventory')
    assert np.allclose(fast[:26], 200.0)
    assert fast[25:].min() < 195.0
    assert np.ptp(slow[60:]) < np.ptp(fast[60:]) / 5


def test_car_dealer_inventory_oscillates_before_settling_after_a_demand_step():
    """Test that a 10% step in demand makes the inventory oscillate about its new level and then settle there."""
    trajectory = _run_seeded(car_dealer(), 150.0, 1.0)
    inventory, sales = trajectory.values('dealer.inventory'), trajectory.values('dealer.sales')
    assert np.allclose(inventory[:26], 200.0)
    assert sales[30] == pytest.approx(22.0)
    settled = 10.0 * 22.0
    signs = np.sign(inventory[26:] - settled)
    assert np.count_nonzero(np.diff(signs[signs != 0])) >= 3
    assert inventory[-1] == pytest.approx(settled, abs=1.0)


def test_car_dealer_delays_shape_the_oscillation():
    """Test that a longer response delay damps the oscillation and a shorter perception delay barely changes it."""
    amplitudes = {}
    for name, settings in {'base': {}, 'slow': {'response_delay': 6.0}, 'alert': {'perception_delay': 2.0}}.items():
        inventory = _run_seeded(car_dealer(**settings), 100.0, 1.0).values('dealer.inventory')[25:]
        amplitudes[name] = np.ptp(inventory)
    assert amplitudes['slow'] < amplitudes['base']
    assert amplitudes['alert'] == pytest.approx(amplitudes['base'], rel=0.15)
