"""Test the characteristic behaviour of the system traps, the inventory, commodity, and beer game models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from itertools import pairwise

import numpy as np
import pytest

from skcomplex.core.analysis import loops
from skcomplex.core.runtime import Simulation
from skcomplex.models.economics.markets import beer_game, car_dealer, commodity_cycle
from skcomplex.models.social.organizations import escalation, shifting_the_burden, success_to_the_successful


def _run(system, until, dt):
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt, random_state=0).run(until=until).trajectory_


def _polarities(system):
    """Return the polarities of the feedback loops of a system at its initial state."""
    return sorted(loop.polarity for loop in loops(system).loops)


def _peaks(values):
    """Return the indices of the local maxima of a series."""
    return np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])) + 1


def test_car_dealer_inventory_oscillates_before_settling_after_a_demand_step():
    """Test that a 10% step in demand makes the inventory oscillate about its new level and then settle there."""
    trajectory = _run(car_dealer(), 150.0, 1.0)
    inventory, sales = trajectory['dealer.inventory'], trajectory['dealer.sales']
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
        inventory = _run(car_dealer(**settings), 100.0, 1.0)['dealer.inventory'][25:]
        amplitudes[name] = np.ptp(inventory)
    assert amplitudes['slow'] < amplitudes['base']
    assert amplitudes['alert'] == pytest.approx(amplitudes['base'], rel=0.15)


def test_success_to_the_successful_gives_everything_to_the_initially_larger():
    """Test that the initially larger competitor takes the whole budget, through two reinforcing loops."""
    for a, b in ((55.0, 45.0), (45.0, 55.0)):
        trajectory = _run(success_to_the_successful(a=a, b=b), 100.0, 5.0)
        winner, loser = ('contest.a', 'contest.b') if a > b else ('contest.b', 'contest.a')
        assert trajectory[winner][-1] > 99.0
        assert trajectory[loser][-1] < 1.0
    assert _polarities(success_to_the_successful()) == ['balancing', 'balancing', 'reinforcing', 'reinforcing']


def test_shifting_the_burden_lets_the_quick_fix_take_over():
    """Test that the quick fix replaces the capability of the system, which builds up to the goal without it."""
    addicted = _run(shifting_the_burden(), 40.0, 1.0)
    assert addicted['burden.capability'][-1] < 0.01
    assert addicted['burden.intervention'][-1] == pytest.approx(1.0, abs=0.01)
    assert abs(addicted['burden.symptom'][-1]) < 0.01
    unaided = _run(shifting_the_burden(fix_rate=0.0), 40.0, 1.0)
    assert unaided['burden.capability'][-1] == pytest.approx(1.0, abs=0.01)
    assert 'reinforcing' in _polarities(shifting_the_burden())


def test_escalation_grows_both_stocks_exponentially():
    """Test that both stocks grow at the rate of the reinforcing loop through the two parties."""
    trajectory = _run(escalation(), 100.0, 5.0)
    rate = (-3.0 + np.sqrt(9.0 + 0.8)) / 4.0
    for path in ('race.a', 'race.b'):
        assert np.log(trajectory[path][-1] / trajectory[path][-5]) / 20.0 == pytest.approx(rate, rel=1e-3)
    assert 'reinforcing' in _polarities(escalation())


def test_commodity_cycle_oscillates_after_a_demand_step():
    """Test that the market rests until demand steps up, then the capacity and the price cycle."""
    trajectory = _run(commodity_cycle(), 40.0, 0.5)
    capacity, price = trajectory['market.capacity'], trajectory['market.price']
    assert np.allclose(price[:4], 1.0)
    assert len(_peaks(capacity)) >= 2
    assert price.max() > 1.2
    assert price.min() < 0.7


def test_beer_game_amplifies_orders_upstream():
    """Test that the chain rests until demand steps up, and the order amplitude grows from retailer to factory."""
    trajectory = _run(beer_game(), 36.0, 0.5)
    echelons = ('retailer', 'wholesaler', 'distributor', 'factory')
    for echelon in echelons:
        assert np.allclose(trajectory[f'{echelon}.orders'][:11], 4.0)
    peaks = [trajectory[f'{echelon}.orders'].max() for echelon in echelons]
    assert all(upstream > downstream for downstream, upstream in pairwise(peaks))
    assert peaks[-1] > 3 * peaks[0]
