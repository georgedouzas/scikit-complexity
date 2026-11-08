"""Test the `_commodities` module of `skcomplex.systems.economics.markets`."""

import numpy as np

from skcomplex.systems.economics.markets import CommodityCycleSystem

from ...conftest import _run_seeded


def _peaks(values):
    """Return the indices of the local maxima of a series."""
    return np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])) + 1


def test_commodity_cycle_oscillates_after_a_demand_step():
    """Test that the market rests until demand steps up, then the capacity and the price cycle."""
    trajectory = _run_seeded(CommodityCycleSystem().prepare().system_, 40.0, 0.5, CommodityCycleSystem())
    capacity, price = trajectory.values('market.capacity'), trajectory.values('market.price')
    assert np.allclose(price[:4], 1.0)
    assert len(_peaks(capacity)) >= 2
    assert price.max() > 1.2
    assert price.min() < 0.7
