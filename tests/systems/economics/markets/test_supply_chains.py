"""Test the `_supply_chains` module of `skcomplex.systems.economics.markets`."""

from itertools import pairwise

import numpy as np

from skcomplex.systems.economics.markets import beer_game

from ...conftest import _run_seeded


def test_beer_game_amplifies_orders_upstream():
    """Test that the chain rests until demand steps up, and the order amplitude grows from retailer to factory."""
    trajectory = _run_seeded(beer_game(), 36.0, 0.5, 'beer_game')
    echelons = ('retailer', 'wholesaler', 'distributor', 'factory')
    for echelon in echelons:
        assert np.allclose(trajectory.values(f'{echelon}.orders')[:11], 4.0)
    peaks = [trajectory.values(f'{echelon}.orders').max() for echelon in echelons]
    assert all(upstream > downstream for downstream, upstream in pairwise(peaks))
    assert peaks[-1] > 3 * peaks[0]
