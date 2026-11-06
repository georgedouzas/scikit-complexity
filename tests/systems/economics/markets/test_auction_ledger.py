"""Test the `_auction_ledger` module of `skcomplex.systems.economics.markets`."""

import numpy as np

from skcomplex.systems.economics.markets import maximum_surplus


def test_maximum_surplus():
    """The maximum surplus pairs the highest values with the lowest costs."""
    assert maximum_surplus(np.array([10.0, 6.0, 2.0]), np.array([1.0, 4.0, 8.0])) == 11.0
