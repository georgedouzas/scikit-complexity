"""Test the `_assets` module of `skcomplex.models.economics.finance`."""

import numpy as np

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.economics.finance import Asset, GeometricBrownianMotion


def test_asset_price_stays_positive_and_reproducible():
    """Test that geometric Brownian motion stays positive and is reproducible."""
    system = System(
        components=[Asset('stock', price=100.0)],
        interactions=[GeometricBrownianMotion('gbm', target='stock', mu=0.05, sigma=0.2)],
    )
    first = Simulation(system, dt=0.1, random_state=7).run(until=5).trajectory_['stock.price']
    second = Simulation(system, dt=0.1, random_state=7).run(until=5).trajectory_['stock.price']
    assert np.all(first > 0)
    assert np.array_equal(first, second)
