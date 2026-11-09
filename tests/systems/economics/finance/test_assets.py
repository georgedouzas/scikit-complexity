"""Test the `_assets` module of `skcomplex.systems.economics.finance`."""

import numpy as np

from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.finance import AssetElement, GeometricBrownianMotionInteraction


def test_asset_price_stays_positive_and_reproducible():
    """Test that geometric Brownian motion stays positive and is reproducible."""
    system = CompositeSystem(
        elements=[AssetElement('stock', price=100.0)],
        interactions=[GeometricBrownianMotionInteraction('gbm', target='stock', mu=0.05, sigma=0.2)],
    )
    first = simulate(system, until=5, dt=0.1, random_state=7).values('stock.price')
    second = simulate(system, until=5, dt=0.1, random_state=7).values('stock.price')
    assert np.all(first > 0)
    assert np.array_equal(first, second)
