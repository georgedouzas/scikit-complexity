"""Test the `_cobweb` module of `skcomplex.systems.economics.markets`."""

import pytest

from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement


def test_cobweb_converges_to_equilibrium():
    """Test that a cobweb market with `d < b` converges to `(a - c) / (b + d)`."""
    a, b, c, d = 10.0, 2.0, 1.0, 1.0
    system = CompositeSystem(
        elements=[MarketElement('m', price=1.0)],
        interactions=[CobwebInteraction('c', target='m', a=a, b=b, c=c, d=d)],
    )
    price = simulate(system, until=60).get_values('m.price')[-1]
    assert price == pytest.approx((a - c) / (b + d))
