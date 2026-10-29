"""Test the `_cobweb` module of `skcomplex.models.economics.markets`."""

import pytest

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.economics.markets import CobwebInteraction, MarketElement


def test_cobweb_converges_to_equilibrium():
    """Test that a cobweb market with `d < b` converges to `(a - c) / (b + d)`."""
    a, b, c, d = 10.0, 2.0, 1.0, 1.0
    system = System(
        elements=[MarketElement('m', price=1.0)],
        interactions=[CobwebInteraction('c', target='m', a=a, b=b, c=c, d=d)],
    )
    price = Simulation(system).check().prepare().run(until=60).trajectory_.values('m.price')[-1]
    assert price == pytest.approx((a - c) / (b + d))
