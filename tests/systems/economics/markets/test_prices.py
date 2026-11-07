"""Test the `_prices` module of `skcomplex.systems.economics.markets`."""

import numpy as np
import pytest

from skcomplex.systems.economics.markets import (
    RiccatiPricesSystem,
)

from ..conftest import _run


def test_riccati_prices_converge_to_the_lower_equilibrium():
    """Test that a price below the larger root converges to the smaller root of the quadratic excess demand."""
    price = _run(RiccatiPricesSystem(1.0), 10.0, dt=0.1).values('market.P')
    roots = np.sort(np.roots([0.1, -3.0, 9.0]))
    assert price[-1] == pytest.approx(roots[0], rel=1e-3)
