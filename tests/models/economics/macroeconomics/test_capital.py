"""Test the `_capital` module of `skcomplex.models.economics.macroeconomics`."""

import numpy as np
import pytest

from skcomplex.models.economics.macroeconomics import capital_stock

from ..conftest import _run


@pytest.mark.parametrize(('lifetime', 'sign'), [(10.0, -1), (15.0, 0), (20.0, 1)])
def test_capital_stock_grows_with_the_lifetime_of_capital(lifetime, sign):
    """Test that capital declines, stays, or grows exponentially as its lifetime is below, at, or above balance."""
    capital = _run(capital_stock(lifetime=lifetime), 50, dt=1.0)['economy.K']
    rate = 0.2 / 3 - 1 / lifetime
    assert capital[-1] == pytest.approx(100 * np.exp(rate * 50), rel=1e-3)
    assert np.sign(round(capital[-1] - 100, 6)) == sign
