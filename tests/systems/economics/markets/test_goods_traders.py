"""Test the `_goods_traders` module of `skcomplex.systems.economics.markets`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.markets import GoodsTradersSystem


def test_goods_traders_trades_conserve_goods():
    """Test that 50 steps of barter keep the total of every good while utility rises and prices converge."""
    trajectory = simulate(GoodsTradersSystem(), until=50)
    for good in ('sugar', 'spice', 'salt'):
        np.testing.assert_allclose(trajectory.get_values(good), trajectory.get_values(good)[0], rtol=1e-9)
    dispersion = trajectory.get_values('exchange.dispersion')
    assert np.nanmean(dispersion[-10:]) < 0.25 * np.nanmean(dispersion[1:6])
    assert trajectory.get_values('utility')[-1] > trajectory.get_values('utility')[0]


def test_goods_traders_of_two_goods_trade_no_salt():
    """Test that toy traders of two goods hold no salt and still trade."""
    trajectory = simulate(GoodsTradersSystem(goods=2), until=5)
    assert np.all(trajectory.get_values('salt') == 0)
    assert np.nansum(trajectory.get_values('exchange.volume')) > 0


def test_goods_traders_reject_other_numbers_of_goods():
    """Test that the traders barter two or three goods only."""
    with pytest.raises(ValueError, match='2 or 3 goods'):
        GoodsTradersSystem(goods=4).prepare()
