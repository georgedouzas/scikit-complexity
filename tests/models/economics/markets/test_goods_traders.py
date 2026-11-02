"""Test the `_goods_traders` module of `skcomplex.models.economics.markets`."""

import numpy as np
import pytest

from skcomplex.models import run
from skcomplex.models.economics.markets import goods_traders


def test_goods_traders_trades_conserve_goods():
    """Test that 50 steps of barter keep the total of every good while utility rises and prices converge."""
    trajectory = run('goods_traders', settings={'until': 50}).trajectory_
    for good in ('sugar', 'spice', 'salt'):
        np.testing.assert_allclose(trajectory.values(good), trajectory.values(good)[0], rtol=1e-9)
    dispersion = trajectory.values('exchange.dispersion')
    assert np.nanmean(dispersion[-10:]) < 0.25 * np.nanmean(dispersion[1:6])
    assert trajectory.values('utility')[-1] > trajectory.values('utility')[0]


def test_goods_traders_of_two_goods_trade_no_salt():
    """Test that toy traders of two goods hold no salt and still trade."""
    trajectory = run('goods_traders', goods=2, settings={'until': 5}).trajectory_
    assert np.all(trajectory.values('salt') == 0)
    assert np.nansum(trajectory.values('exchange.volume')) > 0


def test_goods_traders_reject_other_numbers_of_goods():
    """Test that the traders barter two or three goods only."""
    with pytest.raises(ValueError, match='2 or 3 goods'):
        goods_traders(goods=4)
