"""Test the `_order_book` module of `skcomplex.systems.economics.finance`."""

import numpy as np
import pytest
from scipy import stats

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.finance import OrderBookSystem


def test_order_book_chartists_raise_volatility_and_conserve_holdings():
    """Test that chartists make returns more volatile and that trades keep the total money and shares."""
    returns = {}
    for chartists in (0.0, 0.5):
        trajectory = simulate(OrderBookSystem(chartists=chartists, imitation=0.0), until=150)
        prices = trajectory.get_values('book.price')[1:]
        returns[chartists] = np.diff(np.log(prices))
        traders = trajectory.get_agents('traders', 150).values
        assert traders['money'].sum() == pytest.approx(400 * 1000.0)
        assert traders['shares'].sum() == pytest.approx(400 * 10.0)
    assert returns[0.5].std() > 1.5 * returns[0.0].std()
    assert abs(stats.trim_mean(returns[0.0], 0.1)) < 0.01


def test_order_book_imitation_drives_out_chartists():
    """Test that traders copy the more successful fundamentalists, so the share of chartists falls."""
    trajectory = simulate(OrderBookSystem(chartists=0.5), until=150, random_state=0)
    assert trajectory.get_values('chartists')[-1] < 0.6 * trajectory.get_values('chartists')[0]
