"""Test the `_order_book` module of `skcomplex.models.economics.finance`."""

import numpy as np
import pytest
from scipy import stats

from skcomplex.models import run


def test_order_book_chartists_raise_volatility_and_conserve_holdings():
    """Test that chartists make returns more volatile and that trades keep the total money and shares."""
    returns = {}
    for chartists in (0.0, 0.5):
        trajectory = run('order_book', chartists=chartists, imitation=0.0, settings={'until': 150}).trajectory_
        prices = trajectory.values('book.price')[1:]
        returns[chartists] = np.diff(np.log(prices))
        traders = trajectory.agents('traders', 150).values
        assert traders['money'].sum() == pytest.approx(400 * 1000.0)
        assert traders['shares'].sum() == pytest.approx(400 * 10.0)
    assert returns[0.5].std() > 1.5 * returns[0.0].std()
    assert abs(stats.trim_mean(returns[0.0], 0.1)) < 0.01


def test_order_book_imitation_drives_out_chartists():
    """Test that traders copy the more successful fundamentalists, so the share of chartists falls."""
    trajectory = run('order_book', chartists=0.5, settings={'until': 150}).trajectory_
    assert trajectory.values('chartists')[-1] < 0.6 * trajectory.values('chartists')[0]
