"""Test the `_adaptive_auction` module of `skcomplex.systems.economics.markets`."""

import numpy as np

from skcomplex.systems.economics.markets import GoodingDoubleAuctionSystem

from ..conftest import _run


def test_gooding_double_auction_settles_near_the_competitive_price_in_the_short_run():
    """Test that the 500 units trade, after the quotes meet, at prices near 70, halfway between value and cost."""
    trajectory = _run(GoodingDoubleAuctionSystem(), 300)
    prices, volume = trajectory.values('auction.price'), trajectory.values('auction.volume')
    assert np.nansum(volume) == 500
    assert np.flatnonzero(volume > 0)[0] > 20
    assert abs(np.nanmean(prices[volume > 0]) - 70) < 5


def test_gooding_double_auction_keeps_trading_between_cost_and_value_without_limits():
    """Test that unlimited traders trade on, at prices that stay between the costs and the values but wander."""
    trajectory = _run(GoodingDoubleAuctionSystem(5, 5, unlimited=True), 3000)
    prices, volume = trajectory.values('auction.price')[200:], trajectory.values('auction.volume')
    assert volume[-100:].sum() > 0
    assert np.nanmin(prices) > 18
    assert np.nanmax(prices) < 132
    assert np.nanmax(prices) - np.nanmin(prices) > 2
