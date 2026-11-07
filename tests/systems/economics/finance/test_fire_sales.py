"""Test the `_fire_sales` module of `skcomplex.systems.economics.finance`."""

import numpy as np

from skcomplex.systems.economics.finance import overlapping_portfolio_fire_sales

from ..conftest import _agents, _run


def test_fire_sales_market_impact_spreads_defaults_beyond_the_shocked_holders():
    """Test that a small shock stays absorbed, and that market impact carries a large one past the direct holders."""
    assert _run(overlapping_portfolio_fire_sales(), 30).values('sales.defaults')[-1] == 0
    direct = _run(overlapping_portfolio_fire_sales(shock=0.6, alpha=0.0), 30).values('sales.defaults')[-1]
    system = overlapping_portfolio_fire_sales(shock=0.6)
    holders = np.count_nonzero(np.asarray(system.interactions[0].holdings)[:, 0])
    assert direct == holders
    assert _run(system, 30).values('sales.defaults')[-1] > 2 * holders


def test_fire_sales_leverage_targeting_sells_where_thresholds_stay_passive():
    """Test that leverage targeting sells and lowers prices after a shock that threshold banks absorb."""
    passive = _run(overlapping_portfolio_fire_sales(alpha=0.02), 30)
    active = _run(overlapping_portfolio_fire_sales(alpha=0.02, selling='leverage'), 30)
    assert np.nansum(passive.values('sales.sold')) == 0
    assert np.nansum(active.values('sales.sold')) > 0
    assert _agents(active, 'assets', 30)['price'].mean() < _agents(passive, 'assets', 30)['price'].mean()
