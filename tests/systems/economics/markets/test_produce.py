"""Test the `_produce` module of `skcomplex.systems.economics.markets`."""

import numpy as np

from skcomplex.systems.economics.markets import (
    ProduceMarketSystem,
)

from ..conftest import _agents, _run


def test_produce_market_lists_shrink_until_every_shopper_is_done():
    """Test that shopping lists only shrink, and that stalls selling everything serve every list at their markups."""
    trajectory = _run(
        ProduceMarketSystem(30, 5, n_items_stocked=4, random_state=1).prepare().system_,
        6,
        observables=ProduceMarketSystem().get_run_settings()['observables'],
    )
    assert np.all(np.diff(trajectory.values('items_left')) <= 0)
    assert trajectory.values('items_left')[-1] == 0
    complete = _run(
        ProduceMarketSystem(30, 5, n_items_stocked=12, random_state=1).prepare().system_,
        1,
        observables=ProduceMarketSystem().get_run_settings()['observables'],
    )
    assert complete.values('items_left')[-1] == 0
    initial = (
        ProduceMarketSystem(30, 5, n_items_stocked=12, random_state=1).prepare().system_.elements[1].values['wanted']
    )
    prices = (0.6, 0.5, 0.7, 0.9, 1.2, 1.5, 0.8, 0.4, 0.5, 0.3, 0.6, 1.0)
    costs = np.array([sum(price for item, price in enumerate(prices) if int(mask) >> item & 1) for mask in initial])
    ratio = _agents(complete, 'shoppers', 1)['spent'] / costs
    assert np.all((ratio >= 1.01 - 1e-9) & (ratio <= 1.3 + 1e-9))
