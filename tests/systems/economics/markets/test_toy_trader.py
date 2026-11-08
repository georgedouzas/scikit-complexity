"""Test the `_toy_trader` module of `skcomplex.systems.economics.markets`."""

import pytest

from skcomplex.systems.economics.markets import ToyTraderSystem

from ..conftest import _agents, _run


def test_toy_trader_price_depends_on_market_sight_with_fixed_supply_and_demand():
    """Test that money and toys are conserved and that the price falls with a short sight and rises with a long one."""
    near = _run(ToyTraderSystem(sight=1.0), 300)
    assert _agents(near, 'traders', 300)['money'].sum() == pytest.approx(40000.0)
    assert _agents(near, 'traders', 300)['toys'].sum() == pytest.approx(2000.0)
    assert near.values('market.average_price')[-1] < 9.0
    far = _run(ToyTraderSystem(sight=40.0), 300)
    assert far.values('market.average_price')[-1] > 11.0
    assert near.values('market.money_gini')[-1] > 0.3


def test_toy_trader_asset_fitness_inflates_the_price_and_three_goods_conserve_stocks():
    """Test asset price inflation under evolution with an asset fitness test, and the stocks of the three goods."""
    evolving = _run(ToyTraderSystem(mutation=1 / 180, fitness='assets'), 300)
    assert evolving.values('market.average_price')[-1] > 20.0
    three = _agents(_run(ToyTraderSystem(200, n_goods=3), 50), 'traders', 50)
    assert three['crayons'].sum() == pytest.approx(200 * 30.0)
    assert three['blackboards'].sum() == pytest.approx(200 * 15.0)
