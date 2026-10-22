"""Test the market, credit, and banking models built from markets, phases, and institutions."""

import numpy as np
import pytest

from skcomplex.core.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.economics.macroeconomics import credit_networks, fractional_reserve_banking, metaphoria
from skcomplex.models.economics.markets import double_auction, maximum_surplus, wilhite_trade_networks


def _run(system, until, observables=()):
    """Return the trajectory of a model run with a fixed random state."""
    return Simulation(system, random_state=0, observables=observables).run(until=until).trajectory_


@pytest.mark.parametrize('protocol', ['order_book', 'call_auction'])
def test_double_auction_budget_constraint_is_efficient(protocol):
    """Zero-intelligence traders with budget constraints realise over 90% of the surplus, without them much less."""
    constrained = _run(double_auction(30, 30, protocol=protocol), 10)['ledger.efficiency'][1:]
    unconstrained = _run(double_auction(30, 30, protocol=protocol, constrained=False), 10)['ledger.efficiency'][1:]
    assert constrained.mean() > 0.9
    assert unconstrained.mean() < constrained.mean() - 0.1


def test_double_auction_rejects_unknown_protocol():
    """An unknown protocol raises an error."""
    with pytest.raises(ValueError, match='protocol'):
        double_auction(protocol='barter')


def test_maximum_surplus():
    """The maximum surplus pairs the highest values with the lowest costs."""
    assert maximum_surplus(np.array([10.0, 6.0, 2.0]), np.array([1.0, 4.0, 8.0])) == 11.0


def test_credit_switching_concentrates_loans():
    """A higher intensity of switching concentrates loans in fewer banks and narrows the spread of rates."""
    slow = _run(
        built := credit_networks(60, 8, intensity=0.5),
        40,
        observables=describe('credit_networks').observables_of(built),
    )
    fast = _run(
        built := credit_networks(60, 8, intensity=20.0),
        40,
        observables=describe('credit_networks').observables_of(built),
    )
    assert fast['herfindahl'][-20:].mean() > slow['herfindahl'][-20:].mean()
    assert fast['credit.spread'][-20:].mean() < slow['credit.spread'][-20:].mean()


def test_credit_net_worth_moves_between_firms_and_banks():
    """Interest and bad debt move net worth between firms and banks, so the total changes by returns and entries."""
    trajectory = _run(
        built := credit_networks(40, 5),
        20,
        observables=describe('credit_networks').observables_of(built),
    )
    total = trajectory['bank_equity'] + trajectory['firm_worth']
    changes = trajectory['returns'][1:] + trajectory['injected'][1:]
    np.testing.assert_allclose(np.diff(total), changes, rtol=1e-6, atol=1e-6)


def test_fractional_reserve_money_multiplier():
    """Broad money settles at base money over the reserve ratio, and moves to the new multiplier when it changes."""
    trajectory = _run(
        built := fractional_reserve_banking(ratio=0.1, new_ratio=0.2, change_at=100.0),
        200,
        observables=describe('fractional_reserve_banking').observables_of(built),
    )
    np.testing.assert_allclose(trajectory['reserves'], 1000.0)
    assert trajectory['broad_money'][90:100].mean() == pytest.approx(10_000.0, rel=0.05)
    assert trajectory['broad_money'][190:].mean() == pytest.approx(5_000.0, rel=0.05)
    ratio = trajectory['banks.reserve_ratio']
    assert ratio[99] == pytest.approx(0.1)
    assert ratio[-1] == pytest.approx(0.2)


def _barter(topology):
    """Return the trajectory of Wilhite's barter of 100 traders in 5 groups on one topology."""
    system = wilhite_trade_networks(100, topology=topology, n_groups=5, shortcuts=3, random_state=1)
    return _run(system, 30, describe('wilhite_trade_networks').observables_of(system))


def test_wilhite_trade_networks_small_world_converges_with_few_searches():
    """Isolated groups keep different prices, a small world unifies them with a fraction of the complete searches."""
    isolated, small_world, complete = (_barter(topology) for topology in ('isolated', 'small_world', 'complete'))
    assert small_world['dispersion'][-1] < 0.1
    assert isolated['dispersion'][-1] > 2 * small_world['dispersion'][-1]
    assert complete['barter.searches'][1:].sum() > 4 * small_world['barter.searches'][1:].sum()
    goods = [small_world.agents('traders', frame).values for frame in (0, 30)]
    for good in ('good1', 'good2'):
        np.testing.assert_allclose(np.sum(goods[1][good]), np.sum(goods[0][good]))
    assert small_world['utility'][-1] > small_world['utility'][0]


def test_wilhite_trade_networks_rejects_unknown_topology():
    """An unknown topology raises an error."""
    with pytest.raises(ValueError, match='topology'):
        wilhite_trade_networks(10, topology='ring')


def _metaphoria(until, **arguments):
    """Return the trajectory of Metaphoria with its standard observables."""
    system = metaphoria(100, random_state=0, **arguments)
    return _run(system, until, describe('metaphoria').observables_of(system))


def test_metaphoria_hunter_gatherers_thrive_where_money_dies_out():
    """Hunter-gatherers keep a stable population on land where the monetary population dies out, as in Gooding."""
    gatherers = _metaphoria(150)['population']
    monetary = _metaphoria(150, mode='monetary')['population']
    assert gatherers[100:].min() > 50
    assert monetary[-1] < gatherers[-1] / 4


def test_metaphoria_monetary_money_is_stock_flow_consistent():
    """With abundant food the monetary economy lives with starvation, and all money held is matched by loans."""
    trajectory = _metaphoria(100, mode='monetary', max_food=50.0, max_resistance=0.0)
    assert trajectory['population'][-1] > 100
    assert np.nansum(trajectory['census.starvations']) > 0
    assert trajectory['money_gini'][-1] > 0.3
    holders = [trajectory.agents(group, 100).values for group in ('people', 'businesses', 'bank')]
    money = sum(np.sum(values['money']) for values in holders)
    loans = sum(np.sum(values['loans']) for values in holders)
    assert abs(money) < 1e-6 * np.sum(holders[2]['loans'])
    assert abs(loans) < 1e-6 * np.sum(holders[2]['loans'])
    assert np.sum(holders[2]['loans']) > 0


def test_metaphoria_rejects_unknown_mode():
    """An unknown mode raises an error."""
    with pytest.raises(ValueError, match='mode'):
        metaphoria(mode='barter')
