"""Test the `_markets` module of `skcomplex.models.economics.realizations`."""

import math

import numpy as np
import pytest

from skcomplex.core.base import BaseInteraction, role, state
from skcomplex.core.components import ElementComponent
from skcomplex.core.space import DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.agents import MatchLaw, Phase, PhasesSchedule, RuleLaw, matches
from skcomplex.models.economics.realizations import (
    BilateralSearch,
    CallAuction,
    DeferredAcceptance,
    MarketInteraction,
    Order,
    OrderBook,
    RationedQueue,
    SampleAndSwitch,
    Walrasian,
)

from .conftest import TraderComponent


def _orders(limits, budgets=None):
    """Return orders of plain agents with ids, limits, and quantities of one."""

    class Agent:
        def __init__(self, identity):
            self.id = identity

    budgets = budgets or [math.inf] * len(limits)
    return [
        Order(Agent(index), limit, 1.0, budget)
        for index, (limit, budget) in enumerate(zip(limits, budgets, strict=True))
    ]


class _Model:
    """Minimal model view for protocols."""

    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.storage = {}


def _zero_intelligence(n_traders, rng):
    """Return the values, costs, and a market of zero-intelligence traders with budget constraints."""
    values = rng.uniform(50, 200, n_traders)
    costs = rng.uniform(0, 150, n_traders)

    class BuyerComponent(TraderComponent):
        def laws(self, s):
            return [RuleLaw(self.quote, phase='quote')]

        def quote(self, agent, model):
            agent.bid = model.rng.uniform(0, agent.value) if agent.demand > 0 else 0.0

    class SellerComponent(TraderComponent):
        def laws(self, s):
            return [RuleLaw(self.quote, phase='quote')]

        def quote(self, agent, model):
            agent.ask = model.rng.uniform(agent.value, 200) if agent.supply > 0 else 200.0

    buyers = BuyerComponent.group('buyers', n_traders, value=values, demand=1.0, money=values)
    sellers = SellerComponent.group('sellers', n_traders, value=costs, supply=1.0)
    market = MarketInteraction('auction', buyers='buyers', sellers='sellers', protocol=CallAuction(), phase='clear')
    system = System(
        components=[buyers, sellers],
        interactions=[market],
        time=DiscreteTime(schedule=PhasesSchedule([Phase('quote'), Phase('clear')])),
    )
    return values, costs, system


class LimitTraderComponent(ElementComponent):
    """Trader with a limit price on both sides of the book."""

    bid: float = state(0.0)
    ask: float = state(math.inf)

    def laws(self, s):
        """Return no rules."""
        return []


class BookInteraction(BaseInteraction):
    """Interaction that pairs buyers with sellers by an order book and records the pairs."""

    buyers: str = role()
    sellers: str = role()

    def laws(self, s):
        """Return the matching and the rule that records it."""
        return [
            MatchLaw('buyers', 'sellers', pairing=OrderBook(arrival='sequential'), label='trades'),
            RuleLaw(self.record),
        ]

    def record(self, model):
        """Record the ids of the buyer and seller of every pair, and the id of the earliest seller."""
        PAIRS.extend((buyer.id, seller.id) for buyer, seller in matches(model, 'trades'))
        EARLIEST.append(min(agent.id for agent in model.agents('sellers')))


PAIRS: list[tuple[int, int]] = []


EARLIEST: list[int] = []


def test_order_book_priority():
    """Of two sell orders at the same price, the earlier one trades first."""
    sellers = _orders([10.0, 10.0])
    buyers = _orders([10.0])
    buyers[0].agent.id = 5
    trades = OrderBook(arrival='sequential').match(buyers, sellers, _Model())
    assert len(trades) == 1
    assert trades[0].seller is sellers[0].agent
    assert trades[0].price == 10.0


def test_order_book_trades_at_resting_price():
    """An incoming bid trades at the best resting asks, cheapest first."""
    sellers = _orders([12.0, 9.0, 11.0])
    buyers = _orders([11.5])
    buyers[0].quantity = 3.0
    buyers[0].agent.id = 9
    trades = OrderBook(arrival='sequential').match(buyers, sellers, _Model())
    assert [trade.price for trade in trades] == [9.0, 11.0]


def test_call_auction_uniform_price():
    """A call auction pairs the highest bids with the lowest asks at one price."""
    buyers, sellers = _orders([10.0, 8.0, 3.0]), _orders([2.0, 5.0, 9.0])
    trades = CallAuction().match(buyers, sellers, _Model())
    assert len(trades) == 2
    assert {trade.price for trade in trades} == {6.5}


def test_rationed_queue_and_walrasian():
    """A queue serves buyers at the cheapest asks, and the auctioneer clears at a price between the limits."""
    buyers, sellers = _orders([10.0, 9.0, 1.0]), _orders([4.0, 6.0])
    trades = RationedQueue(order='bid').match(buyers, sellers, _Model())
    assert [(trade.price, trade.buyer.id) for trade in trades] == [(4.0, 0), (6.0, 1)]
    buyers, sellers = _orders([10.0, 9.0, 1.0]), _orders([4.0, 6.0])
    trades = Walrasian().match(buyers, sellers, _Model())
    assert len(trades) == 2
    assert all(6.0 <= trade.price <= 9.0 for trade in trades)


def test_bilateral_search_and_budget():
    """Bilateral search trades at the midpoint, within the buyer's budget."""
    buyers, sellers = _orders([10.0], [3.0]), _orders([2.0])
    trades = BilateralSearch(tries=2).match(buyers, sellers, _Model())
    assert trades[0].price == 6.0
    assert trades[0].quantity == pytest.approx(0.5)


def test_deferred_acceptance_is_stable():
    """The stable matching gives the most wanted seller to its favourite buyer."""
    buyers, sellers = _orders([10.0, 12.0]), _orders([5.0, 7.0])
    for index, order in enumerate(sellers):
        order.agent.id = 10 + index

    class Agent:
        def __init__(self, identity, bid=0.0, ask=0.0):
            self.id, self.bid, self.ask = identity, bid, ask

    for order in buyers:
        order.agent = Agent(order.agent.id, bid=order.limit)
    for order in sellers:
        order.agent = Agent(order.agent.id, ask=order.limit)
    trades = DeferredAcceptance().match(buyers, sellers, _Model())
    pairs = {(trade.buyer.id, trade.seller.id) for trade in trades}
    assert pairs == {(1, 10), (0, 11)}


def test_sample_and_switch_keeps_partner():
    """A buyer stays with a partner that asks no more than the sampled sellers."""

    class Agent:
        def __init__(self, identity, partner=-1.0):
            self.id, self.partner = identity, partner

    buyers = [Order(Agent(0, partner=11.0), 10.0, 1.0)]
    sellers = [Order(Agent(10), 5.0, 1.0), Order(Agent(11), 4.0, 1.0)]
    trades = SampleAndSwitch(sample=2).match(buyers, sellers, _Model())
    assert trades[0].seller.id == 11


def test_double_auction_efficiency():
    """Zero-intelligence traders with budget constraints realise over 90% of the maximum surplus."""
    rng = np.random.default_rng(1)
    values, costs, system = _zero_intelligence(50, rng)
    simulation = Simulation(system, random_state=0).run(until=30)
    buyers = simulation.trajectory_.agents('buyers', 30)
    sellers = simulation.trajectory_.agents('sellers', 30)
    bought = buyers.values['goods'] > 0
    sold = sellers.values['goods'] < 0
    realised = buyers.values['value'][bought].sum() - sellers.values['value'][sold].sum()
    sorted_values, sorted_costs = np.sort(values)[::-1], np.sort(costs)
    maximum = np.sum(np.maximum(sorted_values - sorted_costs, 0))
    assert realised > 0.9 * maximum
    money = buyers.values['money'].sum() + sellers.values['money'].sum()
    assert money == pytest.approx(values.sum())
    assert np.nansum(simulation.trajectory_.values('auction.volume')) == bought.sum()


def test_order_book_is_a_pairing_of_the_matching_law():
    """Of two same-price sell orders, the earlier one trades first when the order book pairs a matching law."""
    PAIRS.clear()
    EARLIEST.clear()
    system = System(
        components=[
            LimitTraderComponent.group('buyers', 1, bid=10.0),
            LimitTraderComponent.group('sellers', 2, ask=10.0),
        ],
        interactions=[BookInteraction('book', buyers='buyers', sellers='sellers')],
    )
    Simulation(system).run(1)
    assert len(PAIRS) == 1
    assert PAIRS[0][1] == EARLIEST[0]
