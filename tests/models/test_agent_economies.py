"""Test phases, markets, institutions, and learning blocks of agent-based models."""

import math

import numpy as np
import pytest

from skcomplex.core.base import SimulationError, parameter, role, state
from skcomplex.core.components import Element, memory
from skcomplex.core.interactions import Interaction, Match, matches
from skcomplex.core.laws import Rule
from skcomplex.core.runtime import Simulation
from skcomplex.core.space import DiscreteTime, LatticeSpace, Phase, Phases
from skcomplex.core.system import Observable, System
from skcomplex.models.economics.realizations import (
    BilateralSearch,
    CallAuction,
    DeferredAcceptance,
    Institution,
    Market,
    Order,
    OrderBook,
    RationedQueue,
    SampleAndSwitch,
    Walrasian,
    evolve,
    heuristics,
    logit_choice,
    logit_probabilities,
)
from skcomplex.models.social.realizations import imitate


class Employer(Element):
    """Test helper Employer."""

    offer: float = state(0.0)

    def laws(self, s):
        """Test helper laws."""
        return [Rule(self.post, phase='offers')]

    def post(self, agent, model):
        """Test helper post."""
        agent.offer = model.time + 1.0


class Worker(Element):
    """Test helper Worker."""

    accepted: float = state(-1.0)
    stale: float = state(0.0)

    def laws(self, s):
        """Test helper laws."""
        return [Rule(self.accept, phase='acceptance')]

    def accept(self, agent, model):
        """Test helper accept."""
        best = max(employer.offer for employer in model.agents('employers'))
        agent.stale += float(best != model.time + 1.0)
        agent.accepted = best


def test_offer_before_acceptance():
    """Every acceptance refers to an offer made earlier in the same step."""
    system = System(
        components=[Worker.group('workers', 5), Employer.group('employers', 3)],
        time=DiscreteTime(schedule=Phases([Phase('offers'), Phase('acceptance')])),
    )
    trajectory = Simulation(system, random_state=0).run(until=10).trajectory_
    workers = trajectory.agents('workers', 10)
    assert np.all(workers.values['stale'] == 0)
    assert np.all(workers.values['accepted'] == trajectory.agents('employers', 10).values['offer'].max())


def test_undeclared_phase_raises():
    """A rule in a phase that the system does not declare raises an error."""
    system = System(components=[Employer.group('employers', 2)])
    with pytest.raises(SimulationError, match='offers'):
        Simulation(system).run(until=1)


def test_phases_keep_unphased_models():
    """Rules without a phase run in the default phase."""

    class Counter(Element):
        count: float = state(0.0)

        def laws(self, s):
            return [Rule(self.tick)]

        def tick(self, agent, model):
            agent.count += 1

    system = System(
        components=[Counter.group('counters', 3)],
        time=DiscreteTime(schedule=Phases([Phase('extra', repeat=2)])),
    )
    trajectory = Simulation(system, random_state=0).run(until=4).trajectory_
    assert trajectory.agents('counters', 4).values['count'].tolist() == [4.0, 4.0, 4.0]


class Imitator(Element):
    """Test helper Imitator."""

    strategy: float = state(0.0)
    payoff: float = state(0.0)

    def laws(self, s):
        """Test helper laws."""
        return [Rule(self.copy, phase='imitation')]

    def copy(self, agent, model):
        """Test helper copy."""
        imitate(agent, agent.neighbors(), 'payoff', ('strategy', 'payoff'))


def test_simultaneous_imitation():
    """Every agent reads its neighbours' strategies from the start of a simultaneous phase."""
    width = 9
    locations = [(x, 0) for x in range(width)]
    payoffs = np.zeros(width)
    payoffs[0] = 1.0
    system = System(
        components=[Imitator.group('players', width, location=locations, strategy=payoffs, payoff=payoffs)],
        space=LatticeSpace((width, 1), boundary='fixed', neighborhood='von_neumann'),
        time=DiscreteTime(schedule=Phases([Phase('imitation', activation='simultaneous')])),
    )
    trajectory = Simulation(system, random_state=0).run(until=3).trajectory_
    for frame in range(4):
        players = trajectory.agents('players', frame)
        order = np.argsort(np.asarray(players.locations)[:, 0])
        assert players.values['strategy'][order].tolist() == [1.0] * (frame + 1) + [0.0] * (width - frame - 1)


def test_sequential_and_by_type_order():
    """Sequential activation follows creation, and by-type activation follows the listed groups."""
    seen = []

    class First(Element):
        def laws(self, s):
            return [Rule(lambda agent, model: seen.append(('first', agent.id)), phase='go')]

    class Second(Element):
        def laws(self, s):
            return [Rule(lambda agent, model: seen.append(('second', agent.id)), phase='go')]

    system = System(
        components=[First.group('first', 3), Second.group('second', 3)],
        time=DiscreteTime(schedule=Phases([Phase('go', activation='by-type', groups=['second', 'first'])])),
    )
    Simulation(system, random_state=0).run(until=1)
    assert [group for group, _ in seen] == ['second'] * 3 + ['first'] * 3
    seen.clear()
    sequential = System(
        components=[First.group('first', 3), Second.group('second', 3)],
        time=DiscreteTime(schedule=Phases([Phase('go', 'sequential')])),
    )
    Simulation(sequential, random_state=0).run(until=1)
    assert [identity for _, identity in seen] == sorted(identity for _, identity in seen)


def test_phase_checks_arguments():
    """A phase rejects an unknown activation and fewer than one round."""
    with pytest.raises(ValueError, match='activation'):
        Phase('x', activation='parallel')
    with pytest.raises(ValueError, match='round'):
        Phase('x', repeat=0)


class Trader(Element):
    """Test helper Trader."""

    bid: float = state(0.0)
    ask: float = state(0.0)
    demand: float = state(0.0)
    supply: float = state(0.0)
    money: float = state(0.0)
    goods: float = state(0.0)
    value: float = state(0.0)


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


def _zero_intelligence(n_traders, rng):
    """Return the values, costs, and a market of zero-intelligence traders with budget constraints."""
    values = rng.uniform(50, 200, n_traders)
    costs = rng.uniform(0, 150, n_traders)

    class Buyer(Trader):
        def laws(self, s):
            return [Rule(self.quote, phase='quote')]

        def quote(self, agent, model):
            agent.bid = model.rng.uniform(0, agent.value) if agent.demand > 0 else 0.0

    class Seller(Trader):
        def laws(self, s):
            return [Rule(self.quote, phase='quote')]

        def quote(self, agent, model):
            agent.ask = model.rng.uniform(agent.value, 200) if agent.supply > 0 else 200.0

    buyers = Buyer.group('buyers', n_traders, value=values, demand=1.0, money=values)
    sellers = Seller.group('sellers', n_traders, value=costs, supply=1.0)
    market = Market('auction', buyers='buyers', sellers='sellers', protocol=CallAuction(), phase='clear')
    system = System(
        components=[buyers, sellers],
        interactions=[market],
        time=DiscreteTime(schedule=Phases([Phase('quote'), Phase('clear')])),
    )
    return values, costs, system


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
    assert np.nansum(simulation.trajectory_['auction.volume']) == bought.sum()


class Shop(Element):
    """Shop whose price grows by a drift and noise, slowed by the interest rate."""

    price: float = state(1.0)
    drift: float = parameter(0.03)

    def laws(self, s):
        """Return the pricing rule."""
        return [Rule(self.inflate)]

    def inflate(self, agent, model):
        """Raise the price."""
        agent.price *= 1 + model.parameter('drift') + model.rng.normal(0, 0.02) - model.parameters['bank.rate']


class CentralBank(Institution):
    """Central bank that raises its rate whenever inflation of the price level exceeds the target."""

    reports = ('inflation',)
    rate: float = parameter(0.0)
    target: float = parameter(0.02)

    def act(self, model):
        """Measure inflation and raise the rate above the target."""
        level = model.observable('price_level')
        inflation = level / model.storage.get('level', 1.0) - 1
        model.storage['level'] = level
        model.report('inflation', inflation)
        if inflation > model.parameter('target'):
            model.set_parameter('rate', model.parameter('rate') + 0.005)


def test_interest_rate_rule():
    """The interest rate changes only at steps where inflation exceeds the target."""
    system = System(
        components=[Shop.group('shops', 20)],
        interactions=[CentralBank('bank', phase='policy')],
        time=DiscreteTime(schedule=Phases([Phase('policy')])),
    )
    observables = [Observable('price_level', group='shops', variable='price')]
    trajectory = Simulation(system, random_state=0, observables=observables).run(until=40).trajectory_
    rate, inflation = trajectory['bank.rate'], trajectory['bank.inflation']
    changed = np.diff(rate) != 0
    assert changed.any()
    assert not changed.all()
    np.testing.assert_array_equal(changed, inflation[1:] > 0.02)
    assert np.isnan(inflation[0])


def test_undeclared_report_raises():
    """Reporting a name the owner does not declare raises an error."""
    system = System(components=[Trader.group('traders', 1)], interactions=[Institution('reader', rule=_report_total)])
    with pytest.raises(SimulationError, match='declare'):
        Simulation(system).run(until=1)


def _report_total(model):
    """Report an undeclared value."""
    model.report('total', 1.0)


def test_observable_and_set_parameter_by_view():
    """Institutions read observables and reject unknown parameters."""

    class Reader(Institution):
        reports = ('total',)

        def act(self, model):
            model.report('total', model.observable('wealth'))
            with pytest.raises(SimulationError, match='No parameter'):
                model.set_parameter('missing', 1.0)

    system = System(
        components=[Trader.group('traders', 4, money=2.0)],
        interactions=[Reader('reader')],
    )
    observables = [Observable('wealth', group='traders', variable='money', reduce='sum')]
    trajectory = Simulation(system, random_state=0, observables=observables).run(until=2).trajectory_
    assert trajectory['reader.total'][-1] == 8.0


class Saver(Element):
    """Test helper Saver."""

    remembers = {'money': 3}  # noqa: RUF012
    money: float = state(0.0)
    seen: float = state(0.0)

    def laws(self, s):
        """Test helper laws."""
        return [Rule(self.save)]

    def save(self, agent, model):
        """Test helper save."""
        agent.seen = float(len(memory(agent, 'money')))
        agent.money += 1


def test_memory_window():
    """Agents remember the last values of the states their type lists."""
    trajectory = Simulation(System(components=[Saver.group('savers', 2)]), random_state=0).run(until=5).trajectory_
    assert trajectory.agents('savers', 5).values['seen'].tolist() == [3.0, 3.0]


def test_memory_of_unremembered_state_raises():
    """Reading the memory of a state the type does not remember raises an error."""

    class Forgetful(Element):
        money: float = state(0.0)

        def laws(self, s):
            return [Rule(lambda agent, model: memory(agent, 'money'))]

    with pytest.raises(KeyError, match='remember'):
        Simulation(System(components=[Forgetful.group('agents', 1)])).run(until=1)


def test_heuristics_and_logit():
    """Forecasting heuristics and logit choice follow their formulas."""
    assert heuristics.adaptive([2.0], 1.0, weight=0.5) == 1.5
    assert heuristics.trend([1.0, 2.0], strength=1.0) == 3.0
    assert heuristics.anchoring([1.0, 3.0]) == 4.5
    assert heuristics.fundamental([3.0], fundamental=1.0, reversion=0.5) == 2.0
    probabilities = logit_probabilities([1.0, 0.0], np.log(3.0))
    np.testing.assert_allclose(probabilities, [0.75, 0.25])
    assert logit_probabilities([1.0, 2.0], math.inf).tolist() == [0.0, 1.0]
    assert logit_choice([0.0, 50.0], 1.0, np.random.default_rng(0)) == 1


def test_evolve_operators():
    """Evolution keeps elites, mutates bits, and rejects unknown selections."""
    rng = np.random.default_rng(0)
    genomes = rng.integers(0, 2, (20, 8))
    fitness = genomes.sum(axis=1).astype(float)
    child = evolve(genomes, fitness, rng, elite=2)
    assert child.shape == genomes.shape
    assert set(np.unique(child)) <= {0.0, 1.0}
    np.testing.assert_array_equal(child[0], genomes[np.argmax(fitness)])
    real = evolve(rng.normal(size=(6, 3)), np.arange(6.0), rng, select='roulette', alphabet=None, mutation=1.0)
    assert real.shape == (6, 3)
    with pytest.raises(ValueError, match='selection'):
        evolve(genomes, fitness, rng, select='lottery')


class LimitTrader(Element):
    """Trader with a limit price on both sides of the book."""

    bid: float = state(0.0)
    ask: float = state(math.inf)

    def laws(self, s):
        """Return no rules."""
        return []


class Book(Interaction):
    """Interaction that pairs buyers with sellers by an order book and records the pairs."""

    buyers: str = role()
    sellers: str = role()

    def laws(self, s):
        """Return the matching and the rule that records it."""
        return [Match('buyers', 'sellers', pairing=OrderBook(arrival='sequential'), label='trades'), Rule(self.record)]

    def record(self, model):
        """Record the ids of the buyer and seller of every pair, and the id of the earliest seller."""
        PAIRS.extend((buyer.id, seller.id) for buyer, seller in matches(model, 'trades'))
        EARLIEST.append(min(agent.id for agent in model.agents('sellers')))


PAIRS = []
EARLIEST = []


def test_order_book_is_a_pairing_of_the_matching_law():
    """Of two same-price sell orders, the earlier one trades first when the order book pairs a matching law."""
    PAIRS.clear()
    EARLIEST.clear()
    system = System(
        components=[LimitTrader.group('buyers', 1, bid=10.0), LimitTrader.group('sellers', 2, ask=10.0)],
        interactions=[Book('book', buyers='buyers', sellers='sellers')],
    )
    Simulation(system).run(1)
    assert len(PAIRS) == 1
    assert PAIRS[0][1] == EARLIEST[0]
