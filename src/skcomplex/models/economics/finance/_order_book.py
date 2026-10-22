"""Finance models: a stock market whose fundamentalists and chartists trade in a limit order book and imitate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from typing import Any, ClassVar, cast

import numpy as np

from ....core.backends import AgentView, ModelView
from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction, Match, matches
from ....core.laws import Invariant, Law, Rule, Symbols, transfer
from ....core.runtime import register_model
from ....core.space import DiscreteTime, LatticeSpace, Phase, Phases
from ....core.system import Observable, System
from ..realizations import OrderBook, Trade

LOWEST_PRICE = 0.01


def _total(holding: str, model: Any) -> float:  # noqa: ANN401
    """Return the total of a holding over the traders bound to a book."""
    return float(sum(getattr(agent, holding) for agent in model.agents('traders')))


def _order_book_observables(system: System) -> list[Observable]:
    """Return the share of chartists and the mean wealth of the traders."""
    del system
    return [
        Observable('chartists', group='traders', variable='strategy', reduce='mean'),
        Observable('mean_wealth', group='traders', variable='wealth', reduce='mean'),
    ]


class StockTrader(Element):
    """Trader who forecasts the price as a fundamentalist or a chartist, quotes a limit order, and imitates.

    A fundamentalist expects the price `p` to revert to the fundamental value `F`, `p + reversion (F - p)`, and a
    chartist expects the last change to go on, `p + extrapolation (p - p')`, both with a relative noise. The trader
    bids its expectation for one share when it is above the price and it can pay, and asks its expectation for one
    share when it is below and it holds one. After trading, a trader whose wealth grew less than that of a random
    neighbour copies the neighbour's strategy with the probability `imitation`.

    Args:
        label:
            The label of the agent or group.

        strategy:
            0 for a fundamentalist, 1 for a chartist.

        money:
            The money of the trader.

        shares:
            The shares of the trader.

        bid:
            The limit price of a buy order.

        ask:
            The limit price of a sell order.

        demand:
            The shares wanted in the step.

        supply:
            The shares offered in the step.

        wealth:
            The money plus the shares at the last price.

        gain:
            The change of the wealth in the last step.

        fundamental:
            The fundamental value of the share.

        reversion:
            The share of the gap to the fundamental value a fundamentalist expects to close.

        extrapolation:
            The share of the last change of the price a chartist expects to repeat.

        noise:
            The standard deviation of the relative noise of a forecast.

        imitation:
            The probability of copying a more successful neighbour.

    Examples:
        >>> from skcomplex.models.economics.finance import StockTrader
        >>> StockTrader.group('traders', 4, strategy=1.0).size
        4
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'strategy', 'color_range': (0.0, 1.0)})
    strategy: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    money: float = state(default=1000.0, domain=Domain.REAL)
    shares: float = state(default=10.0, domain=Domain.REAL)
    bid: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    ask: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    demand: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    supply: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    wealth: float = state(default=0.0, domain=Domain.REAL)
    gain: float = state(default=0.0, domain=Domain.REAL)
    fundamental: float = parameter(default=100.0, domain=Domain.POSITIVE)
    reversion: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    extrapolation: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    noise: float = parameter(default=0.02, domain=Domain.NONNEGATIVE)
    imitation: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the quoting, accounting, and imitation rules in their phases.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Rule(self.quote, phase='quote'),
            Rule(self.account, phase='account'),
            Rule(self.imitate, phase='imitation'),
        ]

    def _prices(self, model: ModelView) -> tuple[float, float]:
        """Return the last two prices of the book, the fundamental value before the first."""
        history = model.history('book.price', 2)
        prices = [value for value in history if math.isfinite(value)] or [model.parameter('fundamental')]
        return prices[-1], prices[0]

    def quote(self, agent: AgentView, model: ModelView) -> None:
        """Forecast the price and quote a limit order for one share.

        Args:
            agent:
                The trader.

            model:
                The model.
        """
        price, previous = self._prices(model)
        if agent.strategy:
            expected = price + model.parameter('extrapolation') * (price - previous)
        else:
            expected = price + model.parameter('reversion') * (model.parameter('fundamental') - price)
        expected = max(LOWEST_PRICE, expected * (1 + model.parameter('noise') * model.rng.normal()))
        buying = expected > price
        agent.bid, agent.ask = (expected, 0.0) if buying else (0.0, expected)
        agent.demand = float(buying and agent.money >= expected)
        agent.supply = float(not buying and agent.shares >= 1)

    def account(self, agent: AgentView, model: ModelView) -> None:
        """Value the wealth at the last price and record its change.

        Args:
            agent:
                The trader.

            model:
                The model.
        """
        price, _ = self._prices(model)
        wealth = agent.money + agent.shares * price
        agent.gain = wealth - agent.wealth if agent.wealth else 0.0
        agent.wealth = wealth

    def imitate(self, agent: AgentView, model: ModelView) -> None:
        """Copy the strategy of a random neighbour whose wealth grew more, with the probability of imitation.

        Args:
            agent:
                The trader.

            model:
                The model.
        """
        neighbours = agent.neighbors()
        if not neighbours or model.rng.random() >= model.parameter('imitation'):
            return
        other = neighbours[int(model.rng.integers(len(neighbours)))]
        if other.gain > agent.gain:
            agent.strategy = other.strategy


class Book(Interaction):
    """Limit order book that matches the orders of the traders with price-time priority and settles the trades.

    The matching law pairs buy and sell orders with the order book of `skcomplex.models.economics.realizations`, and
    every trade is two transfers, money from buyer to seller and shares from seller to buyer, so the invariants of the
    book check that the total money and the total shares stay the same. The book reports the last traded `price`, the
    previous price when nothing trades, and the traded `volume`.

    Args:
        label:
            The label of the book.

        traders:
            The path of the group of traders.

        opening:
            The price before the first trade.

    Examples:
        >>> from skcomplex.models.economics.finance import Book
        >>> Book('book', traders='traders').reports
        ('price', 'volume')
    """

    reports: ClassVar[tuple[str, ...]] = ('price', 'volume')
    traders: str = role()
    opening: float = 100.0

    def laws(self, s: Symbols) -> list[Any]:
        """Return the matching of orders, the settlement rule, and the invariants of money and shares.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Match('traders', pairing=OrderBook(), label='trades', phase='trade'),
            Rule(self.settle, phase='trade'),
            *(
                Invariant(f'total {name}', functools.partial(_total, name), tolerance=1e-6)
                for name in ('money', 'shares')
            ),
        ]

    def settle(self, model: Any) -> None:  # noqa: ANN401
        """Transfer the money and the shares of every trade and report the last price and the volume.

        Args:
            model:
                The model view of the book.
        """
        trades = cast('list[Trade]', matches(model, 'trades'))
        for trade in trades:
            transfer(trade.buyer, trade.seller, trade.quantity * trade.price, 'money')
            transfer(trade.seller, trade.buyer, trade.quantity, 'shares')
        last = model.history('price', 1)
        price = trades[-1].price if trades else (float(last[-1]) if len(last) else self.opening)
        model.report('price', price)
        model.report('volume', sum(trade.quantity for trade in trades))


@register_model(
    field='economics/finance',
    tags=('stock market', 'agents', 'order book', 'imitation', 'fundamentalists', 'chartists'),
    references=('Caiani et al. 2016 ch. 3', 'Lux and Marchesi 1999'),
    defaults={'until': 200},
    observables=_order_book_observables,
)
def order_book(
    width: int = 20,
    height: int = 20,
    *,
    chartists: float = 0.3,
    fundamental: float = 100.0,
    reversion: float = 0.2,
    extrapolation: float = 1.0,
    noise: float = 0.02,
    imitation: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a stock market of fundamentalists and chartists on a grid who trade in a limit order book and imitate.

    Every step every trader forecasts the price, as a fundamentalist who expects it to revert to the fundamental value
    or as a chartist who expects the last change to go on, and quotes a limit order for one share at its forecast. The
    order book matches the orders with price-time priority, and every trade moves money and a share between the two
    traders. Then traders copy the strategy of a neighbour whose wealth grew more. Fundamentalists hold the price near
    the fundamental value, chartists amplify its swings, and imitation lets the share of chartists wander, so the
    returns come in bursts with fat tails (Caiani et al. 2016 ch. 3).

    Args:
        width:
            The number of columns of the grid, with one trader per cell.

        height:
            The number of rows of the grid.

        chartists:
            The initial share of chartists.

        fundamental:
            The fundamental value of the share.

        reversion:
            The share of the gap to the fundamental value a fundamentalist expects to close.

        extrapolation:
            The share of the last change of the price a chartist expects to repeat.

        noise:
            The standard deviation of the relative noise of a forecast.

        imitation:
            The probability that a trader looks at a neighbour in a step.

        random_state:
            The seed or generator of the initial strategies.

    Returns:
        system:
            The model, with the group `traders` and the book `book`, whose reporters are `book.price` and
            `book.volume`.

    Examples:
        >>> from skcomplex.models.economics.finance import order_book
        >>> order_book(5, 5).elements[0].size
        25
    """
    rng = np.random.default_rng(random_state)
    cells = [(x, y) for x in range(width) for y in range(height)]
    traders = StockTrader.group(
        'traders',
        len(cells),
        location=cells,
        strategy=(rng.random(len(cells)) < chartists).astype(float),
        money=10 * fundamental,
        shares=10.0,
        wealth=20 * fundamental,
        fundamental=fundamental,
        reversion=reversion,
        extrapolation=extrapolation,
        noise=noise,
        imitation=imitation,
    )
    return System(
        components=[traders],
        interactions=[Book('book', traders='traders', opening=fundamental)],
        space=LatticeSpace((width, height), boundary='periodic', neighborhood='von_neumann'),
        time=DiscreteTime(
            schedule=Phases(
                [Phase('quote'), Phase('trade'), Phase('account'), Phase('imitation', activation='simultaneous')],
            ),
        ),
    )
