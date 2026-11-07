"""Finance systems: a stock market whose fundamentalists and chartists trade in a limit order book and imitate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import ImitationInteraction, MapInteraction, MatchingInteraction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import System
from ....core.components.types import MatchingRule
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import (
    ContinuousMatchingSpec,
    MatchSpec,
    ObservableSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    RandomSearchSpec,
    RoleSpec,
    StateSpec,
    VisualSpec,
)
from ..._catalog import register_system
from ..common.forecasting import heuristics


class StockTraderView(AgentView):
    """Stock trader as the rules see it, with its strategy, holdings, order, wealth, gain, and fundamental value.

    The order is the limit prices `bid` and `ask` and the quantities `demand` and `supply`.
    """

    strategy: float
    money: float
    shares: float
    bid: float
    ask: float
    demand: float
    supply: float
    wealth: float
    gain: float
    fundamental: float


LOWEST_PRICE = 0.01


class StockTraderElement(Element):
    """Trader who forecasts the price as a fundamentalist or a chartist, quotes a limit order, and imitates.

    A fundamentalist expects the price `p` to revert to the fundamental value `F`, `p + reversion (F - p)`, and a
    chartist expects the last change to go on, `p + extrapolation (p - p')`, both with a relative noise. The trader
    bids its expectation for one share when it is above the price and it can pay, and asks its expectation for one
    share when it is below and it holds one. After trading, an `ImitationInteraction` of rule `'better'` lets a trader
    whose wealth grew less than that of a random neighbour copy the neighbour's strategy.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.finance import StockTraderElement
        >>> GroupElement('traders', element=StockTraderElement, size=4, values={'strategy': 1.0}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'strategy', 'color_range': (0.0, 1.0)}),
        *,
        strategy: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        money: Annotated[float, StateSpec()] = 1000.0,
        shares: Annotated[float, StateSpec()] = 10.0,
        bid: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        ask: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        demand: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        supply: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        wealth: Annotated[float, StateSpec()] = 0.0,
        gain: Annotated[float, StateSpec()] = 0.0,
        fundamental: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        reversion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        extrapolation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        noise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
    ) -> None:
        super().__init__(label, visual=visual)
        self.strategy = strategy
        self.money = money
        self.shares = shares
        self.bid = bid
        self.ask = ask
        self.demand = demand
        self.supply = supply
        self.wealth = wealth
        self.gain = gain
        self.fundamental = fundamental
        self.reversion = reversion
        self.extrapolation = extrapolation
        self.noise = noise

    @classmethod
    def _get_view(cls) -> type[StockTraderView]:
        """Return the view of the agents of the element."""
        return StockTraderView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the quoting rule in its phase.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.quote, phase='quote')]

    @staticmethod
    def _prices(agent: StockTraderView, system: SystemView) -> tuple[float, float]:
        """Return the last two prices of the book, the fundamental value before the first."""
        history = system.history('book.price', 2)
        prices = [value for value in history if math.isfinite(value)] or [agent.fundamental]
        return prices[-1], prices[0]

    def quote(self, agent: StockTraderView, system: SystemView) -> None:
        """Forecast the price and quote a limit order for one share.

        Args:
            agent:
                The trader.

            system:
                The system.
        """
        price, previous = self._prices(agent, system)
        if agent.strategy:
            expected = heuristics.trend([previous, price], strength=system.parameter('extrapolation'))
        else:
            reversion = 1 - system.parameter('reversion')
            expected = heuristics.fundamental([price], fundamental=system.parameter('fundamental'), reversion=reversion)
        expected = max(LOWEST_PRICE, expected * (1 + system.parameter('noise') * system.rng.normal()))
        buying = expected > price
        new_bid, new_ask = (expected, 0.0) if buying else (0.0, expected)
        agent.bid = new_bid
        agent.ask = new_ask
        agent.demand = float(buying and agent.money >= expected)
        agent.supply = float(not buying and agent.shares >= 1)

    @staticmethod
    def get_gain(agent: StockTraderView, system: SystemView) -> float:
        """Return the change of the wealth valued at the last price, zero before the first valuation.

        Args:
            agent:
                The trader.

            system:
                The system view of the accounting.

        Returns:
            gain:
                The change of the wealth in the step.
        """
        return StockTraderElement.get_wealth(agent, system) - agent.wealth if agent.wealth else 0.0

    @staticmethod
    def get_wealth(agent: StockTraderView, system: SystemView) -> float:
        """Return the money plus the shares valued at the last price.

        Args:
            agent:
                The trader.

            system:
                The system view of the accounting.

        Returns:
            wealth:
                The wealth of the trader.
        """
        price, _ = StockTraderElement._prices(agent, system)
        return agent.money + agent.shares * price


class BookInteraction(MatchingInteraction[SystemView]):
    """Limit order book that matches the orders of the traders and reports the last traded price.

    The book is a matching of the traders with themselves, which binds the requesters' limit to their `bid` and the
    providers' to their `ask`, and every trade is two transfers, money from buyer to seller and shares from seller to
    buyer, so the total money and the total shares stay the same, which the default run checks as invariants. The book
    reports the last traded `price`, the previous price when nothing trades, and the traded `volume`.

    Args:
        label:
            The label of the book.

        requesters:
            The path of the group of buying traders.

        providers:
            The path of the group of selling traders.

        rule:
            The spec of the algorithm that matches the orders, such as `ContinuousMatchingSpec()`.

        limit:
            The states with the limit prices of the buyers and of the sellers.

        quantity:
            The states with the quantities the buyers want and the sellers offer.

        allowance:
            The state of a buyer with the money it can spend.

        transfers:
            The states every trade moves, money paid by the buyer and shares delivered to it.

        phase:
            The phase the book clears in.

        on_match:
            A function of every trade and the system view, run after the transfers, or `None`.

        opening:
            The price before the first trade.

    Examples:
        >>> from skcomplex.systems.economics.finance import BookInteraction
        >>> BookInteraction('book', requesters='traders', providers='traders').reports
        ('price', 'volume')
    """

    reports: ClassVar[tuple[str, ...]] = ('price', 'volume')

    def __init__(
        self,
        label: str,
        *,
        requesters: Annotated[str, RoleSpec()],
        providers: Annotated[str, RoleSpec()],
        rule: MatchingRule = RandomSearchSpec(),
        limit: str | tuple[str, str] = ('bid', 'ask'),
        quantity: str | tuple[str, str] = ('demand', 'supply'),
        allowance: str = 'money',
        transfers: Sequence[tuple[str, int]] = (('money', -1), ('shares', 1)),
        phase: str | None = 'trade',
        on_match: Callable[[MatchSpec, SystemView], None] | None = None,
        opening: float = 100.0,
    ) -> None:
        super().__init__(
            label,
            requesters=requesters,
            providers=providers,
            rule=rule,
            limit=limit,
            quantity=quantity,
            allowance=allowance,
            transfers=transfers,
            phase=phase,
            on_match=on_match,
        )
        self.opening = opening

    def _report_matches(self, matches: list[MatchSpec], unmatched: float, system: SystemView) -> None:
        """Report the last traded price, the previous price when nothing trades, and the traded volume.

        Args:
            matches:
                The trades of the clearing.

            unmatched:
                The quantity of both sides left after the clearing, which the book does not report.

            system:
                The system view of the book.
        """
        del unmatched
        last = system.history('price', 1)
        price = matches[-1].value if matches else (float(last[-1]) if len(last) else self.opening)
        system.report('price', price)
        system.report('volume', sum(match.quantity for match in matches))


HOLDINGS = ('money', 'shares')


def _order_book_observables(system: System) -> list[ObservableSpec]:
    """Return the share of chartists, the mean wealth of the traders, and their total money and shares."""
    del system
    return [
        ObservableSpec('chartists', group='traders', variable='strategy', reduce='mean'),
        ObservableSpec('mean_wealth', group='traders', variable='wealth', reduce='mean'),
        *(ObservableSpec(f'total_{name}', group='traders', variable=name, reduce='sum') for name in HOLDINGS),
    ]


@register_system(
    field='economics/finance',
    tags=('stock market', 'agents', 'order book', 'imitation', 'fundamentalists', 'chartists'),
    references=('Lux and Marchesi 1999', 'Caiani et al. 2016 §3.2.1'),
    defaults={'until': 200, 'invariants': {f'total_{name}': 1e-6 for name in HOLDINGS}},
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
    returns come in bursts with fat tails.

    The two strategies are the fundamentalists and chartists of Lux and Marchesi (1999), and imitating a wealthier
    neighbour on the grid stands in for their switching between strategies. The order book is the order-driven market
    of Caiani et al. (2016, section 3.2.1). The traders of Caiani's chapter 3, noise traders who follow gurus of an
    evolving network, are a different model.

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
            The system, with the group `traders` and the book `book`, whose reporters are `book.price` and
            `book.volume`, and the imitation `imitation` of the strategies of more successful neighbours.

    Examples:
        >>> from skcomplex.systems.economics.finance import order_book
        >>> order_book(5, 5).elements[0].prepare().size_
        25
    """
    rng = np.random.default_rng(random_state)
    cells = [(x, y) for x in range(width) for y in range(height)]
    traders = GroupElement(
        'traders',
        element=StockTraderElement,
        size=len(cells),
        values={
            'strategy': (rng.random(len(cells)) < chartists).astype(float),
            'money': 10 * fundamental,
            'shares': 10.0,
            'wealth': 20 * fundamental,
            'fundamental': fundamental,
            'reversion': reversion,
            'extrapolation': extrapolation,
            'noise': noise,
        },
        location=cells,
    )
    accounts = [
        MapInteraction(
            f'{name}_account',
            target='traders',
            state=name,
            function=function,
            form='agent',
            phase='account',
        )
        for name, function in (('gain', StockTraderElement.get_gain), ('wealth', StockTraderElement.get_wealth))
    ]
    return System(
        elements=[traders],
        interactions=[
            BookInteraction(
                'book',
                requesters='traders',
                providers='traders',
                rule=ContinuousMatchingSpec(),
                opening=fundamental,
            ),
            *accounts,
            ImitationInteraction(
                'imitation',
                target='traders',
                state='strategy',
                payoff='gain',
                rule='better',
                form='agent',
                probability=imitation,
                phase='imitation',
            ),
        ],
        space=LatticeSpace((width, height), boundary='periodic', neighborhood='von_neumann'),
        time=DiscreteTime(
            schedule=PhasesScheduleSpec(
                [
                    PhaseSpec('quote'),
                    PhaseSpec('trade'),
                    PhaseSpec('account'),
                    PhaseSpec('imitation', activation='simultaneous'),
                ],
            ),
        ),
    )
