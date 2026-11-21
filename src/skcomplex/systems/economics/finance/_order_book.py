"""Finance systems: a stock market of fundamentalists and chartists.

They trade in a limit order book and imitate.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, Protocol, cast

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import BaseSolver, MesaSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.agents import ImitationInteraction
from ...general.matching import ContinuousMatchingSpec, MatchingInteraction, MatchSpec, RandomSearchSpec
from ...general.matching.types import MatchingRule
from ..common.forecasting import heuristics

LOWEST_PRICE = 0.01
HOLDINGS = ('money', 'shares')


class StockTraderView(AgentViewProtocol, Protocol):
    """Stock trader as the rules see it.

    It holds its strategy, holdings, order, wealth, gain, and the parameters of its forecast. The order is the limit
    prices `bid` and `ask` and the quantities `demand` and `supply`.
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
    reversion: float
    extrapolation: float
    noise: float


class StockTraderElement(Element):
    """Trader who forecasts the price, quotes a limit order, and imitates.

    It forecasts as a fundamentalist or a chartist. A fundamentalist expects the price `p` to revert to the fundamental
    value `F`, `p + reversion (F - p)`, and a chartist expects the last change to go on, `p + extrapolation (p - p')`,
    both with a relative noise. The trader bids its expectation for one share when it is above the price and it can pay,
    and asks its expectation for one share when it is below and it holds one. After trading, an `ImitationInteraction`
    of rule `'better'` lets a trader whose wealth grew less than that of a random neighbour copy the neighbour's
    strategy.

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

        values:
            Further values by name, such as the cell `x` and `y` of every member.

        size:
            The number of members, or `None` for one.

        space:
            The grid whose axes `x` and `y` place the members, or `None` for none.

    Examples:
        >>> from skcomplex.systems.economics.finance import StockTraderElement
        >>> StockTraderElement('traders', strategy=1.0, size=4).size
        4
    """

    visual = VisualSpec('agents', {'color_by': 'strategy', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        strategy: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        money: object = 1000.0,
        shares: object = 10.0,
        bid: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        ask: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        demand: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        supply: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        wealth: object = 0.0,
        gain: object = 0.0,
        fundamental: Annotated[object, RealSet.open(0, oo)] = 100.0,
        reversion: Annotated[object, RealSet.closed_open(0, oo)] = 0.2,
        extrapolation: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        noise: Annotated[object, RealSet.closed_open(0, oo)] = 0.02,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
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


class LimitOrderInteraction(Interaction):
    """Limit orders of the traders of the order book.

    Every trader forecasts the price and quotes a limit order for one share. A fundamentalist expects the price to close
    the share `reversion` of its gap to the fundamental value, and a chartist expects the last change of the price to
    repeat by the share `extrapolation`, both with a relative normal noise of standard deviation `noise`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of traders.

        phase:
            The phase the traders quote in.

    Examples:
        >>> from skcomplex.systems.economics.finance import LimitOrderInteraction
        >>> LimitOrderInteraction('quotes', target='traders').phase
        'quote'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role, phase: str | None = 'quote') -> None:
        super().__init__(label)
        self.target = target
        self.phase = phase

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the quote of every trader as its rule.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self.quote, phase=self.phase)]

    @staticmethod
    def get_prices(agent: StockTraderView, system: SystemView) -> tuple[float, float]:
        """Return the last two prices of the book, the fundamental value before the first.

        Args:
            agent:
                The trader.

            system:
                The system view of the rule.

        Returns:
            prices:
                The last price and the one before it.
        """
        history = system.get_history('book.price', 2)
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
        price, previous = self.get_prices(agent, system)
        if agent.strategy:
            expected = heuristics.trend([previous, price], strength=agent.extrapolation)
        else:
            reversion = 1 - agent.reversion
            expected = heuristics.fundamental(
                [price],
                fundamental=agent.fundamental,
                reversion=reversion,
            )
        expected = max(LOWEST_PRICE, expected * (1 + agent.noise * system.rng.normal()))
        buying = expected > price
        new_bid, new_ask = (expected, 0.0) if buying else (0.0, expected)
        agent.bid = new_bid
        agent.ask = new_ask
        agent.demand = float(buying and agent.money >= expected)
        agent.supply = float(not buying and agent.shares >= 1)


class AccountingInteraction(Interaction):
    """Accounts of the traders of the order book after every step of trade.

    Every trader values its money and shares at the last price, and its gain is the change of this wealth in the step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of traders.

        phase:
            The phase the accounts run in.

    Examples:
        >>> from skcomplex.systems.economics.finance import AccountingInteraction
        >>> AccountingInteraction('accounts', target='traders').phase
        'account'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role, phase: str | None = 'account') -> None:
        super().__init__(label)
        self.target = target
        self.phase = phase

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the accounts of every step as the rule of the system.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseComponent] = [UpdateLaw(context, self.account, phase=self.phase)]

    def account(self, system: SystemView) -> None:
        """Set the gain of every trader, and then its wealth at the last price.

        Args:
            system:
                The system view of the accounts.
        """
        traders = cast(list[StockTraderView], system.get_agents('target'))
        for trader in traders:
            trader.gain = self.get_gain(trader, system)
        for trader in traders:
            trader.wealth = self.get_wealth(trader, system)

    @staticmethod
    def get_gain(agent: StockTraderView, system: SystemView) -> float:
        """Return the change of the wealth valued at the last price.

        It is zero before the first valuation.

        Args:
            agent:
                The trader.

            system:
                The system view of the accounting.

        Returns:
            gain:
                The change of the wealth in the step.
        """
        return AccountingInteraction.get_wealth(agent, system) - agent.wealth if agent.wealth else 0.0

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
        price, _ = LimitOrderInteraction.get_prices(agent, system)
        return agent.money + agent.shares * price


class BookInteraction(MatchingInteraction[SystemView]):
    """Limit order book that matches the orders of the traders.

    It reports the last traded price. The book is a matching of the traders with themselves, which binds the requesters'
    limit to their `bid` and the providers' to their `ask`, and every trade is two transfers, money from buyer to seller
    and shares from seller to buyer, so the total money and the total shares stay the same, which the default run checks
    as invariants. The book reports the last traded `price`, the previous price when nothing trades, and the traded
    `volume`.

    Args:
        label:
            The label of the book.

        requesters:
            The path of the group of buying traders.

        providers:
            The path of the group of selling traders.

        rule:
            The spec of the algorithm that matches the orders, such as `ContinuousMatchingSpec()`.

        reserve:
            The states with the limit prices of the buyers and of the sellers.

        quantity:
            The states with the quantities the buyers want and the sellers offer.

        cap:
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
        requesters: Role,
        providers: Role,
        rule: MatchingRule = RandomSearchSpec(),
        reserve: str | tuple[str, str] = ('bid', 'ask'),
        quantity: str | tuple[str, str] = ('demand', 'supply'),
        cap: str = 'money',
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
            reserve=reserve,
            quantity=quantity,
            cap=cap,
            transfers=transfers,
            phase=phase,
            on_match=on_match,
        )
        self.opening = opening

    def _report_matches(self, matches: list[MatchSpec], unmatched: float, system: SystemView) -> None:
        """Report the last traded price and the traded volume.

        When nothing trades, it reports the previous price.

        Args:
            matches:
                The trades of the clearing.

            unmatched:
                The quantity of both sides left after the clearing, which the book does not report.

            system:
                The system view of the book.
        """
        del unmatched
        last = system.get_history('price', 1)
        price = matches[-1].value if matches else (float(last[-1]) if len(last) else self.opening)
        system.report('price', price)
        system.report('volume', sum(match.quantity for match in matches))


class OrderBookSystem(BaseSystem):
    """A stock market of fundamentalists and chartists on a grid.

    They trade in a limit order book and imitate. Every step every trader forecasts the price, as a fundamentalist who
    expects it to revert to the fundamental value or as a chartist who expects the last change to go on, and quotes a
    limit order for one share at its forecast. The order book matches the orders with price-time priority, and every
    trade moves money and a share between the two traders. Then traders copy the strategy of a neighbour whose wealth
    grew more. Fundamentalists hold the price near the fundamental value, chartists amplify its swings, and imitation
    lets the share of chartists wander, so the returns come in bursts with fat tails.

    The two strategies are the fundamentalists and chartists of Lux and Marchesi (1999), and imitating a wealthier
    neighbour on the grid stands in for their switching between strategies. The order book is the order-driven market of
    Caiani et al. (2016, section 3.2.1). The traders of Caiani's chapter 3, noise traders who follow gurus of an
    evolving network, are a different model. The system has the group `traders` and the book `book`, whose reporters are
    `book.price` and `book.volume`, and the imitation `imitation` of the strategies of more successful neighbours.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.finance import OrderBookSystem
        >>> OrderBookSystem(5, 5).prepare().system_.elements[0].size
        25
    """

    field = 'economics/finance'
    topics = ('stock market', 'agents', 'order book', 'imitation', 'fundamentalists', 'chartists')
    references = ('Lux and Marchesi 1999', 'Caiani et al. 2016 §3.2.1')
    until = 200
    invariants: ClassVar[Mapping[str, float]] = {f'total_{name}': 1e-6 for name in HOLDINGS}
    observables = (
        Observable('chartists', group='traders', variable='strategy', reduce='mean'),
        Observable('mean_wealth', group='traders', variable='wealth', reduce='mean'),
        *(Observable(f'total_{name}', group='traders', variable=name, reduce='sum') for name in HOLDINGS),
    )

    def __init__(
        self,
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
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.width = width
        self.height = height
        self.chartists = chartists
        self.fundamental = fundamental
        self.reversion = reversion
        self.extrapolation = extrapolation
        self.noise = noise
        self.imitation = imitation
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        space = LatticeSpace((self.width, self.height), boundary='periodic', metric='manhattan')
        x, y = np.array(cells).T
        traders = StockTraderElement(
            'traders',
            strategy=(rng.random(len(cells)) < self.chartists).astype(float),
            money=10 * self.fundamental,
            shares=10.0,
            wealth=20 * self.fundamental,
            fundamental=self.fundamental,
            reversion=self.reversion,
            extrapolation=self.extrapolation,
            noise=self.noise,
            values={'x': x, 'y': y},
            size=len(cells),
            space=space,
        )
        self.system_ = System(
            self.label,
            elements=[traders],
            interactions=[
                LimitOrderInteraction('quotes', target='traders'),
                BookInteraction(
                    'book',
                    requesters='traders',
                    providers='traders',
                    rule=ContinuousMatchingSpec(),
                    opening=self.fundamental,
                ),
                AccountingInteraction('accounts', target='traders'),
                ImitationInteraction(
                    'imitation',
                    target='traders',
                    state='strategy',
                    payoff='gain',
                    rule='better',
                    schedule='synchronous',
                    probability=self.imitation,
                    phase='imitation',
                ),
            ],
            space=space,
            solver=MesaSolver(
                phases=[
                    PhaseSpec('quote'),
                    PhaseSpec('trade'),
                    PhaseSpec('account'),
                    PhaseSpec('imitation', activation='simultaneous'),
                ],
            ),
        )
