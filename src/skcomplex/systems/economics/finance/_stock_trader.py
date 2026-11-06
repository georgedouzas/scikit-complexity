"""Trader who forecasts the price as a fundamentalist or a chartist, quotes a limit order, and imitates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ..common.forecasting import heuristics
from ._stock_trader_view import StockTraderView

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
