"""Shop that offers its capacity at its price, searches for the price of the largest profit, and may close."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.engine.solvers import SystemView
from ....core.specs import (
    MatchSpec,
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._store_view import StoreView


class StoreElement(Element):
    """Shop that offers its capacity at its price, searches for the price of the largest profit, and may close.

    The shop searches for its price by trial and error: it moves the price by `adjustment` in the same direction as
    long as its profit grows, and turns back when the profit falls, never below the unit cost. It pays the unit cost
    of every unit sold and the fixed cost every step, and closes when its money runs out.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        price:
            The price of a unit.

        ask:
            The price offered in the current step.

        supply:
            The units still offered in the step.

        sales:
            The units sold in the last step.

        money:
            The money of the shop.

        profit:
            The profit of the last step.

        direction:
            The direction of the last change of the price, 1 up or -1 down.

        capacity:
            The units the shop can sell in a step.

        unit_cost:
            The cost of a unit sold.

        fixed_cost:
            The cost of running the shop for a step.

        adjustment:
            The relative change of the price in a step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import StoreElement
        >>> GroupElement('shops', element=StoreElement, size=4, values={'price': 10.0}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'price'}),
        *,
        price: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        ask: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        supply: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        sales: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        money: Annotated[float, StateSpec()] = 100.0,
        profit: Annotated[float, StateSpec()] = 0.0,
        direction: Annotated[float, StateSpec()] = 1.0,
        capacity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
        unit_cost: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        fixed_cost: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        adjustment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.price = price
        self.ask = ask
        self.supply = supply
        self.sales = sales
        self.money = money
        self.profit = profit
        self.direction = direction
        self.capacity = capacity
        self.unit_cost = unit_cost
        self.fixed_cost = fixed_cost
        self.adjustment = adjustment

    @classmethod
    def _get_view(cls) -> type[StoreView]:
        """Return the view of a shop, with its states and cost parameters."""
        return StoreView

    @staticmethod
    def get_profit(agent: StoreView) -> float:
        """Return the profit of the step, the margin on the units sold less the fixed cost.

        Args:
            agent:
                The shop.

        Returns:
            profit:
                The profit of the step.
        """
        return agent.sales * (agent.ask - agent.unit_cost) - agent.fixed_cost

    @staticmethod
    def get_opening() -> tuple[tuple[str, Callable[[StoreView, SystemView], float]], ...]:
        """Return the states a shop sets when it opens for the market, its capacity offered at its price.

        Returns:
            opening:
                The name of every state and the function of its value, of the shop and the system view.
        """
        return (
            ('supply', lambda agent, system: agent.capacity),
            ('ask', lambda agent, system: agent.price),
            ('sales', lambda agent, system: 0.0),
        )

    @staticmethod
    def get_settlement() -> tuple[tuple[str, Callable[[StoreView, SystemView], float]], ...]:
        """Return the states a shop sets when it settles its accounts, in their order.

        The shop turns its price back when its profit falls, records the profit, and pays its costs. The pricing of
        the system then steps the price by the sign of the direction.

        Returns:
            settlement:
                The name of every state and the function of its value, of the shop and the system view.
        """
        profit = StoreElement.get_profit
        return (
            ('direction', lambda agent, system: -agent.direction if profit(agent) < agent.profit else agent.direction),
            ('profit', lambda agent, system: profit(agent)),
            ('money', lambda agent, system: agent.money - agent.sales * agent.unit_cost - agent.fixed_cost),
        )

    @staticmethod
    def record_sale(match: MatchSpec, system: SystemView) -> None:
        """Count the units of a purchase in the sales of the shop.

        Args:
            match:
                The purchase of a shopper, the requester, from a shop, the provider.

            system:
                The system view of the market.
        """
        shop = match.provider
        if isinstance(shop, StoreView):
            shop.sales += match.quantity
