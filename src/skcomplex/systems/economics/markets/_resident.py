"""Shopper who wants one unit a step and shops in person, paying for travel, or online, where distance is free."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.engine.solvers import SystemView
from ....core.specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._resident_view import ResidentView
from ._store_view import StoreView


class ResidentElement(Element):
    """Shopper who wants one unit a step and shops in person, paying for travel, or online, where distance is free.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        bid:
            The highest price the shopper pays.

        demand:
            The units still wanted in the step.

        money:
            The money of the shopper.

        travel:
            The travel cost per cell of the current step, zero online.

        distance_cost:
            The travel cost per cell of shopping in person.

        online:
            The probability of shopping online in a step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import ResidentElement
        >>> GroupElement('shoppers', element=ResidentElement, size=3, values={'online': 1.0}).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'travel', 'color_range': (0.0, 1.0)}),
        *,
        bid: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        demand: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        money: Annotated[float, StateSpec()] = 1000.0,
        travel: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        distance_cost: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        online: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.bid = bid
        self.demand = demand
        self.money = money
        self.travel = travel
        self.distance_cost = distance_cost
        self.online = online

    @classmethod
    def _get_view(cls) -> type[ResidentView]:
        """Return the view of a shopper, with its states and shopping parameters."""
        return ResidentView

    @staticmethod
    def get_travel(agent: ResidentView, system: SystemView) -> float:
        """Return the travel cost per cell of the step, zero when the shopper shops online with its probability.

        Args:
            agent:
                The shopper.

            system:
                The system view of the plan.

        Returns:
            travel:
                The travel cost per cell.
        """
        return 0.0 if system.rng.random() < agent.online else agent.distance_cost

    @staticmethod
    def get_preference(shopper: object, shop: object) -> float:
        """Return the preference of a shopper for a shop, the opposite of the delivered price.

        The delivered price is the shop's ask plus the shopper's travel cost per cell times the distance to the shop.

        Args:
            shopper:
                The shopper.

            shop:
                The shop.

        Returns:
            preference:
                The opposite of the delivered price, higher for a preferred shop.
        """
        if not isinstance(shopper, ResidentView) or not isinstance(shop, StoreView):
            return -math.inf
        return -(shop.ask + shopper.travel * shopper.distance_to(shop))
