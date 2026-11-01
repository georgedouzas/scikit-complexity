"""Trader with a bundle of two goods and the Cobb-Douglas utility `good1 good2`, who records its rate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....engine.solvers import ModelView
from ....specs import (
    StateSpec,
    VisualSpec,
)
from ._barter_trader_view import BarterTraderView


class BarterTraderElement(Element):
    """Trader with a bundle of two goods and the Cobb-Douglas utility `good1 good2`, who records its rate.

    Its marginal rate of substitution `good2 / good1` is the price of the first good, in units of the second, at
    which it is indifferent to trade. Every step it records the rate of its current bundle.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        good1:
            The holding of the first good.

        good2:
            The holding of the second good.

        mrs:
            The marginal rate of substitution of the bundle.

        links:
            The number of trading partners the trader can search.

        partners:
            The number of partners that can match with the trader in a step, without limit.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.markets import BarterTraderElement
        >>> GroupElement('traders', element=BarterTraderElement, size=4).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        good1: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 10.0,
        good2: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 10.0,
        mrs: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        links: Annotated[float, StateSpec(domain=NN)] = 0.0,
        partners: Annotated[float, StateSpec()] = math.inf,
    ) -> None:
        super().__init__(label, visual=visual)
        self.good1 = good1
        self.good2 = good2
        self.mrs = mrs
        self.links = links
        self.partners = partners

    @classmethod
    def _get_view(cls) -> type[BarterTraderView]:
        """Return the view of a trader, with its holdings, rate, and links."""
        return BarterTraderView

    @staticmethod
    def is_partner(trader: object, other: object) -> bool:
        """Return whether another trader is a trading partner of a trader, a neighbour on the trade network.

        Args:
            trader:
                The trader that searches.

            other:
                Another trader.

        Returns:
            partner:
                Whether the other trader is in a node next to the trader's node.
        """
        if not isinstance(trader, BarterTraderView) or not isinstance(other, BarterTraderView):
            return False
        return other.id != trader.id and other.location in trader.neighbor_cells()

    @staticmethod
    def get_rate_gap(trader: object, other: object) -> float:
        """Return the gap between the logarithms of the rates of substitution of a trader and a trading partner.

        The gap is minus infinity for a trader that is not a partner, so a trader takes a partner whenever it has one.

        Args:
            trader:
                The trader that searches.

            other:
                Another trader.

        Returns:
            gap:
                The absolute difference of the logarithms of the two rates.
        """
        if not BarterTraderElement.is_partner(trader, other):
            return -math.inf
        if not isinstance(trader, BarterTraderView) or not isinstance(other, BarterTraderView):
            return -math.inf
        return abs(math.log(other.good2 / other.good1) - math.log(trader.good2 / trader.good1))

    @staticmethod
    def get_rate(agent: BarterTraderView, model: ModelView) -> float:
        """Return the marginal rate of substitution of the bundle, `good2 / good1`.

        Args:
            agent:
                The trader.

            model:
                The model view of the appraisal.

        Returns:
            mrs:
                The marginal rate of substitution.
        """
        return agent.good2 / agent.good1
