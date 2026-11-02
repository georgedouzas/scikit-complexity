"""Buyer of one unit per trading period who bids a random price, below its value under the budget constraint."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....engine.solvers import ModelView
from ....specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._zero_intelligence_buyer_view import ZeroIntelligenceBuyerView


class ZeroIntelligenceBuyerElement(Element):
    """Buyer of one unit per trading period who bids a random price, below its value under the budget constraint.

    In every round of a trading period, a buyer that has not bought yet bids a price drawn uniformly from 0 to its
    value, or to the price ceiling without the constraint, the zero-intelligence traders of Gode and Sunder (1993).

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        value:
            The value of the unit to the buyer, its redemption value.

        bid:
            The current bid.

        demand:
            The number of units still wanted in the trading period.

        profit:
            The accumulated surplus of the buyer's trades.

        ceiling:
            The highest price of the market.

        constrained:
            One when the buyer never bids above its value, zero otherwise.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.markets import ZeroIntelligenceBuyerElement
        >>> GroupElement('buyers', element=ZeroIntelligenceBuyerElement, size=5, values={'value': 100.0}).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        bid: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        demand: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        profit: Annotated[float, StateSpec()] = 0.0,
        ceiling: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 200.0,
        constrained: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value
        self.bid = bid
        self.demand = demand
        self.profit = profit
        self.ceiling = ceiling
        self.constrained = constrained

    @classmethod
    def _get_view(cls) -> type[ZeroIntelligenceBuyerView]:
        """Return the view of a buyer, with its value, bid, demand, and profit."""
        return ZeroIntelligenceBuyerView

    @staticmethod
    def get_bid(agent: ZeroIntelligenceBuyerView, model: ModelView) -> float:
        """Return a bid drawn uniformly from 0 to the value, or to the price ceiling without the constraint.

        Args:
            agent:
                The buyer.

            model:
                The model view of the quotes.

        Returns:
            bid:
                The bid of the round.
        """
        return float(model.rng.uniform(0.0, agent.value if agent.constrained else agent.ceiling))
