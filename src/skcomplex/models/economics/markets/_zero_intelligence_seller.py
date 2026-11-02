"""Seller of one unit per trading period who asks a random price, above its cost under the budget constraint."""

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
from ._zero_intelligence_seller_view import ZeroIntelligenceSellerView


class ZeroIntelligenceSellerElement(Element):
    """Seller of one unit per trading period who asks a random price, above its cost under the budget constraint.

    In every round of a trading period, a seller that has not sold yet asks a price drawn uniformly from its cost, or
    from 0 without the constraint, to the price ceiling.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        value:
            The cost of the unit to the seller.

        ask:
            The current ask.

        supply:
            The number of units still offered in the trading period.

        profit:
            The accumulated surplus of the seller's trades.

        ceiling:
            The highest price of the market.

        constrained:
            One when the seller never asks below its cost, zero otherwise.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.markets import ZeroIntelligenceSellerElement
        >>> GroupElement('sellers', element=ZeroIntelligenceSellerElement, size=5, values={'value': 50.0}).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        ask: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 200.0,
        supply: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        profit: Annotated[float, StateSpec()] = 0.0,
        ceiling: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 200.0,
        constrained: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value
        self.ask = ask
        self.supply = supply
        self.profit = profit
        self.ceiling = ceiling
        self.constrained = constrained

    @classmethod
    def _get_view(cls) -> type[ZeroIntelligenceSellerView]:
        """Return the view of a seller, with its cost, ask, supply, and profit."""
        return ZeroIntelligenceSellerView

    @staticmethod
    def get_ask(agent: ZeroIntelligenceSellerView, model: ModelView) -> float:
        """Return an ask drawn uniformly from the cost, or from 0 without the constraint, to the price ceiling.

        Args:
            agent:
                The seller.

            model:
                The model view of the quotes.

        Returns:
            ask:
                The ask of the round.
        """
        bottom = agent.value if agent.constrained else 0.0
        return float(model.rng.uniform(bottom, max(bottom, agent.ceiling)))
