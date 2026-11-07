"""Firm of the Mark-0 economy, with its workforce, demand, price, wage, cash balance, and profit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw
from ....core.specs import StateSpec, VisualSpec
from ._mark0_firm_view import Mark0FirmView


class Mark0FirmElement(Element):
    """Firm of the Mark-0 economy, with its workforce, demand, price, wage, cash balance, and profit.

    The production is the productivity times the workforce, a negative cash balance is a debt, and the economy of the
    system updates every firm at once.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        workforce:
            The workers of the firm, as a share of the households.

        demand:
            The demand for the goods of the firm.

        price:
            The price of the goods of the firm.

        wage:
            The wage the firm pays.

        cash:
            The cash balance, negative for a debt.

        profit:
            The profit of the last step.

        active:
            1 for a firm in business, 0 for a bankrupt firm waiting to be revived.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.macroeconomics import Mark0FirmElement
        >>> GroupElement('firms', element=Mark0FirmElement, size=3).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        workforce: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        demand: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        price: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        wage: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        cash: Annotated[float, StateSpec()] = 0.0,
        profit: Annotated[float, StateSpec()] = 0.0,
        active: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.workforce = workforce
        self.demand = demand
        self.price = price
        self.wage = wage
        self.cash = cash
        self.profit = profit
        self.active = active

    @classmethod
    def _get_view(cls) -> type[Mark0FirmView]:
        """Return the view of the agents of the element."""
        return Mark0FirmView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the economy updates every firm at once.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []
