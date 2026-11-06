"""Households sector, whose deposits are money, and which consumes out of its wages and its deposits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class HouseholdsElement(Element):
    """Households sector, whose deposits are money, and which consumes out of its wages and its deposits.

    Args:
        label:
            The label of the sector.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        deposits:
            The deposits of the households.

        alpha_income:
            The propensity to consume out of wages.

        alpha_wealth:
            The propensity to consume out of deposits.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import HouseholdsElement
        >>> HouseholdsElement('households').deposits
        60.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        deposits: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='money')] = 60.0,
        alpha_income: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        alpha_wealth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
    ) -> None:
        super().__init__(label, visual=visual)
        self.deposits = deposits
        self.alpha_income = alpha_income
        self.alpha_wealth = alpha_wealth
