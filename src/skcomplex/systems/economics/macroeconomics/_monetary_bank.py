"""Bank of a stock-flow consistent economy, whose own funds are money, and which lends and charges interest."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class MonetaryBankElement(Element):
    """Bank of a stock-flow consistent economy, whose own funds are money, and which lends and charges interest.

    Args:
        label:
            The label of the bank.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        deposits:
            The own funds of the bank.

        interest:
            The interest rate on loans per period.

        payout:
            The share of the bank's funds paid out as dividends per period.

        credit:
            The new loans per unit of sales.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import MonetaryBankElement
        >>> MonetaryBankElement('bank').deposits
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        deposits: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='money')] = 10.0,
        interest: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        payout: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        credit: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.deposits = deposits
        self.interest = interest
        self.payout = payout
        self.credit = credit
