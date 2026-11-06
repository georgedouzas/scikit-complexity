"""Firms sector, with its deposits, its loans, and the sales of the last period, which pays wages and profits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class FirmsElement(Element):
    """Firms sector, with its deposits, its loans, and the sales of the last period, which pays wages and profits.

    Args:
        label:
            The label of the sector.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        deposits:
            The deposits of the firms, which are money.

        loans:
            The loans of the firms.

        sales:
            The sales of the last period.

        wage_share:
            The share of sales paid as wages.

        distribution:
            The share of the firms' deposits distributed as profits per period.

        repayment:
            The share of loans repaid per period.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import FirmsElement
        >>> FirmsElement('firms').loans
        50.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        deposits: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='money')] = 20.0,
        loans: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        sales: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 40.0,
        wage_share: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.7,
        distribution: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        repayment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
    ) -> None:
        super().__init__(label, visual=visual)
        self.deposits = deposits
        self.loans = loans
        self.sales = sales
        self.wage_share = wage_share
        self.distribution = distribution
        self.repayment = repayment
