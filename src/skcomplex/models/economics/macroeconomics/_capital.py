"""Macroeconomics models: physical capital that reinvests its output and depreciates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import RateInteraction
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class CapitalStockElement(Element):
    """Capital stock that reinvests a fraction of its output and depreciates over its lifetime.

    Args:
        label:
            The label of the stock.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        K:
            The capital stock.

        investment_fraction:
            The fraction of the output that is reinvested.

        output_per_capital:
            The output per unit of capital and unit time.

        lifetime:
            The average lifetime of the capital.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import CapitalStockElement
        >>> CapitalStockElement('economy', lifetime=20.0).lifetime
        20.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        K: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        investment_fraction: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        output_per_capital: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1 / 3,
        lifetime: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 15.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.K = K
        self.investment_fraction = investment_fraction
        self.output_per_capital = output_per_capital
        self.lifetime = lifetime


def _net_investment(s: SimpleNamespace) -> Expression:
    """Return the reinvested output minus the depreciation of the capital stock."""
    stock = s.target
    rate: Expression = stock.investment_fraction * stock.output_per_capital * stock.K - stock.K / stock.lifetime
    return rate


@register_model(
    field='economics/macroeconomics',
    tags=('growth', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 50.0, 'dt': 1.0},
)
def capital_stock(
    capital: float = 100.0,
    *,
    investment_fraction: float = 0.2,
    output_per_capital: float = 1 / 3,
    lifetime: float = 15.0,
) -> System:
    """Return Meadows' capital stock, grown by reinvested output and drained by depreciation.

    The stock follows `dK/dt = investment_fraction * output_per_capital * K - K / lifetime`. It grows exponentially when
    the capital lives longer than `1 / (investment_fraction * output_per_capital)`, declines when it lives shorter, and
    stays in dynamic equilibrium otherwise.

    Args:
        capital:
            The initial capital stock.

        investment_fraction:
            The fraction of the output that is reinvested.

        output_per_capital:
            The annual output per unit of capital.

        lifetime:
            The average lifetime of the capital.

    Returns:
        system:
            The model, with the capital stock `economy.K`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import capital_stock
        >>> capital_stock(lifetime=20.0).elements[0].lifetime
        20.0
    """
    economy = CapitalStockElement(
        'economy',
        K=capital,
        investment_fraction=investment_fraction,
        output_per_capital=output_per_capital,
        lifetime=lifetime,
    )
    return System(
        elements=[economy],
        interactions=[RateInteraction('investment', target='economy', state='K', rate=_net_investment)],
    )
