"""Macroeconomics systems: physical capital that reinvests and depreciates.

It reinvests its output.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class CapitalStockElement(Element):
    """Capital stock that reinvests a fraction of its output and depreciates.

    It depreciates over its lifetime.

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
        >>> from skcomplex.systems.economics.macroeconomics import CapitalStockElement
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


class CapitalStockSystem(BaseSystem):
    """Meadows' capital stock, grown by reinvested output and drained by depreciation.

    The stock follows `dK/dt = investment_fraction * output_per_capital * K - K / lifetime`. It grows exponentially when
    the capital lives longer than `1 / (investment_fraction * output_per_capital)`, declines when it lives shorter, and
    stays in dynamic equilibrium otherwise. The system has the capital stock `economy.K`.

    Args:
        capital:
            The initial capital stock.

        investment_fraction:
            The fraction of the output that is reinvested.

        output_per_capital:
            The annual output per unit of capital.

        lifetime:
            The average lifetime of the capital.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import CapitalStockSystem
        >>> CapitalStockSystem(lifetime=20.0).prepare().system_.elements[0].lifetime
        20.0
    """

    field = 'economics/macroeconomics'
    topics = ('growth', 'system dynamics', 'ode')
    references = ('Meadows 2008',)
    until = 50.0
    dt = 1.0

    def __init__(
        self,
        capital: float = 100.0,
        *,
        investment_fraction: float = 0.2,
        output_per_capital: float = 1 / 3,
        lifetime: float = 15.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.capital = capital
        self.investment_fraction = investment_fraction
        self.output_per_capital = output_per_capital
        self.lifetime = lifetime

    @staticmethod
    def _net_investment(s: SimpleNamespace) -> Expression:
        """Return the reinvested output minus the depreciation of the capital stock."""
        stock = s.target
        rate: Expression = stock.investment_fraction * stock.output_per_capital * stock.K - stock.K / stock.lifetime
        return rate

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        economy = CapitalStockElement(
            'economy',
            K=self.capital,
            investment_fraction=self.investment_fraction,
            output_per_capital=self.output_per_capital,
            lifetime=self.lifetime,
        )
        return CompositeSystem(
            self.label,
            elements=[economy],
            interactions=[RateInteraction('investment', target='economy', state='K', rate=self._net_investment)],
        )
