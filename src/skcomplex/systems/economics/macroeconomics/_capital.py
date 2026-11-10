"""Macroeconomics systems: physical capital that reinvests and depreciates.

It reinvests its output.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class CapitalStockSystem(DynamicalSystem):
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
        >>> CapitalStockSystem(lifetime=20.0).prepare().system_.elements[0].get_value('lifetime')
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
        super().__init__(label=label)
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

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'economy'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'K': DeclaredVariableSpec(self.capital, StateSpec(domain=RealSet.closed_open(0, oo))),
            'investment_fraction': DeclaredVariableSpec(
                self.investment_fraction,
                ParameterSpec(domain=RealSet.closed_open(0, oo)),
            ),
            'output_per_capital': DeclaredVariableSpec(
                self.output_per_capital,
                ParameterSpec(domain=RealSet.open(0, oo)),
            ),
            'lifetime': DeclaredVariableSpec(self.lifetime, ParameterSpec(domain=RealSet.open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the parts of the system built from its model parameters."""
        return {
            'K': self._net_investment,
        }
