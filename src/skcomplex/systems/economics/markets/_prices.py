"""Markets systems: prices that adjust continuously to excess demand."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class RiccatiPricesSystem(DynamicalSystem):
    """A price that moves with excess demand.

    Demand is linear and supply is quadratic and saturating. The price follows the Riccati equation `dP/dt = k (D - S)`
    with demand `D = beta - lam P` and supply `S = delta + alpha P + gamma P^2`. When
    `(lam + alpha)^2 > 4 gamma (delta - beta)` there are two equilibrium prices, and prices below the larger one
    converge to the smaller one. Otherwise the price blows up in finite time. The system has the price `market.P`.

    Args:
        price:
            The initial price.

        k:
            The speed of the price adjustment.

        beta:
            The demand at price zero.

        lam:
            The slope of the demand.

        delta:
            The supply at price zero.

        alpha:
            The linear intensity of the supply.

        gamma:
            The quadratic intensity of the supply, negative for saturation.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.markets import RiccatiPricesSystem
        >>> RiccatiPricesSystem(gamma=-0.2).prepare().system_.elements[0].get_value('gamma')
        -0.2
    """

    field = 'economics/markets'
    topics = ('market', 'riccati', 'ode')
    references = ('Forrest 2018',)
    until = 10.0
    dt = 0.1

    def __init__(
        self,
        price: float = 1.0,
        *,
        k: float = 1.0,
        beta: float = 10.0,
        lam: float = 1.0,
        delta: float = 1.0,
        alpha: float = 2.0,
        gamma: float = -0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.price = price
        self.k = k
        self.beta = beta
        self.lam = lam
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def _get_excess_demand(s: SimpleNamespace) -> Expression:
        """Return the rate of the price.

        It is `k` times the demand `beta - lam P` less supply `delta + alpha P + g P^2`.
        """
        market = s.target
        demand = market.beta - market.lam * market.P
        supply = market.delta + market.alpha * market.P + market.gamma * market.P**2
        return market.k * (demand - supply)

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'market'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'P': DeclaredVariableSpec(self.price, StateSpec()),
            'k': DeclaredVariableSpec(self.k, ParameterSpec(domain=RealSet.open(0, oo))),
            'beta': DeclaredVariableSpec(self.beta, ParameterSpec(domain=RealSet.open(0, oo))),
            'lam': DeclaredVariableSpec(self.lam, ParameterSpec(domain=RealSet.open(0, oo))),
            'delta': DeclaredVariableSpec(self.delta, ParameterSpec(domain=RealSet.open(0, oo))),
            'alpha': DeclaredVariableSpec(self.alpha, ParameterSpec(domain=RealSet.open(0, oo))),
            'gamma': DeclaredVariableSpec(self.gamma, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return the components of the system built from its model parameters."""
        return {
            'P': self._get_excess_demand,
        }
