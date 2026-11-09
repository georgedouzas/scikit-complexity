"""Optimal growth: the consumption that maximises discounted utility.

The economy accumulates capital.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, exp, log, oo
from sage.sets.real_set import RealSet

from ....core.engine.solvers import CollocationSolver
from ....core.specs import DeclaredVariableSpec, ObservableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations

MIN_CONSUMPTION = 0.05
MAX_CONSUMPTION = 5.0


def _saving_rate(s: SimpleNamespace) -> Expression:
    """Return the share of output saved, `1 - c / (A k^alpha)`."""
    economy = s.economy
    rate: Expression = 1 - economy.c / (economy.A * economy.k**economy.alpha)
    return rate


def _utility(s: SimpleNamespace) -> Expression:
    """Return the discounted utility of consumption per unit time.

    It is `exp(-rho t) log(c)`.
    """
    economy = s.economy
    utility: Expression = exp(-economy.rho * s.t) * log(economy.c)
    return utility


def _capital_value(s: SimpleNamespace) -> Expression:
    """Return the discounted value of the capital left at the horizon.

    It is `exp(-rho T) v k(T)`.
    """
    economy = s.economy
    value: Expression = exp(-economy.rho * s.t) * economy.v * economy.k
    return value


class OptimalConsumptionSystem(DynamicalSystem):
    """The Ramsey economy that chooses its consumption path.

    The path maximises its discounted utility. Run with an optimal control solver from little capital, consumption rises
    with capital along the saddle path towards the modified golden rule, and the economy saves a falling share of its
    output. The system has the capital `economy.k`, the consumption `economy.c`, and the observable `saving_rate`.

    Args:
        k:
            The initial capital per worker.

        A:
            The total factor productivity.

        alpha:
            The capital share of output.

        delta:
            The depreciation rate.

        rho:
            The discount rate.

        v:
            The value per unit of capital left at the horizon.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import OptimalConsumptionSystem
        >>> OptimalConsumptionSystem(rho=0.05).prepare().system_.elements[0].get_value('rho')
        0.05
    """

    field = 'economics/macroeconomics'
    topics = ('optimal control', 'growth', 'capital', 'collocation', 'ode')
    references = ('Dilão 2023',)
    until = 40.0
    dt = 0.5
    solver = CollocationSolver(objective=_utility, terminal=_capital_value, sense='max', nodes=81)
    observables = (ObservableSpec('saving_rate', _saving_rate),)

    def __init__(
        self,
        k: float = 1.0,
        *,
        A: float = 1.0,
        alpha: float = 0.3,
        delta: float = 0.05,
        rho: float = 0.03,
        v: float = 0.7,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.k = k
        self.A = A
        self.alpha = alpha
        self.delta = delta
        self.rho = rho
        self.v = v

    @staticmethod
    def _net_saving(s: SimpleNamespace) -> Expression:
        """Return the output not consumed minus depreciation, `A k^alpha - delta k - c`."""
        economy = s.target
        rate: Expression = economy.A * economy.k**economy.alpha - economy.delta * economy.k - economy.c
        return rate

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'economy'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'k': DeclaredVariableSpec(self.k, StateSpec(domain=RealSet.open(0, oo))),
            'c': DeclaredVariableSpec(0.5, ParameterSpec(adaptive=True, bounds=(MIN_CONSUMPTION, MAX_CONSUMPTION))),
            'A': DeclaredVariableSpec(self.A, ParameterSpec(domain=RealSet.open(0, oo))),
            'alpha': DeclaredVariableSpec(self.alpha, ParameterSpec(domain=RealSet.open(0, oo))),
            'delta': DeclaredVariableSpec(self.delta, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'rho': DeclaredVariableSpec(self.rho, ParameterSpec(domain=RealSet.open(0, oo))),
            'v': DeclaredVariableSpec(self.v, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the parts of the system built from its model parameters."""
        return {
            'k': self._net_saving,
        }
