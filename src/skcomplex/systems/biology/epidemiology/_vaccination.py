"""Optimal vaccination of an epidemic.

The vaccination rate balances the burden of an epidemic against the cost of vaccines.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.engine.solvers import PontryaginSolver
from ....core.specs import DeclaredVariableSpec, ObservableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations

MAX_RATE = 0.9


def _get_cost(s: SimpleNamespace) -> Expression:
    """Return the running cost of the epidemic.

    The cost is the infected fraction and the effort of vaccination.
    """
    population = s.population
    cost: Expression = population.a * population.I + population.b * population.u**2 / 2
    return cost


def _get_vaccinated(s: SimpleNamespace) -> Expression:
    """Return the susceptible people vaccinated per unit time, `u S`."""
    population = s.population
    vaccinated: Expression = population.u * population.S
    return vaccinated


class SIRVaccinationSystem(DynamicalSystem):
    """An SIR epidemic with the optimal vaccination rate.

    The rate minimises the burden of infections and vaccines.

    Run with an optimal control solver, vaccination is strongest early, while many are susceptible, and the epidemic
    peaks lower than without vaccination. With no weight `a` on the infections the optimal rate is 0, the epidemic
    without control. The system has the fractions `population.S`, `population.I`, and `population.R`, the vaccination
    rate `population.u`, and the observables `vaccinated` and `cost`, the running cost.

    Args:
        S:
            The initial susceptible fraction.

        I:
            The initial infected fraction.

        beta:
            The transmission rate.

        gamma:
            The recovery rate.

        a:
            The weight of the infected fraction in the cost.

        b:
            The weight of the squared vaccination rate in the cost.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import SIRVaccinationSystem
        >>> SIRVaccinationSystem(beta=0.3).prepare().system_.elements[0].get_value('beta')
        0.3
    """

    field = 'biology/epidemiology'
    topics = ('optimal control', 'vaccination', 'sir', 'pontryagin', 'ode')
    references = ('Lenhart and Workman 2007',)
    until = 30.0
    dt = 0.2
    solver = PontryaginSolver(objective=_get_cost)
    observables = (
        ObservableSpec('vaccinated', _get_vaccinated),
        ObservableSpec('cost', _get_cost),
    )

    def __init__(
        self,
        S: float = 0.99,
        I: float = 0.01,
        *,
        beta: float = 0.5,
        gamma: float = 0.1,
        a: float = 1.0,
        b: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.S = S
        self.I = I
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.b = b

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'population'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'S': DeclaredVariableSpec(self.S, StateSpec(domain=RealSet.closed_open(0, oo))),
            'I': DeclaredVariableSpec(self.I, StateSpec(domain=RealSet.closed_open(0, oo))),
            'R': DeclaredVariableSpec(max(1 - self.S - self.I, 0.0), StateSpec(domain=RealSet.closed_open(0, oo))),
            'u': DeclaredVariableSpec(0.0, ParameterSpec(adaptive=True, bounds=(0.0, MAX_RATE))),
            'beta': DeclaredVariableSpec(self.beta, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'gamma': DeclaredVariableSpec(self.gamma, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'a': DeclaredVariableSpec(self.a, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'b': DeclaredVariableSpec(self.b, ParameterSpec(domain=RealSet.open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the flows between the fractions."""
        return {
            'S': lambda s: -self._infection(s.target) - s.target.u * s.target.S,
            'I': lambda s: self._infection(s.target) - s.target.gamma * s.target.I,
            'R': lambda s: s.target.gamma * s.target.I + s.target.u * s.target.S,
        }

    @staticmethod
    def _infection(population: SimpleNamespace) -> Expression:
        """Return the new infections per unit time, `beta S I`."""
        return population.beta * population.S * population.I
