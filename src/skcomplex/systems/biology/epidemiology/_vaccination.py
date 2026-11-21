"""Optimal vaccination of an epidemic.

The vaccination rate balances the burden of an epidemic against the cost of vaccines.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import ObjectiveLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, PontryaginSolver
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

    The rate minimises the burden of infections and vaccines, the objective law
    `ObjectiveLaw(a I + b u^2 / 2, controls=[s.target.u])`, with `u` between 0 and 0.9.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
    observables = (
        Observable('vaccinated', _get_vaccinated),
        Observable('cost', _get_cost),
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.S = S
        self.I = I
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.b = b

    def _get_solver(self) -> BaseSolver:
        """Return the solver the system is tuned to."""
        return PontryaginSolver()

    def _get_interactions(self) -> list[BaseInteraction]:
        """Return the flows between the fractions, and the burden that the vaccination rate minimises."""
        burden = Interaction(
            'burden',
            roles={'target': self._get_element_label()},
            laws=lambda s: [ObjectiveLaw(_get_cost(SimpleNamespace(population=s.target)), controls=[s.target.u])],
        )
        return [*super()._get_interactions(), burden]

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'population'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'S': self.S,
            'I': self.I,
            'R': max(1 - self.S - self.I, 0.0),
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the vaccination rate, which the burden steers, and the parameters of the model."""
        return {'u': 0.0, 'beta': self.beta, 'gamma': self.gamma, 'a': self.a, 'b': self.b}

    def _get_allowed(self) -> dict[str, object]:
        """Return the allowed values of the fractions, the vaccination rate, and the parameters."""
        nonnegative = RealSet.closed_open(0, oo)
        return {
            'S': nonnegative,
            'I': nonnegative,
            'R': nonnegative,
            'u': RealSet.closed(0, MAX_RATE),
            'beta': nonnegative,
            'gamma': nonnegative,
            'a': nonnegative,
            'b': RealSet.open(0, oo),
        }

    def _get_equations(self) -> Equations:
        """Return the flows between the fractions."""
        return {
            'S': lambda s: -self._get_infection(s.target) - s.target.u * s.target.S,
            'I': lambda s: self._get_infection(s.target) - s.target.gamma * s.target.I,
            'R': lambda s: s.target.gamma * s.target.I + s.target.u * s.target.S,
        }

    @staticmethod
    def _get_infection(population: SimpleNamespace) -> Expression:
        """Return the new infections per unit time, `beta S I`."""
        return population.beta * population.S * population.I
