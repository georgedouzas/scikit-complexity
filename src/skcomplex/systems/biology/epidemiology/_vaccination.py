"""Optimal vaccination: the vaccination rate that balances the burden of an epidemic against the cost of vaccines."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import PontryaginSolver
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec

MAX_RATE = 0.9


def _vaccinated(s: SimpleNamespace) -> Expression:
    """Return the susceptible people vaccinated per unit time, `u S`."""
    population = s.population
    vaccinated: Expression = population.u * population.S
    return vaccinated


def _cost(s: SimpleNamespace) -> Expression:
    """Return the running cost of the epidemic, the infected fraction and the effort of vaccination."""
    population = s.population
    cost: Expression = population.a * population.I + population.b * population.u**2 / 2
    return cost


class VaccinatedPopulationElement(Element):
    """Susceptible, infected, and recovered fractions of a population vaccinated at a bounded rate.

    The fractions follow `S' = -beta S I - u S`, `I' = beta S I - gamma I`, and `R' = gamma I + u S`, where the
    vaccination rate `u` is a control input between 0 and its largest value. The planner minimises the burden of the
    infections and the cost of the vaccines, `∫ (a I + b u^2 / 2) dt`, after Lenhart and Workman.

    Args:
        label:
            The label of the population.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        S:
            The initial susceptible fraction.

        I:
            The initial infected fraction.

        R:
            The initial recovered fraction.

        u:
            The vaccination rate, the control input.

        beta:
            The transmission rate.

        gamma:
            The recovery rate.

        a:
            The weight of the infected fraction in the cost.

        b:
            The weight of the squared vaccination rate in the cost.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import VaccinatedPopulationElement
        >>> VaccinatedPopulationElement('population', I=0.01).I
        0.01
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        S: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.99,
        I: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        R: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u: Annotated[float, ParameterSpec(adaptive=True, bounds=(0.0, MAX_RATE))] = 0.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.S = S
        self.I = I
        self.R = R
        self.u = u
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.b = b

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the flows between the fractions.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        infection, recovery, vaccination = s.beta * s.S * s.I, s.gamma * s.I, s.u * s.S
        return [
            RateLaw(s.S, -infection - vaccination),
            RateLaw(s.I, infection - recovery),
            RateLaw(s.R, recovery + vaccination),
        ]


class SIRVaccinationSystem(BaseSystem):
    """An SIR epidemic with the vaccination rate that minimises the burden of infections and vaccines.

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
        >>> SIRVaccinationSystem(beta=0.3).prepare().system_.elements[0].beta
        0.3
    """

    field = 'biology/epidemiology'
    topics = ('optimal control', 'vaccination', 'sir', 'pontryagin', 'ode')
    references = ('Lenhart and Workman 2007',)
    until = 30.0
    dt = 0.2
    solver = PontryaginSolver(objective=_cost)
    observables = (
        ObservableSpec('vaccinated', _vaccinated),
        ObservableSpec('cost', _cost),
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
        super().__init__(label)
        self.S = S
        self.I = I
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.b = b

    def _build_system(self) -> System:
        """Return the parts of an SIR epidemic with the vaccination rate."""
        population = VaccinatedPopulationElement(
            'population',
            S=self.S,
            I=self.I,
            R=max(1 - self.S - self.I, 0.0),
            beta=self.beta,
            gamma=self.gamma,
            a=self.a,
            b=self.b,
        )
        return System(self.label, elements=[population])
