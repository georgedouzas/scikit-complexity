"""Optimal vaccination: the vaccination rate that balances the burden of an epidemic against the cost of vaccines."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import BaseLaw, Expression, NumericDomain, parameter, state
from ....core.components import Element
from ....core.laws import RateLaw, Symbols
from ....core.system import Observable, System
from ....engine.runtime import register_model
from ....formalisms.control import PontryaginSolver

MAX_RATE = 0.9


def _vaccinated(s: Symbols) -> Expression:
    """Return the susceptible people vaccinated per unit time, `u S`."""
    population = s.population
    vaccinated: Expression = population.u * population.S
    return vaccinated


def _cost(s: Symbols) -> Expression:
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
        >>> from skcomplex.models.biology.epidemiology import VaccinatedPopulationElement
        >>> VaccinatedPopulationElement('population', I=0.01).I
        0.01
    """

    S: float = state(default=0.99, domain=NumericDomain.NONNEGATIVE)
    I: float = state(default=0.01, domain=NumericDomain.NONNEGATIVE)
    R: float = state(default=0.0, domain=NumericDomain.NONNEGATIVE)
    u: float = parameter(default=0.0, adaptive=True, bounds=(0.0, MAX_RATE))
    beta: float = parameter(default=0.5, domain=NumericDomain.NONNEGATIVE)
    gamma: float = parameter(default=0.1, domain=NumericDomain.NONNEGATIVE)
    a: float = parameter(default=1.0, domain=NumericDomain.NONNEGATIVE)
    b: float = parameter(default=1.0, domain=NumericDomain.POSITIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
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


@register_model(
    field='biology/epidemiology',
    tags=('optimal control', 'vaccination', 'sir', 'pontryagin', 'ode'),
    references=('Lenhart and Workman 2007',),
    defaults={'until': 30.0, 'dt': 0.2, 'solver': PontryaginSolver(objective=_cost)},
    observables=lambda system: [Observable('vaccinated', _vaccinated), Observable('cost', _cost)],
)
def sir_vaccination(
    S: float = 0.99,
    I: float = 0.01,
    *,
    beta: float = 0.5,
    gamma: float = 0.1,
    a: float = 1.0,
    b: float = 1.0,
) -> System:
    """Return an SIR epidemic with the vaccination rate that minimises the burden of infections and vaccines.

    Run with an optimal control solver, vaccination is strongest early, while many are susceptible, and the epidemic
    peaks lower than without vaccination. With no weight `a` on the infections the optimal rate is 0, the epidemic
    without control.

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

    Returns:
        system:
            The model, with the fractions `population.S`, `population.I`, and `population.R`, the vaccination rate
            `population.u`, and the observables `vaccinated` and `cost`, the running cost.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import sir_vaccination
        >>> sir_vaccination(beta=0.3).components[0].beta
        0.3
    """
    population = VaccinatedPopulationElement(
        'population',
        S=S,
        I=I,
        R=max(1 - S - I, 0.0),
        beta=beta,
        gamma=gamma,
        a=a,
        b=b,
    )
    return System(components=[population])
