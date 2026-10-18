"""Model epidemics as compartments that exchange individuals."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...core import Domain, Expression, parameter, role, state
from ...model import Element, Interaction, Jump, Law, Rate, Symbols


def _transfer(source: Expression, destination: Expression, rate: Expression, *, stochastic: bool) -> list[Law]:
    """Return the laws that move individuals from one compartment to another at a rate."""
    if stochastic:
        return [Jump(rate, {source: -1, destination: 1})]
    return [Rate(source, -rate), Rate(destination, rate)]


class Compartment(Element):
    """Group of individuals in one epidemiological state, such as susceptible, infected, or recovered.

    Args:
        label:
            The label of the compartment.

        n:
            The initial number of individuals.

    Examples:
        >>> from skcomplex.models.epidemiology import Compartment
        >>> Compartment('S', n=990).n
        990
    """

    n: float = state(default=0.0, domain=Domain.NONNEGATIVE)


class Infection(Interaction):
    """Mass-action infection, which moves `beta * S * I` individuals per unit time from susceptible to infected.

    In stochastic mode each infection is a random event that moves one individual.

    Args:
        label:
            The label of the interaction.

        susceptible:
            The path of the susceptible compartment.

        infected:
            The path of the infected compartment.

        beta:
            The transmission rate.

        stochastic:
            Whether infections are random events.

    Examples:
        >>> from skcomplex.models.epidemiology import Infection
        >>> Infection('infection', susceptible='S', infected='I', beta=0.001).beta
        0.001
    """

    susceptible: str = role()
    infected: str = role()
    beta: float = parameter(default=0.001, domain=Domain.NONNEGATIVE)
    stochastic: bool = False

    def laws(self, s: Symbols) -> list[Law]:
        """Return the transfer of `beta * S * I` from susceptible to infected.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        rate = s.beta * s.susceptible.n * s.infected.n
        return _transfer(s.susceptible.n, s.infected.n, rate, stochastic=self.stochastic)


class Recovery(Interaction):
    """Recovery, which moves `gamma * I` individuals per unit time from infected to recovered.

    In stochastic mode each recovery is a random event that moves one individual.

    Args:
        label:
            The label of the interaction.

        infected:
            The path of the infected compartment.

        recovered:
            The path of the recovered compartment.

        gamma:
            The recovery rate.

        stochastic:
            Whether recoveries are random events.

    Examples:
        >>> from skcomplex.models.epidemiology import Recovery
        >>> Recovery('recovery', infected='I', recovered='R', gamma=0.1).gamma
        0.1
    """

    infected: str = role()
    recovered: str = role()
    gamma: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    stochastic: bool = False

    def laws(self, s: Symbols) -> list[Law]:
        """Return the transfer of `gamma * I` from infected to recovered.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return _transfer(s.infected.n, s.recovered.n, s.gamma * s.infected.n, stochastic=self.stochastic)
