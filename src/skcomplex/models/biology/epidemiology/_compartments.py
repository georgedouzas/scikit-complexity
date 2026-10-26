"""Epidemiology models: compartmental models of infection and recovery."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

from ....core.base import BaseInteraction, BaseLaw, Expression, NumericDomain, parameter, role, state
from ....core.components import Element
from ....core.laws import JumpLaw, RateLaw, Symbols


def _transfer(source: Expression, destination: Expression, rate: Expression, *, stochastic: bool) -> list[BaseLaw]:
    """Return the laws that move individuals from one compartment to another at a rate."""
    if stochastic:
        return [JumpLaw(rate, {source: -1, destination: 1})]
    return [RateLaw(source, -rate), RateLaw(destination, rate)]


class CompartmentElement(Element):
    """Group of individuals in one epidemiological state, such as susceptible, infected, or recovered.

    Args:
        label:
            The label of the compartment.

        n:
            The initial number of individuals.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import CompartmentElement
        >>> CompartmentElement('S', n=990).n
        990
    """

    n: float = state(default=0.0, domain=NumericDomain.NONNEGATIVE)


class InfectionInteraction(BaseInteraction):
    """Mass-action infection, which moves `beta * S * I` individuals per unit time from susceptible to infected.

    In stochastic mode each infection is a random event that moves one individual.

    Args:
        label:
            The label of the interaction.

        stochastic:
            Whether infections are random events.

        **values:
            The values of the declared variables, by name:

            susceptible:
                The path of the susceptible compartment.

            infected:
                The path of the infected compartment.

            beta:
                The transmission rate.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import InfectionInteraction
        >>> InfectionInteraction('infection', susceptible='S', infected='I', beta=0.001).beta
        0.001
    """

    susceptible: str = role()
    infected: str = role()
    beta: float = parameter(default=0.001, domain=NumericDomain.NONNEGATIVE)

    def __init__(
        self,
        label: str,
        *,
        stochastic: bool = False,
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.stochastic = stochastic

    def laws(self, s: Symbols) -> list[BaseLaw]:
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


class RecoveryInteraction(BaseInteraction):
    """Recovery, which moves `gamma * I` individuals per unit time from infected to recovered.

    In stochastic mode each recovery is a random event that moves one individual.

    Args:
        label:
            The label of the interaction.

        stochastic:
            Whether recoveries are random events.

        **values:
            The values of the declared variables, by name:

            infected:
                The path of the infected compartment.

            recovered:
                The path of the recovered compartment.

            gamma:
                The recovery rate.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import RecoveryInteraction
        >>> RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.1).gamma
        0.1
    """

    infected: str = role()
    recovered: str = role()
    gamma: float = parameter(default=0.1, domain=NumericDomain.NONNEGATIVE)

    def __init__(
        self,
        label: str,
        *,
        stochastic: bool = False,
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.stochastic = stochastic

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the transfer of `gamma * I` from infected to recovered.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return _transfer(s.infected.n, s.recovered.n, s.gamma * s.infected.n, stochastic=self.stochastic)


class ImmunityLossInteraction(BaseInteraction):
    """Loss of immunity, which moves `xi * R` individuals per unit time from recovered back to susceptible.

    In stochastic mode each loss is a random event that moves one individual.

    Args:
        label:
            The label of the interaction.

        stochastic:
            Whether losses of immunity are random events.

        **values:
            The values of the declared variables, by name:

            recovered:
                The path of the recovered compartment.

            susceptible:
                The path of the susceptible compartment.

            xi:
                The rate at which immunity wanes.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import ImmunityLossInteraction
        >>> ImmunityLossInteraction('waning', recovered='R', susceptible='S', xi=0.05).xi
        0.05
    """

    recovered: str = role()
    susceptible: str = role()
    xi: float = parameter(default=0.05, domain=NumericDomain.NONNEGATIVE)

    def __init__(
        self,
        label: str,
        *,
        stochastic: bool = False,
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.stochastic = stochastic

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the transfer of `xi * R` from recovered to susceptible.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return _transfer(s.recovered.n, s.susceptible.n, s.xi * s.recovered.n, stochastic=self.stochastic)
