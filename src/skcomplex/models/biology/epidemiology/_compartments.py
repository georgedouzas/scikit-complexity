"""Epidemiology models: compartmental models of infection and recovery."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, JumpLaw, RateLaw
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


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

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n


class InfectionInteraction(Interaction):
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

    def __init__(
        self,
        label: str,
        *,
        susceptible: Annotated[str, RoleSpec()],
        infected: Annotated[str, RoleSpec()],
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        stochastic: bool = False,
    ) -> None:
        super().__init__(label)
        self.susceptible = susceptible
        self.infected = infected
        self.beta = beta
        self.stochastic = stochastic

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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


class RecoveryInteraction(Interaction):
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

    def __init__(
        self,
        label: str,
        *,
        infected: Annotated[str, RoleSpec()],
        recovered: Annotated[str, RoleSpec()],
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        stochastic: bool = False,
    ) -> None:
        super().__init__(label)
        self.infected = infected
        self.recovered = recovered
        self.gamma = gamma
        self.stochastic = stochastic

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the transfer of `gamma * I` from infected to recovered.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return _transfer(s.infected.n, s.recovered.n, s.gamma * s.infected.n, stochastic=self.stochastic)


class ImmunityLossInteraction(Interaction):
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

    def __init__(
        self,
        label: str,
        *,
        recovered: Annotated[str, RoleSpec()],
        susceptible: Annotated[str, RoleSpec()],
        xi: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        stochastic: bool = False,
    ) -> None:
        super().__init__(label)
        self.recovered = recovered
        self.susceptible = susceptible
        self.xi = xi
        self.stochastic = stochastic

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the transfer of `xi * R` from recovered to susceptible.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return _transfer(s.recovered.n, s.susceptible.n, s.xi * s.recovered.n, stochastic=self.stochastic)
