"""Loss of immunity, the reaction `R -> S` at the rate `xi R` from recovered back to susceptible."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import ReactionInteraction
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import Stoichiometry
from ._infection import MODES


class ImmunityLossInteraction(ReactionInteraction):
    """Loss of immunity, the reaction `R -> S` at the rate `xi R` from recovered back to susceptible.

    It is a `ReactionInteraction` between the states `n` of the two compartments its roles bind, a rate equation, or
    in stochastic mode a random event that moves one individual.

    Args:
        label:
            The label of the interaction.

        recovered:
            The path of the recovered compartment.

        susceptible:
            The path of the susceptible compartment.

        xi:
            The rate at which immunity wanes.

        stochastic:
            Whether losses of immunity are random events.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import ImmunityLossInteraction
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

    def _get_reactants(self) -> Stoichiometry:
        """Return the states `n` of the compartments the reaction consumes."""
        return {(self.recovered, 'n'): 1}

    def _get_products(self) -> Stoichiometry:
        """Return the states `n` of the compartments the reaction produces."""
        return {(self.susceptible, 'n'): 1}

    def _get_mode(self) -> str:
        """Return the mode of the reaction, random events when stochastic and a rate equation otherwise."""
        return MODES[self.stochastic]

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the mass action of the reaction at the rate constant `xi`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the compartments.

        Returns:
            rate:
                The propensity of the reaction.
        """
        return self._get_mass_action(namespace, namespace.xi)
