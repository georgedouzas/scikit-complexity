"""Recovery, the reaction `I -> R` at the rate `gamma I` from infected to recovered."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.interactions import ReactionInteraction
from ....specs import ParameterSpec, RoleSpec
from ....types import Stoichiometry
from ._infection import MODES


class RecoveryInteraction(ReactionInteraction):
    """Recovery, the reaction `I -> R` at the rate `gamma I` from infected to recovered.

    It is a `ReactionInteraction` between the states `n` of the two compartments its roles bind, a rate equation, or
    in stochastic mode a random event that moves one individual.

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
        >>> from skcomplex.core.elements import CompartmentElement
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.biology.epidemiology import RecoveryInteraction
        >>> system = System(
        ...     elements=[CompartmentElement('I', n=10.0), CompartmentElement('R')],
        ...     interactions=[RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.5)],
        ... )
        >>> trajectory = Simulation(system, dt=0.01).check().prepare().run(until=2.0).trajectory_
        >>> round(float(trajectory.values('I.n')[-1]), 3)
        3.679
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

    def _get_reactants(self) -> Stoichiometry:
        """Return the states `n` of the compartments the reaction consumes."""
        return {(self.infected, 'n'): 1}

    def _get_products(self) -> Stoichiometry:
        """Return the states `n` of the compartments the reaction produces."""
        return {(self.recovered, 'n'): 1}

    def _get_mode(self) -> str:
        """Return the mode of the reaction, random events when stochastic and a rate equation otherwise."""
        return MODES[self.stochastic]

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the mass action of the reaction at the rate constant `gamma`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the compartments.

        Returns:
            rate:
                The propensity of the reaction.
        """
        return self._get_mass_action(namespace, namespace.gamma)
