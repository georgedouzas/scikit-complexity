"""Mass-action infection, the reaction `S + I -> 2 I` at the rate `beta S I` from susceptible to infected."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.interactions import ReactionInteraction
from ....specs import ParameterSpec, RoleSpec
from ....types import Stoichiometry

MODES = {False: 'rate', True: 'jump'}


class InfectionInteraction(ReactionInteraction):
    """Mass-action infection, the reaction `S + I -> 2 I` at the rate `beta S I` from susceptible to infected.

    It is a `ReactionInteraction` between the states `n` of the two compartments its roles bind, a rate equation, or
    in stochastic mode a random event that moves one individual.

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
        >>> from skcomplex.core.elements import CompartmentElement
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.biology.epidemiology import InfectionInteraction
        >>> system = System(
        ...     elements=[CompartmentElement('S', n=99.0), CompartmentElement('I', n=1.0)],
        ...     interactions=[InfectionInteraction('infection', susceptible='S', infected='I', beta=0.01)],
        ... )
        >>> trajectory = Simulation(system, dt=0.1).check().prepare().run(until=5.0).trajectory_
        >>> round(float(trajectory.values('S.n')[-1] + trajectory.values('I.n')[-1]), 6)
        100.0
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

    def _get_reactants(self) -> Stoichiometry:
        """Return the states `n` of the compartments the reaction consumes."""
        return {(self.susceptible, 'n'): 1, (self.infected, 'n'): 1}

    def _get_products(self) -> Stoichiometry:
        """Return the states `n` of the compartments the reaction produces."""
        return {(self.infected, 'n'): 2}

    def _get_mode(self) -> str:
        """Return the mode of the reaction, random events when stochastic and a rate equation otherwise."""
        return MODES[self.stochastic]

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the mass action of the reaction at the rate constant `beta`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the compartments.

        Returns:
            rate:
                The propensity of the reaction.
        """
        return self._get_mass_action(namespace, namespace.beta)
