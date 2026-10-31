"""Epidemiology models: transfers between the compartments of an epidemic, as mass-action reactions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import Interaction, ReactionInteraction
from ....core.laws import BaseLaw
from ....specs import ParameterSpec, RoleSpec

MODES = {False: 'rate', True: 'jump'}


class InfectionInteraction(Interaction):
    """Mass-action infection, the reaction `S + I -> 2 I` at the rate `beta S I` from susceptible to infected.

    Its laws are those of a `ReactionInteraction` between the states `n` of the two compartments, a rate equation, or
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

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the reaction `S + I -> 2 I` at the rate constant `beta`.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the susceptible and infected compartments.

        Returns:
            laws:
                The rate laws, or the jump law, of the reaction.
        """
        reaction = ReactionInteraction(
            self.label,
            reactants={(self.susceptible, 'n'): 1, (self.infected, 'n'): 1},
            products={(self.infected, 'n'): 2},
            mode=MODES[self.stochastic],
        )
        species = SimpleNamespace(
            reactants=(namespace.susceptible, namespace.infected),
            products=(namespace.infected,),
            modifiers=(),
            k=namespace.beta,
        )
        return list(reaction.check().prepare().run(species).laws_)


class RecoveryInteraction(Interaction):
    """Recovery, the reaction `I -> R` at the rate `gamma I` from infected to recovered.

    Its laws are those of a `ReactionInteraction` between the states `n` of the two compartments, a rate equation, or
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

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the reaction `I -> R` at the rate constant `gamma`.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the infected and recovered compartments.

        Returns:
            laws:
                The rate laws, or the jump law, of the reaction.
        """
        reaction = ReactionInteraction(
            self.label,
            reactants={(self.infected, 'n'): 1},
            products={(self.recovered, 'n'): 1},
            mode=MODES[self.stochastic],
        )
        species = SimpleNamespace(
            reactants=(namespace.infected,),
            products=(namespace.recovered,),
            modifiers=(),
            k=namespace.gamma,
        )
        return list(reaction.check().prepare().run(species).laws_)


class ImmunityLossInteraction(Interaction):
    """Loss of immunity, the reaction `R -> S` at the rate `xi R` from recovered back to susceptible.

    Its laws are those of a `ReactionInteraction` between the states `n` of the two compartments, a rate equation, or
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
        >>> from skcomplex.models.biology.epidemiology import ImmunityLossInteraction
        >>> ImmunityLossInteraction('waning', recovered='R', susceptible='S', xi=0.05).check().xi
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

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the reaction `R -> S` at the rate constant `xi`.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the recovered and susceptible compartments.

        Returns:
            laws:
                The rate laws, or the jump law, of the reaction.
        """
        reaction = ReactionInteraction(
            self.label,
            reactants={(self.recovered, 'n'): 1},
            products={(self.susceptible, 'n'): 1},
            mode=MODES[self.stochastic],
        )
        species = SimpleNamespace(
            reactants=(namespace.recovered,),
            products=(namespace.susceptible,),
            modifiers=(),
            k=namespace.xi,
        )
        return list(reaction.check().prepare().run(species).laws_)
