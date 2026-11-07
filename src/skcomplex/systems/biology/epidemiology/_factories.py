"""Epidemiology systems: compartmental systems of infection and recovery."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import CompartmentElement
from ....core.components.interactions import ReactionInteraction
from ....core.components.system import System
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import Stoichiometry
from ..._catalog import register_system

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
        >>> from skcomplex.core.components.elements import CompartmentElement
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.biology.epidemiology import InfectionInteraction
        >>> system = System(
        ...     elements=[CompartmentElement('S', n=99.0), CompartmentElement('I', n=1.0)],
        ...     interactions=[InfectionInteraction('infection', susceptible='S', infected='I', beta=0.01)],
        ... )
        >>> trajectory = Simulation(system, dt=0.1).prepare().run(until=5.0).trajectory_
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
        >>> from skcomplex.core.components.elements import CompartmentElement
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.biology.epidemiology import RecoveryInteraction
        >>> system = System(
        ...     elements=[CompartmentElement('I', n=10.0), CompartmentElement('R')],
        ...     interactions=[RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.5)],
        ... )
        >>> trajectory = Simulation(system, dt=0.01).prepare().run(until=2.0).trajectory_
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


@register_system(
    field='biology/epidemiology',
    tags=('compartments', 'ode', 'jump'),
    references=('Kermack and McKendrick 1927',),
    defaults={'until': 160.0, 'dt': 1.0},
)
def sir(
    susceptible: float = 990.0,
    infected: float = 10.0,
    recovered: float = 0.0,
    *,
    beta: float = 0.0003,
    gamma: float = 0.1,
    stochastic: bool = False,
) -> System:
    """Return an epidemic that moves individuals from susceptible to infected to recovered.

    Args:
        susceptible:
            The initial number of susceptible individuals.

        infected:
            The initial number of infected individuals.

        recovered:
            The initial number of recovered individuals.

        beta:
            The infection rate per susceptible-infected pair.

        gamma:
            The recovery rate.

        stochastic:
            Whether infections and recoveries are random events of single individuals.

    Returns:
        system:
            The system, with the compartments `S`, `I`, and `R`.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import sir
        >>> [element.label for element in sir().elements]
        ['S', 'I', 'R']
    """
    return System(
        elements=[
            CompartmentElement('S', n=susceptible),
            CompartmentElement('I', n=infected),
            CompartmentElement('R', n=recovered),
        ],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=beta, stochastic=stochastic),
            RecoveryInteraction('recovery', infected='I', recovered='R', gamma=gamma, stochastic=stochastic),
        ],
    )


@register_system(
    field='biology/epidemiology',
    tags=('compartments', 'ode', 'jump', 'endemic'),
    references=('Kermack and McKendrick 1932', 'Argun et al. 2021', 'Datseris and Parlitz 2022'),
    defaults={'until': 100.0, 'dt': 1.0},
)
def sis(
    susceptible: float = 990.0,
    infected: float = 10.0,
    *,
    beta: float = 0.0003,
    gamma: float = 0.1,
    stochastic: bool = False,
) -> System:
    """Return an infection without immunity, where the recovered become susceptible again and the disease is endemic.

    Above the threshold `beta N > gamma`, the infected settle at `I = N - gamma / beta`, and below it they die out.

    Args:
        susceptible:
            The initial number of susceptible individuals.

        infected:
            The initial number of infected individuals.

        beta:
            The infection rate per susceptible-infected pair.

        gamma:
            The recovery rate.

        stochastic:
            Whether infections and recoveries are random events of single individuals.

    Returns:
        system:
            The system, with the compartments `S` and `I`.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import sis
        >>> [element.label for element in sis().elements]
        ['S', 'I']
    """
    return System(
        elements=[CompartmentElement('S', n=susceptible), CompartmentElement('I', n=infected)],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=beta, stochastic=stochastic),
            RecoveryInteraction('recovery', infected='I', recovered='S', gamma=gamma, stochastic=stochastic),
        ],
    )


@register_system(
    field='biology/epidemiology',
    tags=('compartments', 'ode', 'jump', 'endemic', 'waning immunity'),
    references=('Kermack and McKendrick 1932', 'Argun et al. 2021'),
    defaults={'until': 300.0, 'dt': 1.0},
)
def sirs(
    susceptible: float = 990.0,
    infected: float = 10.0,
    recovered: float = 0.0,
    *,
    beta: float = 0.0003,
    gamma: float = 0.1,
    xi: float = 0.02,
    stochastic: bool = False,
) -> System:
    """Return an epidemic whose immunity wanes, which returns in damped waves to an endemic equilibrium.

    The equilibrium has `S = gamma / beta` susceptible and `I = xi (N - S) / (gamma + xi)` infected.

    Args:
        susceptible:
            The initial number of susceptible individuals.

        infected:
            The initial number of infected individuals.

        recovered:
            The initial number of recovered individuals.

        beta:
            The infection rate per susceptible-infected pair.

        gamma:
            The recovery rate.

        xi:
            The rate at which immunity wanes.

        stochastic:
            Whether infections, recoveries, and losses of immunity are random events of single individuals.

    Returns:
        system:
            The system, with the compartments `S`, `I`, and `R`.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import sirs
        >>> sirs(xi=0.1).interactions[-1].xi
        0.1
    """
    return System(
        elements=[
            CompartmentElement('S', n=susceptible),
            CompartmentElement('I', n=infected),
            CompartmentElement('R', n=recovered),
        ],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=beta, stochastic=stochastic),
            RecoveryInteraction('recovery', infected='I', recovered='R', gamma=gamma, stochastic=stochastic),
            ImmunityLossInteraction('waning', recovered='R', susceptible='S', xi=xi, stochastic=stochastic),
        ],
    )


@register_system(
    field='biology/epidemiology',
    tags=('compartments', 'ode', 'jump', 'outbreak', 'zombies'),
    references=('Munz et al. 2009', 'Raman 2021'),
    defaults={'until': 10.0, 'dt': 0.05},
)
def zombie_outbreak(
    susceptible: float = 500.0,
    zombies: float = 1.0,
    removed: float = 0.0,
    *,
    birth: float = 0.0,
    beta: float = 0.0095,
    alpha: float = 0.005,
    delta: float = 0.0001,
    zeta: float = 0.0001,
    stochastic: bool = False,
) -> System:
    """Return the outbreak of zombies of Munz et al., where the dead rise again and humans are eventually overrun.

    Humans `S` are born at the rate `birth` and die naturally at the rate `delta`. A zombie `Z` bites a human at the
    rate `beta S Z`, and humans destroy zombies at the rate `alpha S Z`. The dead and destroyed `R` rise as zombies at
    the rate `zeta`. Without births the total `S + Z + R` is conserved, and with resurrection the only stable state
    is the doomsday state without humans, whatever the initial numbers, whenever `beta > alpha`.

    Args:
        susceptible:
            The initial number of humans.

        zombies:
            The initial number of zombies.

        removed:
            The initial number of dead and destroyed.

        birth:
            The birth rate of humans.

        beta:
            The rate at which zombies turn humans.

        alpha:
            The rate at which humans destroy zombies.

        delta:
            The rate of natural death of humans.

        zeta:
            The rate at which the removed rise as zombies.

        stochastic:
            Whether every event moves one individual at random times.

    Returns:
        system:
            The system, with the compartments `S`, `Z`, and `R`.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import zombie_outbreak
        >>> [element.label for element in zombie_outbreak().elements]
        ['S', 'Z', 'R']
    """
    mode = 'jump' if stochastic else 'rate'
    return System(
        elements=[
            CompartmentElement('S', n=susceptible),
            CompartmentElement('Z', n=zombies),
            CompartmentElement('R', n=removed),
        ],
        interactions=[
            InfectionInteraction('bites', susceptible='S', infected='Z', beta=beta, stochastic=stochastic),
            ReactionInteraction(
                'destruction',
                reactants={('Z', 'n'): 1},
                products={('R', 'n'): 1},
                modifiers={('S', 'n'): 1},
                k=alpha,
                mode=mode,
            ),
            ReactionInteraction('death', reactants={('S', 'n'): 1}, products={('R', 'n'): 1}, k=delta, mode=mode),
            ReactionInteraction('resurrection', reactants={('R', 'n'): 1}, products={('Z', 'n'): 1}, k=zeta, mode=mode),
            ReactionInteraction('birth', products={('S', 'n'): 1}, k=birth, mode=mode),
        ],
    )
