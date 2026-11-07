"""Epidemiology systems: compartmental systems of infection and recovery."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.components.elements import CompartmentElement
from ....core.components.interactions import ReactionInteraction
from ....core.components.system import System
from ..._catalog import register_system
from ._immunity_loss import ImmunityLossInteraction
from ._infection import InfectionInteraction
from ._recovery import RecoveryInteraction


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
