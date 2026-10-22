"""Epidemiology models: compartmental models of infection and recovery."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.model import System
from ....core.runtime import register_model
from ._compartments import Compartment, ImmunityLoss, Infection, Recovery


@register_model(
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
            The model, with the compartments `S`, `I`, and `R`.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import sir
        >>> [element.label for element in sir().elements]
        ['S', 'I', 'R']
    """
    return System(
        components=[Compartment('S', n=susceptible), Compartment('I', n=infected), Compartment('R', n=recovered)],
        interactions=[
            Infection('infection', susceptible='S', infected='I', beta=beta, stochastic=stochastic),
            Recovery('recovery', infected='I', recovered='R', gamma=gamma, stochastic=stochastic),
        ],
    )


@register_model(
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
            The model, with the compartments `S` and `I`.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import sis
        >>> [element.label for element in sis().elements]
        ['S', 'I']
    """
    return System(
        components=[Compartment('S', n=susceptible), Compartment('I', n=infected)],
        interactions=[
            Infection('infection', susceptible='S', infected='I', beta=beta, stochastic=stochastic),
            Recovery('recovery', infected='I', recovered='S', gamma=gamma, stochastic=stochastic),
        ],
    )


@register_model(
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
            The model, with the compartments `S`, `I`, and `R`.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import sirs
        >>> sirs(xi=0.1).interactions[-1].xi
        0.1
    """
    return System(
        components=[Compartment('S', n=susceptible), Compartment('I', n=infected), Compartment('R', n=recovered)],
        interactions=[
            Infection('infection', susceptible='S', infected='I', beta=beta, stochastic=stochastic),
            Recovery('recovery', infected='I', recovered='R', gamma=gamma, stochastic=stochastic),
            ImmunityLoss('waning', recovered='R', susceptible='S', xi=xi, stochastic=stochastic),
        ],
    )
