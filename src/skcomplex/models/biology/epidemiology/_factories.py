"""Epidemiology models: compartmental models of infection and recovery."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.model import System
from ....core.runtime import register_model
from ._compartments import Compartment, Infection, Recovery


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
        elements=[Compartment('S', n=susceptible), Compartment('I', n=infected), Compartment('R', n=recovered)],
        interactions=[
            Infection('infection', susceptible='S', infected='I', beta=beta, stochastic=stochastic),
            Recovery('recovery', infected='I', recovered='R', gamma=gamma, stochastic=stochastic),
        ],
    )
