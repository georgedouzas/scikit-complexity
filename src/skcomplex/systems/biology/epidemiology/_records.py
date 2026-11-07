"""Epidemiology systems: outbreaks that strike only when a strain beats every earlier one, by record dynamics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import DiscreteTime
from ....core.components.system import System
from ..._catalog import register_system
from ._pathogen import PathogenElement
from ._resident import ResidentElement
from ._resistance_infection import ResistanceInfectionInteraction


@register_system(
    field='biology/epidemiology',
    tags=('record dynamics', 'intermittency', 'immunity', 'stochastic map'),
    references=('Jensen 2023',),
    defaults={'until': 1000, 'dt': 1},
)
def record_dynamics_epidemic(n: int = 10_000, *, seed: int | None = 0) -> System:
    """Return yearly strains that infect every resident less resistant than their strength, who become immune.

    Every resident keeps a resistance `R` drawn uniformly from `[0, 1]`. Once a year a strain of strength `eta`, drawn
    uniformly from `[0, 1)`, infects the residents with `R < eta`, whose resistance becomes 1. In a large population an
    outbreak happens only in a year whose strain is stronger than every earlier one, a record, so outbreaks thin out
    as time goes on: the mean number of outbreaks in `t` years is close to the harmonic number `H_t`, about
    `ln t + 0.577`, a Poisson process in the logarithm of time.

    Args:
        n:
            The number of residents.

        seed:
            The seed of the resistances.

    Returns:
        system:
            The system, with the resistances `residents.R`, the infections of the last year `residents.infected`, and
            the strength `pathogen.eta` of the strain of the coming year.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import record_dynamics_epidemic
        >>> record_dynamics_epidemic(100).elements[0].prepare().size_
        100
    """
    resistance = np.random.default_rng(seed).random(n)
    residents = GroupElement('residents', element=ResidentElement, size=n, values={'R': resistance})
    return System(
        elements=[residents, PathogenElement('pathogen')],
        interactions=[ResistanceInfectionInteraction('infection', residents='residents', pathogen='pathogen')],
        time=DiscreteTime(),
    )
