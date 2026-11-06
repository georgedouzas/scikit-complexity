"""Evolutionary computation systems: bit strings evolving on a fitness landscape."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.system import System
from ..._catalog import register_system
from ._bit_string import BitStringElement
from ._reproduction import ReproductionInteraction


@register_system(
    field='computer_science/evolutionary_computation',
    tags=('evolution', 'agents', 'fitness landscape', 'genetic algorithm'),
    references=('Downey 2018',),
    defaults={'until': 100},
)
def bit_string_evolution(
    n_agents: int = 100,
    *,
    n_bits: int = 8,
    death: float = 0.1,
    mutation: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a population of bit strings that evolves on a random additive fitness landscape.

    Every step each agent dies with a constant probability, and the dead are replaced by copies of parents chosen in
    proportion to their fitness, with one bit flipped with a small probability, so the mean fitness climbs.

    Args:
        n_agents:
            The size of the population, which stays constant.

        n_bits:
            The number of bits of a genome.

        death:
            The probability that an agent dies in a step.

        mutation:
            The probability that a newborn has one random bit flipped.

        random_state:
            The seed or generator of the landscape and of the random initial genomes.

    Returns:
        system:
            The system, with the group `genomes` of states `genome` and `fitness`.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import bit_string_evolution
        >>> bit_string_evolution(10, n_bits=4).interactions[0].target
        'genomes'
    """
    rng = np.random.default_rng(random_state)
    zero_values, one_values = rng.random(n_bits), rng.random(n_bits)
    genomes = rng.integers(0, 2**n_bits, size=n_agents).astype(float)
    fitness = ReproductionInteraction._fitness_of(genomes, zero_values, one_values)
    reproduction = ReproductionInteraction(
        'reproduction',
        target='genomes',
        death=death,
        mutation=mutation,
        zero_values=tuple(zero_values.tolist()),
        one_values=tuple(one_values.tolist()),
    )
    return System(
        elements=[
            GroupElement(
                'genomes',
                element=BitStringElement,
                size=n_agents,
                values={'genome': genomes, 'fitness': fitness},
            ),
        ],
        interactions=[reproduction],
    )
