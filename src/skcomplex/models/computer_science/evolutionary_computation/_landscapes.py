"""Evolutionary computation models: bit strings evolving on a fitness landscape."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.model import AgentType, Interaction, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import ModelView

type Array = npt.NDArray[np.float64]


def _bits_of(genomes: Array, n_bits: int) -> npt.NDArray[np.int64]:
    """Return the bits of integer genomes, one row per genome."""
    return (genomes.astype(np.int64)[:, None] >> np.arange(n_bits)) & 1


def _fitness_of(genomes: Array, zero_values: Array, one_values: Array) -> Array:
    """Return the mean over the loci of the value of each genome's allele."""
    bits = _bits_of(genomes, zero_values.size)
    return np.where(bits == 1, one_values, zero_values).mean(axis=1)


class BitString(AgentType):
    """Agent whose genome is a string of bits, stored as the integer it spells, with the fitness of that genome.

    Args:
        label:
            The label of the agent type or group.

        location:
            Unused, since genomes have no space.

        genome:
            The initial genome, an integer whose binary digits are the bits.

        fitness:
            The initial fitness.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import BitString
        >>> BitString.group('genomes', 4, genome=[0, 1, 2, 3])
        Group('genomes', element=BitString, size=4)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'fitness'})
    genome: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    fitness: float = state(default=0.0, domain=Domain.NONNEGATIVE)


class Reproduction(Interaction):
    """Death at a constant rate and replacement by mutated copies of parents chosen in proportion to their fitness.

    The fitness of a genome is the mean over its loci of a value given to each allele of each locus, an additive
    landscape with one value for a 0 and one for a 1 at every locus.

    Args:
        label:
            The label of the interaction.

        death:
            The probability that an agent dies in a step.

        mutation:
            The probability that a newborn has one random bit flipped.

        population:
            The path of the group of bit strings.

        zero_values:
            The value of a 0 at every locus.

        one_values:
            The value of a 1 at every locus.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import Reproduction
        >>> Reproduction('reproduction', population='genomes', zero_values=(0.1, 0.2), one_values=(0.9, 0.3)).mutation
        0.05
    """

    death: float = parameter(0.1, domain=Domain.NONNEGATIVE)
    mutation: float = parameter(0.05, domain=Domain.NONNEGATIVE)
    population: str = role()
    zero_values: tuple[float, ...] = ()
    one_values: tuple[float, ...] = ()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule of one generation step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.reproduce)]

    def reproduce(self, model: ModelView) -> None:
        """Replace the dead by mutated copies of parents chosen by fitness, and update every fitness.

        Args:
            model:
                The model.
        """
        agents = model.agents('population')
        genomes = np.array([agent.genome for agent in agents], dtype=float)
        fitness = np.array([agent.fitness for agent in agents], dtype=float)
        rng = model.rng
        dead = np.flatnonzero(rng.random(len(agents)) < model.parameter('death'))
        parents = rng.choice(len(agents), size=dead.size, p=fitness / fitness.sum())
        flips = rng.random(dead.size) < model.parameter('mutation')
        loci = rng.integers(0, len(self.zero_values), size=dead.size)
        genomes[dead] = genomes[parents].astype(np.int64) ^ (flips.astype(np.int64) << loci)
        zero_values, one_values = np.asarray(self.zero_values), np.asarray(self.one_values)
        for agent, genome, value in zip(agents, genomes, _fitness_of(genomes, zero_values, one_values), strict=True):
            agent.genome = float(genome)
            agent.fitness = float(value)


@register_model(
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
            The model, with the group `genomes` of states `genome` and `fitness`.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import bit_string_evolution
        >>> bit_string_evolution(10, n_bits=4).interactions[0].population
        'genomes'
    """
    rng = np.random.default_rng(random_state)
    zero_values, one_values = rng.random(n_bits), rng.random(n_bits)
    genomes = rng.integers(0, 2**n_bits, size=n_agents).astype(float)
    fitness = _fitness_of(genomes, zero_values, one_values)
    reproduction = Reproduction(
        'reproduction',
        population='genomes',
        death=death,
        mutation=mutation,
        zero_values=tuple(zero_values.tolist()),
        one_values=tuple(one_values.tolist()),
    )
    return System(
        components=[BitString.group('genomes', n_agents, genome=genomes, fitness=fitness)],
        interactions=[reproduction],
    )
