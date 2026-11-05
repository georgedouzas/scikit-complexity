"""One generation of a population that climbs a random fitness landscape as a whole, under strong selection."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.sets.real_set import RealSet

from ....core.components.interactions import EvolutionInteraction
from ....core.components.types import SystemViewProtocol
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import Array


class AdaptiveClimbingInteraction(EvolutionInteraction):
    """One generation of a population that climbs a random fitness landscape as a whole, under strong selection.

    All individuals share one genome. Each mutates one random locus with probability `u`, to a genome of random fitness
    in [0, 1], with probability `mutation`. If a mutant is fitter, the whole population takes the fittest mutant. If
    every individual mutated to a less fit genome, the population escapes to the fittest of them. Otherwise it stays.
    It is the best-mutant
    replacement of `EvolutionInteraction` on genomes stored as the integers their bits spell.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of individuals.

        mutation:
            The probability that an individual mutates in a generation.

        length:
            The number of loci of a genome.

        seed:
            The seed of the fitness landscape.

    Examples:
        >>> from skcomplex.systems.biology.evolution import AdaptiveClimbingInteraction
        >>> AdaptiveClimbingInteraction('climbing', target='population', mutation=0.1).replacement
        'best-mutant'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        mutation: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.1,
        length: int = 20,
        seed: int = 0,
    ) -> None:
        super().__init__(label, target=target, strategy='trait', replacement='best-mutant', mutation=mutation)
        self.length = length
        self.seed = seed

    def _evaluate(self, strategies: Array, system: SystemViewProtocol) -> Array:
        """Return the random fitness of every genome.

        Args:
            strategies:
                The genomes.

            system:
                The system view of the interaction.

        Returns:
            fitness:
                The fitness of every genome.
        """
        return np.array(
            [AdaptiveClimbingInteraction._random_fitness(int(genome), self.seed) for genome in np.ravel(strategies)],
        )

    def _mutate(self, children: Array, probability: float, rng: np.random.Generator, population: Array) -> Array:
        """Return the genomes, each with one random locus flipped with the probability of mutation.

        Args:
            children:
                The genomes, one row per individual.

            probability:
                The probability that a genome mutates.

            rng:
                The random number generator.

            population:
                The genomes of the population.

        Returns:
            children:
                The mutated genomes.
        """
        flips = (rng.random(len(children)) < probability).astype(np.int64)
        loci = rng.integers(self.length, size=len(children))
        return (children.astype(np.int64) ^ (flips << loci)[:, None]).astype(float)

    @staticmethod
    def _random_fitness(genome: int, seed: int) -> float:
        """Return the fitness of a genome in the random energy system, uniform in [0, 1] and fixed by the seed."""
        return float(np.random.default_rng([seed, genome]).random())
