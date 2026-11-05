"""Death at a constant rate and replacement by mutated copies of parents chosen in proportion to their fitness."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.sets.real_set import RealSet

from ....core.components.interactions import EvolutionInteraction
from ....core.components.types import SystemViewProtocol
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import Array


class ReproductionInteraction(EvolutionInteraction):
    """Death at a constant rate and replacement by mutated copies of parents chosen in proportion to their fitness.

    It is the death-birth replacement of `EvolutionInteraction`, with parents drawn among all agents, on genomes stored
    as the integers their bits spell. A newborn has one random bit flipped with probability `mutation`. The fitness of
    a genome is the mean over its loci of a value given to each allele of each locus, an additive landscape with one
    value for a 0 and one for a 1 at every locus.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of bit strings.

        death:
            The probability that an agent dies in a step.

        mutation:
            The probability that a newborn has one random bit flipped.

        zero_values:
            The value of a 0 at every locus.

        one_values:
            The value of a 1 at every locus.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import ReproductionInteraction
        >>> ReproductionInteraction(
        ...     'reproduction', target='genomes', zero_values=(0.1, 0.2), one_values=(0.9, 0.3)
        ... ).mutation
        0.05
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        death: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.1,
        mutation: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.05,
        zero_values: tuple[float, ...] = (),
        one_values: tuple[float, ...] = (),
    ) -> None:
        super().__init__(
            label,
            target=target,
            strategy='genome',
            selection='proportional',
            replacement='death-birth',
            mutation=mutation,
            death=death,
            parents='all',
        )
        self.zero_values = zero_values
        self.one_values = one_values

    def _evaluate(self, strategies: Array, system: SystemViewProtocol) -> Array:
        """Return the fitness of every genome on the additive landscape.

        Args:
            strategies:
                The genomes.

            system:
                The system view of the interaction.

        Returns:
            fitness:
                The fitness of every genome.
        """
        return ReproductionInteraction._fitness_of(
            np.ravel(strategies),
            np.asarray(self.zero_values),
            np.asarray(self.one_values),
        )

    def _mutate(self, children: Array, probability: float, rng: np.random.Generator, population: Array) -> Array:
        """Return the genomes of the newborns, each with one random bit flipped with the probability of mutation.

        Args:
            children:
                The genomes, one row per newborn.

            probability:
                The probability that a newborn mutates.

            rng:
                The random number generator.

            population:
                The genomes of the population.

        Returns:
            children:
                The mutated genomes.
        """
        flips = (rng.random(len(children)) < probability).astype(np.int64)
        loci = rng.integers(0, len(self.zero_values), size=len(children))
        return (children.astype(np.int64) ^ (flips << loci)[:, None]).astype(float)

    @staticmethod
    def _bits_of(genomes: Array, n_bits: int) -> npt.NDArray[np.int64]:
        """Return the bits of integer genomes, one row per genome."""
        return (genomes.astype(np.int64)[:, None] >> np.arange(n_bits)) & 1

    @staticmethod
    def _fitness_of(genomes: Array, zero_values: Array, one_values: Array) -> Array:
        """Return the mean over the loci of the value of each genome's allele."""
        bits = ReproductionInteraction._bits_of(genomes, zero_values.size)
        return np.where(bits == 1, one_values, zero_values).mean(axis=1)
