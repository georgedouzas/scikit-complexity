"""Evolutionary computation systems: bit strings evolving on a fitness landscape."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ...general.agents import EvolutionInteraction


class BitStringView(AgentView):
    """View of an individual of a fitness landscape, with its states as attributes."""

    genome: float
    fitness: float


class BitStringElement(Element):
    """Agent whose genome is a string of bits, with the fitness of that genome.

    The genome is stored as the integer it spells.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        genome:
            The initial genome, an integer whose binary digits are the bits.

        fitness:
            The initial fitness.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.computer_science.evolutionary_computation import BitStringElement
        >>> group = GroupElement('genomes', element=BitStringElement, size=4, values={'genome': [0, 1, 2, 3]})
        >>> group.label, group.element.__name__, group.size
        ('genomes', 'BitStringElement', 4)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'fitness'}),
        *,
        genome: Annotated[float, StateSpec(domain=NN)] = 0.0,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.genome = genome
        self.fitness = fitness

    @classmethod
    def _get_view(cls) -> type[BitStringView]:
        """Return the view of the individuals.

        It types their states.
        """
        return BitStringView


class ReproductionInteraction(EvolutionInteraction):
    """Death at a constant rate and replacement by mutated copies of fit parents.

    The parents are chosen in proportion to their fitness. It is the death-birth replacement of `EvolutionInteraction`,
    with parents drawn among all agents, on genomes stored as the integers their bits spell. A newborn has one random
    bit flipped with probability `mutation`. The fitness of a genome is the mean over its loci of a value given to each
    allele of each locus, an additive landscape with one value for a 0 and one for a 1 at every locus.

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
        """Return the genomes of the newborns, each with one random bit flipped.

        The bit is flipped with the probability of mutation.

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


class BitStringEvolutionSystem(BaseSystem):
    """A population of bit strings that evolves on a random additive fitness landscape.

    Every step each agent dies with a constant probability, and the dead are replaced by copies of parents chosen in
    proportion to their fitness, with one bit flipped with a small probability, so the mean fitness climbs. The system
    has the group `genomes` of states `genome` and `fitness`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import BitStringEvolutionSystem
        >>> BitStringEvolutionSystem(10, n_bits=4).prepare().system_.interactions[0].target
        'genomes'
    """

    field = 'computer_science/evolutionary_computation'
    topics = ('evolution', 'agents', 'fitness landscape', 'genetic algorithm')
    references = ('Downey 2018',)
    until = 100

    def __init__(
        self,
        n_agents: int = 100,
        *,
        n_bits: int = 8,
        death: float = 0.1,
        mutation: float = 0.05,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.n_bits = n_bits
        self.death = death
        self.mutation = mutation
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return a population of bit strings on a random additive fitness landscape.

        The population evolves on the landscape.
        """
        rng = np.random.default_rng(self.random_state)
        zero_values, one_values = rng.random(self.n_bits), rng.random(self.n_bits)
        genomes = rng.integers(0, 2**self.n_bits, size=self.n_agents).astype(float)
        fitness = ReproductionInteraction._fitness_of(genomes, zero_values, one_values)
        reproduction = ReproductionInteraction(
            'reproduction',
            target='genomes',
            death=self.death,
            mutation=self.mutation,
            zero_values=tuple(zero_values.tolist()),
            one_values=tuple(one_values.tolist()),
        )
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'genomes',
                    element=BitStringElement,
                    size=self.n_agents,
                    values={'genome': genomes, 'fitness': fitness},
                ),
            ],
            interactions=[reproduction],
        )
