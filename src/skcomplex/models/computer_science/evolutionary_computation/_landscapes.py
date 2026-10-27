"""Evolutionary computation models: bit strings evolving on a fitness landscape."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import ModelView

type Array = npt.NDArray[np.float64]


def _bits_of(genomes: Array, n_bits: int) -> npt.NDArray[np.int64]:
    """Return the bits of integer genomes, one row per genome."""
    return (genomes.astype(np.int64)[:, None] >> np.arange(n_bits)) & 1


def _fitness_of(genomes: Array, zero_values: Array, one_values: Array) -> Array:
    """Return the mean over the loci of the value of each genome's allele."""
    bits = _bits_of(genomes, zero_values.size)
    return np.where(bits == 1, one_values, zero_values).mean(axis=1)


class BitStringElement(Element):
    """Agent whose genome is a string of bits, stored as the integer it spells, with the fitness of that genome.

    Args:
        label:
            The label of the agent or group.

        genome:
            The initial genome, an integer whose binary digits are the bits.

        fitness:
            The initial fitness.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import BitStringElement
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


class ReproductionInteraction(Interaction):
    """Death at a constant rate and replacement by mutated copies of parents chosen in proportion to their fitness.

    The fitness of a genome is the mean over its loci of a value given to each allele of each locus, an additive
    landscape with one value for a 0 and one for a 1 at every locus.

    Args:
        label:
            The label of the interaction.

        zero_values:
            The value of a 0 at every locus.

        one_values:
            The value of a 1 at every locus.

        **values:
            The values of the declared variables, by name:

            death:
                The probability that an agent dies in a step.

            mutation:
                The probability that a newborn has one random bit flipped.

            population:
                The path of the group of bit strings.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import ReproductionInteraction
        >>> ReproductionInteraction(
        ...     'reproduction', population='genomes', zero_values=(0.1, 0.2), one_values=(0.9, 0.3)
        ... ).mutation
        0.05
    """

    def __init__(
        self,
        label: str,
        *,
        death: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        mutation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        population: Annotated[str, RoleSpec()],
        zero_values: tuple[float, ...] = (),
        one_values: tuple[float, ...] = (),
    ) -> None:
        super().__init__(label)
        self.death = death
        self.mutation = mutation
        self.population = population
        self.zero_values = zero_values
        self.one_values = one_values

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of one generation step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.reproduce)]

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
    reproduction = ReproductionInteraction(
        'reproduction',
        population='genomes',
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
