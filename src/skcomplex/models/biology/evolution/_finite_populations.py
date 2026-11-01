"""Evolution models: finite populations of individuals that reproduce, mutate, and climb fitness landscapes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import lru_cache
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import EvolutionInteraction
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....core.types import ModelViewProtocol
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ._climber_view import ClimberView
from ._individual_view import IndividualView
from ._tangled_nature_life_cycle import TangledNatureLifeCycleInteraction

ORGANISMS = 'organisms'


@lru_cache(maxsize=16)
def _nk_tables(seed: int, n: int, k: int) -> Array:
    """Return the fitness contributions of every locus for every configuration of itself and its `k` neighbours."""
    return np.random.default_rng(seed).random((n, 2 ** (k + 1)))


def _nk_fitness(genome: int, n: int, k: int, seed: int) -> float:
    """Return the NK fitness of a genome, the mean contribution of its loci, each read with the next `k` loci."""
    tables = _nk_tables(seed, n, k)
    bits = [(genome >> locus) & 1 for locus in range(n)]
    total = 0.0
    for locus in range(n):
        index = sum(bits[(locus + offset) % n] << offset for offset in range(k + 1))
        total += tables[locus, index]
    return total / n


def _random_fitness(genome: int, seed: int) -> float:
    """Return the fitness of a genome in the random energy model, uniform in [0, 1] and fixed by the seed."""
    return float(np.random.default_rng([seed, genome]).random())


class IndividualElement(Element):
    """Individual of a finite population, with an integer trait or genome.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        trait:
            The trait, a type index or the bits of a genome.

        fitness:
            The fitness of the trait.

    Examples:
        >>> from skcomplex.models.biology.evolution import IndividualElement
        >>> GroupElement('population', element=IndividualElement, size=10, values={'trait': 0.0}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'trait'}),
        *,
        trait: Annotated[float, StateSpec(domain=ZZ)] = 0.0,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.trait = trait
        self.fitness = fitness

    @classmethod
    def _get_view(cls) -> type[IndividualView]:
        """Return the view of an individual, `IndividualView`."""
        return IndividualView


class MoranStepInteraction(EvolutionInteraction):
    """One generation of the Moran process: as many birth and death events as there are individuals.

    At every event an individual reproduces with probability proportional to its fitness `(1 + advantage)^trait`,
    its offspring mutates to the trait one above or one below with probability `mutation / 2` each, and it replaces an
    individual chosen uniformly at random, so the size of the population stays constant. It is the Moran replacement
    of `EvolutionInteraction` with proportional selection and steps of the trait.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of individuals.

        advantage:
            The selective advantage of one unit of the trait.

        mutation:
            The probability that an offspring mutates.

        phase:
            The phase of the step the generation runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.biology.evolution import MoranStepInteraction
        >>> MoranStepInteraction('moran', target='population', advantage=0.1).check().replacement
        'moran'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        advantage: Annotated[float, ParameterSpec()] = 0.0,
        mutation: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            strategy='trait',
            selection='proportional',
            replacement='moran',
            mutation=mutation,
            alphabet=None,
            change='step',
            phase=phase,
        )
        self.advantage = advantage

    def _evaluate(self, strategies: Array, model: ModelViewProtocol) -> Array:
        """Return the fitness `(1 + advantage)^trait` of every trait.

        Args:
            strategies:
                The traits.

            model:
                The model view of the interaction.

        Returns:
            fitness:
                The fitness of every trait.
        """
        return (1 + model.parameter('advantage')) ** np.ravel(strategies)


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
        >>> from skcomplex.models.biology.evolution import AdaptiveClimbingInteraction
        >>> AdaptiveClimbingInteraction('climbing', target='population', mutation=0.1).check().replacement
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

    def _evaluate(self, strategies: Array, model: ModelViewProtocol) -> Array:
        """Return the random fitness of every genome.

        Args:
            strategies:
                The genomes.

            model:
                The model view of the interaction.

        Returns:
            fitness:
                The fitness of every genome.
        """
        return np.array([_random_fitness(int(genome), self.seed) for genome in np.ravel(strategies)])

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


class ClimberElement(Element):
    """Agent that walks an NK fitness landscape, flipping one random bit of its genome when that raises its fitness.

    The fitness of a genome of `n` bits is the mean of the contributions of its bits, each a random function of the bit
    and the next `k` bits, fixed by `seed`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        genome:
            The bits of the genome.

        fitness:
            The fitness of the genome.

        n:
            The number of bits.

        k:
            The number of other bits each bit interacts with.

        seed:
            The seed of the landscape.

    Examples:
        >>> from skcomplex.models.biology.evolution import ClimberElement
        >>> GroupElement('climbers', element=ClimberElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'fitness', 'color_range': (0.0, 1.0)}),
        *,
        genome: Annotated[float, StateSpec(domain=NN)] = 0.0,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        k: Annotated[float, ParameterSpec(domain=NN)] = 2.0,
        seed: Annotated[float, ParameterSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.genome = genome
        self.fitness = fitness
        self.n = n
        self.k = k
        self.seed = seed

    @classmethod
    def _get_view(cls) -> type[ClimberView]:
        """Return the view of a climber, `ClimberView`."""
        return ClimberView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the climbing rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.climb)]

    def climb(self, agent: ClimberView, model: ModelView) -> None:
        """Flip a random bit, and keep the flip when it raises the fitness.

        Args:
            agent:
                The climber.

            model:
                The model.
        """
        n, k, seed = int(agent.n), int(agent.k), int(agent.seed)
        mutant = int(agent.genome) ^ (1 << int(model.rng.integers(n)))
        fitness = _nk_fitness(mutant, n, k, seed)
        if fitness > agent.fitness:
            agent.genome = float(mutant)
            agent.fitness = fitness


@register_model(
    field='biology/evolution',
    tags=('population genetics', 'agents', 'fixation', 'genetic drift', 'mutation'),
    references=('Moran 1958', 'Nowak 2006', 'Jensen 2023'),
    defaults={'until': 100},
)
def moran_process(
    size: int = 50,
    *,
    mutants: int = 1,
    selection: float = 0.0,
    mutation: float = 0.0,
) -> System:
    """Return the Moran process, a population of constant size where fitter traits fix by selection and drift.

    The population starts with `mutants` individuals of trait 1 among individuals of trait 0. Without mutation one trait
    eventually fixes, the mutant with probability `(1 - 1 / r^m) / (1 - 1 / r^N)` for the relative fitness
    `r = 1 + selection`. With mutation the traits diffuse along the trait axis, in clusters rather than as a spreading
    Gaussian, which distinguishes evolutionary diffusion from a random walk.

    Args:
        size:
            The number of individuals.

        mutants:
            The initial number of individuals of trait 1.

        selection:
            The selective advantage of one unit of the trait.

        mutation:
            The probability that an offspring's trait mutates by one unit.

    Returns:
        system:
            The model, with the group `population`.

    Examples:
        >>> from skcomplex.models.biology.evolution import moran_process
        >>> float(moran_process(10, mutants=3).elements[0].values['trait'].sum())
        3.0
    """
    traits = np.zeros(size)
    traits[:mutants] = 1.0
    return System(
        elements=[
            GroupElement(
                'population',
                element=IndividualElement,
                size=size,
                values={'trait': traits, 'fitness': (1 + selection) ** traits},
                variable=True,
            ),
        ],
        interactions=[MoranStepInteraction('moran', target='population', advantage=selection, mutation=mutation)],
    )


@register_model(
    field='biology/evolution',
    tags=('coevolution', 'agents', 'punctuated equilibrium', 'ecosystems'),
    references=('Christensen et al. 2002', 'Jensen 2023'),
    defaults={'until': 50},
)
def tangled_nature(
    organisms: int = 100,
    *,
    k: float = 5.0,
    mu: float = 0.01,
    p_mut: float = 0.01,
    p_kill: float = 0.2,
    length: int = 8,
    seed: int = 0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Tangled Nature model, where organisms whose types interact through random couplings coevolve.

    Organisms of one random type start the run. Mutants spread through the space of genomes, and the population
    settles into quasi-stable configurations of a few dominant types, the q-ESS, that end in abrupt transitions, a
    punctuated equilibrium. The trait of an organism holds the bits of its genome. A generation is `N / p_kill`
    visits to organisms drawn with replacement, where `N` is the population at its start.

    Args:
        organisms:
            The initial number of organisms.

        k:
            The strength of the couplings.

        mu:
            The limit of resources, which lowers reproduction in large populations.

        p_mut:
            The probability that a bit of an offspring's genome flips.

        p_kill:
            The probability that a chosen organism dies.

        length:
            The number of bits of a genome.

        seed:
            The seed of the couplings.

        random_state:
            The seed or generator of the initial type.

    Returns:
        system:
            The model, with the group `organisms`.

    Examples:
        >>> from skcomplex.models.biology.evolution import tangled_nature
        >>> tangled_nature(20).elements[0].check().prepare().size_
        20
    """
    genome = float(np.random.default_rng(random_state).integers(2**length))
    life_cycle = TangledNatureLifeCycleInteraction(
        'evolution',
        target=ORGANISMS,
        k=k,
        mu=mu,
        p_mut=p_mut,
        p_kill=p_kill,
        length=length,
        seed=seed,
    )
    visual = VisualSpec('histograms', {'color_by': 'trait', 'shown': ('distinct',)})
    return System(
        elements=[
            GroupElement(
                ORGANISMS,
                element=IndividualElement,
                size=organisms,
                values={'trait': genome},
                variable=True,
                visual=visual,
            ),
        ],
        interactions=[life_cycle],
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('generation', activation='visits', visits=1 / p_kill)]),
        ),
    )


@register_model(
    field='biology/evolution',
    tags=('fitness landscape', 'agents', 'random energy model', 'stochastic escape'),
    references=('Gros 2015',),
    defaults={'until': 200},
)
def adaptive_climbing(
    size: int = 10,
    *,
    u: float = 0.1,
    length: int = 20,
    seed: int = 0,
) -> System:
    """Return a finite population that climbs a random fitness landscape by mutation and strong selection.

    The population climbs to a local optimum in an adaptive walk, ever slower near the top. When every individual
    mutates in the same generation, which is likely only in small populations with a large `u`, it escapes to a lower
    genome and starts a new walk, the wandering regime.

    Args:
        size:
            The number of individuals.

        u:
            The probability that an individual mutates in a generation.

        length:
            The number of loci of a genome.

        seed:
            The seed of the fitness landscape.

    Returns:
        system:
            The model, with the group `population`.

    Examples:
        >>> from skcomplex.models.biology.evolution import adaptive_climbing
        >>> adaptive_climbing(5).elements[0].check().prepare().size_
        5
    """
    fitness = _random_fitness(0, seed)
    return System(
        elements=[
            GroupElement(
                'population',
                element=IndividualElement,
                size=size,
                values={'trait': 0.0, 'fitness': fitness},
                variable=True,
                visual=VisualSpec('agents', {'color_by': 'fitness'}),
            ),
        ],
        interactions=[
            AdaptiveClimbingInteraction('climbing', target='population', mutation=u, length=length, seed=seed),
        ],
    )


@register_model(
    field='biology/evolution',
    tags=('fitness landscape', 'agents', 'NK model', 'hill climbing', 'ruggedness'),
    references=('Kauffman and Levin 1987', 'Downey 2018', 'Thurner et al. 2018'),
    defaults={'until': 50},
)
def nk_landscape(
    climbers: int = 50,
    *,
    n: int = 10,
    k: int = 2,
    seed: int = 0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return agents that climb Kauffman's NK fitness landscape from random genomes until they reach local peaks.

    The landscape grows more rugged with `k`, from a single peak at `k = 0` to a random landscape at `k = n - 1`, so
    the climbers end on more peaks of lower fitness.

    Args:
        climbers:
            The number of agents.

        n:
            The number of bits of a genome.

        k:
            The number of other bits each bit interacts with.

        seed:
            The seed of the landscape.

        random_state:
            The seed or generator of the initial genomes.

    Returns:
        system:
            The model, with the group `climbers`.

    Examples:
        >>> from skcomplex.models.biology.evolution import nk_landscape
        >>> nk_landscape(5).elements[0].check().prepare().size_
        5
    """
    genomes = np.random.default_rng(random_state).integers(0, 2**n, size=climbers)
    fitness = [_nk_fitness(int(genome), n, k, seed) for genome in genomes]
    group = GroupElement(
        'climbers',
        element=ClimberElement,
        size=climbers,
        values={
            'genome': genomes.astype(float),
            'fitness': np.asarray(fitness),
            'n': float(n),
            'k': float(k),
            'seed': float(seed),
        },
        variable=True,
    )
    return System(elements=[group])
