"""Evolution systems: finite populations of individuals.

The individuals reproduce, mutate, and climb fitness landscapes.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from functools import lru_cache
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ, oo
from sage.sets.real_set import RealSet
from scipy.special import expit

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import DiscreteTime
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ...general.agents import BirthDeathInteraction, EvolutionInteraction

ORGANISMS = 'organisms'


class ClimberView(AgentView):
    """View of a climber, with its states and parameters as attributes.

    `ClimberElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    genome: float
    fitness: float
    n: float
    k: float
    seed: float


class ClimberElement(Element):
    """Agent that walks an NK fitness landscape.

    It flips one random bit of its genome when that raises its fitness.

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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.evolution import ClimberElement
        >>> GroupElement('climbers', element=ClimberElement, size=10).size
        10
    """

    draws = True

    law_classes = (RuleLaw,)

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

    def climb(self, agent: ClimberView, system: SystemView) -> None:
        """Flip a random bit, and keep the flip when it raises the fitness.

        Args:
            agent:
                The climber.

            system:
                The system.
        """
        n, k, seed = int(agent.n), int(agent.k), int(agent.seed)
        mutant = int(agent.genome) ^ (1 << int(system.rng.integers(n)))
        fitness = ClimberElement._nk_fitness(mutant, n, k, seed)
        if fitness > agent.fitness:
            agent.genome = float(mutant)
            agent.fitness = fitness

    @staticmethod
    @lru_cache(maxsize=16)
    def _nk_tables(seed: int, n: int, k: int) -> Array:
        """Return the fitness contributions of every locus.

        There is one contribution for every configuration of the locus and its `k` neighbours.
        """
        return np.random.default_rng(seed).random((n, 2 ** (k + 1)))

    @staticmethod
    def _nk_fitness(genome: int, n: int, k: int, seed: int) -> float:
        """Return the NK fitness of a genome.

        It is the mean contribution of its loci, each read with the next `k` loci.
        """
        tables = ClimberElement._nk_tables(seed, n, k)
        bits = [(genome >> locus) & 1 for locus in range(n)]
        total = 0.0
        for locus in range(n):
            index = sum(bits[(locus + offset) % n] << offset for offset in range(k + 1))
            total += tables[locus, index]
        return total / n


class IndividualView(AgentView):
    """View of an individual, with its states as attributes.

    `IndividualElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    trait: float
    fitness: float


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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.evolution import IndividualElement
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


class AdaptiveClimbingInteraction(EvolutionInteraction):
    """One generation of a population that climbs a random fitness landscape.

    The population climbs as a whole, under strong selection.

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
        """Return the genomes, each with one random locus flipped.

        Each locus flips with the probability of mutation.

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
        """Return the fitness of a genome in the random energy system.

        The fitness is uniform in [0, 1] and fixed by the seed.
        """
        return float(np.random.default_rng([seed, genome]).random())


class MoranStepInteraction(EvolutionInteraction):
    """One generation of the Moran process.

    A generation is as many birth and death events as there are individuals.

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
        >>> from skcomplex.systems.biology.evolution import MoranStepInteraction
        >>> MoranStepInteraction('moran', target='population', advantage=0.1).replacement
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

    def _evaluate(self, strategies: Array, system: SystemViewProtocol) -> Array:
        """Return the fitness `(1 + advantage)^trait` of every trait.

        Args:
            strategies:
                The traits.

            system:
                The system view of the interaction.

        Returns:
            fitness:
                The fitness of every trait.
        """
        return (1 + system.parameter('advantage')) ** np.ravel(strategies)


class TangledNatureLifeCycleInteraction(BirthDeathInteraction):
    """Life cycle of the organisms of the Tangled Nature system.

    The cycle is run by the visits of a phase drawn with replacement.

    A visited organism of type `S` reproduces with probability `1 / (1 + exp(-H(S)))`, where
    `H(S) = k / N sum_S' J(S, S') n(S') - mu N` reads the composition `n` of the population of `N` organisms, and is
    replaced by two offspring, whose genome bits flip with probability `p_mut` each. Then a random organism dies with
    probability `p_kill`. The couplings `J` are uniform in [-1, 1] between different types, 0 within a type, and fixed
    by `seed`.

    Args:
        label:
            The label of the life cycle.

        target:
            The path of the variable group of organisms.

        k:
            The strength of the couplings.

        mu:
            The limit of resources, which lowers reproduction in large populations.

        p_mut:
            The probability that a bit of an offspring's genome flips.

        p_kill:
            The probability that a visit kills a random organism.

        length:
            The number of bits of a genome.

        seed:
            The seed of the couplings.

        phase:
            The phase whose visits run the life cycle.

    Examples:
        >>> from skcomplex.systems.biology.evolution import TangledNatureLifeCycleInteraction
        >>> TangledNatureLifeCycleInteraction('evolution', target='organisms').offspring
        2
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        p_mut: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        p_kill: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        length: int = 8,
        seed: int = 0,
        phase: str | None = 'generation',
    ) -> None:
        super().__init__(
            label,
            target=target,
            birth=self.compute_birth,
            offspring=2,
            inherit=('trait',),
            mutate=self.mutate_genome,
            schedule='member',
            phase=phase,
            replace=True,
            cull=self.compute_kill,
            pick='random',
        )
        self.k = k
        self.mu = mu
        self.p_mut = p_mut
        self.p_kill = p_kill
        self.length = length
        self.seed = seed

    @staticmethod
    @lru_cache(maxsize=16)
    def _get_couplings(seed: int, length: int) -> Array:
        """Return the random couplings `J(S, S')`, uniform in [-1, 1] and 0 within a type."""
        size = 2**length
        couplings = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(size, size))
        np.fill_diagonal(couplings, 0.0)
        return couplings

    def compute_birth(self, agent: IndividualView, system: SystemView) -> float:
        """Return the probability that an organism of type `S` reproduces.

        The probability is `1 / (1 + exp(-H(S)))`.

        Args:
            agent:
                The organism.

            system:
                The system view of the life cycle.

        Returns:
            probability:
                The probability of reproduction.
        """
        composition = system.composition('target', 'trait')
        size = sum(composition.values())
        row = self._get_couplings(self.seed, self.length)[int(agent.trait)]
        coupling = sum(row[int(np.asarray(genome))] * count for genome, count in composition.items())
        return float(expit(system.parameter('k') / size * coupling - system.parameter('mu') * size))

    def compute_kill(self, agent: IndividualView, system: SystemView) -> float:
        """Return the probability `p_kill` that a visit kills a random organism.

        Args:
            agent:
                The visited organism.

            system:
                The system view of the life cycle.

        Returns:
            probability:
                The probability of a death.
        """
        return system.parameter('p_kill')

    def mutate_genome(self, values: dict[str, object], system: SystemViewProtocol) -> dict[str, object]:
        """Return the values of an offspring with mutated genome bits.

        Every genome bit flips with probability `p_mut`.

        Args:
            values:
                The values of the offspring, with the genome of its parent as `trait`.

            system:
                The system view of the life cycle.

        Returns:
            values:
                The values with the mutated genome.
        """
        genome = int(np.asarray(values['trait'], dtype=float))
        flips = np.flatnonzero(system.rng.random(self.length) < system.parameter('p_mut'))
        return {**values, 'trait': float(genome ^ sum(1 << int(locus) for locus in flips))}


class MoranProcessSystem(BaseSystem):
    """The Moran process, a population of constant size.

    Fitter traits fix by selection and drift.

    The population starts with `mutants` individuals of trait 1 among individuals of trait 0. Without mutation one trait
    eventually fixes, the mutant with probability `(1 - 1 / r^m) / (1 - 1 / r^N)` for the relative fitness `r = 1 +
    selection`. With mutation the traits diffuse along the trait axis, in clusters rather than as a spreading Gaussian,
    which distinguishes evolutionary diffusion from a random walk. The system has the group `population`.

    Args:
        size:
            The number of individuals.

        mutants:
            The initial number of individuals of trait 1.

        selection:
            The selective advantage of one unit of the trait.

        mutation:
            The probability that an offspring's trait mutates by one unit.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import MoranProcessSystem
        >>> float(MoranProcessSystem(10, mutants=3).prepare().system_.elements[0].values['trait'].sum())
        3.0
    """

    field = 'biology/evolution'
    topics = ('population genetics', 'agents', 'fixation', 'genetic drift', 'mutation')
    references = ('Moran 1958', 'Nowak 2006', 'Jensen 2023')
    until = 100

    def __init__(
        self,
        size: int = 50,
        *,
        mutants: int = 1,
        selection: float = 0.0,
        mutation: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.mutants = mutants
        self.selection = selection
        self.mutation = mutation

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the Moran process."""
        traits = np.zeros(self.size)
        traits[: self.mutants] = 1.0
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'population',
                    element=IndividualElement,
                    size=self.size,
                    values={'trait': traits, 'fitness': (1 + self.selection) ** traits},
                    variable=True,
                ),
            ],
            interactions=[
                MoranStepInteraction('moran', target='population', advantage=self.selection, mutation=self.mutation),
            ],
        )


class TangledNatureSystem(BaseSystem):
    """The Tangled Nature system of coevolving organisms.

    The types of the organisms interact through random couplings.

    Organisms of one random type start the run. Mutants spread through the space of genomes, and the population settles
    into quasi-stable configurations of a few dominant types, the q-ESS, that end in abrupt transitions, a punctuated
    equilibrium. The trait of an organism holds the bits of its genome. A generation is `N / p_kill` visits to organisms
    drawn with replacement, where `N` is the population at its start. The system has the group `organisms`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import TangledNatureSystem
        >>> TangledNatureSystem(20).prepare().system_.elements[0].prepare().size_
        20
    """

    field = 'biology/evolution'
    topics = ('coevolution', 'agents', 'punctuated equilibrium', 'ecosystems')
    references = ('Christensen et al. 2002', 'Jensen 2023')
    until = 50

    def __init__(
        self,
        organisms: int = 100,
        *,
        k: float = 5.0,
        mu: float = 0.01,
        p_mut: float = 0.01,
        p_kill: float = 0.2,
        length: int = 8,
        seed: int = 0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.organisms = organisms
        self.k = k
        self.mu = mu
        self.p_mut = p_mut
        self.p_kill = p_kill
        self.length = length
        self.seed = seed
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the Tangled Nature system."""
        genome = float(np.random.default_rng(self.random_state).integers(2**self.length))
        life_cycle = TangledNatureLifeCycleInteraction(
            'evolution',
            target=ORGANISMS,
            k=self.k,
            mu=self.mu,
            p_mut=self.p_mut,
            p_kill=self.p_kill,
            length=self.length,
            seed=self.seed,
        )
        visual = VisualSpec('histograms', {'color_by': 'trait', 'shown': ('distinct',)})
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    ORGANISMS,
                    element=IndividualElement,
                    size=self.organisms,
                    values={'trait': genome},
                    variable=True,
                    visual=visual,
                ),
            ],
            interactions=[life_cycle],
            time=DiscreteTime(
                schedule=PhasesScheduleSpec([PhaseSpec('generation', activation='visits', visits=1 / self.p_kill)]),
            ),
        )


class AdaptiveClimbingSystem(BaseSystem):
    """A finite population that climbs a random fitness landscape.

    It climbs by mutation and strong selection.

    The population climbs to a local optimum in an adaptive walk, ever slower near the top. When every individual
    mutates in the same generation, which is likely only in small populations with a large `u`, it escapes to a lower
    genome and starts a new walk, the wandering regime. The system has the group `population`.

    Args:
        size:
            The number of individuals.

        u:
            The probability that an individual mutates in a generation.

        length:
            The number of loci of a genome.

        seed:
            The seed of the fitness landscape.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import AdaptiveClimbingSystem
        >>> AdaptiveClimbingSystem(5).prepare().system_.elements[0].prepare().size_
        5
    """

    field = 'biology/evolution'
    topics = ('fitness landscape', 'agents', 'random energy system', 'stochastic escape')
    references = ('Gros 2015',)
    until = 200

    def __init__(
        self,
        size: int = 10,
        *,
        u: float = 0.1,
        length: int = 20,
        seed: int = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.u = u
        self.length = length
        self.seed = seed

    def _build_system(self) -> CompositeSystem:
        """Return the parts of a finite population."""
        fitness = AdaptiveClimbingInteraction._random_fitness(0, self.seed)
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'population',
                    element=IndividualElement,
                    size=self.size,
                    values={'trait': 0.0, 'fitness': fitness},
                    variable=True,
                    visual=VisualSpec('agents', {'color_by': 'fitness'}),
                ),
            ],
            interactions=[
                AdaptiveClimbingInteraction(
                    'climbing',
                    target='population',
                    mutation=self.u,
                    length=self.length,
                    seed=self.seed,
                ),
            ],
        )


class NKLandscapeSystem(BaseSystem):
    """Agents that climb Kauffman's NK fitness landscape until they reach local peaks.

    The agents start from random genomes.

    The landscape grows more rugged with `k`, from a single peak at `k = 0` to a random landscape at `k = n - 1`, so the
    climbers end on more peaks of lower fitness. The system has the group `climbers`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import NKLandscapeSystem
        >>> NKLandscapeSystem(5).prepare().system_.elements[0].prepare().size_
        5
    """

    field = 'biology/evolution'
    topics = ('fitness landscape', 'agents', 'NK system', 'hill climbing', 'ruggedness')
    references = ('Kauffman and Levin 1987', 'Downey 2018', 'Thurner et al. 2018')
    until = 50

    def __init__(
        self,
        climbers: int = 50,
        *,
        n: int = 10,
        k: int = 2,
        seed: int = 0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.climbers = climbers
        self.n = n
        self.k = k
        self.seed = seed
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system."""
        genomes = np.random.default_rng(self.random_state).integers(0, 2**self.n, size=self.climbers)
        fitness = [ClimberElement._nk_fitness(int(genome), self.n, self.k, self.seed) for genome in genomes]
        group = GroupElement(
            'climbers',
            element=ClimberElement,
            size=self.climbers,
            values={
                'genome': genomes.astype(float),
                'fitness': np.asarray(fitness),
                'n': float(self.n),
                'k': float(self.k),
                'seed': float(self.seed),
            },
            variable=True,
        )
        return CompositeSystem(self.label, elements=[group])
