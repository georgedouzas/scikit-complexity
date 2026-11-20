"""Evolution systems: finite populations of individuals.

The individuals reproduce, mutate, and climb fitness landscapes.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Mapping, Sequence
from functools import lru_cache
from types import SimpleNamespace
from typing import Annotated, Protocol

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, ZZ, oo
from sage.sets.real_set import RealSet
from scipy.special import expit

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import BaseSolver, MesaSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ...general.agents import BirthDeathInteraction, EvolutionInteraction

ORGANISMS = 'organisms'


class ClimberView(AgentViewProtocol, Protocol):
    """View of a climber, with its states and parameters as attributes.

    The rules of `ClimberElement` read them with checked types.
    """

    genome: float
    fitness: float
    n: float
    k: float
    seed: float


class ClimberElement(Element):
    """Agent that walks an NK fitness landscape.

    It flips one random bit of its genome when that raises its fitness, by `ClimbingInteraction`.

    The fitness of a genome of `n` bits is the mean of the contributions of its bits, each a random function of the bit
    and the next `k` bits, fixed by `seed`.

    Args:
        label:
            The label of the agent or group.

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

        values:
            Further values by name, or `None` for none.

        size:
            The number of climbers, or `None` for one.

        space:
            The space of the climbers, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.evolution import ClimberElement
        >>> ClimberElement('climbers', size=10).size
        10
    """

    visual = VisualSpec('agents', {'color_by': 'fitness', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        genome: Annotated[float | npt.ArrayLike, NN] = 0.0,
        fitness: Annotated[float | npt.ArrayLike, RealSet.closed_open(0, oo)] = 0.0,
        n: Annotated[float, RealSet.open(0, oo)] = 10.0,
        k: Annotated[float, NN] = 2.0,
        seed: Annotated[float, NN] = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.genome = genome
        self.fitness = fitness
        self.n = n
        self.k = k
        self.seed = seed

    @staticmethod
    @lru_cache(maxsize=16)
    def _get_nk_tables(seed: int, n: int, k: int) -> Array:
        """Return the fitness contributions of every locus.

        There is one contribution for every configuration of the locus and its `k` neighbours.
        """
        return np.random.default_rng(seed).random((n, 2 ** (k + 1)))

    @staticmethod
    def _get_nk_fitness(genome: int, n: int, k: int, seed: int) -> float:
        """Return the NK fitness of a genome.

        It is the mean contribution of its loci, each read with the next `k` loci.
        """
        tables = ClimberElement._get_nk_tables(seed, n, k)
        bits = [(genome >> locus) & 1 for locus in range(n)]
        total = 0.0
        for locus in range(n):
            index = sum(bits[(locus + offset) % n] << offset for offset in range(k + 1))
            total += tables[locus, index]
        return total / n


class IndividualView(AgentViewProtocol, Protocol):
    """View of an individual, with its states as attributes.

    The rules of `IndividualElement` read them with checked types.
    """

    trait: float
    fitness: float


class IndividualElement(Element):
    """Individual of a finite population, with an integer trait or genome.

    Args:
        label:
            The label of the agent or group.

        trait:
            The trait, a type index or the bits of a genome.

        fitness:
            The fitness of the trait.

        values:
            Further values by name, or `None` for none.

        size:
            The number of individuals, or `None` for one.

        space:
            The space of the individuals, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.evolution import IndividualElement
        >>> IndividualElement('population', trait=0.0, size=10).size
        10
    """

    visual = VisualSpec('agents', {'color_by': 'trait'})

    def __init__(
        self,
        label: str,
        *,
        trait: Annotated[float | npt.ArrayLike, ZZ] = 0.0,
        fitness: Annotated[float | npt.ArrayLike, RealSet.closed_open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.trait = trait
        self.fitness = fitness


class ClimbingIndividualElement(IndividualElement):
    """Individual of a climbing population, drawn by its fitness."""

    visual = VisualSpec('agents', {'color_by': 'fitness'})


class TangledOrganismElement(IndividualElement):
    """Organism of the Tangled Nature system, drawn by its distinct genomes."""

    visual = VisualSpec('histograms', {'color_by': 'trait', 'shown': ('distinct',)})


class ClimbingInteraction(Interaction):
    """Hill climbing of agents on an NK fitness landscape.

    At every step each climber flips one random bit of its genome and keeps the flip when it raises its fitness.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of climbers.

    Examples:
        >>> from skcomplex.systems.biology.evolution import ClimbingInteraction
        >>> ClimbingInteraction('climbing', target='climbers').target
        'climbers'
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the climbing rule of every climber.

        Args:
            context:
                The namespace of the climbers.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context.target, self.climb)]

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
        fitness = ClimberElement._get_nk_fitness(mutant, n, k, seed)
        if fitness > agent.fitness:
            agent.genome = float(mutant)
            agent.fitness = fitness


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
        target: Role,
        mutation: Annotated[float, RealSet.closed(0, 1)] = 0.1,
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
            [
                AdaptiveClimbingInteraction._get_random_fitness(int(genome), self.seed)
                for genome in np.ravel(strategies)
            ],
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
    def _get_random_fitness(genome: int, seed: int) -> float:
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
        target: Role,
        advantage: float = 0.0,
        mutation: Annotated[float, RealSet.closed(0, 1)] = 0.0,
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
        return (1 + system.get_parameter('advantage')) ** np.ravel(strategies)


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
        target: Role,
        k: Annotated[float, RealSet.closed_open(0, oo)] = 5.0,
        mu: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
        p_mut: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
        p_kill: Annotated[float, RealSet.closed_open(0, oo)] = 0.2,
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
        composition = system.get_composition('target', 'trait')
        size = sum(composition.values())
        row = self._get_couplings(self.seed, self.length)[int(agent.trait)]
        coupling = sum(row[int(np.asarray(genome))] * count for genome, count in composition.items())
        return float(expit(system.get_parameter('k') / size * coupling - system.get_parameter('mu') * size))

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
        return system.get_parameter('p_kill')

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
        flips = np.flatnonzero(system.rng.random(self.length) < system.get_parameter('p_mut'))
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import MoranProcessSystem
        >>> float(MoranProcessSystem(10, mutants=3).prepare().system_.elements[0].trait.sum())
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.mutants = mutants
        self.selection = selection
        self.mutation = mutation

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Moran process.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        traits = np.zeros(self.size)
        traits[: self.mutants] = 1.0
        self.system_ = System(
            self.label,
            elements=[
                IndividualElement('population', trait=traits, fitness=(1 + self.selection) ** traits, size=self.size),
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import TangledNatureSystem
        >>> TangledNatureSystem(20).prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.organisms = organisms
        self.k = k
        self.mu = mu
        self.p_mut = p_mut
        self.p_kill = p_kill
        self.length = length
        self.seed = seed
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Tangled Nature system.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
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
        self.system_ = System(
            self.label,
            elements=[TangledOrganismElement(ORGANISMS, trait=genome, size=self.organisms)],
            interactions=[life_cycle],
            solver=MesaSolver(phases=[PhaseSpec('generation', activation='visits', visits=1 / self.p_kill)]),
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import AdaptiveClimbingSystem
        >>> AdaptiveClimbingSystem(5).prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.u = u
        self.length = length
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a finite population.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        fitness = AdaptiveClimbingInteraction._get_random_fitness(0, self.seed)
        self.system_ = System(
            self.label,
            elements=[ClimbingIndividualElement('population', trait=0.0, fitness=fitness, size=self.size)],
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import NKLandscapeSystem
        >>> NKLandscapeSystem(5).prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.climbers = climbers
        self.n = n
        self.k = k
        self.seed = seed
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of agents on an NK landscape.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        genomes = np.random.default_rng(self.random_state).integers(0, 2**self.n, size=self.climbers)
        fitness = [ClimberElement._get_nk_fitness(int(genome), self.n, self.k, self.seed) for genome in genomes]
        climbers = ClimberElement(
            'climbers',
            genome=genomes.astype(float),
            fitness=np.asarray(fitness),
            n=float(self.n),
            k=float(self.k),
            seed=float(self.seed),
            size=self.climbers,
        )
        self.system_ = System(
            self.label,
            elements=[climbers],
            interactions=[ClimbingInteraction('climbing', target='climbers')],
        )
