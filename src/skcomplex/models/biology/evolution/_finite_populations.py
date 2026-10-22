"""Evolution models: finite populations of individuals that reproduce, mutate, and climb fitness landscapes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import lru_cache
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.model import AgentType, Interaction, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

type Array = npt.NDArray[np.float64]

ORGANISMS = 'organisms'


@lru_cache(maxsize=16)
def _couplings(seed: int, length: int) -> Array:
    """Return the random couplings `J(S, S')` of the Tangled Nature model, uniform in [-1, 1] and 0 within a type."""
    size = 2**length
    couplings = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(size, size))
    np.fill_diagonal(couplings, 0.0)
    return couplings


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


def _mutate(genome: int, length: int, probability: float, rng: np.random.Generator) -> int:
    """Return a genome whose every locus flips with a probability."""
    flips = np.flatnonzero(rng.random(length) < probability)
    return int(genome ^ sum(1 << int(locus) for locus in flips))


class Individual(AgentType):
    """Individual of a finite population, with an integer trait or genome.

    Args:
        label:
            The label of the agent type or group.

        trait:
            The trait, a type index or the bits of a genome.

        fitness:
            The fitness of the trait.

    Examples:
        >>> from skcomplex.models.biology.evolution import Individual
        >>> Individual.group('population', 10, trait=0.0).size
        10
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'trait'})
    trait: float = state(default=0.0, domain=Domain.INTEGER)
    fitness: float = state(default=1.0, domain=Domain.NONNEGATIVE)


class MoranStep(Interaction):
    """One generation of the Moran process: as many birth and death events as there are individuals.

    At every event an individual reproduces with probability proportional to its fitness `(1 + selection)^trait`,
    its offspring mutates to the trait one above or one below with probability `mutation / 2` each, and it replaces an
    individual chosen uniformly at random, so the size of the population stays constant.

    Args:
        label:
            The label of the interaction.

        population:
            The path of the group of individuals.

        selection:
            The selective advantage of one unit of the trait.

        mutation:
            The probability that an offspring mutates.

    Examples:
        >>> from skcomplex.models.biology.evolution import MoranStep
        >>> MoranStep('moran', population='population', selection=0.1).selection
        0.1
    """

    population: str = role()
    selection: float = parameter(default=0.0, domain=Domain.REAL)
    mutation: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the generation rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.generation)]

    def generation(self, model: ModelView) -> None:
        """Run one birth and death event per individual.

        Args:
            model:
                The model.
        """
        individuals = model.agents(self.population)
        selection = model.parameter('selection')
        mutation = model.parameter('mutation')
        traits = np.asarray([individual.trait for individual in individuals], dtype=float)
        rng = model.rng
        for _ in range(len(traits)):
            weights = (1 + selection) ** traits
            parent = rng.choice(len(traits), p=weights / weights.sum())
            child = traits[parent]
            if rng.random() < mutation:
                child += 1 if rng.random() < 0.5 else -1  # noqa: PLR2004
            traits[rng.integers(len(traits))] = child
        for individual, trait in zip(individuals, traits, strict=True):
            individual.trait = trait
            individual.fitness = (1 + selection) ** trait


class TangledNatureStep(Interaction):
    """One generation of the Tangled Nature model, `N / p_kill` reproduction and killing attempts.

    At every attempt a random organism of type `S` reproduces with probability `1 / (1 + exp(-H(S)))`, where
    `H(S) = k / N sum_S' J(S, S') n(S') - mu N`, and is replaced by two offspring, whose genome bits flip with
    probability `p_mut` each. Then a random organism dies with probability `p_kill`. The couplings `J` are uniform in
    [-1, 1] between different types, 0 within a type, and fixed by `seed`.

    Args:
        label:
            The label of the interaction.

        organisms:
            The path of the group of organisms.

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

    Examples:
        >>> from skcomplex.models.biology.evolution import TangledNatureStep
        >>> TangledNatureStep('evolution', organisms='organisms').length
        8
    """

    organisms: str = role()
    k: float = parameter(default=5.0, domain=Domain.NONNEGATIVE)
    mu: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    p_mut: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    p_kill: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    length: int = 8
    seed: int = 0

    def _reproduce(self, model: ModelView, alive: list[AgentView], counts: Array, chosen: int) -> None:
        """Replace an organism by two offspring with probability `1 / (1 + exp(-H))`."""
        parent = alive[chosen]
        genome = int(parent.trait)
        size = len(alive)
        weight = model.parameter('k') / size * _couplings(self.seed, self.length)[genome] @ counts
        weight -= model.parameter('mu') * size
        if model.rng.random() >= 1 / (1 + np.exp(-weight)):
            return
        parent.remove()
        alive.pop(chosen)
        counts[genome] -= 1
        for _ in range(2):
            child = _mutate(genome, self.length, model.parameter('p_mut'), model.rng)
            alive.append(model.create(self.organisms, trait=float(child)))
            counts[child] += 1

    def laws(self, s: Symbols) -> list[Law]:
        """Return the generation rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.generation)]

    def generation(self, model: ModelView) -> None:
        """Run the reproduction and killing attempts of one generation.

        Args:
            model:
                The model.
        """
        alive = model.agents(self.organisms)
        counts = np.bincount([int(organism.trait) for organism in alive], minlength=2**self.length).astype(float)
        p_kill = model.parameter('p_kill')
        for _ in range(int(len(alive) / p_kill)):
            if not alive:
                return
            self._reproduce(model, alive, counts, int(model.rng.integers(len(alive))))
            if alive and model.rng.random() < p_kill:
                victim = int(model.rng.integers(len(alive)))
                counts[int(alive[victim].trait)] -= 1
                alive.pop(victim).remove()


class AdaptiveClimbing(Interaction):
    """One generation of a population that climbs a random fitness landscape as a whole, under strong selection.

    All individuals share one genome. Each mutates one random locus with probability `u`, to a genome of random fitness
    in [0, 1]. If a mutant is fitter, the whole population takes the fittest mutant. If every individual mutated to a
    less fit genome, the population escapes to the fittest of them. Otherwise it stays.

    Args:
        label:
            The label of the interaction.

        population:
            The path of the group of individuals.

        u:
            The probability that an individual mutates in a generation.

        length:
            The number of loci of a genome.

        seed:
            The seed of the fitness landscape.

    Examples:
        >>> from skcomplex.models.biology.evolution import AdaptiveClimbing
        >>> AdaptiveClimbing('climbing', population='population', u=0.1).u
        0.1
    """

    population: str = role()
    u: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    length: int = 20
    seed: int = 0

    def laws(self, s: Symbols) -> list[Law]:
        """Return the generation rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.generation)]

    def generation(self, model: ModelView) -> None:
        """Mutate, then move the population to a fitter mutant or let it escape.

        Args:
            model:
                The model.
        """
        individuals = model.agents(self.population)
        genome = int(individuals[0].trait)
        current = _random_fitness(genome, self.seed)
        rng = model.rng
        mutated = rng.random(len(individuals)) < model.parameter('u')
        mutants = [genome ^ (1 << int(rng.integers(self.length))) for _ in range(int(mutated.sum()))]
        if not mutants:
            return
        best = max(mutants, key=lambda mutant: _random_fitness(mutant, self.seed))
        if _random_fitness(best, self.seed) <= current and not mutated.all():
            return
        for individual in individuals:
            individual.trait = float(best)
            individual.fitness = _random_fitness(best, self.seed)


class Climber(AgentType):
    """Agent that walks an NK fitness landscape, flipping one random bit of its genome when that raises its fitness.

    The fitness of a genome of `n` bits is the mean of the contributions of its bits, each a random function of the bit
    and the next `k` bits, fixed by `seed`.

    Args:
        label:
            The label of the agent type or group.

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
        >>> from skcomplex.models.biology.evolution import Climber
        >>> Climber.group('climbers', 10).size
        10
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'fitness', 'color_range': (0.0, 1.0)})
    genome: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    fitness: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    n: float = parameter(default=10.0, domain=Domain.POSITIVE)
    k: float = parameter(default=2.0, domain=Domain.NONNEGATIVE_INTEGER)
    seed: float = parameter(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the climbing rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.climb)]

    def climb(self, agent: AgentView, model: ModelView) -> None:
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
        components=[Individual.group('population', size, trait=traits, fitness=(1 + selection) ** traits)],
        interactions=[MoranStep('moran', population='population', selection=selection, mutation=mutation)],
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
    punctuated equilibrium. The trait of an organism holds the bits of its genome.

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
        >>> tangled_nature(20).elements[0].size
        20
    """
    genome = float(np.random.default_rng(random_state).integers(2**length))
    step = TangledNatureStep(
        'evolution',
        organisms=ORGANISMS,
        k=k,
        mu=mu,
        p_mut=p_mut,
        p_kill=p_kill,
        length=length,
        seed=seed,
    )
    visual = VisualSpec('traits', {'color_by': 'trait', 'shown': ('distinct',)})
    return System(components=[Individual.group(ORGANISMS, organisms, trait=genome, visual=visual)], interactions=[step])


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
        >>> adaptive_climbing(5).elements[0].size
        5
    """
    fitness = _random_fitness(0, seed)
    return System(
        components=[
            Individual.group(
                'population',
                size,
                trait=0.0,
                fitness=fitness,
                visual=VisualSpec('agents', {'color_by': 'fitness'}),
            ),
        ],
        interactions=[AdaptiveClimbing('climbing', population='population', u=u, length=length, seed=seed)],
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
        >>> nk_landscape(5).elements[0].size
        5
    """
    genomes = np.random.default_rng(random_state).integers(0, 2**n, size=climbers)
    fitness = [_nk_fitness(int(genome), n, k, seed) for genome in genomes]
    group = Climber.group(
        'climbers',
        climbers,
        genome=genomes.astype(float),
        fitness=np.asarray(fitness),
        n=float(n),
        k=float(k),
        seed=float(seed),
    )
    return System(components=[group])
