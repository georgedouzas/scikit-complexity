"""Evolution models: finite populations of individuals that reproduce, mutate, and climb fitness landscapes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....specs import PhaseSpec, PhasesScheduleSpec, VisualSpec
from ..._catalog import register_model
from ._adaptive_climbing import AdaptiveClimbingInteraction
from ._climber import ClimberElement
from ._individual import IndividualElement
from ._moran_step import MoranStepInteraction
from ._tangled_nature_life_cycle import TangledNatureLifeCycleInteraction

ORGANISMS = 'organisms'


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
        >>> tangled_nature(20).elements[0].prepare().size_
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
        >>> adaptive_climbing(5).elements[0].prepare().size_
        5
    """
    fitness = AdaptiveClimbingInteraction._random_fitness(0, seed)
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
        >>> nk_landscape(5).elements[0].prepare().size_
        5
    """
    genomes = np.random.default_rng(random_state).integers(0, 2**n, size=climbers)
    fitness = [ClimberElement._nk_fitness(int(genome), n, k, seed) for genome in genomes]
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
