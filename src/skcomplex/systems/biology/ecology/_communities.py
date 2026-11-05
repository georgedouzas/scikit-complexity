"""Ecology systems: competition, mutualism, and food webs of several species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

from ....core.components.interactions import BaseInteraction
from ....core.components.system import System
from ..._catalog import register_system
from ..common.populations import LogisticGrowthInteraction, MortalityInteraction, PopulationElement
from ._competition import CompetitionInteraction
from ._mutualism import MutualismInteraction
from ._predation_interaction import PredationInteraction

WEB_POPULATIONS = {'frogs': 50.0, 'mice': 50.0, 'snakes': 10.0, 'hawks': 5.0}


WEB_GROWTH = {'frogs': (1.0, 100.0), 'mice': (1.2, 100.0)}


WEB_MORTALITY = {'snakes': 0.3, 'hawks': 0.2}


WEB_LINKS = (
    ('snakes', 'frogs', 0.02, 0.01),
    ('snakes', 'mice', 0.01, 0.005),
    ('hawks', 'snakes', 0.01, 0.005),
    ('hawks', 'mice', 0.002, 0.001),
)


@register_system(
    field='biology/ecology',
    tags=('competition', 'ode', 'exclusion'),
    references=('Lotka 1925', 'Volterra 1926', 'Argun et al. 2021'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def competition(
    first: float = 10.0,
    second: float = 10.0,
    *,
    r1: float = 1.0,
    r2: float = 0.8,
    K1: float = 100.0,
    K2: float = 80.0,
    alpha12: float = 0.5,
    alpha21: float = 0.6,
) -> System:
    """Return two species that compete for the same resources, which coexist or exclude each other.

    The species coexist at `n1 = (K1 - alpha12 K2) / (1 - alpha12 alpha21)` and
    `n2 = (K2 - alpha21 K1) / (1 - alpha12 alpha21)` when both are positive and `alpha12 alpha21 < 1`. Otherwise one
    species excludes the other.

    Args:
        first:
            The initial size of the first population.

        second:
            The initial size of the second population.

        r1:
            The growth rate of the first population.

        r2:
            The growth rate of the second population.

        K1:
            The carrying capacity of the first population.

        K2:
            The carrying capacity of the second population.

        alpha12:
            The effect of one member of the second population on the first, relative to one of its own.

        alpha21:
            The effect of one member of the first population on the second, relative to one of its own.

    Returns:
        system:
            The system, with the populations `first` and `second`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import competition
        >>> competition(alpha12=0.2).interactions[0].alpha12
        0.2
    """
    return System(
        elements=[PopulationElement('first', n=first), PopulationElement('second', n=second)],
        interactions=[
            CompetitionInteraction(
                'competition',
                first='first',
                second='second',
                r1=r1,
                r2=r2,
                K1=K1,
                K2=K2,
                alpha12=alpha12,
                alpha21=alpha21,
            ),
        ],
    )


@register_system(
    field='biology/ecology',
    tags=('mutualism', 'ode', 'handling time'),
    references=('May 1981', 'Wright 1989', 'Argun et al. 2021'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def mutualism(
    first: float = 10.0,
    second: float = 10.0,
    *,
    r1: float = 1.0,
    r2: float = 1.0,
    K1: float = 50.0,
    K2: float = 50.0,
    a: float = 0.1,
    b: float = 0.5,
    handling: float = 0.5,
) -> System:
    """Return two species that grow logistically and feed each other, above their carrying capacities.

    With a positive handling time the benefit saturates and the populations settle above their capacities. Without it,
    strong mutualism grows without bound.

    Args:
        first:
            The initial size of the first population.

        second:
            The initial size of the second population.

        r1:
            The growth rate of the first population.

        r2:
            The growth rate of the second population.

        K1:
            The carrying capacity of the first population alone.

        K2:
            The carrying capacity of the second population alone.

        a:
            The effective search rate.

        b:
            The rate of encounters.

        handling:
            The handling time of the resource of the partner, 0 for type I mutualism.

    Returns:
        system:
            The system, with the populations `first` and `second`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import mutualism
        >>> [interaction.label for interaction in mutualism().interactions]
        ['growth_first', 'growth_second', 'mutualism']
    """
    return System(
        elements=[PopulationElement('first', n=first), PopulationElement('second', n=second)],
        interactions=[
            LogisticGrowthInteraction('growth_first', target='first', r=r1, K=K1),
            LogisticGrowthInteraction('growth_second', target='second', r=r2, K=K2),
            MutualismInteraction('mutualism', first='first', second='second', a=a, b=b, handling=handling),
        ],
    )


@register_system(
    field='biology/ecology',
    tags=('food web', 'predator-prey', 'ode'),
    references=('Argun et al. 2021', 'Pimm 1982'),
    defaults={'until': 100.0, 'dt': 0.5},
)
def food_webs(
    populations: Mapping[str, float] | None = None,
    *,
    growth: Mapping[str, tuple[float, float]] | None = None,
    mortality: Mapping[str, float] | None = None,
    links: Sequence[tuple[str, str, float, float]] | None = None,
) -> System:
    """Return a food web where basal species grow logistically and consumers live on Lotka-Volterra predation.

    The default web has frogs and mice at its base, snakes that eat frogs and mice, and hawks that eat snakes and
    mice. Its four species coexist at a stable equilibrium.

    Args:
        populations:
            The initial size of every species by name, or `None` for the default web.

        growth:
            The growth rate and the carrying capacity of every basal species, or `None` for the default web.

        mortality:
            The death rate of every consumer, or `None` for the default web.

        links:
            The links `(predator, prey, a, b)`, with the attack rate `a` and the conversion rate `b`, or `None` for the
            default web.

    Returns:
        system:
            The system, with one population per species.

    Examples:
        >>> from skcomplex.systems.biology.ecology import food_webs
        >>> [element.label for element in food_webs().elements]
        ['frogs', 'mice', 'snakes', 'hawks']
    """
    populations = WEB_POPULATIONS if populations is None else populations
    growth = WEB_GROWTH if growth is None else growth
    mortality = WEB_MORTALITY if mortality is None else mortality
    links = WEB_LINKS if links is None else links
    interactions: list[BaseInteraction] = [
        LogisticGrowthInteraction(f'growth_{name}', target=name, r=r, K=K) for name, (r, K) in growth.items()
    ]
    interactions += [MortalityInteraction(f'death_{name}', target=name, d=d) for name, d in mortality.items()]
    interactions += [
        PredationInteraction(f'{predator}_eat_{prey}', predator=predator, prey=prey, a=a, b=b)
        for predator, prey, a, b in links
    ]
    return System(
        elements=[PopulationElement(name, n=n) for name, n in populations.items()],
        interactions=interactions,
    )
