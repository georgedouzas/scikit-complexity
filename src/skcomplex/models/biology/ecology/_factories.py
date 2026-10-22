"""Ecology models: population growth, predation, and invasion fronts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Update, lag
from ....core.runtime import register_model
from ....core.system import System
from ._populations import (
    DelayedLogisticGrowth,
    ExponentialGrowth,
    LogisticGrowth,
    Mortality,
    Population,
    Predation,
    SeasonalLogisticGrowth,
)


@register_model(
    field='biology/ecology',
    tags=('population', 'ode'),
    references=('Malthus 1798',),
    defaults={'until': 20.0, 'dt': 0.1},
)
def exponential_growth(n: float = 10.0, r: float = 0.1) -> System:
    """Return a population that grows or declines at a constant per-capita rate.

    Args:
        n:
            The initial size.

        r:
            The per-capita growth rate.

    Returns:
        system:
            The model, with the population `population`.

    Examples:
        >>> from skcomplex.models.biology.ecology import exponential_growth
        >>> exponential_growth(r=0.2).interactions[0].r
        0.2
    """
    return System(
        components=[Population('population', n=n)],
        interactions=[ExponentialGrowth('growth', target='population', r=r)],
    )


@register_model(
    field='biology/ecology',
    tags=('population', 'ode'),
    references=('Verhulst 1838',),
    defaults={'until': 20.0, 'dt': 0.1},
)
def logistic_growth(n: float = 10.0, r: float = 1.0, K: float = 100.0) -> System:
    """Return a population that grows logistically towards its carrying capacity.

    Args:
        n:
            The initial size.

        r:
            The intrinsic growth rate.

        K:
            The carrying capacity.

    Returns:
        system:
            The model, with the population `population`.

    Examples:
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> logistic_growth(K=500.0).interactions[0].K
        500.0
    """
    return System(
        components=[Population('population', n=n)],
        interactions=[LogisticGrowth('growth', target='population', r=r, K=K)],
    )


@register_model(
    field='biology/ecology',
    tags=('population', 'ode', 'forcing', 'periodic'),
    references=('Coleman 1979', 'Trzaska 2023'),
    defaults={'until': 20.0, 'dt': 0.05},
)
def varying_capacity_logistic(
    n: float = 10.0,
    *,
    r: float = 2.0,
    K: float = 100.0,
    amplitude: float = 0.5,
    period: float = 1.0,
) -> System:
    """Return logistic growth under a seasonal carrying capacity, which settles on a cycle of the season's period.

    Whatever its initial size, the population tends to a unique periodic solution, which lags behind the capacity and
    varies less than it.

    Args:
        n:
            The initial size.

        r:
            The intrinsic growth rate.

        K:
            The mean carrying capacity.

        amplitude:
            The relative amplitude of the seasonal variation of the capacity, below 1.

        period:
            The period of the seasons.

    Returns:
        system:
            The model, with the population `population`.

    Examples:
        >>> from skcomplex.models.biology.ecology import varying_capacity_logistic
        >>> varying_capacity_logistic(amplitude=0.2).interactions[0].amplitude
        0.2
    """
    return System(
        components=[Population('population', n=n)],
        interactions=[
            SeasonalLogisticGrowth('growth', target='population', r=r, K=K, amplitude=amplitude, period=period),
        ],
    )


@register_model(
    field='biology/ecology',
    tags=('population', 'delay', 'oscillation'),
    references=('Hutchinson 1948', 'Erneux 2009'),
    defaults={'until': 100.0, 'dt': 0.1},
)
def hutchinson(n: float = 50.0, *, r: float = 1.0, K: float = 100.0, tau: float = 2.0) -> System:
    """Return Hutchinson's delayed logistic equation, which oscillates around `K` when `r * tau` exceeds π/2.

    The history before the start is the initial size.

    Args:
        n:
            The initial size, also its history.

        r:
            The per-capita growth rate at low density.

        K:
            The carrying capacity.

        tau:
            The delay of the density feedback.

    Returns:
        system:
            The model, with the population `population`.

    Examples:
        >>> from skcomplex.models.biology.ecology import hutchinson
        >>> hutchinson(tau=1.0).interactions[0].tau
        1.0
    """
    return System(
        components=[Population('population', n=n)],
        interactions=[DelayedLogisticGrowth('growth', target='population', r=r, K=K, tau=tau)],
    )


@register_model(
    field='biology/ecology',
    tags=('predator-prey', 'ode', 'oscillation'),
    references=('Lotka 1925', 'Volterra 1926'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def lotka_volterra(
    prey: float = 40.0,
    predators: float = 9.0,
    *,
    r: float = 1.1,
    a: float = 0.4,
    b: float = 0.1,
    d: float = 0.4,
    K: float | None = None,
) -> System:
    """Return prey and predators whose populations oscillate through predation.

    Args:
        prey:
            The initial number of prey.

        predators:
            The initial number of predators.

        r:
            The growth rate of the prey.

        a:
            The rate at which predation removes prey.

        b:
            The rate at which predation feeds predators.

        d:
            The mortality rate of the predators.

        K:
            The carrying capacity of the prey, or `None` for exponential growth of the prey.

    Returns:
        system:
            The model, with the populations `prey` and `predators`.

    Examples:
        >>> from skcomplex.models.biology.ecology import lotka_volterra
        >>> [element.label for element in lotka_volterra().elements]
        ['prey', 'predators']
    """
    growth: Interaction = (
        ExponentialGrowth('growth', target='prey', r=r)
        if K is None
        else LogisticGrowth('growth', target='prey', r=r, K=K)
    )
    return System(
        components=[Population('prey', n=prey), Population('predators', n=predators)],
        interactions=[
            growth,
            Predation('predation', predator='predators', prey='prey', a=a, b=b),
            Mortality('death', target='predators', d=d),
        ],
    )


Rabbits = Element.define(
    'Rabbits',
    states={'pairs': state(default=1.0, domain=Domain.NONNEGATIVE, lags=1, initial=(0.0,))},
    laws=lambda s: [Update(s.pairs, s.pairs + lag(s.pairs, 1))],
)


@register_model(
    field='biology/ecology',
    tags=('population', 'discrete map', 'second order', 'age structure'),
    references=('Fibonacci 1202',),
    defaults={'until': 20, 'dt': 1},
)
def fibonacci_rabbits(pairs: float = 1.0, previous: float = 0.0) -> System:
    """Return Fibonacci's rabbits, `N_{t+1} = N_t + N_{t-1}`, where every pair older than one month breeds a new pair.

    The previous population counts the pairs old enough to breed, so the recurrence is an age structure of two classes.

    Args:
        pairs:
            The number of pairs this month.

        previous:
            The number of pairs last month.

    Returns:
        system:
            The model, with the population `rabbits.pairs` and last month's `rabbits.pairs_lag1`.

    Examples:
        >>> from skcomplex.models.biology.ecology import fibonacci_rabbits
        >>> fibonacci_rabbits().elements[0].pairs
        1.0
    """
    return System(components=[Rabbits('rabbits', pairs=pairs, pairs_lag1=previous)])
