"""Build the ready-made ecology models from populations and their interactions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...core import Domain, state
from ...model import Element, Interaction, System, Update, lag
from ...runtime import register_model
from ._populations import ExponentialGrowth, LogisticGrowth, Mortality, Population, Predation


@register_model(
    field='ecology',
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
        >>> from skcomplex.models.ecology import exponential_growth
        >>> exponential_growth(r=0.2).interactions[0].r
        0.2
    """
    return System(
        elements=[Population('population', n=n)],
        interactions=[ExponentialGrowth('growth', target='population', r=r)],
    )


@register_model(
    field='ecology',
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
        >>> from skcomplex.models.ecology import logistic_growth
        >>> logistic_growth(K=500.0).interactions[0].K
        500.0
    """
    return System(
        elements=[Population('population', n=n)],
        interactions=[LogisticGrowth('growth', target='population', r=r, K=K)],
    )


@register_model(
    field='ecology',
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
        >>> from skcomplex.models.ecology import lotka_volterra
        >>> [element.label for element in lotka_volterra().elements]
        ['prey', 'predators']
    """
    growth: Interaction = (
        ExponentialGrowth('growth', target='prey', r=r)
        if K is None
        else LogisticGrowth('growth', target='prey', r=r, K=K)
    )
    return System(
        elements=[Population('prey', n=prey), Population('predators', n=predators)],
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
    field='ecology',
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
        >>> from skcomplex.models.ecology import fibonacci_rabbits
        >>> fibonacci_rabbits().elements[0].pairs
        1.0
    """
    return System(elements=[Rabbits('rabbits', pairs=pairs, pairs_lag1=previous)])
