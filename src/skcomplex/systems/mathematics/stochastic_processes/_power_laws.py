"""Stochastic process systems: power laws from exponential growth observed at random times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.components.elements import GroupElement
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ..._catalog import register_system
from ._observed_growth import ObservedGrowthElement


@register_system(
    field='mathematics/stochastic_processes',
    tags=('power law', 'renewal', 'exponential growth', 'random observation'),
    references=('Reed and Hughes 2002', 'Thurner et al. 2018'),
    defaults={'until': 20.0, 'dt': 1.0, 'solver': RK4Solver(dt=0.01)},
)
def exponential_growth_at_random_times(
    n_samples: int = 2000,
    *,
    mu: float = 1.0,
    lam: float = 1.5,
    x0: float = 1.0,
) -> System:
    """Return exponential growth observed at exponentially distributed times, whose observed values follow a power law.

    Every sample grows as `x0 exp(mu t)` and is observed and restarted after a waiting time drawn from the
    exponential distribution of rate `lam`. The observed values, and the values of the samples at any later time,
    follow the power law `p(x) ~ x^(-lam / mu - 1)` (Thurner et al. 2018, section 3.3.6.1, after Reed and Hughes 2002).

    Args:
        n_samples:
            The number of independent samples.

        mu:
            The growth rate.

        lam:
            The rate of the observation times.

        x0:
            The value every growth starts from.

    Returns:
        system:
            The system, with the group `samples` of state `x` and the observations `samples.observations`.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import exponential_growth_at_random_times
        >>> exponential_growth_at_random_times(10).elements[0].prepare().size_
        10
    """
    return System(
        elements=[
            GroupElement(
                'samples',
                element=ObservedGrowthElement,
                size=n_samples,
                values={'x': x0, 'x0': x0, 'mu': mu, 'lam': lam},
            ),
        ],
    )
