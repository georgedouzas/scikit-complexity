"""Anomalous diffusion: random walks with power-law waiting times and flights, heterogeneity, and memory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, sqrt, vector

from ....core.components.elements import GroupElement
from ....core.components.interactions import JumpInteraction, MapInteraction, RateInteraction
from ....core.components.laws.utils import draw, normal
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver, FractionalEulerMaruyamaSolver, RK4Solver
from ..._catalog import register_system
from ._continuous_time_walker import ContinuousTimeWalkerElement
from ._fractional_walker import FractionalWalkerElement
from ._levy_flier import LevyFlierElement
from ._levy_walker import LevyWalkerElement
from ._transient_walker import TransientWalkerElement


def _get_flight(s: SimpleNamespace) -> object:
    """Return the position after a jump of power-law length `l0 r^(-1 / mu)` in a random direction."""
    length = draw('length', 'pareto', b=s.target.mu, scale=s.target.l0)
    steps = _direction('direction', len(list(s.target.x)))
    return vector(SR, [item + length * step for item, step in zip(s.target.x, steps, strict=True)])


def _direction(name: str, size: int) -> list[Expression]:
    """Return a unit vector of uniformly random direction drawn from standard normal components."""
    components = [normal(f'{name}{index}') for index in range(size)]
    length = sqrt(sum(component**2 for component in components))
    return [component / length for component in components]


@register_system(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'renewal process', 'subdiffusion', 'random walk'),
    references=('Argun et al. 2021', 'Metzler and Klafter 2000', 'Montroll and Weiss 1965', 'Bertin 2021'),
    defaults={'until': 1000.0, 'dt': 5.0},
)
def continuous_time_random_walk(
    n_walkers: int = 500,
    *,
    alpha: float = 0.5,
    tau0: float = 1.0,
    sigma: float = 1.0,
    n_dim: int = 2,
) -> System:
    """Return the continuous-time random walk, Gaussian jumps after waiting times with the tail `tau^(-1 - alpha)`.

    For `alpha < 1` the waiting times have no mean, so the number of jumps grows as `t^alpha` and the ensemble mean
    squared displacement grows as `t^alpha`, slower than Brownian motion. The walkers renew on their own clocks.

    Args:
        n_walkers:
            The number of independent walkers.

        alpha:
            The tail exponent of the waiting times.

        tau0:
            The shortest waiting time.

        sigma:
            The standard deviation of the jumps along every axis.

        n_dim:
            The number of dimensions.

    Returns:
        system:
            The system, with the positions `walkers.x` and the jump count `jumps.renewals`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import continuous_time_random_walk
        >>> continuous_time_random_walk(10, alpha=0.7).elements[0].values['alpha']
        0.7
    """
    walkers = GroupElement(
        'walkers',
        element=ContinuousTimeWalkerElement,
        size=n_walkers,
        values={'x': np.zeros((n_walkers, n_dim)), 'alpha': alpha, 'tau0': tau0, 'sigma': sigma},
    )
    jumps = JumpInteraction(
        'jumps',
        target='walkers',
        waiting=lambda s: draw('wait', 'pareto', b=s.target.alpha, scale=s.target.tau0),
        reset={
            'x': lambda s: [item + normal(f'jump{index}', 0, s.target.sigma) for index, item in enumerate(s.target.x)],
        },
    )
    return System(elements=[walkers], interactions=[jumps], space=ContinuousSpace(n_dim=n_dim))


@register_system(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'renewal process', 'superdiffusion', 'random walk'),
    references=('Argun et al. 2021', 'Zaburdaev, Denisov, and Klafter 2015', 'Shlesinger, West, and Klafter 1987'),
    defaults={'until': 500.0, 'dt': 2.5, 'solver': RK4Solver(dt=0.5)},
)
def levy_walk(
    n_walkers: int = 500,
    *,
    sigma: float = 1.5,
    tau0: float = 1.0,
    speed: float = 1.0,
    n_dim: int = 2,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Lévy walk, flights at constant speed in random directions for times with the tail `tau^(-1 - sigma)`.

    For `1 < sigma < 2` the flight times have a mean but no variance, so long flights dominate and the ensemble mean
    squared displacement grows as `t^(3 - sigma)`, faster than Brownian motion and slower than ballistic motion.

    Args:
        n_walkers:
            The number of independent walkers.

        sigma:
            The tail exponent of the flight times.

        tau0:
            The shortest flight time.

        speed:
            The speed of every flight.

        n_dim:
            The number of dimensions.

        random_state:
            The seed or generator of the directions of the first flights.

    Returns:
        system:
            The system, with the positions `walkers.x`, the velocities `walkers.v`, and the turn count `turns.renewals`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import levy_walk
        >>> levy_walk(10, sigma=1.2).elements[0].values['sigma']
        1.2
    """
    directions = np.random.default_rng(random_state).standard_normal((n_walkers, n_dim))
    velocities = speed * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    walkers = GroupElement(
        'walkers',
        element=LevyWalkerElement,
        size=n_walkers,
        values={'x': np.zeros((n_walkers, n_dim)), 'v': velocities, 'sigma': sigma, 'tau0': tau0, 'speed': speed},
    )
    flight = RateInteraction('flight', target='walkers', state='x', rate=lambda s: s.target.v)
    turns = JumpInteraction(
        'turns',
        target='walkers',
        waiting=lambda s: draw('flight', 'pareto', b=s.target.sigma, scale=s.target.tau0),
        reset={'v': lambda s: [s.target.speed * item for item in _direction('direction', len(list(s.target.v)))]},
    )
    return System(elements=[walkers], interactions=[flight, turns], space=ContinuousSpace(n_dim=n_dim))


@register_system(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'renewal process', 'heterogeneity', 'sde'),
    references=('Argun et al. 2021', 'Massignan et al. 2014'),
    defaults={'until': 500.0, 'dt': 2.5, 'solver': EulerMaruyamaSolver(dt=0.5)},
)
def annealed_transient_time_motion(
    n_walkers: int = 500,
    *,
    sigma: float = 0.5,
    gamma: float = 1.0,
    n_dim: int = 2,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return annealed transient time motion, Brownian walkers whose diffusion coefficient changes at random times.

    Every walker diffuses with a coefficient `D` of density `sigma D^(sigma - 1)` on `(0, 1)` and keeps it for an
    exponential time of mean `D^(-gamma)`, so slow walkers stay slow for long. For `sigma < gamma` the ensemble mean
    squared displacement grows as `t^(sigma / gamma)`.

    Args:
        n_walkers:
            The number of independent walkers.

        sigma:
            The exponent of the density of diffusion coefficients near zero.

        gamma:
            The exponent of the mean waiting time `D^(-gamma)`.

        n_dim:
            The number of dimensions.

        random_state:
            The seed or generator of the first diffusion coefficients.

    Returns:
        system:
            The system, with the positions `walkers.x`, the coefficients `walkers.D`, and the change count
            `changes.renewals`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import annealed_transient_time_motion
        >>> annealed_transient_time_motion(10, gamma=2.0).elements[0].values['gamma']
        2.0
    """
    coefficients = np.random.default_rng(random_state).beta(sigma, 1.0, n_walkers)
    walkers = GroupElement(
        'walkers',
        element=TransientWalkerElement,
        size=n_walkers,
        values={'x': np.zeros((n_walkers, n_dim)), 'D': coefficients, 'sigma': sigma, 'gamma': gamma},
    )
    noise = RateInteraction('noise', target='walkers', noise=lambda s: sqrt(2 * s.target.D))
    changes = JumpInteraction(
        'changes',
        target='walkers',
        waiting=lambda s: draw('wait', 'expon', scale=s.target.D ** (-s.target.gamma)),
        reset={'D': lambda s: draw('coefficient', 'beta', a=s.target.sigma, b=1)},
    )
    return System(elements=[walkers], interactions=[noise, changes], space=ContinuousSpace(n_dim=n_dim))


@register_system(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'fractional noise', 'long memory', 'sde'),
    references=('Argun et al. 2021', 'Mandelbrot and Van Ness 1968'),
    defaults={'until': 10.0, 'dt': 0.05, 'solver': FractionalEulerMaruyamaSolver(dt=0.01)},
)
def fractional_brownian_motion(
    n_walkers: int = 500,
    *,
    H: float = 0.3,
    D: float = 1.0,
    n_dim: int = 2,
) -> System:
    """Return fractional Brownian motion, `dx = sqrt(2 D) dB_H`, with `<x^2> = 2 D t^(2 H)` along every axis.

    The increments are correlated over the whole run: they anticorrelate for `H < 1/2`, which makes the walkers
    subdiffusive, and correlate for `H > 1/2`, which makes them superdiffusive.

    Args:
        n_walkers:
            The number of independent walkers.

        H:
            The Hurst index.

        D:
            The diffusion coefficient.

        n_dim:
            The number of dimensions.

    Returns:
        system:
            The system, with the positions `walkers.x`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import fractional_brownian_motion
        >>> fractional_brownian_motion(10, H=0.7).elements[0].values['H']
        0.7
    """
    walkers = GroupElement(
        'walkers',
        element=FractionalWalkerElement,
        size=n_walkers,
        values={'x': np.zeros((n_walkers, n_dim)), 'D': D, 'H': H},
    )
    noise = RateInteraction(
        'noise',
        target='walkers',
        noise=lambda s: sqrt(2 * s.target.D),
        hurst=lambda s: s.target.H,
    )
    return System(elements=[walkers], interactions=[noise], space=ContinuousSpace(n_dim=n_dim))


@register_system(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'superdiffusion', 'random walk', 'stochastic map'),
    references=('Bertin 2021', 'Shlesinger, Zaslavsky, and Frisch 1995'),
    defaults={'until': 200.0, 'dt': 1.0},
)
def levy_flight(
    n_walkers: int = 500,
    *,
    mu: float = 1.2,
    l0: float = 1.0,
    n_dim: int = 2,
) -> System:
    """Return the Lévy flight, a walk whose jumps have lengths with the tail `l^(-1 - mu)` in random directions.

    For `mu < 2` the jump lengths have no variance, so the walk is dominated by its longest jumps and the typical
    distance travelled after `n` steps grows as `n^(1 / mu)`, faster than the `n^(1/2)` of a random walk.

    Args:
        n_walkers:
            The number of independent walkers.

        mu:
            The tail exponent of the jump lengths.

        l0:
            The shortest jump length.

        n_dim:
            The number of dimensions.

    Returns:
        system:
            The system, with the positions `walkers.x`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import levy_flight
        >>> levy_flight(10, mu=1.5).elements[0].values['mu']
        1.5
    """
    walkers = GroupElement(
        'walkers',
        element=LevyFlierElement,
        size=n_walkers,
        values={'x': np.zeros((n_walkers, n_dim)), 'mu': mu, 'l0': l0},
    )
    flights = MapInteraction('flights', target='walkers', state='x', function=_get_flight)
    return System(elements=[walkers], interactions=[flights], space=ContinuousSpace(n_dim=n_dim))
