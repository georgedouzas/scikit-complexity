"""Dynamical systems systems: ensembles of chaotic maps, their invariant densities, and chaotic walks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import GroupElement
from ....core.components.system import System
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ._cubic_walk import CubicWalkElement
from ._doubling_map import DOUBLING_DENOMINATOR, DoublingMapElement
from ._skew_tent_map import SkewTentMapElement


def _square_displacement(member: SimpleNamespace) -> Expression:
    """Return the squared position of a walker."""
    return member.y**2


def _draw_cloud(
    size: int,
    start: float,
    spread: float,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    """Return points drawn uniformly from the interval `[start, start + spread]`."""
    return np.random.default_rng(random_state).uniform(start, start + spread, size)


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'random walk', 'diffusion'),
    references=('Bertin 2021',),
    defaults={'until': 200, 'dt': 1},
    observables=lambda system: [
        ObservableSpec('displacement', group='walkers', variable='y'),
        ObservableSpec('square_displacement', _square_displacement, group='walkers'),
    ],
)
def cubic_chaotic_map(n_walkers: int = 100, *, lam: float = 15.0) -> System:
    """Return Bertin's cubic chaotic map and the chaotic walk it drives, which spreads like a random walk.

    Every walker follows the map `x' = λ x (x - 1/2)(x - 1) + x` on the unit interval and moves by the step
    `y' = y + 2x - 1`. The fixed point `x = 1/2` is stable for `λ < 8`, a period-two cycle follows, and the map is
    chaotic at `λ = 15`. The walkers start at `y = 0` with values of `x` spread regularly over the unit interval, so
    the mean position stays near 0 while the mean square position grows linearly in time, and the invariant density of
    `x` is not uniform.

    Args:
        n_walkers:
            The number of walkers of the ensemble.

        lam:
            The parameter `λ` of the map, between 0 and 16.

    Returns:
        system:
            The system, with the states `walkers.x` and `walkers.y` and the observables `displacement`, the mean
            position, and `square_displacement`, the mean square position.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import cubic_chaotic_map
        >>> cubic_chaotic_map(4, lam=10.0).elements[0].prepare().size_
        4
    """
    start = (np.arange(n_walkers) + 1 / 4) / n_walkers
    walkers = GroupElement(
        'walkers',
        element=CubicWalkElement,
        size=n_walkers,
        values={'x': start, 'y': 0.0, 'lam': lam},
    )
    return System(
        elements=[walkers],
    )


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'piecewise linear', 'invariant density'),
    references=('Bertin 2021',),
    defaults={'until': 50, 'dt': 1},
)
def skew_tent_map(
    n_points: int = 100,
    *,
    a: float = 0.3,
    x: float = 0.4,
    spread: float = 0.01,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return an ensemble of the skew tent map, whose density relaxes to the uniform invariant density for any `a`.

    The map is `x' = x / a` for `x < a` and `x' = (1 - x) / (1 - a)` otherwise. Every value has one preimage on each
    branch, with slopes `1 / a` and `1 / (1 - a)`, so the uniform density is invariant. The points start in a narrow
    interval and spread over the unit interval until their histogram is flat. For `a = 1/2` both slopes are 2, and in
    floating point every orbit reaches 0 within about 55 steps, so `a` should differ from 1/2.

    Args:
        n_points:
            The number of points of the ensemble.

        a:
            The position of the top of the tent, between 0 and 1.

        x:
            The left end of the interval of initial points.

        spread:
            The width of the interval of initial points.

        random_state:
            The seed or generator of the initial points.

    Returns:
        system:
            The system, with the states `points.x`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import skew_tent_map
        >>> skew_tent_map(10, a=0.7).elements[0].values['a']
        0.7
    """
    start = _draw_cloud(n_points, x, spread, random_state)
    return System(
        elements=[GroupElement('points', element=SkewTentMapElement, size=n_points, values={'x': start, 'a': a})],
    )


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'bernoulli shift', 'invariant density'),
    references=('Bertin 2021',),
    defaults={'until': 50, 'dt': 1},
)
def doubling_map(
    n_points: int = 100,
    *,
    x: float = 0.4,
    spread: float = 0.01,
    q: float = DOUBLING_DENOMINATOR,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return an ensemble of the doubling map `x' = 2x (mod 1)`, iterated exactly on fractions with an odd denominator.

    In floating point `2x (mod 1)` shifts out one binary digit per step, and every orbit reaches 0 within 53 steps.
    The system keeps every point on the fractions `k / q` of an odd denominator `q` and applies the map to them exactly,
    `k' = 2k (mod q)`, through the update `x' = (2 round(q x) mod q) / q`. The orbits are the exact orbits of these
    fractions. The default `q` is a prime below `2^50` of which 2 is a primitive root, so every orbit except that of 0
    has the period `q - 1`, about `10^15`. The points start in a narrow interval and spread until their histogram
    matches the uniform invariant density.

    Args:
        n_points:
            The number of points of the ensemble.

        x:
            The left end of the interval of initial points, which are rounded to the nearest fraction `k / q`.

        spread:
            The width of the interval of initial points.

        q:
            The odd denominator of the fractions the points stay on, an integer below `2^51`.

        random_state:
            The seed or generator of the initial points.

    Returns:
        system:
            The system, with the states `points.x`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import doubling_map
        >>> float(doubling_map(10, q=101.0).elements[0].values['x'][0] * 101)
        41.0
    """
    start = np.floor(q * _draw_cloud(n_points, x, spread, random_state) + 1 / 2) / q
    return System(
        elements=[GroupElement('points', element=DoublingMapElement, size=n_points, values={'x': start, 'q': q})],
    )
