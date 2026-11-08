"""Dynamical systems of chaotic ensembles.

The module covers ensembles of chaotic maps, their invariant densities, and chaotic walks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, floor, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import mod, where
from ....core.components.system import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


class CubicWalkElement(Element):
    """Walker driven by Bertin's cubic chaotic map `x' = lam x (x - 1/2)(x - 1) + x`.

    The walker steps by `2 x - 1`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value of the map, between 0 and 1.

        y:
            The initial position of the walker.

        lam:
            The nonlinearity of the map.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CubicWalkElement
        >>> CubicWalkElement('walker', lam=8.0).lam
        8.0
    """

    law_classes = (UpdateLaw,)

    visual = VisualSpec('points', {'position': 'y', 'states': ()})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.3,
        y: Annotated[object, StateSpec(role='position')] = 0.0,
        lam: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 15.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.lam = lam

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the next value of the map and the next position.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [
            UpdateLaw(s.x, s.lam * s.x * (s.x - 1 / 2) * (s.x - 1) + s.x),
            UpdateLaw(s.y, s.y + 2 * s.x - 1),
        ]


DOUBLING_DENOMINATOR = 1125899906840747.0


class DoublingMapElement(Element):
    """Doubling map `x' = 2 x (mod 1)` on the grid of the fractions of denominator `q`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, between 0 and 1.

        q:
            The denominator of the grid.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import DoublingMapElement
        >>> DoublingMapElement('point', q=1024.0).q
        1024.0
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.4,
        q: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = DOUBLING_DENOMINATOR,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.q = q

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the next value, rounded to the grid.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [UpdateLaw(s.x, mod(2 * floor(s.q * s.x + 1 / 2), s.q) / s.q)]


class SkewTentMapElement(Element):
    """Skew tent map `x' = x / a` below the peak `a` and `(1 - x) / (1 - a)` above it.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, between 0 and 1.

        a:
            The position of the peak, between 0 and 1.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SkewTentMapElement
        >>> SkewTentMapElement('point', a=0.5).a
        0.5
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.4,
        a: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.a = a

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the next value.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [UpdateLaw(s.x, where(s.x < s.a, s.x / s.a, (1 - s.x) / (1 - s.a)))]


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


class CubicChaoticMapSystem(BaseSystem):
    """Bertin's cubic chaotic map and the chaotic walk it drives.

    The walk spreads like a random walk.

    Every walker follows the map `x' = λ x (x - 1/2)(x - 1) + x` on the unit interval and moves by the step `y' = y + 2x
    - 1`. The fixed point `x = 1/2` is stable for `λ < 8`, a period-two cycle follows, and the map is chaotic at `λ =
    15`. The walkers start at `y = 0` with values of `x` spread regularly over the unit interval, so the mean position
    stays near 0 while the mean square position grows linearly in time, and the invariant density of `x` is not uniform.
    The system has the states `walkers.x` and `walkers.y` and the observables `displacement`, the mean position, and
    `square_displacement`, the mean square position.

    Args:
        n_walkers:
            The number of walkers of the ensemble.

        lam:
            The parameter `λ` of the map, between 0 and 16.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CubicChaoticMapSystem
        >>> CubicChaoticMapSystem(4, lam=10.0).prepare().system_.elements[0].prepare().size_
        4
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'random walk', 'diffusion')
    references = ('Bertin 2021',)
    until = 200
    dt = 1
    observables = (
        ObservableSpec('displacement', group='walkers', variable='y'),
        ObservableSpec('square_displacement', _square_displacement, group='walkers'),
    )

    def __init__(
        self,
        n_walkers: int = 100,
        *,
        lam: float = 15.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_walkers = n_walkers
        self.lam = lam

    def _build_system(self) -> System:
        """Return Bertin's cubic chaotic map and the chaotic walk it drives.

        The walk spreads like a random walk.
        """
        start = (np.arange(self.n_walkers) + 1 / 4) / self.n_walkers
        walkers = GroupElement(
            'walkers',
            element=CubicWalkElement,
            size=self.n_walkers,
            values={'x': start, 'y': 0.0, 'lam': self.lam},
        )
        return System(
            self.label,
            elements=[walkers],
        )


class SkewTentMapSystem(BaseSystem):
    """An ensemble of the skew tent map.

    Its density relaxes to the uniform invariant density for any `a`.

    The map is `x' = x / a` for `x < a` and `x' = (1 - x) / (1 - a)` otherwise. Every value has one preimage on each
    branch, with slopes `1 / a` and `1 / (1 - a)`, so the uniform density is invariant. The points start in a narrow
    interval and spread over the unit interval until their histogram is flat. For `a = 1/2` both slopes are 2, and in
    floating point every orbit reaches 0 within about 55 steps, so `a` should differ from 1/2. The system has the states
    `points.x`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SkewTentMapSystem
        >>> SkewTentMapSystem(10, a=0.7).prepare().system_.elements[0].values['a']
        0.7
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'piecewise linear', 'invariant density')
    references = ('Bertin 2021',)
    until = 50
    dt = 1

    def __init__(
        self,
        n_points: int = 100,
        *,
        a: float = 0.3,
        x: float = 0.4,
        spread: float = 0.01,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_points = n_points
        self.a = a
        self.x = x
        self.spread = spread
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        start = _draw_cloud(self.n_points, self.x, self.spread, self.random_state)
        return System(
            self.label,
            elements=[
                GroupElement(
                    'points',
                    element=SkewTentMapElement,
                    size=self.n_points,
                    values={'x': start, 'a': self.a},
                ),
            ],
        )


class DoublingMapSystem(BaseSystem):
    """An ensemble of the doubling map `x' = 2x (mod 1)`.

    The map is iterated exactly on fractions with an odd denominator.

    In floating point `2x (mod 1)` shifts out one binary digit per step, and every orbit reaches 0 within 53 steps. The
    system keeps every point on the fractions `k / q` of an odd denominator `q` and applies the map to them exactly, `k'
    = 2k (mod q)`, through the update `x' = (2 round(q x) mod q) / q`. The orbits are the exact orbits of these
    fractions. The default `q` is a prime below `2^50` of which 2 is a primitive root, so every orbit except that of 0
    has the period `q - 1`, about `10^15`. The points start in a narrow interval and spread until their histogram
    matches the uniform invariant density. The system has the states `points.x`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import DoublingMapSystem
        >>> float(DoublingMapSystem(10, q=101.0).prepare().system_.elements[0].values['x'][0] * 101)
        41.0
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'bernoulli shift', 'invariant density')
    references = ('Bertin 2021',)
    until = 50
    dt = 1

    def __init__(
        self,
        n_points: int = 100,
        *,
        x: float = 0.4,
        spread: float = 0.01,
        q: float = DOUBLING_DENOMINATOR,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_points = n_points
        self.x = x
        self.spread = spread
        self.q = q
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        start = np.floor(self.q * _draw_cloud(self.n_points, self.x, self.spread, self.random_state) + 1 / 2) / self.q
        return System(
            self.label,
            elements=[
                GroupElement(
                    'points',
                    element=DoublingMapElement,
                    size=self.n_points,
                    values={'x': start, 'q': self.q},
                ),
            ],
        )
