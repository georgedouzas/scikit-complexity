"""Dynamical systems of chaotic ensembles.

The module covers ensembles of chaotic maps, their invariant densities, and chaotic walks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, floor

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import mod, where
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role, VisualSpec
from ....core.systems import BaseSystem, System

DOUBLING_DENOMINATOR = 1125899906840747.0


class CubicWalkElement(Element):
    """Walkers driven by a chaotic map, each with the value `u` of its map and its position `x`.

    Args:
        label:
            The label of the walkers.

        u:
            The initial value of the map, between 0 and 1, one per walker or shared.

        x:
            The initial position, one per walker or shared.

        size:
            The number of walkers, or `None` for one.

        space:
            The line the walkers live on, whose axis `x` makes `x` their position, or `None`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CubicWalkElement
        >>> CubicWalkElement('walker', u=0.4).u
        0.4
    """

    visual = VisualSpec('points', {'position': 'x', 'states': ()})

    def __init__(
        self,
        label: str,
        *,
        u: object = 0.3,
        x: object = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.u = u
        self.x = x


class CubicWalkInteraction(Interaction):
    """Bertin's cubic chaotic map `u' = lam u (u - 1/2)(u - 1) + u` and the step `x' = x + 2 u - 1` it drives.

    Args:
        label:
            The label of the interaction.

        target:
            The walkers whose values `u` and positions `x` the map moves.

        lam:
            The nonlinearity of the map.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CubicWalkInteraction
        >>> CubicWalkInteraction('walk', lam=8.0).lam
        8.0
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'walkers', lam: float = 15.0) -> None:
        super().__init__(label)
        self.target = target
        self.lam = lam

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next value of the map and the next position as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            u, x = context.target.u, context.target.x
            self.laws_: list[BaseComponent] = [
                UpdateLaw(u, context.lam * u * (u - 1 / 2) * (u - 1) + u),
                UpdateLaw(x, x + 2 * u - 1),
            ]


class DoublingMapInteraction(Interaction):
    """The doubling map `x' = 2 x (mod 1)` on the grid of the fractions of denominator `q`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose values `x` the map moves.

        q:
            The denominator of the grid.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import DoublingMapInteraction
        >>> DoublingMapInteraction('doubling', target='points', q=1024.0).q
        1024.0
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'points', q: float = DOUBLING_DENOMINATOR) -> None:
        super().__init__(label)
        self.target = target
        self.q = q

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next value, rounded to the grid, as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, q = context.target.x, context.q
            self.laws_: list[BaseComponent] = [UpdateLaw(x, mod(2 * floor(q * x + 1 / 2), q) / q)]


class SkewTentMapInteraction(Interaction):
    """The skew tent map `x' = x / a` below the peak `a` and `(1 - x) / (1 - a)` above it.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose values `x` the map moves.

        a:
            The position of the peak, between 0 and 1.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SkewTentMapInteraction
        >>> SkewTentMapInteraction('tent', target='points', a=0.5).a
        0.5
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'points', a: float = 0.3) -> None:
        super().__init__(label)
        self.target = target
        self.a = a

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next value as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, a = context.target.x, context.a
            self.laws_: list[BaseComponent] = [UpdateLaw(x, where(x < a, x / a, (1 - x) / (1 - a)))]


def _get_square_displacement(member: SimpleNamespace) -> Expression:
    """Return the squared position of a walker."""
    return member.x**2


class CubicChaoticMapSystem(BaseSystem):
    """Bertin's cubic chaotic map and the chaotic walk it drives.

    The walk spreads like a random walk.

    Every walker follows the map `u' = λ u (u - 1/2)(u - 1) + u` on the unit interval and moves by the step `x' = x + 2u
    - 1`. The fixed point `u = 1/2` is stable for `λ < 8`, a period-two cycle follows, and the map is chaotic at `λ =
    15`. The walkers start at `x = 0` with values of `u` spread regularly over the unit interval, so the mean position
    stays near 0 while the mean square position grows linearly in time, and the invariant density of `u` is not uniform.
    The system has the states `walkers.u` and `walkers.x` and the observables `displacement`, the mean position, and
    `square_displacement`, the mean square position.

    Args:
        n_walkers:
            The number of walkers of the ensemble.

        lam:
            The parameter `λ` of the map, between 0 and 16.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CubicChaoticMapSystem
        >>> CubicChaoticMapSystem(4, lam=10.0).prepare().system_.elements[0].size
        4
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'random walk', 'diffusion')
    references = ('Bertin 2021',)
    until = 200
    dt = 1
    observables = (
        Observable('displacement', group='walkers', variable='x'),
        Observable('square_displacement', _get_square_displacement, group='walkers'),
    )

    def __init__(
        self,
        n_walkers: int = 100,
        *,
        lam: float = 15.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.lam = lam

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the walkers on a line and the map that drives them."""
        super()._prepare(context)
        start = (np.arange(self.n_walkers) + 1 / 4) / self.n_walkers
        walkers = CubicWalkElement('walkers', u=start, size=self.n_walkers, space=ContinuousSpace(n_dim=1))
        self.system_ = System(
            self.label,
            elements=[walkers],
            interactions=[CubicWalkInteraction('walk', target='walkers', lam=self.lam)],
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SkewTentMapSystem
        >>> SkewTentMapSystem(10, a=0.7).prepare().system_.interactions[0].a
        0.7
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'piecewise linear', 'invariant density')
    references = ('Bertin 2021',)
    until = 50
    dt = 1

    @staticmethod
    def _draw_cloud(
        size: int,
        start: float,
        spread: float,
        random_state: int | np.random.Generator | None,
    ) -> np.ndarray:
        """Return points drawn uniformly from the interval `[start, start + spread]`."""
        return np.random.default_rng(random_state).uniform(start, start + spread, size)

    def __init__(
        self,
        n_points: int = 100,
        *,
        a: float = 0.3,
        x: float = 0.4,
        spread: float = 0.01,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_points = n_points
        self.a = a
        self.x = x
        self.spread = spread
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the points and their map."""
        super()._prepare(context)
        start = SkewTentMapSystem._draw_cloud(self.n_points, self.x, self.spread, self.random_state)
        self.system_ = System(
            self.label,
            elements=[Element('points', values={'x': start}, size=self.n_points)],
            interactions=[SkewTentMapInteraction('tent', target='points', a=self.a)],
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_points = n_points
        self.x = x
        self.spread = spread
        self.q = q
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the points and their map."""
        super()._prepare(context)
        start = (
            np.floor(
                self.q * SkewTentMapSystem._draw_cloud(self.n_points, self.x, self.spread, self.random_state) + 1 / 2,
            )
            / self.q
        )
        self.system_ = System(
            self.label,
            elements=[Element('points', values={'x': start}, size=self.n_points)],
            interactions=[DoublingMapInteraction('doubling', target='points', q=self.q)],
        )
