"""Fractals of the complex quadratic map.

The module covers the Mandelbrot set, the filled Julia sets, and their inverse iteration.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import sqrt

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import bernoulli, where
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from .types import Window

MANDELBROT_WINDOW: Window = (-2.2, 0.8, -1.2, 1.2)
JULIA_WINDOW: Window = (-1.6, 1.6, -1.2, 1.2)


class InverseJuliaPointElement(Element):
    """Points of the inverse iteration of `z -> z² + omega`, each with its point `z`.

    An `InverseJuliaInteraction` moves every point to a square root of `z - omega` drawn at random.

    Args:
        label:
            The label of the points.

        z:
            The initial point, one per member or shared.

        size:
            The number of points, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import InverseJuliaPointElement
        >>> InverseJuliaPointElement('point', z=0.5j).z
        0.5j
    """

    def __init__(self, label: str, *, z: object = 0.5, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.z = z


class InverseJuliaInteraction(Interaction):
    """Move of a point to a random square root of `z - omega`.

    The point moves to `+√(z - omega)` with probability `p` and to `-√(z - omega)` otherwise, with the principal square
    root, drawn anew at every step for every member of a bound group.

    Args:
        label:
            The label of the interaction.

        target:
            The points whose value `z` the interaction moves.

        omega:
            The constant of the quadratic map.

        p:
            The probability of the positive square root.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import InverseJuliaInteraction
        >>> InverseJuliaInteraction('inverse', p=0.3).p
        0.3
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'points',
        omega: complex = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.omega = omega
        self.p = p

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the move to the square root drawn in the step as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            z = context.target.z
            root = sqrt(z - context.omega)
            self.laws_: list[BaseLaw] = [UpdateLaw(z, (2 * bernoulli('branch', context.p) - 1) * root)]


class QuadraticMapElement(Element):
    """Points of the complex quadratic map `z' = z^2 + c`, each with a count `n` of its steps.

    A `QuadraticMapInteraction` iterates the points while they stay within the escape radius.

    Args:
        label:
            The label of the element.

        z:
            The initial point.

        n:
            The initial number of steps within the escape radius.

        c:
            The constant of the map.

        space:
            The space with one point per site, or `None` for one point.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import QuadraticMapElement
        >>> QuadraticMapElement('point', c=0.25).c
        0.25
    """

    visual = VisualSpec('image', {'states': ('n',), 'low': 0.0})

    def __init__(
        self,
        label: str,
        *,
        z: object = 0.0,
        n: int = 0,
        c: object = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.z = z
        self.n = n
        self.c = c


class QuadraticMapInteraction(Interaction):
    """Iteration of `z' = z^2 + c` with a count of the steps within the escape radius.

    The point stays once `|z|` exceeds the escape radius, and the count `n` stops at its escape time.

    Args:
        label:
            The label of the interaction.

        target:
            The points whose values `z` and `n` the map sets.

        radius:
            The escape radius.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import QuadraticMapInteraction
        >>> QuadraticMapInteraction('escape', radius=3.0).radius
        3.0
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'plane', radius: float = 2.0) -> None:
        super().__init__(label)
        self.target = target
        self.radius = radius

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next point and the next count as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            z, n, c = context.target.z, context.target.n, context.target.c
            inside = abs(z) <= context.radius
            self.laws_: list[BaseLaw] = [
                UpdateLaw(z, where(inside, z**2 + c, z)),
                UpdateLaw(n, n + where(inside, 1, 0)),
            ]


class MandelbrotSystem(BaseSystem):
    """The Mandelbrot set (Dilão 2023 §5.5).

    The set holds the points `c` whose orbit of `z -> z^2 + c` from 0 stays bounded.

    Every site holds one point `c` of the window and iterates the map, and `n` counts the steps the orbit stays within
    the escape radius. The points of the set never escape, so their count grows with every step, while the count of an
    escaped point stops at its escape time and colors the bands around the set. The system has the field `plane`, its
    orbits `plane.z`, and its escape times `plane.n`.

    Args:
        shape:
            The number of sites along the imaginary and the real axis.

        window:
            The smallest and largest real part, then the smallest and largest imaginary part, of the points.

        radius:
            The escape radius, 2 for the quadratic map.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import MandelbrotSystem
        >>> MandelbrotSystem((3, 3), window=(-1, 1, -1, 1)).prepare().system_.elements[0].c[3:6].tolist()
        [(-1+0j), 0j, (1+0j)]
    """

    field = 'mathematics/fractals'
    topics = ('fractal', 'complex', 'map', 'lattice', 'escape time')
    references = ('Dilão 2023',)
    until = 100
    dt = 1

    @staticmethod
    def _get_plane(shape: Sequence[int], window: Window) -> npt.NDArray[np.complex128]:
        """Return the complex points of a lattice over a window.

        The real part runs along columns and the imaginary part up.
        """
        rows, columns = (int(item) for item in shape)
        left, right, bottom, top = window
        real = np.linspace(left, right, columns)
        imaginary = np.linspace(top, bottom, rows)
        return real[np.newaxis, :] + 1j * imaginary[:, np.newaxis]

    @staticmethod
    def _get_escape_system(
        label: str,
        shape: Sequence[int],
        z: npt.NDArray[np.complex128] | complex,
        c: npt.NDArray[np.complex128] | complex,
        radius: float,
    ) -> System:
        """Return the system of the quadratic map over a lattice.

        The system comes from the initial points and the constants, given as arrays of the shape of the lattice or as
        one shared value, and read row by row as one value per site.
        """
        space = LatticeSpace(tuple(int(item) for item in shape))
        z, c = (np.ravel(value) if isinstance(value, np.ndarray) else value for value in (z, c))
        return System(
            label,
            elements=[QuadraticMapElement('plane', z=z, c=c, space=space)],
            interactions=[QuadraticMapInteraction('escape', target='plane', radius=radius)],
        )

    def __init__(
        self,
        shape: Sequence[int] = (240, 300),
        *,
        window: Window = MANDELBROT_WINDOW,
        radius: float = 2.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.window = window
        self.radius = radius

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the points of the window and their escape."""
        super()._prepare(context)
        points = MandelbrotSystem._get_plane(self.shape, self.window)
        self.system_ = MandelbrotSystem._get_escape_system(
            self.label,
            self.shape,
            np.zeros_like(points),
            points,
            self.radius,
        )


class JuliaSystem(BaseSystem):
    """The filled Julia set of `z -> z^2 + c` (Dilão 2023 §5.5).

    The set holds the starting points whose orbit stays bounded.

    Every site holds one starting point `z` of the window and iterates the map with the same constant `c`, and `n`
    counts the steps the orbit stays within the escape radius. For `c` in the Mandelbrot set the filled Julia set is
    connected, and outside it the set breaks into dust. The system has the field `plane`, its orbits `plane.z`, and its
    escape times `plane.n`.

    Args:
        c:
            The constant of the map.

        shape:
            The number of sites along the imaginary and the real axis.

        window:
            The smallest and largest real part, then the smallest and largest imaginary part, of the starting points.

        radius:
            The escape radius, at least 2 and at least `|c|`, so escaped orbits never return.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import JuliaSystem
        >>> JuliaSystem(0.0, (3, 3), window=(-1, 1, -1, 1)).prepare().system_.elements[0].z[:3].tolist()
        [(-1+1j), 1j, (1+1j)]
    """

    field = 'mathematics/fractals'
    topics = ('fractal', 'complex', 'map', 'lattice', 'escape time')
    references = ('Dilão 2023',)
    until = 100
    dt = 1

    def __init__(
        self,
        c: complex = -0.8 + 0.156j,
        shape: Sequence[int] = (240, 320),
        *,
        window: Window = JULIA_WINDOW,
        radius: float = 2.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.c = c
        self.shape = shape
        self.window = window
        self.radius = radius

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the starting points of the window and their escape."""
        super()._prepare(context)
        points = MandelbrotSystem._get_plane(self.shape, self.window)
        self.system_ = MandelbrotSystem._get_escape_system(self.label, self.shape, points, complex(self.c), self.radius)


class InverseJuliaSystem(BaseSystem):
    """The random function system `z -> ±√(z - omega)`.

    Its points settle on the Julia set of `z² + omega`.

    At every step each point moves to the positive square root of `z - omega` with probability `p` and to the negative
    one otherwise (Dilão 2023, Exercise 5.8). Both branches invert the quadratic map `z -> z² + omega`, which repels its
    Julia set, so the inverse branches attract the points onto the Julia set, drawn by the chaos game with weights `p`
    and `1 - p`. For `omega = 0` the Julia set is the unit circle, and the default is Douady's rabbit. The system has
    the points `points.z`.

    Args:
        omega:
            The constant of the quadratic map.

        p:
            The probability of the positive square root.

        n_points:
            The number of points, started uniformly in the square of side 2 about the origin.

        random_state:
            The seed or generator of the initial points.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import InverseJuliaSystem
        >>> InverseJuliaSystem(0.0, n_points=10).prepare().system_.elements[0].size
        10
    """

    field = 'mathematics/fractals'
    topics = ('fractal', 'complex', 'iterated function system', 'random map', 'julia set')
    references = ('Barnsley 1988', 'Dilão 2023')
    until = 30
    dt = 1

    def __init__(
        self,
        omega: complex = -0.123 + 0.745j,
        *,
        p: float = 0.5,
        n_points: int = 5000,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.omega = omega
        self.p = p
        self.n_points = n_points
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the points and their random inverse map."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        start = rng.uniform(-1.0, 1.0, self.n_points) + 1j * rng.uniform(-1.0, 1.0, self.n_points)
        inverse = InverseJuliaInteraction('inverse', target='points', omega=complex(self.omega), p=self.p)
        self.system_ = System(
            self.label,
            elements=[InverseJuliaPointElement('points', z=start, size=self.n_points)],
            interactions=[inverse],
        )
