"""Fractals of the complex quadratic map: the Mandelbrot set, the filled Julia sets, and their inverse iteration."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import CC, NN, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement, GroupElement
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import bernoulli, where
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from .types import Window


class InverseJuliaPointElement(Element):
    """Point of the inverse iteration of `z -> z² + omega`, which moves to a square root of `z - omega` drawn at random.

    The point moves to `+√(z - omega)` with probability `p` and to `-√(z - omega)` otherwise, with the principal
    square root.

    Args:
        label:
            The label of the element or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        z:
            The initial point.

        omega:
            The constant of the quadratic map.

        p:
            The probability of the positive square root.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import InverseJuliaPointElement
        >>> InverseJuliaPointElement('point', p=0.3).p
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[object, StateSpec(domain=CC)] = 0.5,
        omega: Annotated[object, ParameterSpec(domain=CC)] = 0.0,
        p: Annotated[object, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.omega = omega
        self.p = p

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the move to the square root drawn in the step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [UpdateLaw(s.z, (2 * bernoulli('branch', s.p) - 1) * sqrt(s.z - s.omega))]


ESCAPE_IMAGE = VisualSpec('image', {'states': ('n',), 'low': 0.0})
MANDELBROT_WINDOW: Window = (-2.2, 0.8, -1.2, 1.2)
JULIA_WINDOW: Window = (-1.6, 1.6, -1.2, 1.2)


class QuadraticMapElement(Element):
    """Complex quadratic map `z' = z^2 + c` while `|z|` stays within the escape radius, with a count of its steps.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        z:
            The initial point.

        n:
            The initial number of steps within the escape radius.

        c:
            The constant of the map.

        radius:
            The escape radius.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import QuadraticMapElement
        >>> QuadraticMapElement('point', c=0.25).c
        0.25
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[object, StateSpec(domain=CC)] = 0.0,
        n: Annotated[object, StateSpec(domain=NN)] = 0.0,
        c: Annotated[object, ParameterSpec(domain=CC)] = 0.0,
        radius: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.n = n
        self.c = c
        self.radius = radius

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the next point, which stays once it escaped, and the next count.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [
            UpdateLaw(s.z, where(abs(s.z) <= s.radius, s.z**2 + s.c, s.z)),
            UpdateLaw(s.n, s.n + where(abs(s.z) <= s.radius, 1, 0)),
        ]


def _plane(shape: Sequence[int], window: Window) -> npt.NDArray[np.complex128]:
    """Return the complex points of a lattice over a window, the real part along columns and the imaginary part up."""
    rows, columns = (int(item) for item in shape)
    left, right, bottom, top = window
    real = np.linspace(left, right, columns)
    imaginary = np.linspace(top, bottom, rows)
    return real[np.newaxis, :] + 1j * imaginary[:, np.newaxis]


def _escape_field(shape: Sequence[int], z: object, c: object, radius: float) -> System:
    """Return the field of the quadratic map over a lattice from the initial points and the constants."""
    space = LatticeSpace(tuple(int(item) for item in shape))
    values = {'z': z, 'c': c, 'radius': radius}
    return System(
        elements=[FieldElement('plane', element=QuadraticMapElement, space=space, values=values, visual=ESCAPE_IMAGE)],
    )


@register_system(
    field='mathematics/fractals',
    tags=('fractal', 'complex', 'map', 'lattice', 'escape time'),
    references=('Dilão 2023',),
    defaults={'until': 100, 'dt': 1},
)
def mandelbrot(
    shape: Sequence[int] = (240, 300),
    *,
    window: Window = MANDELBROT_WINDOW,
    radius: float = 2.0,
) -> System:
    """Return the Mandelbrot set, the points `c` whose orbit of `z -> z^2 + c` from 0 stays bounded (Dilão 2023 §5.5).

    Every site holds one point `c` of the window and iterates the map, and `n` counts the steps the orbit stays within
    the escape radius. The points of the set never escape, so their count grows with every step, while the count of an
    escaped point stops at its escape time and colors the bands around the set.

    Args:
        shape:
            The number of sites along the imaginary and the real axis.

        window:
            The smallest and largest real part, then the smallest and largest imaginary part, of the points.

        radius:
            The escape radius, 2 for the quadratic map.

    Returns:
        system:
            The system, with the field `plane`, its orbits `plane.z`, and its escape times `plane.n`.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import mandelbrot
        >>> mandelbrot((3, 3), window=(-1, 1, -1, 1)).elements[0].values['c'][1].tolist()
        [(-1+0j), 0j, (1+0j)]
    """
    points = _plane(shape, window)
    return _escape_field(shape, np.zeros_like(points), points, radius)


@register_system(
    field='mathematics/fractals',
    tags=('fractal', 'complex', 'map', 'lattice', 'escape time'),
    references=('Dilão 2023',),
    defaults={'until': 100, 'dt': 1},
)
def julia(
    c: complex = -0.8 + 0.156j,
    shape: Sequence[int] = (240, 320),
    *,
    window: Window = JULIA_WINDOW,
    radius: float = 2.0,
) -> System:
    """Return the filled Julia set of `z -> z^2 + c`, the starting points whose orbit stays bounded (Dilão 2023 §5.5).

    Every site holds one starting point `z` of the window and iterates the map with the same constant `c`, and `n`
    counts the steps the orbit stays within the escape radius. For `c` in the Mandelbrot set the filled Julia set is
    connected, and outside it the set breaks into dust.

    Args:
        c:
            The constant of the map.

        shape:
            The number of sites along the imaginary and the real axis.

        window:
            The smallest and largest real part, then the smallest and largest imaginary part, of the starting points.

        radius:
            The escape radius, at least 2 and at least `|c|`, so escaped orbits never return.

    Returns:
        system:
            The system, with the field `plane`, its orbits `plane.z`, and its escape times `plane.n`.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import julia
        >>> julia(0.0, (3, 3), window=(-1, 1, -1, 1)).elements[0].values['z'][0].tolist()
        [(-1+1j), 1j, (1+1j)]
    """
    points = _plane(shape, window)
    return _escape_field(shape, points, complex(c), radius)


@register_system(
    field='mathematics/fractals',
    tags=('fractal', 'complex', 'iterated function system', 'random map', 'julia set'),
    references=('Barnsley 1988', 'Dilão 2023'),
    defaults={'until': 30, 'dt': 1},
)
def inverse_julia(
    omega: complex = -0.123 + 0.745j,
    *,
    p: float = 0.5,
    n_points: int = 5000,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the random function system `z -> ±√(z - omega)`, whose points settle on the Julia set of `z² + omega`.

    At every step each point moves to the positive square root of `z - omega` with probability `p` and to the
    negative one otherwise (Dilão 2023, Exercise 5.8). Both branches invert the quadratic map `z -> z² + omega`, which
    repels its Julia set, so the inverse branches attract the points onto the Julia set, drawn by the chaos game with
    weights `p` and `1 - p`. For `omega = 0` the Julia set is the unit circle, and the default is Douady's rabbit.

    Args:
        omega:
            The constant of the quadratic map.

        p:
            The probability of the positive square root.

        n_points:
            The number of points, started uniformly in the square of side 2 about the origin.

        random_state:
            The seed or generator of the initial points.

    Returns:
        system:
            The system, with the points `points.z`.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import inverse_julia
        >>> inverse_julia(0.0, n_points=10).elements[0].prepare().size_
        10
    """
    rng = np.random.default_rng(random_state)
    start = rng.uniform(-1.0, 1.0, n_points) + 1j * rng.uniform(-1.0, 1.0, n_points)
    values = {'z': start, 'omega': complex(omega), 'p': p}
    return System(elements=[GroupElement('points', element=InverseJuliaPointElement, size=n_points, values=values)])
