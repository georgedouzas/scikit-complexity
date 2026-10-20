"""Escape-time fractals of the complex quadratic map: the Mandelbrot set and the filled Julia sets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import Element, Field, LatticeSpace, System, Update, where
from ....core.runtime import register_model

type Window = tuple[float, float, float, float]

ESCAPE_IMAGE = VisualSpec('image', {'states': ('n',), 'low': 0.0})
MANDELBROT_WINDOW: Window = (-2.2, 0.8, -1.2, 1.2)
JULIA_WINDOW: Window = (-1.6, 1.6, -1.2, 1.2)

QuadraticMap = Element.define(
    'QuadraticMap',
    states={
        'z': state(0.0, domain=Domain.COMPLEX),
        'n': state(0.0, domain=Domain.NONNEGATIVE_INTEGER),
    },
    parameters={
        'c': parameter(0.0, domain=Domain.COMPLEX),
        'radius': parameter(2.0, domain=Domain.POSITIVE),
    },
    laws=lambda s: [
        Update(s.z, where(abs(s.z) <= s.radius, s.z**2 + s.c, s.z)),
        Update(s.n, s.n + where(abs(s.z) <= s.radius, 1, 0)),
    ],
)


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
    return System(elements=[Field('plane', QuadraticMap, space, values=values, visual=ESCAPE_IMAGE)])


@register_model(
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
            The model, with the field `plane`, its orbits `plane.z`, and its escape times `plane.n`.

    Examples:
        >>> from skcomplex.models.mathematics.fractals import mandelbrot
        >>> mandelbrot((3, 3), window=(-1, 1, -1, 1)).elements[0].values['c'][1].tolist()
        [(-1+0j), 0j, (1+0j)]
    """
    points = _plane(shape, window)
    return _escape_field(shape, np.zeros_like(points), points, radius)


@register_model(
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
            The model, with the field `plane`, its orbits `plane.z`, and its escape times `plane.n`.

    Examples:
        >>> from skcomplex.models.mathematics.fractals import julia
        >>> julia(0.0, (3, 3), window=(-1, 1, -1, 1)).elements[0].values['z'][0].tolist()
        [(-1+1j), 1j, (1+1j)]
    """
    points = _plane(shape, window)
    return _escape_field(shape, points, complex(c), radius)
