"""Pattern formation models: amplitude equations and rolls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ....core.model import Element, Field, LatticeSpace, Rate, System, laplacian
from ....core.runtime import register_model
from ....core.solvers import ETDRK4

type Array = npt.NDArray[np.float64]


def _noise(shape: tuple[int, ...], scale: float, random_state: int | np.random.Generator | None) -> Array:
    """Return small uniform noise of a shape."""
    return scale * np.random.default_rng(random_state).uniform(-1.0, 1.0, size=shape)


GinzburgLandau = Element.define(
    'GinzburgLandau',
    states={'u': 0.0, 'v': 0.0},
    parameters={'b': 1.0, 'c': -1.5},
    laws=lambda s: [
        Rate(s.u, s.u + laplacian(s.u) - s.b * laplacian(s.v) - (s.u**2 + s.v**2) * (s.u - s.c * s.v)),
        Rate(s.v, s.v + s.b * laplacian(s.u) + laplacian(s.v) - (s.u**2 + s.v**2) * (s.c * s.u + s.v)),
    ],
)

SwiftHohenberg = Element.define(
    'SwiftHohenberg',
    states={'u': 0.0},
    parameters={'r': 0.3},
    laws=lambda s: [Rate(s.u, (s.r - 1) * s.u - 2 * laplacian(s.u) - laplacian(laplacian(s.u)) - s.u**3)],
)


@register_model(
    field='physics/pattern_formation',
    tags=('field', 'chaos', 'spirals', 'spectral', 'pde'),
    references=('Aranson and Kramer 2002', 'Gros 2015'),
    defaults={'until': 50.0, 'dt': 1.0, 'solver': ETDRK4(dt=0.1)},
)
def complex_ginzburg_landau(
    shape: Sequence[int] = (64, 64),
    *,
    length: float = 64.0,
    b: float = 1.0,
    c: float = -1.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the complex Ginzburg-Landau equation `A_t = A + (1 + i b) laplacian(A) - (1 + i c) |A|^2 A`.

    The amplitude `A = u + i v` is held as its real and imaginary parts, and the linear part couples them per Fourier
    mode. With `1 + b c < 0` the uniform oscillation is unstable and spiral defects turn into turbulence.

    Args:
        shape:
            The number of sites along both axes of the periodic lattice.

        length:
            The side of the square.

        b:
            The linear dispersion.

        c:
            The nonlinear dispersion.

        random_state:
            The seed or generator of the small random initial amplitude.

    Returns:
        system:
            The model, with the real part `medium.u` and the imaginary part `medium.v`.

    Examples:
        >>> from skcomplex.models.physics.pattern_formation import complex_ginzburg_landau
        >>> complex_ginzburg_landau((8, 8)).elements[0].values['c']
        -1.5
    """
    shape = tuple(shape)
    values = {'u': _noise(shape, 0.01, random_state), 'v': _noise(shape, 0.01, None), 'b': b, 'c': c}
    space = LatticeSpace(shape, spacing=length / shape[0])
    return System(elements=[Field('medium', GinzburgLandau, space, values=values, discretization='spectral')])


@register_model(
    field='physics/pattern_formation',
    tags=('field', 'rolls', 'convection', 'spectral', 'pde'),
    references=('Swift and Hohenberg 1977', 'Cross and Hohenberg 1993', 'Datseris and Parlitz 2022'),
    defaults={'until': 200.0, 'dt': 5.0, 'solver': ETDRK4(dt=0.5)},
)
def swift_hohenberg(
    shape: Sequence[int] = (64, 64),
    *,
    length: float = 16 * math.pi,
    r: float = 0.3,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Swift-Hohenberg equation `u_t = r u - (1 + laplacian)^2 u - u^3`, which forms rolls of length 2 pi.

    Above onset, `r > 0`, small noise grows into rolls whose wavenumber is near 1, the fastest growing one.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        length:
            The side of the domain.

        r:
            The distance above onset.

        random_state:
            The seed or generator of the noise.

    Returns:
        system:
            The model, with the field `cells.u`.

    Examples:
        >>> from skcomplex.models.physics.pattern_formation import swift_hohenberg
        >>> swift_hohenberg((8, 8), r=0.1).elements[0].values['r']
        0.1
    """
    shape = tuple(shape)
    values = {'u': _noise(shape, 0.01, random_state), 'r': r}
    space = LatticeSpace(shape, spacing=length / shape[0])
    return System(elements=[Field('cells', SwiftHohenberg, space, values=values, discretization='spectral')])
