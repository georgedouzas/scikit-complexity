"""Pattern formation models: amplitude equations and rolls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ....core.model import Element, Field, LatticeSpace, Rate, System, laplacian
from ....core.runtime import register_model
from ....core.solvers import RK4

type Array = npt.NDArray[np.float64]

GrayScott = Element.define(
    'GrayScott',
    states={'u': 1.0, 'v': 0.0},
    parameters={'Du': 0.16, 'Dv': 0.08, 'F': 0.035, 'k': 0.065},
    laws=lambda s: [
        Rate(s.u, s.Du * laplacian(s.u) - s.u * s.v**2 + s.F * (1 - s.u)),
        Rate(s.v, s.Dv * laplacian(s.v) + s.u * s.v**2 - (s.F + s.k) * s.v),
    ],
)

Brusselator = Element.define(
    'Brusselator',
    states={'u': 1.0, 'v': 3.0},
    parameters={'a': 1.0, 'b': 3.0, 'Du': 1.0, 'Dv': 8.0},
    laws=lambda s: [
        Rate(s.u, s.Du * laplacian(s.u) + s.a - (s.b + 1) * s.u + s.u**2 * s.v),
        Rate(s.v, s.Dv * laplacian(s.v) + s.b * s.u - s.u**2 * s.v),
    ],
)

Schnakenberg = Element.define(
    'Schnakenberg',
    states={'u': 1.0, 'v': 0.9},
    parameters={'a': 0.1, 'b': 0.9, 'Du': 1.0, 'Dv': 40.0},
    laws=lambda s: [
        Rate(s.u, s.Du * laplacian(s.u) + s.a - s.u + s.u**2 * s.v),
        Rate(s.v, s.Dv * laplacian(s.v) + s.b - s.u**2 * s.v),
    ],
)


def _noise(shape: tuple[int, ...], scale: float, random_state: int | np.random.Generator | None) -> Array:
    """Return small uniform noise of a shape."""
    return scale * np.random.default_rng(random_state).uniform(-1.0, 1.0, size=shape)


@register_model(
    field='chemistry/pattern_formation',
    tags=('field', 'reaction-diffusion', 'self-replication', 'pde'),
    references=('Gray and Scott 1984', 'Pearson 1993'),
    defaults={'until': 2000.0, 'dt': 20.0, 'solver': RK4(dt=1.0)},
)
def gray_scott(
    shape: Sequence[int] = (128, 128),
    *,
    Du: float = 0.16,
    Dv: float = 0.08,
    F: float = 0.035,
    k: float = 0.065,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Gray-Scott reaction-diffusion model, seeded in its middle, which grows spots that replicate.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        Du:
            The diffusion of the substrate `u`.

        Dv:
            The diffusion of the activator `v`.

        F:
            The feed rate.

        k:
            The kill rate.

        random_state:
            The seed or generator of the noise on the seed.

    Returns:
        system:
            The model, with the substrate `cells.u` and the activator `cells.v`.

    Examples:
        >>> from skcomplex.models.chemistry.pattern_formation import gray_scott
        >>> gray_scott((32, 32), F=0.04).elements[0].values['F']
        0.04
    """
    shape = tuple(shape)
    u, v = np.ones(shape), np.zeros(shape)
    middle = tuple(slice(size // 2 - size // 10, size // 2 + size // 10) for size in shape)
    u[middle], v[middle] = 0.5, 0.25
    noise = _noise(shape, 0.01, random_state)
    values = {'u': u + noise, 'v': v + np.abs(noise), 'Du': Du, 'Dv': Dv, 'F': F, 'k': k}
    return System(elements=[Field('cells', GrayScott, LatticeSpace(shape), values=values)])


@register_model(
    field='chemistry/pattern_formation',
    tags=('field', 'reaction-diffusion', 'Turing', 'oscillation', 'pde'),
    references=('Prigogine and Lefever 1968', 'Gros 2015'),
    defaults={'until': 100.0, 'dt': 1.0, 'solver': RK4(dt=0.01)},
)
def brusselator(
    shape: Sequence[int] = (64, 64),
    *,
    a: float = 1.0,
    b: float = 3.0,
    Du: float = 1.0,
    Dv: float = 8.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Brusselator reaction-diffusion model from small noise around its steady state `(a, b / a)`.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        a:
            The feed of `u`.

        b:
            The conversion of `u` into `v`.

        Du:
            The diffusion of `u`.

        Dv:
            The diffusion of `v`.

        random_state:
            The seed or generator of the noise.

    Returns:
        system:
            The model, with the concentrations `cells.u` and `cells.v`.

    Examples:
        >>> from skcomplex.models.chemistry.pattern_formation import brusselator
        >>> brusselator((8, 8), b=2.5).elements[0].values['b']
        2.5
    """
    shape = tuple(shape)
    values = {
        'u': a + _noise(shape, 0.01, random_state),
        'v': np.full(shape, b / a),
        'a': a,
        'b': b,
        'Du': Du,
        'Dv': Dv,
    }
    return System(elements=[Field('cells', Brusselator, LatticeSpace(shape), values=values)])


@register_model(
    field='chemistry/pattern_formation',
    tags=('field', 'reaction-diffusion', 'Turing', 'pde'),
    references=('Schnakenberg 1979', 'Turing 1952', 'Murray 2003'),
    defaults={'until': 200.0, 'dt': 5.0, 'solver': RK4(dt=0.005)},
)
def schnakenberg(
    shape: Sequence[int] = (64, 64),
    *,
    a: float = 0.1,
    b: float = 0.9,
    Du: float = 1.0,
    Dv: float = 40.0,
    spacing: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Schnakenberg reaction-diffusion model, which forms Turing patterns from noise when `Dv >> Du`.

    The steady state `u = a + b`, `v = b / (a + b)^2` is stable without diffusion and unstable to perturbations of a
    band of wavelengths with it.

    Args:
        shape:
            The number of sites along every dimension of the periodic lattice.

        a:
            The production of `u`.

        b:
            The production of `v`.

        Du:
            The diffusion of the activator `u`.

        Dv:
            The diffusion of the inhibitor `v`.

        spacing:
            The distance between sites.

        random_state:
            The seed or generator of the noise.

    Returns:
        system:
            The model, with the concentrations `cells.u` and `cells.v`.

    Examples:
        >>> from skcomplex.models.chemistry.pattern_formation import schnakenberg
        >>> schnakenberg((8, 8)).elements[0].values['Dv']
        40.0
    """
    shape = tuple(shape)
    steady_u, steady_v = a + b, b / (a + b) ** 2
    values = {
        'u': steady_u * (1 + _noise(shape, 0.01, random_state)),
        'v': np.full(shape, steady_v),
        'a': a,
        'b': b,
        'Du': Du,
        'Dv': Dv,
    }
    return System(elements=[Field('cells', Schnakenberg, LatticeSpace(shape, spacing=spacing), values=values)])
