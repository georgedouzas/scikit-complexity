"""Bénard-Marangoni convection: the order-parameter equation of a layer with a free, thermocapillary surface."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import numpy as np

from skcomplex.core.declarations import parameter, state

from ....core.components import Element
from ....core.elements import FieldElement
from ....core.laws import RateLaw
from ....core.laws.utils import laplacian
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import ETDRK4Solver
from ...core.base import BaseLaw
from ...core.laws import Symbols


class MarangoniLayerElement(Element):
    """Marangoni layer element."""

    u = state(0.0)
    r = parameter(0.1)
    g = parameter(1.0)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the laws of the marangoni layer element."""
        return [
            RateLaw(s.u, (s.r - 1) * s.u - 2 * laplacian(s.u) - laplacian(laplacian(s.u)) + s.g * s.u**2 - s.u**3),
        ]


@register_model(
    field='physics/fluids',
    tags=('field', 'convection', 'hexagons', 'surface tension', 'spectral', 'pde'),
    references=('Satz 2022', 'Bestehorn 1993', 'Cross and Hohenberg 1993'),
    defaults={'until': 300.0, 'dt': 5.0, 'solver': ETDRK4Solver(dt=0.5)},
)
def benard_marangoni(
    shape: Sequence[int] = (64, 64),
    *,
    length: float = 16 * math.pi,
    r: float = 0.1,
    g: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the order-parameter equation of Bénard-Marangoni convection, which forms hexagonal cells.

    The field `u` is the vertical velocity at mid-depth of a heated layer whose open surface has a surface tension
    that falls with temperature. It follows `u_t = r u - (1 + laplacian)^2 u + g u^2 - u^3`, the Swift-Hohenberg
    equation of convection with the quadratic term that the thermocapillary surface adds by breaking the symmetry
    between rising and sinking fluid. With `g > 0` and small `r > 0`, the cells are hexagons of rising fluid with a
    wavenumber near 1, and with `g = 0`, the layer of a fixed surface, they are rolls.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        length:
            The side of the domain.

        r:
            The distance above onset.

        g:
            The strength of the quadratic term of the thermocapillary surface.

        random_state:
            The seed or generator of the small random initial field.

    Returns:
        system:
            The model, with the field `layer.u`.

    Examples:
        >>> from skcomplex.models.physics.fluids import benard_marangoni
        >>> benard_marangoni((8, 8), g=0.5).elements[0].values['g']
        0.5
    """
    shape = tuple(shape)
    rng = np.random.default_rng(random_state)
    values = {'u': 0.01 * rng.uniform(-1.0, 1.0, shape), 'r': r, 'g': g}
    space = LatticeSpace(shape, spacing=length / shape[0])
    return System(
        elements=[FieldElement('layer', MarangoniLayerElement, space, values=values, discretization='spectral')],
    )
