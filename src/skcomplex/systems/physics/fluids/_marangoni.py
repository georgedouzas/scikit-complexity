"""Bénard-Marangoni convection: the order-parameter equation of a layer with a free, thermocapillary surface."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import GradientInteraction
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import ETDRK4Solver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


def _get_marangoni_density(s: SimpleNamespace) -> Expression:
    """Return the density of the Swift-Hohenberg functional with the quadratic term of the free surface."""
    layer = s.target
    u = layer.u
    potential = -(layer.r - 1) * u**2 / 2 - layer.g * u**3 / 3 + u**4 / 4
    return potential + u * laplacian(u) + laplacian(u) ** 2 / 2


class MarangoniLayerElement(Element):
    """Site of a convecting layer with a free surface, with the order parameter `u`.

    Args:
        label:
            The label of the site.

        visual:
            How the site is drawn, or `None` for none.

        u:
            The initial order parameter.

        r:
            The distance from the onset of convection.

        g:
            The strength of the quadratic term that the free surface adds.

    Examples:
        >>> from skcomplex.systems.physics.fluids import MarangoniLayerElement
        >>> MarangoniLayerElement('site', g=0.5).g
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        r: Annotated[float, ParameterSpec()] = 0.1,
        g: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.r = r
        self.g = g


@register_system(
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
            The system, with the field `layer.u`.

    Examples:
        >>> from skcomplex.systems.physics.fluids import benard_marangoni
        >>> benard_marangoni((8, 8), g=0.5).elements[0].values['g']
        0.5
    """
    shape = tuple(shape)
    rng = np.random.default_rng(random_state)
    values = {'u': 0.01 * rng.uniform(-1.0, 1.0, shape), 'r': r, 'g': g}
    space = LatticeSpace(shape, spacing=length / shape[0])
    return System(
        elements=[
            FieldElement('layer', element=MarangoniLayerElement, space=space, values=values, discretization='spectral'),
        ],
        interactions=[
            GradientInteraction(
                'relaxation',
                target='layer',
                states='u',
                potential=_get_marangoni_density,
                variational=True,
            ),
        ],
    )
