"""Heat transfer systems: diffusion of heat on lattices and meshes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Annotated

import numpy as np
import skfem
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import DiffusionInteraction
from ....core.components.spaces import LatticeSpace, MeshSpace
from ....core.components.system import System
from ....core.engine.solvers import IMEXEulerSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system


class HeatElement(Element):
    """Site of a plate, with the temperature `u` and the diffusion coefficient `D` of heat.

    Args:
        label:
            The label of the site.

        visual:
            How the site is drawn, or `None` for none.

        u:
            The initial temperature.

        D:
            The diffusion coefficient.

    Examples:
        >>> from skcomplex.systems.physics.heat import HeatElement
        >>> HeatElement('site', D=0.5).D
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.D = D


@register_system(
    field='physics/heat',
    tags=('field', 'diffusion', 'pde'),
    references=('Fourier 1822', 'Datseris and Parlitz 2022'),
    defaults={'until': 50.0, 'dt': 1.0},
)
def heat_diffusion(
    shape: Sequence[int] = (64, 64),
    *,
    D: float = 1.0,
    spacing: float = 1.0,
    boundary: str = 'zero-flux',
) -> System:
    """Return heat diffusing on a plate from a hot square in its middle, `du/dt = D laplacian(u)`.

    Args:
        shape:
            The number of sites along every dimension.

        D:
            The diffusion coefficient.

        spacing:
            The distance between sites.

        boundary:
            `'periodic'`, `'zero-flux'` for an insulated plate, or `'fixed'` for edges held at zero.

    Returns:
        system:
            The system, with the temperature `plate.u`.

    Examples:
        >>> from skcomplex.systems.physics.heat import heat_diffusion
        >>> heat_diffusion((10, 10)).elements[0].prepare().size_
        100
    """
    shape = tuple(shape)
    u = np.zeros(shape)
    middle = tuple(slice(size // 2 - max(size // 8, 1), size // 2 + max(size // 8, 1)) for size in shape)
    u[middle] = 1.0
    condition = ('fixed', 0.0) if boundary == 'fixed' else boundary
    space = LatticeSpace(shape, spacing=spacing)
    return System(
        elements=[
            FieldElement('plate', element=HeatElement, space=space, boundary={'u': condition}, values={'u': u, 'D': D}),
        ],
        interactions=[DiffusionInteraction('conduction', target='plate', state='u', coefficient=lambda s: s.target.D)],
    )


def harmonic(x: Array, y: Array) -> Array:
    """Return the harmonic function `exp(x) sin(y)`, a steady temperature known in closed form.

    Args:
        x:
            The first coordinates.

        y:
            The second coordinates.

    Returns:
        values:
            The temperatures.

    Examples:
        >>> from skcomplex.systems.physics.heat import harmonic
        >>> float(harmonic(0.0, 0.0))
        0.0
    """
    return np.exp(x) * np.sin(y)


@register_system(
    field='physics/heat',
    tags=('field', 'diffusion', 'finite element', 'mesh', 'pde'),
    references=('Strang and Fix 1973', 'Gustafsson and McBain 2020'),
    defaults={'until': 2.0, 'dt': 0.1, 'solver': IMEXEulerSolver(dt=0.05)},
)
def heat_l_shape(
    refine: int = 3,
    *,
    D: float = 1.0,
    temperature: Callable[[Array, Array], Array] | float = harmonic,
) -> System:
    """Return heat on the L-shaped domain with fixed boundary temperatures, discretised by linear finite elements.

    The plate starts cold inside, and its temperature relaxes to the solution of Laplace's equation with the
    boundary temperatures. The default boundary temperature is harmonic, so the steady state is known.

    Args:
        refine:
            The number of uniform refinements of the mesh.

        D:
            The diffusion coefficient.

        temperature:
            The boundary temperature, a number or a function of the coordinates.

    Returns:
        system:
            The system, with the temperature `plate.u` at the nodes of the mesh.

    Examples:
        >>> from skcomplex.systems.physics.heat import heat_l_shape
        >>> heat_l_shape(1).elements[0].prepare().size_
        21
    """
    grid = skfem.MeshTri.init_lshaped().refined(refine)
    mesh = MeshSpace(grid.p, grid.t)
    return System(
        elements=[
            FieldElement(
                'plate',
                element=HeatElement,
                space=mesh,
                boundary={'u': ('fixed', temperature)},
                values={'D': D},
                discretization='finite-element',
            ),
        ],
        interactions=[DiffusionInteraction('conduction', target='plate', state='u', coefficient=lambda s: s.target.D)],
    )
