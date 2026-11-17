"""Heat transfer systems.

The module covers diffusion of heat on lattices and meshes.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Annotated

import numpy as np
import skfem
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.spaces import BaseSpace, CellComplexSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, IMEXEulerSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array
from ..common.fields import DiffusivityInteraction


class HeatElement(Element):
    """Sites of a plate with the temperature `u` and the heat diffusion coefficient `D`.

    Args:
        label:
            The label of the plate.

        u:
            The initial temperature, one value per site or one for all.

        D:
            The diffusion coefficient.

        space:
            The lattice or the mesh of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.heat import HeatElement
        >>> HeatElement('site', D=0.5).D
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        u: object = 0.0,
        D: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.u = u
        self.D = D


class HeatDiffusionSystem(BaseSystem):
    """Heat diffusing on a plate from a hot square in its middle.

    The temperature follows `du/dt = D laplacian(u)`.

    The system has the temperature `plate.u`.

    Args:
        shape:
            The number of sites along every dimension.

        D:
            The diffusion coefficient.

        spacing:
            The distance between sites.

        boundary:
            `'periodic'`, `'zero-flux'` for an insulated plate, or `'fixed'` for edges held at zero.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.heat import HeatDiffusionSystem
        >>> HeatDiffusionSystem((10, 10), boundary='fixed').prepare().system_.elements[0].space.boundary
        {'u': ('fixed', 0.0)}
    """

    field = 'physics/heat'
    topics = ('field', 'diffusion', 'pde')
    references = ('Fourier 1822', 'Datseris and Parlitz 2022')
    until = 50.0
    dt = 1.0

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        D: float = 1.0,
        spacing: float = 1.0,
        boundary: str = 'zero-flux',
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.D = D
        self.spacing = spacing
        self.boundary = boundary

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of heat diffusing on a plate from a hot square in its middle.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = tuple(self.shape)
        u = np.zeros(shape)
        middle = tuple(slice(size // 2 - max(size // 8, 1), size // 2 + max(size // 8, 1)) for size in shape)
        u[middle] = 1.0
        condition = ('fixed', 0.0) if self.boundary == 'fixed' else self.boundary
        space = LatticeSpace(shape, spacing=self.spacing, boundary={'u': condition})
        plate = HeatElement('plate', u=u.ravel(), D=self.D, space=space)
        conduction = DiffusivityInteraction('conduction', target='plate', state='u', coefficient=lambda s: s.target.D)
        self.system_ = System(self.label, elements=[plate], interactions=[conduction])


def harmonic(x: Array, y: Array) -> Array:
    """Return the harmonic function `exp(x) sin(y)`.

    It is a steady temperature known in closed form.

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


class HeatLShapeSystem(BaseSystem):
    """Heat on the L-shaped domain with fixed boundary temperatures.

    The domain is discretised by linear finite elements.

    The plate starts cold inside, and its temperature relaxes to the solution of Laplace's equation with the boundary
    temperatures. The default boundary temperature is harmonic, so the steady state is known. The system has the
    temperature `plate.u` at the nodes of the mesh.

    Args:
        refine:
            The number of uniform refinements of the mesh.

        D:
            The diffusion coefficient.

        temperature:
            The boundary temperature, a number or a function of the coordinates.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for implicit Euler steps of 0.05.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.heat import HeatLShapeSystem
        >>> HeatLShapeSystem(1).prepare().system_.elements[0].space.prepare().size_
        21
    """

    field = 'physics/heat'
    topics = ('field', 'diffusion', 'finite element', 'mesh', 'pde')
    references = ('Strang and Fix 1973', 'Gustafsson and McBain 2020')
    until = 2.0
    dt = 0.1

    def __init__(
        self,
        refine: int = 3,
        *,
        D: float = 1.0,
        temperature: Callable[[Array, Array], Array] | float = harmonic,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.refine = refine
        self.D = D
        self.temperature = temperature

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of heat on the L-shaped mesh with fixed boundary temperatures.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        grid = skfem.MeshTri.init_lshaped().refined(self.refine)
        mesh = CellComplexSpace(points=grid.p, cells=grid.t, boundary={'u': ('fixed', self.temperature)})
        plate = HeatElement('plate', D=self.D, space=mesh)
        conduction = DiffusivityInteraction('conduction', target='plate', state='u', coefficient=lambda s: s.target.D)
        self.system_ = System(self.label, elements=[plate], interactions=[conduction], solver=IMEXEulerSolver(dt=0.05))
