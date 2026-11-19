"""Bénard-Marangoni convection.

The module holds the order-parameter equation of a layer with a free, thermocapillary surface.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, ETDRK4Solver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.gradients import GradientInteraction


class MarangoniLayerElement(Element):
    """Sites of a convecting layer with a free surface, with the order parameter `u`.

    Args:
        label:
            The label of the layer.

        u:
            The initial order parameter, one value per site or one for all.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.fluids import MarangoniLayerElement
        >>> MarangoniLayerElement('site', u=0.5).u
        0.5
    """

    def __init__(self, label: str, *, u: object = 0.0, space: BaseSpace | None = None) -> None:
        super().__init__(label, space=space)
        self.u = u


class ThermocapillaryInteraction(GradientInteraction):
    """Relaxation of the order parameter of a convecting layer whose free surface breaks the up-down symmetry.

    The order parameter `u` of the target follows `u_t = r u - (1 + laplacian)^2 u + g u^2 - u^3`, the gradient flow
    of the density `-(r - 1) u^2 / 2 - g u^3 / 3 + u^4 / 4 + u laplacian(u) + laplacian(u)^2 / 2`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the layer.

        r:
            The distance from the onset of convection.

        g:
            The strength of the quadratic term that the free surface adds.

    Examples:
        >>> from skcomplex.systems.physics.fluids import ThermocapillaryInteraction
        >>> ThermocapillaryInteraction('relaxation', target='layer', g=0.5).g
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        r: float = 0.1,
        g: float = 1.0,
    ) -> None:
        super().__init__(label, target=target, states='u', variational=True)
        self.r = r
        self.g = g

    def _get_potential(self, namespace: SimpleNamespace) -> Expression:
        """Return the density of the Swift-Hohenberg functional with the quadratic term of the free surface.

        Args:
            namespace:
                The namespace of the symbols the density can use.

        Returns:
            density:
                The density of the functional.
        """
        u = namespace.target.u
        potential = -(namespace.r - 1) * u**2 / 2 - namespace.g * u**3 / 3 + u**4 / 4
        density: Expression = potential + u * laplacian(u) + laplacian(u) ** 2 / 2
        return density


class BenardMarangoniSystem(BaseSystem):
    """The order-parameter equation of Bénard-Marangoni convection.

    The convection forms hexagonal cells.

    The field `u` is the vertical velocity at mid-depth of a heated layer whose open surface has a surface tension that
    falls with temperature. It follows `u_t = r u - (1 + laplacian)^2 u + g u^2 - u^3`, the Swift-Hohenberg equation of
    convection with the quadratic term that the thermocapillary surface adds by breaking the symmetry between rising and
    sinking fluid. With `g > 0` and small `r > 0`, the cells are hexagons of rising fluid with a wavenumber near 1, and
    with `g = 0`, the layer of a fixed surface, they are rolls. The system has the field `layer.u`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for exponential steps of 0.5.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.fluids import BenardMarangoniSystem
        >>> BenardMarangoniSystem((8, 8), g=0.5).prepare().system_.interactions[0].g
        0.5
    """

    field = 'physics/fluids'
    topics = ('field', 'convection', 'hexagons', 'surface tension', 'spectral', 'pde')
    references = ('Satz 2022', 'Bestehorn 1993', 'Cross and Hohenberg 1993')
    until = 300.0
    dt = 5.0

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        length: float = 16 * math.pi,
        r: float = 0.1,
        g: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.length = length
        self.r = r
        self.g = g
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a convecting layer with a thermocapillary surface, from a small random field.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        u = 0.01 * rng.uniform(-1.0, 1.0, shape)
        space = LatticeSpace(shape, spacing=self.length / shape[0], discretization='spectral')
        self.system_ = System(
            self.label,
            elements=[MarangoniLayerElement('layer', u=u.ravel(), space=space)],
            interactions=[ThermocapillaryInteraction('relaxation', target='layer', r=self.r, g=self.g)],
            solver=ETDRK4Solver(dt=0.5),
        )
