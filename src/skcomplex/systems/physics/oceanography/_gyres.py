"""Wind-driven gyres.

The module holds the circulation of a closed ocean basin under a wind stress on the beta plane.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import inverse_helmholtz, partial
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, RK4Solver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.fields import AdvectionInteraction, DiffusivityInteraction


class QuasiGeostrophicSiteElement(Element):
    """Sites of a quasi-geostrophic ocean basin.

    Every site holds the relative potential vorticity of the water column at one position.

    A site holds the relative potential vorticity `zeta = laplacian(psi) - psi / Rd**2`, the potential vorticity
    without the planetary part `beta y`, with the planetary vorticity gradient `beta`, the deformation radius `Rd`, the
    viscosity `nu`, and the curl of the wind stress `curl` at the site.

    Args:
        label:
            The label of the basin.

        zeta:
            The initial relative potential vorticity, one value or one per site.

        beta:
            The gradient of the planetary vorticity.

        Rd:
            The deformation radius.

        nu:
            The viscosity.

        curl:
            The curl of the wind stress, one value or one per site.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.oceanography import QuasiGeostrophicSiteElement
        >>> QuasiGeostrophicSiteElement('site').beta
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        zeta: object = 0.0,
        beta: Annotated[float, RealSet.open(0, oo)] = 1.0,
        Rd: Annotated[float, RealSet.open(0, oo)] = 1.0,
        nu: Annotated[float, RealSet.open(0, oo)] = 1e-4,
        curl: object = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.zeta = zeta
        self.beta = beta
        self.Rd = Rd
        self.nu = nu
        self.curl = curl


def _get_streamfunction(s: SimpleNamespace) -> Expression:
    """Return the streamfunction of a site.

    It is the inversion of its relative potential vorticity with walls at rest.
    """
    streamfunction: Expression = inverse_helmholtz(s.zeta, 1 / s.Rd, ('fixed', 0))
    return streamfunction


class WindStressInteraction(Interaction):
    """Input of vorticity into a basin by the curl of the wind stress.

    The relative potential vorticity `zeta` of every site of the target grows at the curl `curl` of the wind stress at
    the site, `zeta' = curl`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the basin.

    Examples:
        >>> from skcomplex.systems.physics.oceanography import WindStressInteraction
        >>> WindStressInteraction('wind', target='basin').target
        'basin'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate of the vorticity of the basin to the curl of the wind stress.

        Args:
            context:
                The namespace of the basin and the time `t`.
        """
        super()._prepare(context)
        zeta = context.target.zeta
        self.laws_ = [RateLaw(zeta.diff(context.t), context.target.curl)]


def _get_sites(values: np.ndarray) -> np.ndarray:
    """Return the values of every site, the observed field itself."""
    return np.array(values)


class DoubleGyreSystem(BaseSystem):
    """The quasi-geostrophic double gyre.

    It is the wind-driven circulation of a closed ocean basin.

    The potential vorticity `q = laplacian(psi) - psi / Rd**2 + beta y` of a square basin of side `length` is carried by
    the geostrophic velocity `(-psi_y, psi_x)` of the streamfunction `psi`, driven by the curl of the wind stress
    `-tau0 cos(2 pi y / length)`, and damped by the viscosity `nu`. The state is the relative potential vorticity
    `zeta = q - beta y`, held at zero on the walls for free slip, so that

        zeta_t + divergence(u zeta) = -beta psi_x - tau0 (2 pi / length) sin(2 pi y / length) + nu laplacian(zeta),

    with `psi = inverse_helmholtz(zeta, 1 / Rd, ('fixed', 0))`. In the viscous steady regime the interior follows the
    Sverdrup balance `beta psi_x = curl(tau)`, and the return flow closes in Munk western boundary layers of width `(nu
    / beta) ** (1 / 3)`, so the streamfunction forms a subtropical and a subpolar gyre of opposite sign whose extrema
    lie close to the western wall. The defaults keep the wind weak, so the boundary layers stay viscous rather than
    inertial, and the circulation is steady after a few hundred units of time. The catalog entry streams the
    streamfunction as the observable `psi`. The system has the relative potential vorticity `basin.zeta`.

    Args:
        n_sites:
            The number of sites along each side of the basin.

        length:
            The side of the basin.

        beta:
            The gradient of the planetary vorticity.

        Rd:
            The deformation radius.

        nu:
            The viscosity.

        tau0:
            The amplitude of the wind stress.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.oceanography import DoubleGyreSystem
        >>> DoubleGyreSystem(16).prepare().system_.elements[0].beta
        1.0
    """

    field = 'physics/oceanography'
    topics = ('quasi-geostrophic', 'gyre', 'western boundary current', 'field', 'inversion', 'pde')
    references = ('Munk 1950', 'arXiv:1910.00583')
    until = 400.0
    dt = 0.4
    observables = (Observable('psi', _get_streamfunction, group='basin', reduce=_get_sites),)

    def __init__(
        self,
        n_sites: int = 64,
        *,
        length: float = 1.0,
        beta: float = 1.0,
        Rd: float = 1.0,
        nu: float = 1.25e-4,
        tau0: float = 1e-5,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_sites = n_sites
        self.length = length
        self.beta = beta
        self.Rd = Rd
        self.nu = nu
        self.tau0 = tau0

    @staticmethod
    def _get_velocity(s: SimpleNamespace) -> list[Expression]:
        """Return the geostrophic velocity of a site, `(-psi_y, psi_x)`."""
        streamfunction = _get_streamfunction(s.target)
        return [-partial(streamfunction, 1), partial(streamfunction, 0)]

    @staticmethod
    def _get_beta_drift(s: SimpleNamespace) -> Expression:
        """Return the advection of the planetary vorticity, `-beta psi_x`."""
        drift: Expression = -s.target.beta * partial(_get_streamfunction(s.target), 0)
        return drift

    @staticmethod
    def _get_viscosity(s: SimpleNamespace) -> Expression:
        """Return the viscosity of a site."""
        viscosity: Expression = s.target.nu
        return viscosity

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the wind-driven circulation of a closed ocean basin.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        spacing = self.length / self.n_sites
        latitudes = (np.arange(self.n_sites) + 0.5) * spacing
        curl = -self.tau0 * (2 * np.pi / self.length) * np.sin(2 * np.pi * latitudes / self.length)
        basin = QuasiGeostrophicSiteElement(
            'basin',
            zeta=np.zeros(self.n_sites**2),
            beta=self.beta,
            Rd=self.Rd,
            nu=self.nu,
            curl=np.broadcast_to(curl, (self.n_sites, self.n_sites)).ravel(),
            space=LatticeSpace((self.n_sites, self.n_sites), spacing=spacing, boundary='fixed'),
        )
        transport = AdvectionInteraction(
            'transport',
            target='basin',
            state='zeta',
            drift=self._get_velocity,
            source=self._get_beta_drift,
        )
        wind = WindStressInteraction('wind', target='basin')
        viscosity = DiffusivityInteraction('viscosity', target='basin', state='zeta', coefficient=self._get_viscosity)
        self.system_ = System(
            self.label,
            elements=[basin],
            interactions=[transport, wind, viscosity],
            solver=RK4Solver(dt=0.4),
        )
