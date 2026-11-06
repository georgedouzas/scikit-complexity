"""Wind-driven gyres: the circulation of a closed ocean basin under a wind stress on the beta plane."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import FieldElement
from ....core.components.interactions import AdvectionInteraction, DiffusionInteraction, RateInteraction
from ....core.components.laws.utils import inverse_helmholtz, partial
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ._quasi_geostrophic_site import QuasiGeostrophicSiteElement


def _streamfunction(s: SimpleNamespace) -> Expression:
    """Return the streamfunction of a site, the inversion of its relative potential vorticity with walls at rest."""
    streamfunction: Expression = inverse_helmholtz(s.zeta, 1 / s.Rd, ('fixed', 0))
    return streamfunction


def _velocity(s: SimpleNamespace) -> list[Expression]:
    """Return the geostrophic velocity of a site, `(-psi_y, psi_x)`."""
    streamfunction = _streamfunction(s.target)
    return [-partial(streamfunction, 1), partial(streamfunction, 0)]


def _beta_drift(s: SimpleNamespace) -> Expression:
    """Return the advection of the planetary vorticity, `-beta psi_x`."""
    drift: Expression = -s.target.beta * partial(_streamfunction(s.target), 0)
    return drift


def _wind(s: SimpleNamespace) -> Expression:
    """Return the input of vorticity by the curl of the wind stress."""
    curl: Expression = s.target.curl
    return curl


def _viscosity(s: SimpleNamespace) -> Expression:
    """Return the viscosity of a site."""
    viscosity: Expression = s.target.nu
    return viscosity


def _sites(values: np.ndarray) -> np.ndarray:
    """Return the values of every site, the observed field itself."""
    return np.array(values)


@register_system(
    field='physics/oceanography',
    tags=('quasi-geostrophic', 'gyre', 'western boundary current', 'field', 'inversion', 'pde'),
    references=('Munk 1950', 'arXiv:1910.00583'),
    defaults={'until': 400.0, 'dt': 0.4, 'solver': RK4Solver(dt=0.4)},
    observables=lambda system: [ObservableSpec('psi', _streamfunction, group='basin', reduce=_sites)],
)
def double_gyre(
    n_sites: int = 64,
    *,
    length: float = 1.0,
    beta: float = 1.0,
    Rd: float = 1.0,
    nu: float = 1.25e-4,
    tau0: float = 1e-5,
) -> System:
    """Return the quasi-geostrophic double gyre, the wind-driven circulation of a closed ocean basin.

    The potential vorticity `q = laplacian(psi) - psi / Rd**2 + beta y` of a square basin of side `length` is carried by
    the geostrophic velocity `(-psi_y, psi_x)` of the streamfunction `psi`, driven by the curl of the wind stress
    `-tau0 cos(2 pi y / length)`, and damped by the viscosity `nu`. The state is the relative potential vorticity
    `zeta = q - beta y`, held at zero on the walls for free slip, so that

        zeta_t + divergence(u zeta) = -beta psi_x - tau0 (2 pi / length) sin(2 pi y / length) + nu laplacian(zeta),

    with `psi = inverse_helmholtz(zeta, 1 / Rd, ('fixed', 0))`. In the viscous steady regime the interior follows the
    Sverdrup balance `beta psi_x = curl(tau)`, and the return flow closes in Munk western boundary layers of width
    `(nu / beta) ** (1 / 3)`, so the streamfunction forms a subtropical and a subpolar gyre of opposite sign whose
    extrema lie close to the western wall. The defaults keep the wind weak, so the boundary layers stay viscous rather
    than inertial, and the circulation is steady after a few hundred units of time. The catalog entry streams the
    streamfunction as the observable `psi`.

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

    Returns:
        system:
            The system, with the relative potential vorticity `basin.zeta`.

    Examples:
        >>> from skcomplex.systems.physics.oceanography import double_gyre
        >>> double_gyre(16).elements[0].values['beta']
        1.0
    """
    spacing = length / n_sites
    latitudes = (np.arange(n_sites) + 0.5) * spacing
    curl = -tau0 * (2 * np.pi / length) * np.sin(2 * np.pi * latitudes / length)
    values = {
        'zeta': 0.0,
        'beta': beta,
        'Rd': Rd,
        'nu': nu,
        'curl': np.broadcast_to(curl, (n_sites, n_sites)).copy(),
    }
    basin = FieldElement(
        'basin',
        element=QuasiGeostrophicSiteElement,
        space=LatticeSpace((n_sites, n_sites), spacing=spacing, boundary='fixed'),
        boundary={'zeta': ('fixed', 0.0)},
        values=values,
    )
    transport = AdvectionInteraction(
        'transport',
        target='basin',
        state='zeta',
        drift=_velocity,
        source=_beta_drift,
        limiter=None,
        riemann=None,
    )
    wind = RateInteraction('wind', target='basin', state='zeta', rate=_wind)
    viscosity = DiffusionInteraction('viscosity', target='basin', state='zeta', coefficient=_viscosity)
    return System(elements=[basin], interactions=[transport, wind, viscosity])
