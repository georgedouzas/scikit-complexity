"""Active Model B+: the conserved scalar field theory of motility-induced phase separation with thermal noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
from sage.all__sagemath_symbolics import sqrt

from ....core.base import Domain, parameter
from ....core.components import Element, Field
from ....core.laws import Law, Noise, Rate, Symbols, divergence, gradient, laplacian
from ....core.runtime import register_model
from ....core.solvers import EulerMaruyama
from ....core.space import LatticeSpace
from ....core.system import System


def _active_model_b_laws(s: Symbols) -> list[Law]:
    """Return the rate and the conserved noise of Active Model B+ for one site."""
    phi = s.phi
    slope = gradient(phi)
    potential = -phi + phi**3 - s.kappa * laplacian(phi) + s.lam * sum(item**2 for item in slope)
    active = [s.zeta * laplacian(phi) * item for item in slope]
    return [Rate(phi, laplacian(potential) - divergence(active)), Noise(phi, sqrt(2 * s.D), conserved=True)]


ActiveScalar = Element.define(
    'ActiveScalar',
    states={'phi': 0.0},
    parameters={
        'kappa': parameter(default=1.0, domain=Domain.POSITIVE),
        'lam': 1.0,
        'zeta': 4.0,
        'D': parameter(default=0.05, domain=Domain.NONNEGATIVE),
    },
    laws=_active_model_b_laws,
)


@register_model(
    field='physics/active_matter',
    tags=('field', 'active matter', 'phase separation', 'microphase separation', 'conserved noise', 'model b', 'spde'),
    references=('Fodor, Jack, and Cates 2022', 'Wittkowski et al. 2014', 'Tjhung, Nardini, and Cates 2018'),
    defaults={'until': 1500.0, 'dt': 30.0, 'solver': EulerMaruyama(dt=0.02), 'backend': 'numba'},
)
def active_model_b(
    shape: Sequence[int] = (64, 64),
    *,
    mean: float = 0.4,
    kappa: float = 1.0,
    lam: float = 1.0,
    zeta: float = 4.0,
    D: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Active Model B+, the conserved field theory of the density of active particles.

    The density `phi` follows `phi_t = laplacian(mu) - div(zeta laplacian(phi) grad(phi)) + div(sqrt(2 D) eta)`, with
    the chemical potential `mu = -phi + phi^3 - kappa laplacian(phi) + lam |grad(phi)|^2` and a conserved white noise
    `eta`, Equation 4 of Fodor, Jack, and Cates with unit mobility. The terms `lam` and `zeta` are the lowest-order
    active currents that no free energy gives. With `zeta = 0` it is Active Model B, where `lam` shifts the coexisting
    densities and the phases still coarsen. With `lam = 1`, `zeta = 4`, and a majority of the dense phase, the
    Ostwald ripening of the dilute bubbles reverses, so they stop growing at a finite size, a microphase separation.
    The noise conserves the total of `phi`, so its mean stays fixed. The lattice has unit spacing, and an explicit step
    must stay below about 1/32.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        mean:
            The mean of `phi`, which the dynamics conserve.

        kappa:
            The stiffness of the interfaces.

        lam:
            The active term `lam |grad(phi)|^2` of the chemical potential.

        zeta:
            The active current `zeta laplacian(phi) grad(phi)`.

        D:
            The temperature of the conserved noise.

        random_state:
            The seed or generator of the initial noise of amplitude 0.05 around the mean.

    Returns:
        system:
            The model, with the field `fluid.phi`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import active_model_b
        >>> active_model_b((8, 8), zeta=0.0).elements[0].values['zeta']
        0.0
    """
    shape = tuple(shape)
    rng = np.random.default_rng(random_state)
    values = {
        'phi': mean + 0.05 * rng.uniform(-1.0, 1.0, shape),
        'kappa': kappa,
        'lam': lam,
        'zeta': zeta,
        'D': D,
    }
    return System(components=[Field('fluid', ActiveScalar, LatticeSpace(shape), values=values)])
