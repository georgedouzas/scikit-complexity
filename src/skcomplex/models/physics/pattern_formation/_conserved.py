"""Pattern formation of conserved fields: the nonreciprocal Cahn-Hilliard equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.components import Element, Field
from ....core.laws import Rate, laplacian
from ....core.runtime import register_model
from ....core.space import LatticeSpace
from ....core.system import System

NonreciprocalMixture = Element.define(
    'NonreciprocalMixture',
    states={'phi': 0.0, 'psi': 0.0},
    parameters={'chi': 0.0, 'alpha': 0.3, 'D': 0.1, 'kappa': 1.0},
    laws=lambda s: [
        Rate(s.phi, laplacian(-s.phi + s.phi**3 - s.kappa * laplacian(s.phi) + (s.chi + s.alpha) * s.psi)),
        Rate(s.psi, laplacian(s.D * s.psi + (s.chi - s.alpha) * s.phi)),
    ],
)


@register_model(
    field='physics/pattern_formation',
    tags=('field', 'conserved', 'nonreciprocal', 'traveling waves', 'phase separation', 'pde'),
    references=('Brauns and Marchetti 2024', 'Saha, Agudo-Canalejo, and Golestanian 2020', 'Frohoff-Hülsmann 2021'),
    defaults={'until': 400.0, 'dt': 2.0},
)
def nonreciprocal_cahn_hilliard(
    shape: Sequence[int] = (128,),
    *,
    length: float = 40.0,
    chi: float = 0.0,
    alpha: float = 0.3,
    D: float = 0.1,
    kappa: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the nonreciprocal Cahn-Hilliard equations, a demixing field coupled to a diffusing one.

    The conserved field `phi` phase separates and the conserved field `psi` diffuses, and they are coupled through
    cross-diffusion, `phi_t = laplacian(-phi + phi^3 - kappa laplacian(phi) + (chi + alpha) psi)` and
    `psi_t = laplacian(D psi + (chi - alpha) phi)`, the minimal model of Brauns and Marchetti. The coupling `chi` is
    reciprocal and `alpha` is not. When `alpha^2 - chi^2` exceeds `D^2`, the unstable band of the dispersion relation
    of the mixed state reaches the propagating modes, and the demixed domains travel instead of coarsening.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        length:
            The side of the domain.

        chi:
            The reciprocal part of the cross-diffusion.

        alpha:
            The nonreciprocal part of the cross-diffusion.

        D:
            The diffusivity of `psi`.

        kappa:
            The stiffness of the interfaces of `phi`.

        random_state:
            The seed or generator of the small random initial fields around the mixed state.

    Returns:
        system:
            The model, with the demixing field `mixture.phi` and the diffusing field `mixture.psi`.

    Examples:
        >>> from skcomplex.models.physics.pattern_formation import nonreciprocal_cahn_hilliard
        >>> nonreciprocal_cahn_hilliard((16,), alpha=0.5).elements[0].values['alpha']
        0.5
    """
    shape = tuple(shape)
    rng = np.random.default_rng(random_state)
    values = {
        'phi': 0.01 * rng.uniform(-1.0, 1.0, shape),
        'psi': 0.01 * rng.uniform(-1.0, 1.0, shape),
        'chi': chi,
        'alpha': alpha,
        'D': D,
        'kappa': kappa,
    }
    space = LatticeSpace(shape, spacing=length / shape[0])
    return System(components=[Field('mixture', NonreciprocalMixture, space, values=values)])
