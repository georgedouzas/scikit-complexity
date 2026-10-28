"""Pattern formation of conserved fields: the nonreciprocal Cahn-Hilliard equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.elements import Element, FieldElement
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import laplacian
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class NonreciprocalMixtureElement(Element):
    """Nonreciprocal mixture element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[object, StateSpec()] = 0.0,
        psi: Annotated[object, StateSpec()] = 0.0,
        chi: Annotated[object, ParameterSpec()] = 0.0,
        alpha: Annotated[object, ParameterSpec()] = 0.3,
        D: Annotated[object, ParameterSpec()] = 0.1,
        kappa: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.psi = psi
        self.chi = chi
        self.alpha = alpha
        self.D = D
        self.kappa = kappa

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the nonreciprocal mixture element."""
        return [
            RateLaw(s.phi, laplacian(-s.phi + s.phi**3 - s.kappa * laplacian(s.phi) + (s.chi + s.alpha) * s.psi)),
            RateLaw(s.psi, laplacian(s.D * s.psi + (s.chi - s.alpha) * s.phi)),
        ]


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
    return System(elements=[FieldElement('mixture', element=NonreciprocalMixtureElement, space=space, values=values)])
