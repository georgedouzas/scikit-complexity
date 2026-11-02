"""Active Model B+: the conserved scalar field theory of motility-induced phase separation with thermal noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement
from ....core.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.laws.utils import divergence, gradient, laplacian
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.backends import NumbaBackend
from ....engine.solvers import EulerMaruyamaSolver
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class ActiveScalarElement(Element):
    """Site of Active Model B+, a conserved scalar field `phi` that phase separates with active currents.

    The field follows `phi_t = laplacian(mu) - div(zeta laplacian(phi) grad phi) + div(sqrt(2 D) eta)`, with the
    chemical potential `mu = -phi + phi^3 - kappa laplacian(phi) + lam |grad phi|^2`. The active terms `lam` and `zeta`
    break the detailed balance of the Cahn-Hilliard equation, and a large `zeta` reverses Ostwald ripening.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        phi:
            The initial field.

        kappa:
            The stiffness of the interfaces.

        lam:
            The active term of the chemical potential.

        zeta:
            The active current.

        D:
            The strength of the conserved noise.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.models.physics.active_matter import ActiveScalarElement
        >>> ActiveScalarElement('site').zeta
        4.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[float, StateSpec()] = 0.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        lam: Annotated[float, ParameterSpec()] = 1.0,
        zeta: Annotated[float, ParameterSpec()] = 4.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.kappa = kappa
        self.lam = lam
        self.zeta = zeta
        self.D = D

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the conserved rate and the conserved noise of the field.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate and the noise of `phi`.
        """
        s = namespace
        phi = s.phi
        slope = gradient(phi)
        potential = -phi + phi**3 - s.kappa * laplacian(phi) + s.lam * sum(item**2 for item in slope)
        active = [s.zeta * laplacian(phi) * item for item in slope]
        return [RateLaw(phi, laplacian(potential) - divergence(active)), NoiseLaw(phi, sqrt(2 * s.D), conserved=True)]


@register_model(
    field='physics/active_matter',
    tags=('field', 'active matter', 'phase separation', 'microphase separation', 'conserved noise', 'model b', 'spde'),
    references=('Fodor, Jack, and Cates 2022', 'Wittkowski et al. 2014', 'Tjhung, Nardini, and Cates 2018'),
    defaults={'until': 1500.0, 'dt': 30.0, 'solver': EulerMaruyamaSolver(dt=0.02), 'backend': NumbaBackend()},
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
    return System(
        elements=[FieldElement('fluid', element=ActiveScalarElement, space=LatticeSpace(shape), values=values)],
    )
