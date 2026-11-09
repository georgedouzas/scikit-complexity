"""Pattern formation of conserved fields.

The module holds the nonreciprocal Cahn-Hilliard equations.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.components.elements import Element, FieldElement
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class NonreciprocalMixtureElement(Element):
    """Site of two conserved fields coupled nonreciprocally.

    The fields are a demixing field `phi` and a diffusing field `psi`.

    The fields follow `phi_t = laplacian(-phi + phi^3 - kappa laplacian(phi) + (chi + alpha) psi)` and
    `psi_t = laplacian(D psi + (chi - alpha) phi)`. The reciprocal coupling `chi` derives from a free energy, and the
    nonreciprocal coupling `alpha` does not, so above a threshold the demixed domains travel.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        phi:
            The initial demixing field.

        psi:
            The initial diffusing field.

        chi:
            The reciprocal coupling.

        alpha:
            The nonreciprocal coupling.

        D:
            The diffusion coefficient of `psi`.

        kappa:
            The stiffness of the interfaces of `phi`.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import NonreciprocalMixtureElement
        >>> NonreciprocalMixtureElement('site').alpha
        0.3
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[float, StateSpec()] = 0.0,
        psi: Annotated[float, StateSpec()] = 0.0,
        chi: Annotated[float, ParameterSpec()] = 0.0,
        alpha: Annotated[float, ParameterSpec()] = 0.3,
        D: Annotated[float, ParameterSpec()] = 0.1,
        kappa: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.psi = psi
        self.chi = chi
        self.alpha = alpha
        self.D = D
        self.kappa = kappa

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the conserved rates of both fields.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of `phi` and `psi`.
        """
        s = namespace
        potential = -s.phi + s.phi**3 - s.kappa * laplacian(s.phi) + (s.chi + s.alpha) * s.psi
        return [
            RateLaw(s.phi, laplacian(potential)),
            RateLaw(s.psi, laplacian(s.D * s.psi + (s.chi - s.alpha) * s.phi)),
        ]


class NonreciprocalCahnHilliardSystem(BaseSystem):
    """The nonreciprocal Cahn-Hilliard equations.

    A demixing field is coupled to a diffusing one.

    The conserved field `phi` phase separates and the conserved field `psi` diffuses, and they are coupled through
    cross-diffusion, `phi_t = laplacian(-phi + phi^3 - kappa laplacian(phi) + (chi + alpha) psi)` and `psi_t =
    laplacian(D psi + (chi - alpha) phi)`, the minimal system of Brauns and Marchetti. The coupling `chi` is reciprocal
    and `alpha` is not. When `alpha^2 - chi^2` exceeds `D^2`, the unstable band of the dispersion relation of the mixed
    state reaches the propagating modes, and the demixed domains travel instead of coarsening. The system has the
    demixing field `mixture.phi` and the diffusing field `mixture.psi`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import NonreciprocalCahnHilliardSystem
        >>> NonreciprocalCahnHilliardSystem((16,), alpha=0.5).prepare().system_.elements[0].values['alpha']
        0.5
    """

    field = "physics/pattern_formation"
    topics = ("field", "conserved", "nonreciprocal", "traveling waves", "phase separation", "pde")
    references = ("Brauns and Marchetti 2024", "Saha, Agudo-Canalejo, and Golestanian 2020", "Frohoff-Hülsmann 2021")
    until = 400.0
    dt = 2.0

    def __init__(
        self,
        shape: Sequence[int] = (128,),
        *,
        length: float = 40.0,
        chi: float = 0.0,
        alpha: float = 0.3,
        D: float = 0.1,
        kappa: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.length = length
        self.chi = chi
        self.alpha = alpha
        self.D = D
        self.kappa = kappa
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the nonreciprocal Cahn-Hilliard equations.

        A demixing field is coupled to a diffusing one.
        """
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        values = {
            "phi": 0.01 * rng.uniform(-1.0, 1.0, shape),
            "psi": 0.01 * rng.uniform(-1.0, 1.0, shape),
            "chi": self.chi,
            "alpha": self.alpha,
            "D": self.D,
            "kappa": self.kappa,
        }
        space = LatticeSpace(shape, spacing=self.length / shape[0])
        return System(
            self.label,
            elements=[FieldElement("mixture", element=NonreciprocalMixtureElement, space=space, values=values)],
        )
