"""Site of the Toner-Tu polar active fluid, with a density `rho` and a velocity `(vx, vy)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.laws.utils import advection, divergence, gradient, laplacian
from ....specs import ParameterSpec, StateSpec, VisualSpec


class PolarFluidElement(Element):
    """Site of the Toner-Tu polar active fluid, with a density `rho` and a velocity `(vx, vy)`.

    The density is conserved, `drho/dt + div(rho v) = 0`, and the velocity follows
    `dv/dt + lam (v . grad) v = a (rho - rho_c) v - beta |v|^2 v - sigma grad(rho) + D laplacian(v)` with additive
    noise of intensity `noise`. Above the critical density the fluid moves with the speed
    `sqrt(a (rho - rho_c) / beta)`.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        rho:
            The initial density.

        vx:
            The initial velocity along the first axis.

        vy:
            The initial velocity along the second axis.

        a:
            The rate of the ordering instability per unit density.

        rho_c:
            The critical density of the ordering transition.

        beta:
            The saturation of the speed.

        lam:
            The strength of the self-advection.

        sigma:
            The pressure per unit density.

        D:
            The viscosity of the velocity.

        noise:
            The intensity of the velocity noise.

    Examples:
        >>> from skcomplex.models.physics.active_matter import PolarFluidElement
        >>> PolarFluidElement('site').rho_c
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        vx: Annotated[float, StateSpec()] = 0.0,
        vy: Annotated[float, StateSpec()] = 0.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        rho_c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        lam: Annotated[float, ParameterSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        noise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.vx = vx
        self.vy = vy
        self.a = a
        self.rho_c = rho_c
        self.beta = beta
        self.lam = lam
        self.sigma = sigma
        self.D = D
        self.noise = noise

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the continuity equation and the Toner-Tu equation of each velocity component.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        velocity = [s.vx, s.vy]
        pressure = gradient(s.sigma * s.rho, 2)
        speed_squared = s.vx**2 + s.vy**2
        laws: list[BaseLaw] = [RateLaw(s.rho, -divergence([s.rho * s.vx, s.rho * s.vy]))]
        for axis, component in enumerate(velocity):
            transport = advection(component, [s.lam * item for item in velocity])
            growth = (s.a * (s.rho - s.rho_c) - s.beta * speed_squared) * component
            laws += [
                RateLaw(component, -transport + growth - pressure[axis] + s.D * laplacian(component)),
                NoiseLaw(component, sqrt(2 * s.noise)),
            ]
        return laws
