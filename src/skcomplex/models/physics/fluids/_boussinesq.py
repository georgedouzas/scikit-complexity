"""Vorticity and temperature of Boussinesq convection, in units of the thermal diffusion time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class BoussinesqElement(Element):
    """Vorticity and temperature of Boussinesq convection, in units of the thermal diffusion time.

    The temperature `theta` is the departure from the linear profile between a hot bottom and a cold top, so
    `dtheta/dt + u . grad(theta) = w + laplacian(theta)` and
    `domega/dt + u . grad(omega) = Pr laplacian(omega) + Ra Pr dtheta/dx`. The advections, the diffusions, and the
    buoyancy are interactions of the model.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        omega:
            The initial vorticity.

        theta:
            The initial temperature departure.

        Ra:
            The Rayleigh number.

        Pr:
            The Prandtl number.

    Examples:
        >>> from skcomplex.models.physics.fluids import BoussinesqElement
        >>> BoussinesqElement('site').Ra
        1000.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        omega: Annotated[float, StateSpec()] = 0.0,
        theta: Annotated[float, StateSpec()] = 0.0,
        Ra: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        Pr: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.omega = omega
        self.theta = theta
        self.Ra = Ra
        self.Pr = Pr
