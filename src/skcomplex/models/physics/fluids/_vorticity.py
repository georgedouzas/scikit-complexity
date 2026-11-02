"""Vorticity of an incompressible two-dimensional flow, `domega/dt + u . grad(omega) = nu laplacian(omega) + f`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class VorticityElement(Element):
    """Vorticity of an incompressible two-dimensional flow, `domega/dt + u . grad(omega) = nu laplacian(omega) + f`.

    The velocity comes from the streamfunction, the solution of a Poisson equation, so the flow is divergence free,
    and the advection is written in conservative form so that dealiasing applies to it. The advection, the viscosity,
    and the forcing are interactions of the model.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        omega:
            The initial vorticity.

        nu:
            The kinematic viscosity.

        f:
            The forcing of the vorticity.

    Examples:
        >>> from skcomplex.models.physics.fluids import VorticityElement
        >>> VorticityElement('site').nu
        0.001
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        omega: Annotated[float, StateSpec()] = 0.0,
        nu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        f: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.omega = omega
        self.nu = nu
        self.f = f
