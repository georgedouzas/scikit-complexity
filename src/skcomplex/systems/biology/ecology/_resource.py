"""Resource that consumers take up, either self-renewing with logistic growth or supplied from outside."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ResourceElement(Element):
    """Resource that consumers take up, either self-renewing with logistic growth or supplied from outside.

    A self-renewing resource grows as `r R (K - R) / K`, and a supplied one flows in at `kappa` and is washed out at
    the rate `omega`. The interaction that couples the resource to its consumers chooses which supply applies.

    Args:
        label:
            The label of the resource or of the group of resources.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        R:
            The initial abundance.

        w:
            The energy a unit of the resource carries.

        r:
            The growth rate of a self-renewing resource at low abundance.

        K:
            The carrying capacity of a self-renewing resource.

        kappa:
            The inflow of a supplied resource.

        omega:
            The dilution rate of a supplied resource.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ResourceElement
        >>> ResourceElement('resources', K=4.8).K
        4.8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        R: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        w: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        omega: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.R = R
        self.w = w
        self.r = r
        self.K = K
        self.kappa = kappa
        self.omega = omega
