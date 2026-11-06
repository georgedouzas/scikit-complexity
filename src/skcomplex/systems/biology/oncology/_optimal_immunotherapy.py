"""Effector cells, tumour cells, and IL-2 under adoptive therapy and IL-2 injection at bounded rates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._optimal_chemotherapy import MAX_DOSE


class OptimalImmunotherapyElement(Element):
    """Effector cells, tumour cells, and IL-2 under adoptive therapy and IL-2 injection at bounded rates.

    The Kirschner-Panetta system in scaled units, with the effector and tumour cells in units of `10^5` and IL-2 in
    units of `2 10^7`, follows `x' = alpha_a y - d1 x + a1 u1 + rho1 x z / (1 + z)` for the effector cells,
    `y' = r1 y (1 - b y) - c1 x y / (1 + y)` for the tumour, and `z' = a2 u2 + rho2 x y / (alpha3 + y) - du z` for
    IL-2. The adoptive therapy `u1` and the IL-2 injection `u2` are control inputs between 0 and 1, and the oncologist
    minimises `∫ (y + w1 u1^2 / 2 + w2 u2^2 / 2) dt + y(T)`.

    The element holds the states and parameters and has no laws of its own: the interactions of `optimal_immunotherapy`
    read its parameters.

    Args:
        label:
            The label of the tumour site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial effector cells.

        y:
            The initial tumour cells.

        z:
            The initial IL-2 concentration.

        u1:
            The adoptive therapy rate, a control input.

        u2:
            The IL-2 injection rate, a control input.

        alpha_a:
            The antigenicity of the tumour.

        d1:
            The death rate of effector cells.

        rho1:
            The proliferation rate of effector cells stimulated by IL-2.

        r1:
            The growth rate of the tumour.

        b:
            The inverse carrying capacity of the tumour.

        c1:
            The kill rate of tumour cells by effector cells.

        rho2:
            The IL-2 production rate by effector-tumour encounters.

        alpha3:
            The tumour at half the IL-2 production.

        du:
            The loss rate of IL-2.

        a1:
            The effector cells per unit of adoptive therapy.

        a2:
            The IL-2 per unit of IL-2 injection.

        w1:
            The weight of the squared adoptive therapy rate in the cost.

        w2:
            The weight of the squared IL-2 injection rate in the cost.

    Examples:
        >>> from skcomplex.systems.biology.oncology import OptimalImmunotherapyElement
        >>> OptimalImmunotherapyElement('site').rho1
        0.1245
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        y: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        z: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u1: Annotated[float, ParameterSpec(adaptive=True, bounds=(0.0, MAX_DOSE))] = 0.0,
        u2: Annotated[float, ParameterSpec(adaptive=True, bounds=(0.0, MAX_DOSE))] = 0.0,
        alpha_a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        rho1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1245,
        r1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.18,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        rho2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.025,
        alpha3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        du: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.5,
        w1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        w2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.u1 = u1
        self.u2 = u2
        self.alpha_a = alpha_a
        self.d1 = d1
        self.rho1 = rho1
        self.r1 = r1
        self.b = b
        self.c1 = c1
        self.rho2 = rho2
        self.alpha3 = alpha3
        self.du = du
        self.a1 = a1
        self.a2 = a2
        self.w1 = w1
        self.w2 = w2
