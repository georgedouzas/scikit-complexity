"""Effector cells, tumour cells, and IL-2 under adoptive cellular therapy and IL-2 injection."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class KirschnerPanettaTumourElement(Element):
    """Effector cells, tumour cells, and IL-2 under adoptive cellular therapy and IL-2 injection.

    The effector cells follow `dE/dt = alpha_a A - d1 E + a1 u1 + rho1 E C / (alpha1 + C)`, the tumour
    `dA/dt = r1 A (1 - b A) - c1 E A / (alpha2 + A)`, and the IL-2 `dC/dt = a2 u2 + rho2 E A / (alpha3 + A) - du C`,
    with constant inputs `u1` and `u2`. The element holds the states and parameters and has no laws of its own: the
    rate and reaction interactions of `kirschner_panetta` read its parameters.

    Args:
        label:
            The label of the tumour site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        E:
            The initial effector cells.

        A:
            The initial tumour cells.

        C:
            The initial IL-2 concentration.

        alpha_a:
            The antigenicity of the tumour.

        d1:
            The death rate of effector cells.

        rho1:
            The proliferation rate of effector cells stimulated by IL-2.

        alpha1:
            The IL-2 at half the proliferation.

        r1:
            The growth rate of the tumour.

        b:
            The inverse carrying capacity of the tumour.

        c1:
            The kill rate of tumour cells by effector cells.

        alpha2:
            The tumour at half the kill.

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

        u1:
            The adoptive therapy input.

        u2:
            The IL-2 injection input.

    Examples:
        >>> from skcomplex.systems.biology.oncology import KirschnerPanettaTumourElement
        >>> KirschnerPanettaTumourElement('site').rho1
        0.1245
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        C: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        alpha_a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        rho1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1245,
        alpha1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20000000.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.18,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-09,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        alpha2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100000.0,
        rho2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        alpha3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
        du: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 500.0,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 70000000.0,
        u1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.E = E
        self.A = A
        self.C = C
        self.alpha_a = alpha_a
        self.d1 = d1
        self.rho1 = rho1
        self.alpha1 = alpha1
        self.r1 = r1
        self.b = b
        self.c1 = c1
        self.alpha2 = alpha2
        self.rho2 = rho2
        self.alpha3 = alpha3
        self.du = du
        self.a1 = a1
        self.a2 = a2
        self.u1 = u1
        self.u2 = u2
