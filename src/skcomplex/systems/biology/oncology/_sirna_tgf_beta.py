"""Effector cells, tumour cells, IL-2, TGF-beta, and the siRNA that silences the production of TGF-beta."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class SirnaTgfBetaElement(Element):
    """Effector cells, tumour cells, IL-2, TGF-beta, and the siRNA that silences the production of TGF-beta.

    The system extends Kirschner and Panetta. TGF-beta lowers the recruitment and the IL-2 driven proliferation of
    effector cells and the production of IL-2, and raises the growth of the tumour. The siRNA, infused at the rate
    `u`, divides the production of TGF-beta by `1 + psi G^gamma3`.

    The element holds the states and parameters and has no laws of its own: the interactions of `sirna_tgf_beta` read
    its parameters.

    Args:
        label:
            The label of the tumour site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        E:
            The initial number of effector cells.

        A:
            The initial number of tumour cells.

        C1:
            The initial concentration of IL-2.

        C2:
            The initial concentration of TGF-beta.

        G:
            The initial amount of siRNA.

        alpha:
            The antigenicity of the tumour.

        gamma1:
            The inhibition of the recruitment of effector cells by TGF-beta.

        d1:
            The death rate of effector cells.

        g1:
            The half-saturation constant of the proliferation of effector cells by IL-2.

        rho1:
            The proliferation rate of effector cells without TGF-beta.

        rmax:
            The largest inhibition of that proliferation by TGF-beta.

        g2:
            The half-saturation constant of that inhibition.

        r1:
            The growth rate of tumour cells.

        b:
            The reciprocal carrying capacity of tumour cells.

        a1:
            The kill rate of tumour cells by effector cells.

        g3:
            The half-saturation constant of the kill.

        rho2:
            The largest growth of tumour cells that TGF-beta adds.

        g4:
            The half-saturation constant of that growth.

        rho3:
            The production rate of IL-2.

        g5:
            The half-saturation constant of the production of IL-2.

        gamma2:
            The inhibition of the production of IL-2 by TGF-beta.

        d2:
            The decay rate of IL-2.

        rho4:
            The largest production rate of TGF-beta.

        mu:
            The number of tumour cells at which the production of TGF-beta rises.

        psi:
            The fraction of the siRNA bound to the mRNA.

        gamma3:
            The exponent of the suppression of the mRNA by the siRNA.

        d3:
            The decay rate of TGF-beta.

        u:
            The infusion rate of siRNA.

        du:
            The decay rate of siRNA.

    Examples:
        >>> from skcomplex.systems.biology.oncology import SirnaTgfBetaElement
        >>> SirnaTgfBetaElement('site').rho4
        300000000.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        C1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        C2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        G: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.035,
        gamma1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        g1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20000000.0,
        rho1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1245,
        rmax: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1121,
        g2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2000000.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.18,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-09,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        g3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100000.0,
        rho2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.27,
        g4: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20000000.0,
        rho3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        g5: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
        gamma2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        rho4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 300000000.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000000.0,
        psi: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        gamma3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        u: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        du: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.66,
    ) -> None:
        super().__init__(label, visual=visual)
        self.E = E
        self.A = A
        self.C1 = C1
        self.C2 = C2
        self.G = G
        self.alpha = alpha
        self.gamma1 = gamma1
        self.d1 = d1
        self.g1 = g1
        self.rho1 = rho1
        self.rmax = rmax
        self.g2 = g2
        self.r1 = r1
        self.b = b
        self.a1 = a1
        self.g3 = g3
        self.rho2 = rho2
        self.g4 = g4
        self.rho3 = rho3
        self.g5 = g5
        self.gamma2 = gamma2
        self.d2 = d2
        self.rho4 = rho4
        self.mu = mu
        self.psi = psi
        self.gamma3 = gamma3
        self.d3 = d3
        self.u = u
        self.du = du
