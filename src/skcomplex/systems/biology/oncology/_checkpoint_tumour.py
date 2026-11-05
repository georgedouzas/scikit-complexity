"""Activated T cells, tumour cells, and an anti-PD-1 drug that frees T cells from PD-1-PD-L1 inhibition."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class CheckpointTumourElement(Element):
    """Activated T cells, tumour cells, and an anti-PD-1 drug that frees T cells from PD-1-PD-L1 inhibition.

    The T cells follow `dE/dt = (r1 ENC C1 / (C1_50 + C1) + r2 E C2 / (C2_50 + C2)) / (1 + UPP / psi_pp) - d1 E`, with
    the cytokines at their steady states `C1 = C1_ss` and `C2 = C2_ss`, the tumour
    `dA/dt = r3 A (1 - b A) - a_rho A E`, and the drug `dUAP/dt = u_ap - a_rl UFP1 UAP - du2 UAP`. Free PD-1 is
    `UFP1 = max(rho_r - du1 UAP, 0) E`, free PD-L1 `UFL1 = sigma_l (E + c_l A)`, and the complex
    `UPP = sigma UFP1 UFL1`.

    The element holds the states and parameters and has no laws of its own: the interactions of `anti_pd1_checkpoint`
    read its parameters.

    Args:
        label:
            The label of the tumour site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        E:
            The initial activated T cells.

        A:
            The initial tumour cells.

        UAP:
            The initial drug concentration.

        ENC:
            The naive T cells.

        r1:
            The activation rate of naive T cells by IL-12.

        r2:
            The proliferation rate of T cells by IL-2.

        C1_50:
            The IL-12 at half activation.

        C2_50:
            The IL-2 at half proliferation.

        C1_ss:
            The steady IL-12 concentration.

        C2_ss:
            The steady IL-2 concentration.

        psi_pp:
            The PD-1-PD-L1 complex at half inhibition.

        d1:
            The death rate of T cells.

        r3:
            The growth rate of the tumour.

        b:
            The inverse carrying capacity of the tumour.

        a_rho:
            The kill rate of tumour cells by T cells.

        a_rl:
            The blocking rate of PD-1 by the drug.

        rho_r:
            The expression of PD-1 on T cells.

        du1:
            The depletion of PD-1 per unit of drug.

        du2:
            The degradation rate of the drug.

        sigma_l:
            The expression of PD-L1 on T cells.

        c_l:
            The expression of PD-L1 on tumour cells relative to T cells.

        sigma:
            The ratio of association to dissociation of PD-1 and PD-L1.

        u_ap:
            The infusion rate of the drug.

    Examples:
        >>> from skcomplex.systems.biology.oncology import CheckpointTumourElement
        >>> CheckpointTumourElement('site').a_rho
        57.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.006,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.3968,
        UAP: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        ENC: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 8.81,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        C1_50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5e-10,
        C2_50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.37e-11,
        C1_ss: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5e-10,
        C2_ss: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.37e-11,
        psi_pp: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.365e-18,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        r3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.18,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1 / 0.8,
        a_rho: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 57.5,
        a_rl: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4908500.0,
        rho_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.19e-07,
        du1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        du2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.033,
        sigma_l: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.56e-07,
        c_l: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        u_ap: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-10,
    ) -> None:
        super().__init__(label, visual=visual)
        self.E = E
        self.A = A
        self.UAP = UAP
        self.ENC = ENC
        self.r1 = r1
        self.r2 = r2
        self.C1_50 = C1_50
        self.C2_50 = C2_50
        self.C1_ss = C1_ss
        self.C2_ss = C2_ss
        self.psi_pp = psi_pp
        self.d1 = d1
        self.r3 = r3
        self.b = b
        self.a_rho = a_rho
        self.a_rl = a_rl
        self.rho_r = rho_r
        self.du1 = du1
        self.du2 = du2
        self.sigma_l = sigma_l
        self.c_l = c_l
        self.sigma = sigma
        self.u_ap = u_ap
