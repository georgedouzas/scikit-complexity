"""Tumour, dendritic cells, proliferating and active T cells, IL-2, IL-10, and TGF-beta."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class SuppressedTumourElement(Element):
    """Tumour, dendritic cells, proliferating and active T cells, IL-2, IL-10, and TGF-beta.

    The equations are those of Padmanabhan et al. (4.22) to (4.33). The tumour grows by the smooth minimum
    `A / ((1 / r1)^p + (A^(1 - n) / r2)^p)^(1/p)` of exponential and power growth and is killed through its fraction
    `A* = A / (1 + (A^(1 - q) / g3)^p)^(1/p)` accessible to the cytotoxic T cells `E`, at the rate
    `a1 A* E / (E + A* / g1 + ER / g2) / (1 + C3 / C3_50)`. Antigen matures unlicensed dendritic cells, which helper
    memory cells license, and dendritic cells recruit the memory pools `EMC`, `EMH`, and `EMR` into proliferating
    cytotoxic, helper, and regulatory cells, which IL-2 matures. IL-10 and TGF-beta, from regulatory and tumour cells,
    suppress maturation, IL-2 production, and killing, and TGF-beta converts helper into regulatory cells.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `tumour_immune_suppression` read its parameters.

    Args:
        label:
            The label of the tumour site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial tumour cells.

        EUD:
            The initial unlicensed dendritic cells.

        ED:
            The initial licensed dendritic cells.

        EPC:
            The initial proliferating cytotoxic T cells.

        E:
            The initial active cytotoxic T cells.

        EPH:
            The initial proliferating helper T cells.

        EH:
            The initial active helper T cells.

        EPR:
            The initial proliferating regulatory T cells.

        ER:
            The initial active regulatory T cells.

        C1:
            The initial IL-2 concentration.

        C2:
            The initial IL-10 concentration.

        C3:
            The initial TGF-beta concentration.

        r1:
            The exponential growth rate of the tumour.

        r2:
            The power growth coefficient of the tumour.

        n:
            The exponent of the power growth.

        p:
            The smoothness of the transitions.

        q:
            The exponent of the accessible fraction of a large tumour.

        a1:
            The maximal kill rate of tumour cells.

        g1:
            The ratio of tumour to cytotoxic cells at half kill.

        g2:
            The ratio of regulatory to cytotoxic cells at half kill.

        g3:
            The size scale of the accessible fraction.

        C3_50:
            The TGF-beta at half kill.

        alpha_a:
            The antigenicity of the tumour.

        C2_50:
            The IL-10 at half dendritic maturation.

        ER_50:
            The regulatory cells at half dendritic maturation.

        lambda_a:
            The licensing rate of dendritic cells.

        d1:
            The death rate of unlicensed dendritic cells.

        d2:
            The death rate of licensed dendritic cells.

        a2:
            The recruitment rate of proliferating cytotoxic cells.

        g4:
            The steepness of the recruitment by dendritic cells.

        d3:
            The death rate of proliferating cytotoxic cells.

        a3:
            The maturation rate of cytotoxic cells.

        EC1_50:
            The IL-2 at half maturation.

        EC3_50:
            The TGF-beta at half maturation.

        d4:
            The death rate of active cytotoxic cells.

        a4:
            The recruitment rate of proliferating helper cells.

        d5:
            The death rate of proliferating helper cells.

        a5:
            The maturation rate of helper cells.

        a6:
            The conversion rate of helper into regulatory cells.

        g5:
            The TGF-beta at half conversion.

        d6:
            The death rate of active helper cells.

        a7:
            The recruitment rate of proliferating regulatory cells.

        d7:
            The death rate of proliferating regulatory cells.

        a8:
            The maturation rate of regulatory cells.

        d8:
            The death rate of active regulatory cells.

        a9:
            The IL-2 production by helper cells.

        CC2_50:
            The IL-10 at half suppression of IL-2.

        CC3_50:
            The TGF-beta at half suppression of IL-2.

        tau1:
            The lifetime of IL-2.

        r3:
            The IL-10 production by regulatory cells.

        r4:
            The IL-10 production by tumour cells.

        tau2:
            The lifetime of IL-10.

        r5:
            The TGF-beta production by regulatory cells.

        r6:
            The TGF-beta production by tumour cells.

        tau3:
            The lifetime of TGF-beta.

        EMC:
            The memory cytotoxic T cells.

        EMH:
            The memory helper T cells.

        EMR:
            The memory regulatory T cells.

    Examples:
        >>> from skcomplex.systems.biology.oncology import SuppressedTumourElement
        >>> SuppressedTumourElement('site').lambda_a
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        EUD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1e-05,
        ED: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1e-05,
        EPC: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1e-05,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1e-05,
        EPH: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1e-05,
        EH: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1e-05,
        EPR: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1e-05,
        ER: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1e-05,
        C1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        C2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        C3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.333,
        r2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 333.0,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        p: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        q: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2 / 3,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        g1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.2,
        g2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 11.0,
        g3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        C3_50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.5,
        alpha_a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        C2_50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.4,
        ER_50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20000000.0,
        lambda_a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.14,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 23.0,
        g4: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.33,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 16.0,
        EC1_50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        EC3_50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.9,
        d4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        a4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 9.9,
        d5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        a5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.9,
        a6: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.022,
        g5: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.7,
        d6: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        a7: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.1,
        d7: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        a8: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.1,
        d8: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        a9: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.7e-05,
        CC2_50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.75,
        CC3_50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.9,
        tau1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.08,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.4e-08,
        r4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.3e-10,
        tau2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        r5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.8e-08,
        r6: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.1e-07,
        tau3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.07,
        EMC: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10000.0,
        EMH: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10000.0,
        EMR: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10000.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.EUD = EUD
        self.ED = ED
        self.EPC = EPC
        self.E = E
        self.EPH = EPH
        self.EH = EH
        self.EPR = EPR
        self.ER = ER
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.r1 = r1
        self.r2 = r2
        self.n = n
        self.p = p
        self.q = q
        self.a1 = a1
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.C3_50 = C3_50
        self.alpha_a = alpha_a
        self.C2_50 = C2_50
        self.ER_50 = ER_50
        self.lambda_a = lambda_a
        self.d1 = d1
        self.d2 = d2
        self.a2 = a2
        self.g4 = g4
        self.d3 = d3
        self.a3 = a3
        self.EC1_50 = EC1_50
        self.EC3_50 = EC3_50
        self.d4 = d4
        self.a4 = a4
        self.d5 = d5
        self.a5 = a5
        self.a6 = a6
        self.g5 = g5
        self.d6 = d6
        self.a7 = a7
        self.d7 = d7
        self.a8 = a8
        self.d8 = d8
        self.a9 = a9
        self.CC2_50 = CC2_50
        self.CC3_50 = CC3_50
        self.tau1 = tau1
        self.r3 = r3
        self.r4 = r4
        self.tau2 = tau2
        self.r5 = r5
        self.r6 = r6
        self.tau3 = tau3
        self.EMC = EMC
        self.EMH = EMH
        self.EMR = EMR
