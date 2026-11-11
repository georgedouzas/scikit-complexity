"""Tumour-immune interactions under immunotherapy.

The therapies are adoptive therapy, vaccines, cytokines, and checkpoints.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, log, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import delayed, hill, where
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import ScipyODESolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ...general.reactions import ReactionInteraction
from ..common.kinetics import MichaelisMentenInteraction
from ..common.pharmacokinetics import DosingInteraction

BOOK_REFERENCE = 'Padmanabhan, Meskin, and Al Moustafa 2021'
STIFF_SOLVER = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)


class CheckpointTumourElement(Element):
    """Activated T cells, tumour cells, and an anti-PD-1 drug.

    The drug frees T cells from PD-1-PD-L1 inhibition.

    The T cells follow `dE/dt = (r1 ENC C1 / (C1_50 + C1) + r2 E C2 / (C2_50 + C2)) / (1 + UPP / psi_pp) - d1 E`, with
    the cytokines at their steady states `C1 = C1_ss` and `C2 = C2_ss`, the tumour
    `dA/dt = r3 A (1 - b A) - a_rho A E`, and the drug `dUAP/dt = u_ap - a_rl UFP1 UAP - du2 UAP`. Free PD-1 is
    `UFP1 = max(rho_r - du1 UAP, 0) E`, free PD-L1 `UFL1 = sigma_l (E + c_l A)`, and the complex
    `UPP = sigma UFP1 UFL1`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `AntiPD1CheckpointSystem` read its parameters.

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


class DelayedVaccinatedTumourElement(Element):
    """Tumour, natural killer cells, and adaptive immune cells with IL-2.

    The adaptive cells are naive and mature CD8+ cells, dendritic cells, regulatory cells, and helper cells.

    The equations are those of Padmanabhan et al. (4.13) to (4.21), with the delays `tau1`, `tau2`, and `tau3` of one
    division of natural killer, CD8+, and helper cells. Naive CD8+ cells mature after `7 tau2` and naive helper cells
    after `2 tau3`. The tumour is killed by mature CD8+ cells through `a1 x^5 / (g1 + x^5)` with
    `x = log(E) / log(A)`. The vaccine `u` is `dose` for `width` time units every `period` from `start`, `doses`
    times.

    The element holds the states and parameters and has no laws of its own: the interactions of `DelayedDCVaccineSystem`
    read its parameters.

    Args:
        label:
            The label of the tumour site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial tumour cells.

        ENK:
            The initial natural killer cells.

        ENC:
            The initial naive CD8+ cells.

        E:
            The initial mature CD8+ cells.

        ED:
            The initial dendritic cells.

        C:
            The initial IL-2 concentration.

        ER:
            The initial regulatory T cells.

        ENH:
            The initial naive helper T cells.

        E1H:
            The initial type 1 helper T cells.

        r1:
            The growth rate of the tumour.

        b:
            The carrying capacity of the tumour.

        a1:
            The maximal kill rate by mature CD8+ cells.

        g1:
            The steepness of the kill by mature CD8+ cells.

        a2:
            The kill rate by natural killer cells.

        s1:
            The influx of natural killer cells.

        d1:
            The death rate of natural killer cells.

        rho1:
            The production rate of natural killer cells by IL-2.

        g2:
            The IL-2 at half production of natural killer cells.

        psi1:
            The inactivation rate of natural killer cells by regulatory cells.

        s2:
            The influx of naive CD8+ cells.

        d2:
            The death rate of naive CD8+ cells.

        rho2:
            The activation rate of naive CD8+ cells by dendritic cells.

        g3:
            The dendritic cells at half activation.

        d3:
            The death rate of mature CD8+ cells.

        rho3:
            The production rate of mature CD8+ cells by dendritic cells.

        m1:
            The proliferation rate of mature CD8+ cells with dendritic cells.

        rho4:
            The proliferation rate of mature CD8+ cells by IL-2.

        g4:
            The squared IL-2 at half proliferation.

        psi2:
            The inactivation rate of mature CD8+ cells by regulatory cells.

        d4:
            The death rate of dendritic cells.

        kappa1:
            The production of dendritic cells from tumour antigen.

        g5:
            The tumour at half production of dendritic cells.

        s3:
            The influx of IL-2.

        d5:
            The decay rate of IL-2.

        rho5:
            The secretion rate of IL-2 by mature CD8+ cells.

        g6:
            The squared mature CD8+ cells at half secretion.

        rho6:
            The secretion rate of IL-2 by type 1 helper cells.

        g7:
            The type 1 helper cells at half secretion.

        s4:
            The influx of regulatory cells.

        d6:
            The death rate of regulatory cells.

        rho7:
            The production rate of regulatory cells by the tumour.

        g8:
            The squared tumour at half production of regulatory cells.

        rho8:
            The production rate of regulatory cells by IL-2.

        g9:
            The IL-2 at half production of regulatory cells.

        s5:
            The influx of naive helper cells.

        d7:
            The death rate of naive helper cells.

        rho9:
            The activation rate of naive helper cells by dendritic cells.

        d8:
            The death rate of type 1 helper cells.

        m2:
            The maturation rate of naive helper cells.

        psi3:
            The inactivation rate of type 1 helper cells by regulatory cells.

        rho10:
            The proliferation rate of type 1 helper cells with dendritic cells.

        tau1:
            The division time of natural killer cells.

        tau2:
            The division time of CD8+ cells.

        tau3:
            The division time of helper cells.

        dose:
            The injection rate of dendritic cells during a dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

        doses:
            The number of doses.

    Examples:
        >>> from skcomplex.systems.biology.oncology import DelayedVaccinatedTumourElement
        >>> DelayedVaccinatedTumourElement('site').tau1
        1.26
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 500000.0,
        ENK: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 44000.0,
        ENC: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 120000.0,
        E: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 10.0,
        ED: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        C: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 48.0,
        ER: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1100000.0,
        ENH: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 126000.0,
        E1H: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.16,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5000000000.0,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 6.9,
        g1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.5,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 7.13e-10,
        s1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 13200.0,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0412,
        rho1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0668,
        g2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 25000.0,
        psi1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-07,
        s2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 500000.0,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        rho2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.2,
        g3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10**5.4,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        rho3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 8.98e-07,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.38,
        rho4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.19,
        g4: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 6250000.0,
        psi2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.3e-07,
        d4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.237,
        kappa1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 24380.0,
        g5: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100000.0,
        s3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 280.5,
        d5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.5,
        rho5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.1,
        g6: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100000000000.0,
        rho6: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.99,
        g7: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100000.0,
        s4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 321200.0,
        d6: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        rho7: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.26,
        g8: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.5e17,
        rho8: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.399,
        g9: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10**4.4,
        s5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 550000.0,
        d7: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        rho9: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.05,
        d8: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.44,
        m2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.45,
        psi3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 8e-07,
        rho10: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.4,
        tau1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.26,
        tau2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.45,
        tau3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.47,
        dose: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1000000.0,
        period: float = 7.0,
        width: float = 1.0,
        start: float = 0.0,
        doses: int = 4,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.ENK = ENK
        self.ENC = ENC
        self.E = E
        self.ED = ED
        self.C = C
        self.ER = ER
        self.ENH = ENH
        self.E1H = E1H
        self.r1 = r1
        self.b = b
        self.a1 = a1
        self.g1 = g1
        self.a2 = a2
        self.s1 = s1
        self.d1 = d1
        self.rho1 = rho1
        self.g2 = g2
        self.psi1 = psi1
        self.s2 = s2
        self.d2 = d2
        self.rho2 = rho2
        self.g3 = g3
        self.d3 = d3
        self.rho3 = rho3
        self.m1 = m1
        self.rho4 = rho4
        self.g4 = g4
        self.psi2 = psi2
        self.d4 = d4
        self.kappa1 = kappa1
        self.g5 = g5
        self.s3 = s3
        self.d5 = d5
        self.rho5 = rho5
        self.g6 = g6
        self.rho6 = rho6
        self.g7 = g7
        self.s4 = s4
        self.d6 = d6
        self.rho7 = rho7
        self.g8 = g8
        self.rho8 = rho8
        self.g9 = g9
        self.s5 = s5
        self.d7 = d7
        self.rho9 = rho9
        self.d8 = d8
        self.m2 = m2
        self.psi3 = psi3
        self.rho10 = rho10
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start
        self.doses = doses


class KirschnerPanettaTumourElement(Element):
    """Effector cells, tumour cells, and IL-2 under adoptive therapy and IL-2.

    The therapy is adoptive cellular therapy and IL-2 injection.

    The effector cells follow `dE/dt = alpha_a A - d1 E + a1 u1 + rho1 E C / (alpha1 + C)`, the tumour
    `dA/dt = r1 A (1 - b A) - c1 E A / (alpha2 + A)`, and the IL-2 `dC/dt = a2 u2 + rho2 E A / (alpha3 + A) - du C`,
    with constant inputs `u1` and `u2`. The element holds the states and parameters and has no laws of its own: the
    rate and reaction interactions of `KirschnerPanettaSystem` read its parameters.

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


class SuppressedTumourElement(Element):
    """Tumour, dendritic cells, T cells, and cytokines.

    The T cells are proliferating and active, and the cytokines are IL-2, IL-10, and TGF-beta.

    The equations are those of Padmanabhan et al. (4.22) to (4.33). The tumour grows by the smooth minimum
    `A / ((1 / r1)^p + (A^(1 - n) / r2)^p)^(1/p)` of exponential and power growth and is killed through its fraction
    `A* = A / (1 + (A^(1 - q) / g3)^p)^(1/p)` accessible to the cytotoxic T cells `E`, at the rate
    `a1 A* E / (E + A* / g1 + ER / g2) / (1 + C3 / C3_50)`. Antigen matures unlicensed dendritic cells, which helper
    memory cells license, and dendritic cells recruit the memory pools `EMC`, `EMH`, and `EMR` into proliferating
    cytotoxic, helper, and regulatory cells, which IL-2 matures. IL-10 and TGF-beta, from regulatory and tumour cells,
    suppress maturation, IL-2 production, and killing, and TGF-beta converts helper into regulatory cells.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `TumourImmuneSuppressionSystem` read its parameters.

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


class VaccinatedTumourElement(Element):
    """Helper and cytotoxic T cells, tumour cells, dendritic cells, and IL-2.

    They are under dendritic-cell vaccination.

    The helper cells follow `dEH/dt = s1 + rho1 ED r1 EH (1 - b1 EH) - d1 EH`, the cytotoxic cells
    `dE/dt = s2 + rho2 C (A + ED) r2 E (1 - b2 E) - d2 E`, the tumour `dA/dt = r3 A (1 - b3 A) - c1 A E`, the
    dendritic cells `dED/dt = -c2 E ED + u(t)`, and the IL-2 `dC/dt = r4 EH ED - c3 E C - d3 C`. The vaccine `u` is
    `dose` for `width` time units every `period` from `start`, `doses` times.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `DendriticCellVaccineSystem` read its parameters.

    Args:
        label:
            The label of the tumour site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        EH:
            The initial helper T cells.

        E:
            The initial cytotoxic T cells.

        A:
            The initial tumour cells.

        ED:
            The initial dendritic cells.

        C:
            The initial IL-2 concentration.

        s1:
            The influx of helper T cells.

        s2:
            The influx of cytotoxic T cells.

        rho1:
            The proliferation constant of helper T cells.

        rho2:
            The proliferation constant of cytotoxic T cells.

        r1:
            The growth rate of helper T cells.

        r2:
            The growth rate of cytotoxic T cells.

        r3:
            The growth rate of the tumour.

        r4:
            The IL-2 secretion rate.

        b1:
            The inverse capacity of helper T cells.

        b2:
            The inverse capacity of cytotoxic T cells.

        b3:
            The inverse capacity of the tumour.

        d1:
            The death rate of helper T cells.

        d2:
            The death rate of cytotoxic T cells.

        d3:
            The degradation rate of IL-2.

        c1:
            The kill rate of tumour cells by cytotoxic T cells.

        c2:
            The kill rate of dendritic cells by cytotoxic T cells.

        c3:
            The consumption rate of IL-2 by cytotoxic T cells.

        dose:
            The injection rate of dendritic cells during a dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

        doses:
            The number of doses.

    Examples:
        >>> from skcomplex.systems.biology.oncology import VaccinatedTumourElement
        >>> VaccinatedTumourElement('site').r3
        0.02
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        EH: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        ED: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        C: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        s1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        s2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        rho1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        rho2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        r3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.02,
        r4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        b1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        b2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        b3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        c3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-07,
        dose: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        period: float = 168.0,
        width: float = 1.0,
        start: float = 0.0,
        doses: int = 8,
    ) -> None:
        super().__init__(label, visual=visual)
        self.EH = EH
        self.E = E
        self.A = A
        self.ED = ED
        self.C = C
        self.s1 = s1
        self.s2 = s2
        self.rho1 = rho1
        self.rho2 = rho2
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start
        self.doses = doses


class KirschnerPanettaSystem(BaseSystem):
    """The Kirschner-Panetta system of effector cells, tumour cells, and IL-2.

    The system is under adoptive therapy and IL-2.

    Untreated, a tumour of antigenicity in the oscillatory range, such as the default 0.02, and the effector cells
    oscillate in long cycles in which the tumour nearly vanishes and regrows to about `4 10^5`. The adoptive therapy
    adds `a1 u1` effector cells per day and the IL-2 injection `a2 u2` IL-2 per day. With `u1` and `u2` above 1, the
    book's treatment factors `a1 = 500` and `a2 = 7 10^7` are exceeded and the tumour is cleared. The populations are in
    volume units, the inputs in volume per day, and time in days, and the initial state is the book's. The system has
    the effector cells `site.E`, the tumour cells `site.A`, and the IL-2 `site.C`.

    Args:
        E:
            The initial effector cells.

        A:
            The initial tumour cells.

        C:
            The initial IL-2 concentration.

        alpha_a:
            The antigenicity of the tumour, in the book's range 0 to 0.05.

        r1:
            The growth rate of the tumour.

        c1:
            The kill rate of tumour cells by effector cells.

        a1:
            The effector cells per unit of adoptive therapy.

        a2:
            The IL-2 per unit of IL-2 injection.

        u1:
            The adoptive therapy input.

        u2:
            The IL-2 injection input.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import KirschnerPanettaSystem
        >>> KirschnerPanettaSystem(u1=1.0).prepare().system_.elements[0].u1
        1.0
    """

    field = 'biology/oncology'
    topics = ('immunotherapy', 'ode', 'adoptive therapy', 'IL-2', 'oscillations')
    references = (BOOK_REFERENCE, 'Kirschner and Panetta 1998')
    until = 1000.0
    dt = 1.0
    solver = STIFF_SOLVER

    def __init__(
        self,
        E: float = 1.0,
        A: float = 1.0,
        C: float = 1.0,
        *,
        alpha_a: float = 0.02,
        r1: float = 0.18,
        c1: float = 1.0,
        a1: float = 500.0,
        a2: float = 7e7,
        u1: float = 0.0,
        u2: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.E = E
        self.A = A
        self.C = C
        self.alpha_a = alpha_a
        self.r1 = r1
        self.c1 = c1
        self.a1 = a1
        self.a2 = a2
        self.u1 = u1
        self.u2 = u2

    def _build_system(self) -> CompositeSystem:
        """Return the components of the Kirschner-Panetta system of effector cells."""
        site = KirschnerPanettaTumourElement(
            'site',
            E=self.E,
            A=self.A,
            C=self.C,
            alpha_a=self.alpha_a,
            r1=self.r1,
            c1=self.c1,
            a1=self.a1,
            a2=self.a2,
            u1=self.u1,
            u2=self.u2,
        )
        interactions = [
            RateInteraction('stimulation', target='site', state='E', rate=lambda s: s.target.alpha_a * s.target.A),
            ReactionInteraction(
                'effector_death',
                target='site',
                reactants={'E': 1},
                propensity=lambda s: s.target.d1 * s.target.E,
            ),
            DosingInteraction('adoptive_therapy', target='site', state='E', dose=lambda s: s.target.a1 * s.target.u1),
            MichaelisMentenInteraction(
                'effector_proliferation',
                target='site',
                products={'E': 1},
                substrate='C',
                vmax=lambda s: s.target.rho1,
                km=lambda s: s.target.alpha1,
                modifiers={'E': 1},
            ),
            RateInteraction(
                'tumour_growth',
                target='site',
                state='A',
                rate=lambda s: s.target.r1 * s.target.A * (1 - s.target.b * s.target.A),
            ),
            MichaelisMentenInteraction(
                'kill',
                target='site',
                reactants={'A': 1},
                substrate='A',
                vmax=lambda s: s.target.c1,
                km=lambda s: s.target.alpha2,
                modifiers={'E': 1},
            ),
            DosingInteraction('il2_injection', target='site', state='C', dose=lambda s: s.target.a2 * s.target.u2),
            MichaelisMentenInteraction(
                'il2_production',
                target='site',
                products={'C': 1},
                substrate='A',
                vmax=lambda s: s.target.rho2,
                km=lambda s: s.target.alpha3,
                modifiers={'E': 1},
            ),
            ReactionInteraction(
                'il2_loss',
                target='site',
                reactants={'C': 1},
                propensity=lambda s: s.target.du * s.target.C,
            ),
        ]
        return CompositeSystem(self.label, elements=[site], interactions=interactions)


class DendriticCellVaccineSystem(BaseSystem):
    """Helper and cytotoxic T cells, a tumour, dendritic cells, and IL-2.

    They are under a dendritic-cell vaccine.

    Vaccinated dendritic cells drive the proliferation of helper cells, which secrete IL-2, and IL-2 with the antigen
    drives the proliferation of cytotoxic cells, which kill the tumour. The book gives no initial state or schedule. The
    T cells start at their resting level `s / d = 0.02`, the dendritic cells and IL-2 at 0, and the vaccine is a
    one-hour injection of `dose` per hour every week, 8 times. Without vaccine the tumour settles at `1 - c1 s2 / (d2
    r3) = 0.9`. The populations are in cells per mm³ scaled by the carrying capacity, IL-2 in pg/mm³, and time in hours.
    The system has the helper cells `site.EH`, the cytotoxic cells `site.E`, the tumour `site.A`, the dendritic cells
    `site.ED`, and the IL-2 `site.C`.

    Args:
        A:
            The initial tumour cells.

        rho2:
            The proliferation constant of cytotoxic T cells.

        r4:
            The IL-2 secretion rate.

        dose:
            The injection rate of dendritic cells during a dose, 0 for no vaccine.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

        doses:
            The number of doses.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import DendriticCellVaccineSystem
        >>> DendriticCellVaccineSystem(dose=0.0).prepare().system_.elements[0].dose
        0.0
    """

    field = 'biology/oncology'
    topics = ('immunotherapy', 'ode', 'vaccine', 'dendritic cells', 'dosing')
    references = (BOOK_REFERENCE, 'Castiglione and Piccoli 2006')
    until = 2880.0
    dt = 2.0
    solver = STIFF_SOLVER

    def __init__(
        self,
        A: float = 0.1,
        *,
        rho2: float = 10.0,
        r4: float = 1e-2,
        dose: float = 0.5,
        period: float = 168.0,
        width: float = 1.0,
        start: float = 0.0,
        doses: int = 8,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.A = A
        self.rho2 = rho2
        self.r4 = r4
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start
        self.doses = doses

    def _build_system(self) -> CompositeSystem:
        """Return the components of helper and cytotoxic T cells."""
        site = VaccinatedTumourElement(
            'site',
            A=self.A,
            rho2=self.rho2,
            r4=self.r4,
            dose=self.dose,
            period=self.period,
            width=self.width,
            start=self.start,
            doses=self.doses,
        )
        interactions = [
            DosingInteraction('helper_influx', target='site', state='EH', dose=lambda s: s.target.s1),
            RateInteraction(
                'helper_proliferation',
                target='site',
                state='EH',
                rate=lambda s: s.target.rho1
                * s.target.ED
                * s.target.r1
                * s.target.EH
                * (1 - s.target.b1 * s.target.EH),
            ),
            ReactionInteraction(
                'helper_death',
                target='site',
                reactants={'EH': 1},
                propensity=lambda s: s.target.d1 * s.target.EH,
            ),
            DosingInteraction('cytotoxic_influx', target='site', state='E', dose=lambda s: s.target.s2),
            RateInteraction(
                'cytotoxic_proliferation',
                target='site',
                state='E',
                rate=lambda s: s.target.rho2
                * s.target.C
                * (s.target.A + s.target.ED)
                * s.target.r2
                * s.target.E
                * (1 - s.target.b2 * s.target.E),
            ),
            ReactionInteraction(
                'cytotoxic_death',
                target='site',
                reactants={'E': 1},
                propensity=lambda s: s.target.d2 * s.target.E,
            ),
            RateInteraction(
                'tumour_growth',
                target='site',
                state='A',
                rate=lambda s: s.target.r3 * s.target.A * (1 - s.target.b3 * s.target.A),
            ),
            ReactionInteraction(
                'kill',
                target='site',
                reactants={'A': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.c1 * s.target.A * s.target.E,
            ),
            DosingInteraction(
                'vaccine',
                target='site',
                state='ED',
                dose=lambda s: s.target.dose,
                period=self.period,
                width=self.width,
                start=self.start,
                count=self.doses,
            ),
            ReactionInteraction(
                'dendritic_kill',
                target='site',
                reactants={'ED': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.c2 * s.target.E * s.target.ED,
            ),
            RateInteraction(
                'il2_production',
                target='site',
                state='C',
                rate=lambda s: s.target.r4 * s.target.EH * s.target.ED,
            ),
            ReactionInteraction(
                'il2_consumption',
                target='site',
                reactants={'C': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.c3 * s.target.E * s.target.C,
            ),
            ReactionInteraction(
                'il2_loss',
                target='site',
                reactants={'C': 1},
                propensity=lambda s: s.target.d3 * s.target.C,
            ),
        ]
        return CompositeSystem(self.label, elements=[site], interactions=interactions)


class AntiPD1CheckpointSystem(BaseSystem):
    """Activated T cells, a tumour, and an anti-PD-1 drug.

    The drug frees the T cells from checkpoint inhibition.

    The PD-1-PD-L1 complex inhibits the activation and proliferation of T cells, and the drug lowers the free PD-1 on T
    cells. The cytokines are at their steady states, as in the book. The book gives neither the naive T cells `ENC` nor
    the association ratio `sigma`, which are set so that the T cells settle below the level that controls the tumour.
    The ranged parameters take the low ends of the book's ranges, and the drug is the book's continuous infusion of
    `10^-10` per day. A larger infusion frees more T cells and lowers the tumour. The book's free PD-1 `(rho_r - du1
    UAP) E` turns negative once the drug exceeds `rho_r / du1`, and it is held at 0 there. With all PD-1 blocked nothing
    limits the T cells, which grow without bound and clear the tumour. Cells and the drug are in g/cm³ and time in days.
    The system has the T cells `site.E`, the tumour `site.A`, and the drug `site.UAP`.

    Args:
        E:
            The initial activated T cells.

        A:
            The initial tumour cells.

        u_ap:
            The infusion rate of the drug, 0 for no treatment.

        sigma:
            The ratio of association to dissociation of PD-1 and PD-L1.

        ENC:
            The naive T cells.

        r3:
            The growth rate of the tumour.

        b:
            The inverse carrying capacity of the tumour.

        c_l:
            The expression of PD-L1 on tumour cells relative to T cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import AntiPD1CheckpointSystem
        >>> AntiPD1CheckpointSystem(u_ap=0.0).prepare().system_.elements[0].u_ap
        0.0
    """

    field = 'biology/oncology'
    topics = ('immunotherapy', 'ode', 'checkpoint inhibitor', 'PD-1', 'dosing')
    references = (BOOK_REFERENCE, 'Nikolopoulou et al. 2018')
    until = 150.0
    dt = 0.5
    solver = STIFF_SOLVER

    def __init__(
        self,
        E: float = 6e-3,
        A: float = 0.3968,
        *,
        u_ap: float = 1e-10,
        sigma: float = 5.0,
        ENC: float = 1e-4,
        r3: float = 0.18,
        b: float = 1 / 0.8,
        c_l: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.E = E
        self.A = A
        self.u_ap = u_ap
        self.sigma = sigma
        self.ENC = ENC
        self.r3 = r3
        self.b = b
        self.c_l = c_l

    def _build_system(self) -> CompositeSystem:
        """Return the components of activated T cells."""
        site = CheckpointTumourElement(
            'site',
            E=self.E,
            A=self.A,
            u_ap=self.u_ap,
            sigma=self.sigma,
            ENC=self.ENC,
            r3=self.r3,
            b=self.b,
            c_l=self.c_l,
        )

        def free_receptor(s: SimpleNamespace) -> Expression:
            """Return the free PD-1 on the T cells.

            The drug lowers it.
            """
            expression = s.target.rho_r - s.target.du1 * s.target.UAP
            receptor: Expression = where(expression > 0, expression, 0) * s.target.E
            return receptor

        interactions = [
            RateInteraction(
                'activation',
                target='site',
                state='E',
                rate=lambda s: (
                    s.target.r1 * s.target.ENC * hill(s.target.C1_ss, s.target.C1_50)
                    + s.target.r2 * s.target.E * hill(s.target.C2_ss, s.target.C2_50)
                )
                / (
                    1
                    + s.target.sigma
                    * free_receptor(s)
                    * s.target.sigma_l
                    * (s.target.E + s.target.c_l * s.target.A)
                    / s.target.psi_pp
                ),
            ),
            ReactionInteraction(
                'effector_death',
                target='site',
                reactants={'E': 1},
                propensity=lambda s: s.target.d1 * s.target.E,
            ),
            RateInteraction(
                'tumour_growth',
                target='site',
                state='A',
                rate=lambda s: s.target.r3 * s.target.A * (1 - s.target.b * s.target.A),
            ),
            ReactionInteraction(
                'kill',
                target='site',
                reactants={'A': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.a_rho * s.target.A * s.target.E,
            ),
            DosingInteraction('infusion', target='site', state='UAP', dose=lambda s: s.target.u_ap),
            ReactionInteraction(
                'binding',
                target='site',
                reactants={'UAP': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.a_rl * free_receptor(s) * s.target.UAP,
            ),
            ReactionInteraction(
                'clearance',
                target='site',
                reactants={'UAP': 1},
                propensity=lambda s: s.target.du2 * s.target.UAP,
            ),
        ]
        return CompositeSystem(self.label, elements=[site], interactions=interactions)


class DelayedDCVaccineSystem(BaseSystem):
    """A tumour and eight innate and adaptive immune populations under a vaccine.

    The populations divide with delays.

    Natural killer cells divide in `tau1`, CD8+ cells mature after seven divisions `7 tau2`, and naive helper cells
    become type 1 helper cells after two divisions `2 tau3`. Regulatory T cells, driven by the tumour and IL-2, inhibit
    natural killer, CD8+, and helper cells. The book's growth term `m1 ED E` of mature CD8+ cells has no saturation and
    makes them diverge within a day, so it saturates in the dendritic cells as `m1 ED E / (g3 + ED)`, like every other
    term the dendritic cells drive. The book gives no schedule, and the vaccine is a one-day injection of `dose`
    dendritic cells per day every week, 4 times. The parameters and initial state are the book's, and with them the
    regulatory T cells hold the adaptive response down, so the tumour grows toward its capacity `b`. Cells are counts,
    IL-2 is in IU/L, and time in days. The system has the tumour `site.A`, the natural killer cells `site.ENK`, the
    naive and mature CD8+ cells `site.ENC` and `site.E`, the dendritic cells `site.ED`, the IL-2 `site.C`, the
    regulatory cells `site.ER`, and the naive and type 1 helper cells `site.ENH` and `site.E1H`.

    Args:
        dose:
            The injection rate of dendritic cells during a dose, 0 for no vaccine.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

        doses:
            The number of doses.

        tau1:
            The division time of natural killer cells.

        tau2:
            The division time of CD8+ cells.

        tau3:
            The division time of helper cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import DelayedDCVaccineSystem
        >>> DelayedDCVaccineSystem(tau2=0.5).prepare().system_.elements[0].tau2
        0.5
    """

    field = 'biology/oncology'
    topics = ('immunotherapy', 'delay', 'vaccine', 'dendritic cells', 'regulatory T cells')
    references = (BOOK_REFERENCE, 'Qomlaqi et al. 2017')
    until = 30.0
    dt = 0.1
    solver = STIFF_SOLVER

    def __init__(
        self,
        *,
        dose: float = 1e6,
        period: float = 7.0,
        width: float = 1.0,
        start: float = 0.0,
        doses: int = 4,
        tau1: float = 1.26,
        tau2: float = 0.45,
        tau3: float = 0.47,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start
        self.doses = doses
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3

    def _build_system(self) -> CompositeSystem:
        """Return the components of a tumour and eight innate and adaptive immune populations."""
        site = DelayedVaccinatedTumourElement(
            'site',
            dose=self.dose,
            period=self.period,
            width=self.width,
            start=self.start,
            doses=self.doses,
            tau1=self.tau1,
            tau2=self.tau2,
            tau3=self.tau3,
        )
        interactions = [
            RateInteraction(
                'tumour_growth',
                target='site',
                state='A',
                rate=lambda s: s.target.r1 * s.target.A * log(s.target.b / s.target.A),
            ),
            ReactionInteraction(
                'cytotoxic_kill',
                target='site',
                reactants={'A': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.a1
                * (log(s.target.E) / log(s.target.A)) ** 5
                / ((log(s.target.E) / log(s.target.A)) ** 5 + s.target.g1)
                * s.target.A,
            ),
            ReactionInteraction(
                'nk_kill',
                target='site',
                reactants={'A': 1},
                modifiers={'ENK': 1},
                propensity=lambda s: s.target.a2 * s.target.ENK * s.target.A,
            ),
            DosingInteraction('nk_influx', target='site', state='ENK', dose=lambda s: s.target.s1),
            ReactionInteraction(
                'nk_death',
                target='site',
                reactants={'ENK': 1},
                propensity=lambda s: s.target.d1 * s.target.ENK,
            ),
            RateInteraction(
                'nk_proliferation',
                target='site',
                state='ENK',
                rate=lambda s: s.target.rho1
                * delayed(s.target.ENK, s.target.tau1)
                * delayed(s.target.C, s.target.tau1)
                / (s.target.g2 + delayed(s.target.C, s.target.tau1)),
            ),
            ReactionInteraction(
                'nk_suppression',
                target='site',
                reactants={'ENK': 1},
                modifiers={'ER': 1},
                propensity=lambda s: s.target.psi1 * s.target.ENK * s.target.ER,
            ),
            DosingInteraction('naive_cytotoxic_influx', target='site', state='ENC', dose=lambda s: s.target.s2),
            ReactionInteraction(
                'naive_cytotoxic_death',
                target='site',
                reactants={'ENC': 1},
                propensity=lambda s: s.target.d2 * s.target.ENC,
            ),
            MichaelisMentenInteraction(
                'cytotoxic_priming',
                target='site',
                reactants={'ENC': 1},
                modifiers={'ENC': 1},
                substrate='ED',
                vmax=lambda s: s.target.rho2,
                km=lambda s: s.target.g3,
            ),
            ReactionInteraction(
                'cytotoxic_death',
                target='site',
                reactants={'E': 1},
                propensity=lambda s: s.target.d3 * s.target.E,
            ),
            RateInteraction(
                'primed_expansion',
                target='site',
                state='E',
                rate=lambda s: s.target.rho3
                * delayed(s.target.ENC, 7 * s.target.tau2, 7 * self.tau2)
                * delayed(s.target.ED, 7 * s.target.tau2, 7 * self.tau2)
                / (s.target.g3 + delayed(s.target.ED, 7 * s.target.tau2, 7 * self.tau2)),
            ),
            RateInteraction(
                'dendritic_expansion',
                target='site',
                state='E',
                rate=lambda s: s.target.m1
                * delayed(s.target.ED, s.target.tau2)
                * delayed(s.target.E, s.target.tau2)
                / (s.target.g3 + delayed(s.target.ED, s.target.tau2)),
            ),
            RateInteraction(
                'il2_expansion',
                target='site',
                state='E',
                rate=lambda s: s.target.rho4
                * delayed(s.target.E, s.target.tau2)
                * delayed(s.target.C, s.target.tau2) ** 2
                / (s.target.g4 + delayed(s.target.C, s.target.tau2) ** 2),
            ),
            ReactionInteraction(
                'cytotoxic_suppression',
                target='site',
                reactants={'E': 1},
                modifiers={'ER': 1},
                propensity=lambda s: s.target.psi2 * s.target.E * s.target.ER,
            ),
            ReactionInteraction(
                'dendritic_death',
                target='site',
                reactants={'ED': 1},
                propensity=lambda s: s.target.d4 * s.target.ED,
            ),
            MichaelisMentenInteraction(
                'antigen_presentation',
                target='site',
                products={'ED': 1},
                substrate='A',
                vmax=lambda s: s.target.kappa1,
                km=lambda s: s.target.g5,
            ),
            DosingInteraction(
                'vaccine',
                target='site',
                state='ED',
                dose=lambda s: s.target.dose,
                period=self.period,
                width=self.width,
                start=self.start,
                count=self.doses,
            ),
            DosingInteraction('il2_influx', target='site', state='C', dose=lambda s: s.target.s3),
            ReactionInteraction(
                'il2_loss',
                target='site',
                reactants={'C': 1},
                propensity=lambda s: s.target.d5 * s.target.C,
            ),
            RateInteraction(
                'cytotoxic_secretion',
                target='site',
                state='C',
                rate=lambda s: s.target.rho5 * s.target.E**2 * s.target.C / (s.target.g6 + s.target.E**2),
            ),
            MichaelisMentenInteraction(
                'helper_secretion',
                target='site',
                products={'C': 1},
                modifiers={'C': 1},
                substrate='E1H',
                vmax=lambda s: s.target.rho6,
                km=lambda s: s.target.g7,
            ),
            DosingInteraction('regulatory_influx', target='site', state='ER', dose=lambda s: s.target.s4),
            ReactionInteraction(
                'regulatory_death',
                target='site',
                reactants={'ER': 1},
                propensity=lambda s: s.target.d6 * s.target.ER,
            ),
            RateInteraction(
                'antigen_expansion',
                target='site',
                state='ER',
                rate=lambda s: s.target.rho7
                * delayed(s.target.ER, s.target.tau3)
                * delayed(s.target.A, s.target.tau3) ** 2
                / (s.target.g8 + delayed(s.target.A, s.target.tau3) ** 2),
            ),
            RateInteraction(
                'il2_regulatory_expansion',
                target='site',
                state='ER',
                rate=lambda s: s.target.rho8
                * delayed(s.target.ER, s.target.tau3)
                * delayed(s.target.C, s.target.tau3)
                / (s.target.g9 + delayed(s.target.C, s.target.tau3)),
            ),
            DosingInteraction('naive_helper_influx', target='site', state='ENH', dose=lambda s: s.target.s5),
            ReactionInteraction(
                'naive_helper_death',
                target='site',
                reactants={'ENH': 1},
                propensity=lambda s: s.target.d7 * s.target.ENH,
            ),
            MichaelisMentenInteraction(
                'helper_priming',
                target='site',
                reactants={'ENH': 1},
                modifiers={'ENH': 1},
                substrate='ED',
                vmax=lambda s: s.target.rho9,
                km=lambda s: s.target.g3,
            ),
            ReactionInteraction(
                'helper_death',
                target='site',
                reactants={'E1H': 1},
                propensity=lambda s: s.target.d8 * s.target.E1H,
            ),
            RateInteraction(
                'primed_helper_expansion',
                target='site',
                state='E1H',
                rate=lambda s: s.target.m2
                * delayed(s.target.ED, 2 * s.target.tau3, 2 * self.tau3)
                * delayed(s.target.ENH, 2 * s.target.tau3, 2 * self.tau3)
                / (s.target.g3 + delayed(s.target.ED, 2 * s.target.tau3, 2 * self.tau3)),
            ),
            ReactionInteraction(
                'helper_suppression',
                target='site',
                reactants={'E1H': 1},
                modifiers={'ER': 1},
                propensity=lambda s: s.target.psi3 * s.target.ER * s.target.E1H,
            ),
            RateInteraction(
                'helper_expansion',
                target='site',
                state='E1H',
                rate=lambda s: s.target.rho10
                * delayed(s.target.ED, s.target.tau3)
                * delayed(s.target.E, s.target.tau3)
                / (s.target.g3 + delayed(s.target.ED, s.target.tau3)),
            ),
        ]
        return CompositeSystem(self.label, elements=[site], interactions=interactions)


class TumourImmuneSuppressionSystem(BaseSystem):
    """A tumour against dendritic, cytotoxic, helper, and regulatory T cells.

    The cells act with IL-2, IL-10, and TGF-beta.

    A tumour of one cell grows exponentially, then as a power of its size, and only its accessible fraction is killed.
    Its antigen matures dendritic cells, which recruit T cells from the memory pools, and the regulatory cells with
    IL-10 and TGF-beta suppress the response. A weakly antigenic tumour escapes, a moderately antigenic one oscillates,
    and a strongly antigenic one is held to a few cells. The book's growth rows are garbled, and the exponential rate is
    read as 0.333 per day and the power coefficient as 333 cells^(1/2) per day, within the book's range 100 to 1000, so
    the growth turns from exponential to power at `10^6` cells. The book gives no memory pools, which are taken as
    constant at `10^4` cells each. The other parameters are the book's, the initial immune cells are `10^-5`, cells are
    counts, cytokines in ng/ml, and time in days. The system has the tumour `site.A`, the dendritic cells `site.EUD` and
    `site.ED`, the T cells `site.EPC`, `site.E`, `site.EPH`, `site.EH`, `site.EPR`, and `site.ER`, and the cytokines
    `site.C1`, `site.C2`, and `site.C3`.

    Args:
        alpha_a:
            The antigenicity of the tumour, in the book's range `10^-5` to `10^5`.

        r1:
            The exponential growth rate of the tumour.

        r2:
            The power growth coefficient of the tumour.

        EMC:
            The memory cytotoxic T cells.

        EMH:
            The memory helper T cells.

        EMR:
            The memory regulatory T cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import TumourImmuneSuppressionSystem
        >>> TumourImmuneSuppressionSystem(alpha_a=1.0).prepare().system_.elements[0].alpha_a
        1.0
    """

    field = 'biology/oncology'
    topics = ('immunotherapy', 'ode', 'immunosuppression', 'cytokines', 'antigenicity')
    references = (BOOK_REFERENCE, 'Robertson-Tessi, El-Kareh, and Goriely 2012')
    until = 200.0
    dt = 0.5
    solver = STIFF_SOLVER

    def __init__(
        self,
        *,
        alpha_a: float = 1e-3,
        r1: float = 0.333,
        r2: float = 333.0,
        EMC: float = 1e4,
        EMH: float = 1e4,
        EMR: float = 1e4,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.alpha_a = alpha_a
        self.r1 = r1
        self.r2 = r2
        self.EMC = EMC
        self.EMH = EMH
        self.EMR = EMR

    def _build_system(self) -> CompositeSystem:
        """Return the components of a tumour against dendritic."""
        site = SuppressedTumourElement(
            'site',
            alpha_a=self.alpha_a,
            r1=self.r1,
            r2=self.r2,
            EMC=self.EMC,
            EMH=self.EMH,
            EMR=self.EMR,
        )

        def accessible(s: SimpleNamespace) -> Expression:
            """Return the fraction of the tumour accessible to the cytotoxic T cells."""
            fraction: Expression = s.target.A / (1 + (s.target.A ** (1 - s.target.q) / s.target.g3) ** s.target.p) ** (
                1 / s.target.p
            )
            return fraction

        def maturation(s: SimpleNamespace) -> Expression:
            """Return the maturation by IL-2 of proliferating T cells.

            TGF-beta suppresses it.
            """
            factor: Expression = hill(s.target.C1, s.target.EC1_50) / (1 + s.target.C3 / s.target.EC3_50)
            return factor

        interactions = [
            RateInteraction(
                'tumour_growth',
                target='site',
                state='A',
                rate=lambda s: s.target.A
                / ((1 / s.target.r1) ** s.target.p + (s.target.A ** (1 - s.target.n) / s.target.r2) ** s.target.p)
                ** (1 / s.target.p),
            ),
            ReactionInteraction(
                'kill',
                target='site',
                reactants={'A': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.a1
                * accessible(s)
                * s.target.E
                / (s.target.E + accessible(s) / s.target.g1 + s.target.ER / s.target.g2)
                / (1 + s.target.C3 / s.target.C3_50),
            ),
            RateInteraction(
                'antigen_uptake',
                target='site',
                state='EUD',
                rate=lambda s: s.target.alpha_a
                * accessible(s)
                / ((1 + s.target.C2 / s.target.C2_50) * (1 + s.target.ER / s.target.ER_50)),
            ),
            ReactionInteraction(
                'licensing',
                target='site',
                reactants={'EUD': 1},
                products={'ED': 1},
                propensity=lambda s: s.target.lambda_a * s.target.EUD / (1 + s.target.EUD / s.target.EMH),
            ),
            ReactionInteraction(
                'immature_death',
                target='site',
                reactants={'EUD': 1},
                propensity=lambda s: s.target.d1 * s.target.EUD,
            ),
            ReactionInteraction(
                'licensed_death',
                target='site',
                reactants={'ED': 1},
                propensity=lambda s: s.target.d2 * s.target.ED,
            ),
            RateInteraction(
                'cytotoxic_priming',
                target='site',
                state='EPC',
                rate=lambda s: s.target.a2
                * s.target.EMC
                * s.target.ED
                / (s.target.ED + s.target.g4 * (s.target.EMC + s.target.EMH + s.target.EMR)),
            ),
            ReactionInteraction(
                'cytotoxic_precursor_death',
                target='site',
                reactants={'EPC': 1},
                propensity=lambda s: s.target.d3 * s.target.EPC,
            ),
            RateInteraction(
                'cytotoxic_maturation',
                target='site',
                state='E',
                rate=lambda s: s.target.a3 * s.target.EPC * maturation(s),
            ),
            ReactionInteraction(
                'cytotoxic_death',
                target='site',
                reactants={'E': 1},
                propensity=lambda s: s.target.d4 * s.target.E,
            ),
            RateInteraction(
                'helper_priming',
                target='site',
                state='EPH',
                rate=lambda s: s.target.a4
                * s.target.EMH
                * (s.target.ED + s.target.EUD)
                / (s.target.ED + s.target.EUD + s.target.g4 * (s.target.EMC + s.target.EMH + s.target.EMR)),
            ),
            ReactionInteraction(
                'helper_precursor_death',
                target='site',
                reactants={'EPH': 1},
                propensity=lambda s: s.target.d5 * s.target.EPH,
            ),
            RateInteraction(
                'helper_maturation',
                target='site',
                state='EH',
                rate=lambda s: s.target.a5 * s.target.EPH * maturation(s),
            ),
            ReactionInteraction(
                'conversion',
                target='site',
                reactants={'EH': 1},
                products={'ER': 1},
                modifiers={'C3': 1},
                propensity=lambda s: s.target.a6 * s.target.EH * s.target.C3 / (s.target.g5 + s.target.C3),
            ),
            ReactionInteraction(
                'helper_death',
                target='site',
                reactants={'EH': 1},
                propensity=lambda s: s.target.d6 * s.target.EH,
            ),
            RateInteraction(
                'regulatory_priming',
                target='site',
                state='EPR',
                rate=lambda s: s.target.a7
                * s.target.EMR
                * s.target.ED
                / (s.target.ED + s.target.g4 * (s.target.EMC + s.target.EMH + s.target.EMR)),
            ),
            ReactionInteraction(
                'regulatory_precursor_death',
                target='site',
                reactants={'EPR': 1},
                propensity=lambda s: s.target.d7 * s.target.EPR,
            ),
            RateInteraction(
                'regulatory_maturation',
                target='site',
                state='ER',
                rate=lambda s: s.target.a8 * s.target.EPR * hill(s.target.C1, s.target.EC1_50),
            ),
            ReactionInteraction(
                'regulatory_death',
                target='site',
                reactants={'ER': 1},
                propensity=lambda s: s.target.d8 * s.target.ER,
            ),
            RateInteraction(
                'il2_secretion',
                target='site',
                state='C1',
                rate=lambda s: s.target.a9
                * s.target.EPH
                / ((1 + s.target.C3 / s.target.CC3_50) * (1 + s.target.C2 / s.target.CC2_50)),
            ),
            ReactionInteraction(
                'il2_loss',
                target='site',
                reactants={'C1': 1},
                propensity=lambda s: s.target.C1 / s.target.tau1,
            ),
            RateInteraction(
                'regulatory_il10',
                target='site',
                state='C2',
                rate=lambda s: s.target.r3 * s.target.ER,
            ),
            RateInteraction(
                'tumour_il10',
                target='site',
                state='C2',
                rate=lambda s: s.target.r4 * s.target.A,
            ),
            ReactionInteraction(
                'il10_loss',
                target='site',
                reactants={'C2': 1},
                propensity=lambda s: s.target.C2 / s.target.tau2,
            ),
            RateInteraction(
                'regulatory_tgf_beta',
                target='site',
                state='C3',
                rate=lambda s: s.target.r5 * s.target.ER,
            ),
            RateInteraction(
                'tumour_tgf_beta',
                target='site',
                state='C3',
                rate=lambda s: s.target.r6 * s.target.A,
            ),
            ReactionInteraction(
                'tgf_beta_loss',
                target='site',
                reactants={'C3': 1},
                propensity=lambda s: s.target.C3 / s.target.tau3,
            ),
        ]
        return CompositeSystem(self.label, elements=[site], interactions=interactions)
