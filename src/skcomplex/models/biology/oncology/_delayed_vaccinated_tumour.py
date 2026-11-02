"""Tumour, natural killer cells, naive and mature CD8+ cells, dendritic cells, IL-2, regulatory, and helper cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class DelayedVaccinatedTumourElement(Element):
    """Tumour, natural killer cells, naive and mature CD8+ cells, dendritic cells, IL-2, regulatory, and helper cells.

    The equations are those of Padmanabhan et al. (4.13) to (4.21), with the delays `tau1`, `tau2`, and `tau3` of one
    division of natural killer, CD8+, and helper cells. Naive CD8+ cells mature after `7 tau2` and naive helper cells
    after `2 tau3`. The tumour is killed by mature CD8+ cells through `a1 x^5 / (g1 + x^5)` with
    `x = log(E) / log(A)`. The vaccine `u` is `dose` for `width` time units every `period` from `start`, `doses`
    times.

    The element holds the states and parameters and has no laws of its own: the interactions of `delayed_dc_vaccine`
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
        >>> from skcomplex.models.biology.oncology import DelayedVaccinatedTumourElement
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
