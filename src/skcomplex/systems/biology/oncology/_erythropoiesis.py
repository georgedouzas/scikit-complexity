"""Quiescent, proliferating, and mature red cells and their mean volume under a constant 6-mercaptopurine level."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ErythropoiesisElement(Element):
    """Quiescent, proliferating, and mature red cells and their mean volume under a constant 6-mercaptopurine level.

    The drug sets the death rate of proliferating cells `a1 = a0 U` and the cell-cycle time
    `tau = (g1 U + MCV0) tau_h / MCV_h`. Quiescent cells enter the cycle at the rate
    `m(Q) = m0 Q50^n / (Q50^n + Q^n)`, of which the fraction `exp(-a1 tau)` survives to give two daughters, and leave
    to mature at the rate `r1`, so `dQ/dt = -r1 Q + (2 exp(-a1 tau) - 1) m(Q) Q`. The proliferating cells follow
    `dP/dt = -a1 P + (1 - exp(-a1 tau)) m(Q) Q`, the mature cells, which live `lifespan` days,
    `dL/dt = r1 Q - d1 L(t - lifespan)`, and their mean corpuscular volume
    `dMCV/dt = ((g1 U + MCV0) r1 Q - 0.85 MCV(t - lifespan) d1 L(t - lifespan) - 0.15 MCV d1 L) / L`. Before the start
    the mature cells and their volume keep their initial values.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `erythropoiesis_mercaptopurine` read its parameters.

    Args:
        label:
            The label of the patient.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        L:
            The initial mature red cells.

        Q:
            The initial quiescent cells.

        P:
            The initial proliferating cells.

        MCV:
            The initial mean corpuscular volume.

        r1:
            The rate of differentiation of quiescent cells.

        d1:
            The death rate of mature cells.

        a0:
            The death rate of proliferating cells per unit of drug.

        tau_h:
            The cell-cycle time of healthy cells.

        MCV_h:
            The healthy mean corpuscular volume.

        MCV0:
            The volume of new cells without the drug.

        g1:
            The increase of the volume of new cells per unit of drug.

        m0:
            The maximum rate at which quiescent cells enter the cycle.

        Q50:
            The quiescent cells at which the rate of entry is half its maximum.

        n:
            The sensitivity of the rate of entry to the quiescent cells.

        lifespan:
            The life span of mature cells.

        U:
            The concentration of the drug.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ErythropoiesisElement
        >>> ErythropoiesisElement('patient').lifespan
        120.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        L: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 3.84,
        Q: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 19.6,
        P: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 2.16,
        MCV: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 85.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.00139,
        d1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.0071,
        a0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00022,
        tau_h: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.68,
        MCV_h: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 85.0,
        MCV0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 85.0,
        g1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0317,
        m0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.01,
        Q50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.027,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        lifespan: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 120.0,
        U: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.L = L
        self.Q = Q
        self.P = P
        self.MCV = MCV
        self.r1 = r1
        self.d1 = d1
        self.a0 = a0
        self.tau_h = tau_h
        self.MCV_h = MCV_h
        self.MCV0 = MCV0
        self.g1 = g1
        self.m0 = m0
        self.Q50 = Q50
        self.n = n
        self.lifespan = lifespan
        self.U = U
