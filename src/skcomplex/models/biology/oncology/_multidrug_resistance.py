"""Tumour cells split by their resistance to three drugs, each given by a constant infusion with its toxicity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class MultidrugResistanceElement(Element):
    """Tumour cells split by their resistance to three drugs, each given by a constant infusion with its toxicity.

    The sensitive cells `S`, the cells resistant to one drug `R1`, `R2`, `R3`, to two drugs `R12`, `R13`, `R23`, and
    to all three `R123` grow at the same rate `r`. A fraction `m_i` of the offspring of a population gains resistance
    to drug `i`, so `dS/dt = r (1 - m1 - m2 - m3) S - K S`, `dRi/dt = r ((1 - mj - mk) Ri + mi S) - K Ri`, and so on,
    where `K` sums the kill `a_i (U_i - Uth) F(U_i - Uth)` of every drug the population is sensitive to. The drugs
    follow `dU_i/dt = u_i - k_i U_i` and their toxicities `dTX_i/dt = U_i - eta_i TX_i`.

    The element holds the states and parameters and has no laws of its own: the interactions of `multidrug_resistance`
    read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        S:
            The initial sensitive cells.

        R1:
            The initial cells resistant to drug 1.

        R2:
            The initial cells resistant to drug 2.

        R3:
            The initial cells resistant to drug 3.

        R12:
            The initial cells resistant to drugs 1 and 2.

        R13:
            The initial cells resistant to drugs 1 and 3.

        R23:
            The initial cells resistant to drugs 2 and 3.

        R123:
            The initial cells resistant to all drugs.

        U1:
            The initial concentration of drug 1.

        U2:
            The initial concentration of drug 2.

        U3:
            The initial concentration of drug 3.

        TX1:
            The initial toxicity of drug 1.

        TX2:
            The initial toxicity of drug 2.

        TX3:
            The initial toxicity of drug 3.

        r:
            The growth rate of every population.

        m1:
            The rate of mutation to resistance against drug 1.

        m2:
            The rate of mutation to resistance against drug 2.

        m3:
            The rate of mutation to resistance against drug 3.

        a1:
            The kill rate of drug 1 per unit of concentration above the threshold.

        a2:
            The kill rate of drug 2 per unit of concentration above the threshold.

        a3:
            The kill rate of drug 3 per unit of concentration above the threshold.

        Uth:
            The concentration above which a drug kills.

        k1:
            The decay rate of drug 1, `dU1` in the book.

        k2:
            The decay rate of drug 2, `dU2` in the book.

        k3:
            The decay rate of drug 3, `dU3` in the book.

        eta1:
            The metabolic rate that lowers the toxicity of drug 1.

        eta2:
            The metabolic rate that lowers the toxicity of drug 2.

        eta3:
            The metabolic rate that lowers the toxicity of drug 3.

        u1:
            The infusion rate of drug 1.

        u2:
            The infusion rate of drug 2.

        u3:
            The infusion rate of drug 3.

    Examples:
        >>> from skcomplex.models.biology.oncology import MultidrugResistanceElement
        >>> MultidrugResistanceElement('tumour').Uth
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        S: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 460500000000.0,
        R1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R12: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R13: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R23: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R123: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        U1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        U2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        U3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        TX1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        TX2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        TX3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0099,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.008,
        m2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        m3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.014,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0084,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0074,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0092,
        Uth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        k1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.32,
        k2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.27,
        k3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        eta1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        eta2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        eta3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.45,
        u1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.S = S
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.R12 = R12
        self.R13 = R13
        self.R23 = R23
        self.R123 = R123
        self.U1 = U1
        self.U2 = U2
        self.U3 = U3
        self.TX1 = TX1
        self.TX2 = TX2
        self.TX3 = TX3
        self.r = r
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.Uth = Uth
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.u1 = u1
        self.u2 = u2
        self.u3 = u3
