"""Normal and malignant lineages of quiescent stem cells, progenitors, basal cells, and differentiated cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class StemCellDifferentiationElement(Element):
    """Normal and malignant lineages of quiescent stem cells, progenitors, basal cells, and differentiated cells.

    In each lineage the quiescent stem cells become progenitors at the rate `k12` and progenitors return at the rate
    `k21 (1 - NPr / (NPr + NPinf))`. Progenitors renew logistically, become basal cells, and basal cells
    differentiate. The renewal of normal progenitors gets the feedback `beta0 (1 - ND / (ND + kappa0))` of the
    differentiated cells. Normal progenitors mutate into malignant ones at the rate `m1`. A therapy multiplies the
    death rates of the progenitors and the basal cells by its dose, and that of the normal progenitors also by
    `1 + delta` times their renewal rate. A pulsed therapy multiplies them by 4 more from `start` for `duration`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `stem_cell_differentiation` read its parameters.

    Args:
        label:
            The label of the tissue.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        NSq:
            The initial normal quiescent stem cells.

        NPr:
            The initial normal progenitors.

        NBs:
            The initial normal basal cells.

        ND:
            The initial normal differentiated cells.

        ASq:
            The initial malignant quiescent stem cells.

        APr:
            The initial malignant progenitors.

        ABs:
            The initial malignant basal cells.

        AD:
            The initial malignant differentiated cells.

        d1:
            The death rate of normal quiescent stem cells.

        d2:
            The death rate of normal progenitors.

        d3:
            The death rate of normal basal cells.

        d4:
            The death rate of normal differentiated cells.

        d5:
            The death rate of malignant quiescent stem cells.

        d6:
            The death rate of malignant progenitors.

        d7:
            The death rate of malignant basal cells.

        d8:
            The death rate of malignant differentiated cells.

        k12:
            The rate at which normal quiescent stem cells become progenitors.

        k21:
            The largest rate at which normal progenitors return to quiescence.

        k23:
            The rate at which normal progenitors become basal cells.

        k34:
            The rate at which normal basal cells differentiate.

        k56:
            The rate at which malignant quiescent stem cells become progenitors.

        k65:
            The largest rate at which malignant progenitors return to quiescence.

        k67:
            The rate at which malignant progenitors become basal cells.

        k78:
            The rate at which malignant basal cells differentiate.

        r1:
            The renewal rate of normal progenitors.

        r2:
            The renewal rate of normal basal cells.

        r3:
            The renewal rate of malignant progenitors.

        r4:
            The renewal rate of malignant basal cells.

        m1:
            The mutation rate of normal progenitors.

        NPinf:
            The saturation constant of the return of normal progenitors.

        APinf:
            The saturation constant of the return of malignant progenitors.

        NPmax:
            The largest number of normal progenitors.

        APmax:
            The largest number of malignant progenitors.

        Nbmax:
            The largest number of normal basal cells.

        Abmax:
            The largest number of malignant basal cells.

        beta0:
            The feedback constant.

        kappa0:
            The number of differentiated cells that halves the feedback.

        delta:
            The weight of the renewal in the death the therapy causes to normal progenitors.

        up:
            The dose of the pulsed therapy.

        uc:
            The dose of the continuous therapy.

        therapy:
            The mode of therapy, `'none'`, `'pulsed'`, or `'continuous'`.

        start:
            The time the pulse of a pulsed therapy starts.

        duration:
            The duration of the pulse of a pulsed therapy.

    Examples:
        >>> from skcomplex.systems.biology.oncology import StemCellDifferentiationElement
        >>> StemCellDifferentiationElement('tissue', therapy='pulsed').therapy
        'pulsed'
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        NSq: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        NPr: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        NBs: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 2000.0,
        ND: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 2000.0,
        ASq: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        APr: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        ABs: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        d4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        d5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        d6: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        d7: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        d8: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        k12: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        k21: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        k23: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        k34: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.06,
        k56: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        k65: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        k67: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        k78: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        r4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        NPinf: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        APinf: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        NPmax: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10000.0,
        APmax: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 50000.0,
        Nbmax: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10000.0,
        Abmax: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5000.0,
        beta0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        kappa0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2000.0,
        delta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        up: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 30.0,
        uc: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        therapy: str = 'continuous',
        start: float = 100.0,
        duration: float = 100.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.NSq = NSq
        self.NPr = NPr
        self.NBs = NBs
        self.ND = ND
        self.ASq = ASq
        self.APr = APr
        self.ABs = ABs
        self.AD = AD
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.d5 = d5
        self.d6 = d6
        self.d7 = d7
        self.d8 = d8
        self.k12 = k12
        self.k21 = k21
        self.k23 = k23
        self.k34 = k34
        self.k56 = k56
        self.k65 = k65
        self.k67 = k67
        self.k78 = k78
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.m1 = m1
        self.NPinf = NPinf
        self.APinf = APinf
        self.NPmax = NPmax
        self.APmax = APmax
        self.Nbmax = Nbmax
        self.Abmax = Abmax
        self.beta0 = beta0
        self.kappa0 = kappa0
        self.delta = delta
        self.up = up
        self.uc = uc
        self.therapy = therapy
        self.start = start
        self.duration = duration
