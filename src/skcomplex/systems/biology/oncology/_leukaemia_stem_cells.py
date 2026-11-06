"""Normal and leukaemic stem cells that share a niche, with the progeny that feeds back on them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class LeukaemiaStemCellsElement(Element):
    """Normal and leukaemic stem cells that share a niche, with the progeny that feeds back on them.

    Stem cells renew symmetrically at the rate `r e^-psi1` and differentiate symmetrically at the rate
    `m e^-psi1 e^-psi3`, where `psi1 = beta1 (NS + AS)` counts the stem cells in the niche and
    `psi2 = beta2 (NPr + ND + AD)` and `psi3 = beta3 (NPr + ND + AD)` count their progeny. Asymmetric division at
    the rate `r e^-psi1 e^-psi2` and symmetric differentiation feed the progenitors and the leukaemic cells.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `leukaemia_stem_cells` read its parameters.

    Args:
        label:
            The label of the niche.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        NS:
            The initial normal stem cells.

        NPr:
            The initial normal progenitors.

        ND:
            The initial normal differentiated cells.

        AS:
            The initial leukaemic stem cells.

        AD:
            The initial differentiated leukaemic cells.

        r1:
            The symmetric renewal rate of normal stem cells.

        m1:
            The symmetric differentiation rate of normal stem cells.

        d1:
            The death rate of normal stem cells.

        r2:
            The asymmetric division rate of normal stem cells.

        r3:
            The proliferation rate of progenitors.

        d2:
            The rate at which progenitors leave their compartment.

        m2:
            The fraction of leaving progenitors that become differentiated cells.

        d3:
            The death rate of differentiated cells.

        r4:
            The symmetric renewal rate of leukaemic stem cells.

        m3:
            The symmetric differentiation rate of leukaemic stem cells.

        d4:
            The death rate of leukaemic stem cells.

        r5:
            The asymmetric division rate of leukaemic stem cells.

        d5:
            The death rate of differentiated leukaemic cells.

        beta1:
            The feedback gain of the stem cells.

        beta2:
            The feedback gain of the progeny on asymmetric division.

        beta3:
            The feedback gain of the progeny on differentiation.

    Examples:
        >>> from skcomplex.systems.biology.oncology import LeukaemiaStemCellsElement
        >>> LeukaemiaStemCellsElement('niche').beta1
        0.01
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        NS: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        NPr: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        ND: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        AS: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        m2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        r4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.7,
        m3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        d4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        r5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        d5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        beta1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        beta2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        beta3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.NS = NS
        self.NPr = NPr
        self.ND = ND
        self.AS = AS
        self.AD = AD
        self.r1 = r1
        self.m1 = m1
        self.d1 = d1
        self.r2 = r2
        self.r3 = r3
        self.d2 = d2
        self.m2 = m2
        self.d3 = d3
        self.r4 = r4
        self.m3 = m3
        self.d4 = d4
        self.r5 = r5
        self.d5 = d5
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
