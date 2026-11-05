"""Tumour with NK cells, CD8 T cells, and circulating lymphocytes under chemotherapy, IL-2, and TIL injection."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ChemoimmunotherapyTumourElement(Element):
    """Tumour with NK cells, CD8 T cells, and circulating lymphocytes under chemotherapy, IL-2, and TIL injection.

    The tumour `A` grows logistically and NK cells `N` and CD8 T cells `E` kill it. CD8 T cells kill at the rate
    `D = a1 (E / A)^l / (g1 + (E / A)^l)`. The circulating lymphocytes `C` feed the NK cells. The chemotherapy `M`
    kills a fraction `1 - exp(-M)` of every population at its own rate, IL-2 `I` stimulates the CD8 T cells, and a TIL
    injection adds CD8 T cells. All three drugs follow the same schedule of doses at the rates `u_til`, `u_il2`, and
    `u_chemo`. Cells are counted, and time is in days.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `chemoimmunotherapy` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial number of tumour cells.

        N:
            The initial number of NK cells.

        E:
            The initial number of CD8 T cells.

        C:
            The initial number of circulating lymphocytes.

        I:
            The initial amount of IL-2.

        M:
            The initial amount of chemotherapeutic drug.

        r1:
            The growth rate of the tumour.

        b1:
            The reciprocal carrying capacity of the tumour.

        c1:
            The rate at which NK cells kill tumour cells.

        a1:
            The highest rate at which CD8 T cells kill tumour cells.

        g1:
            The half-saturation of the CD8 T cell kill.

        l:
            The exponent of the CD8 T cell kill.

        a2:
            The chemotherapy kill rate of tumour cells.

        m1:
            The rate at which circulating lymphocytes become NK cells.

        d1:
            The death rate of NK cells.

        r2:
            The highest recruitment rate of NK cells by the tumour.

        g2:
            The half-saturation of the NK cell recruitment, in cells².

        psi1:
            The rate at which the tumour inactivates NK cells.

        a3:
            The chemotherapy kill rate of NK cells.

        r3:
            The highest recruitment rate of CD8 T cells by the tumour kill.

        g3:
            The half-saturation of the CD8 T cell recruitment, in cells².

        d2:
            The death rate of CD8 T cells.

        psi2:
            The rate at which the tumour inactivates CD8 T cells.

        psi3:
            The rate at which NK cells regulate CD8 T cells.

        r4:
            The rate at which the NK cell kill of tumour cells activates CD8 T cells.

        r5:
            The rate at which circulating lymphocytes meeting tumour cells activate CD8 T cells.

        a4:
            The chemotherapy kill rate of CD8 T cells.

        r6:
            The highest rate at which IL-2 stimulates CD8 T cells.

        g4:
            The half-saturation of the IL-2 stimulation.

        s1:
            The influx of circulating lymphocytes.

        d3:
            The death rate of circulating lymphocytes.

        a5:
            The chemotherapy kill rate of circulating lymphocytes.

        d_u1:
            The decay rate of IL-2.

        d_u2:
            The decay rate of the chemotherapeutic drug.

        u_til:
            The rate of TIL injection during a dose, in cells per day.

        u_il2:
            The rate of IL-2 injection during a dose.

        u_chemo:
            The rate of chemotherapy infusion during a dose.

        start:
            The time of the first dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        doses:
            The number of doses.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ChemoimmunotherapyTumourElement
        >>> ChemoimmunotherapyTumourElement('tumour').r1
        0.431
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000000.0,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 50000.0,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        C: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 11000000.0,
        I: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        M: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.431,
        b1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.17e-08,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 7.13e-10,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 8.17,
        g1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.618,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.657,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00129,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0412,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4198,
        g2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20200000.0,
        psi1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-07,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.996,
        g3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 303000.0,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        psi2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.42e-10,
        psi3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.8e-08,
        r4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.1e-07,
        r5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3e-11,
        a4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        r6: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.125,
        g4: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20000000.0,
        s1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 121000.0,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.012,
        a5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        d_u1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        d_u2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.9,
        u_til: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10000000.0,
        u_il2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10000000.0,
        u_chemo: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        start: float = 7.0,
        period: float = 7.0,
        width: float = 1.0,
        doses: int = 3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.N = N
        self.E = E
        self.C = C
        self.I = I
        self.M = M
        self.r1 = r1
        self.b1 = b1
        self.c1 = c1
        self.a1 = a1
        self.g1 = g1
        self.l = l
        self.a2 = a2
        self.m1 = m1
        self.d1 = d1
        self.r2 = r2
        self.g2 = g2
        self.psi1 = psi1
        self.a3 = a3
        self.r3 = r3
        self.g3 = g3
        self.d2 = d2
        self.psi2 = psi2
        self.psi3 = psi3
        self.r4 = r4
        self.r5 = r5
        self.a4 = a4
        self.r6 = r6
        self.g4 = g4
        self.s1 = s1
        self.d3 = d3
        self.a5 = a5
        self.d_u1 = d_u1
        self.d_u2 = d_u2
        self.u_til = u_til
        self.u_il2 = u_il2
        self.u_chemo = u_chemo
        self.start = start
        self.period = period
        self.width = width
        self.doses = doses
