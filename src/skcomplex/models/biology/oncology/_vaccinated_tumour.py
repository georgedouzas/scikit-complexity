"""Helper T cells, cytotoxic T cells, tumour cells, dendritic cells, and IL-2 under dendritic-cell vaccination."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class VaccinatedTumourElement(Element):
    """Helper T cells, cytotoxic T cells, tumour cells, dendritic cells, and IL-2 under dendritic-cell vaccination.

    The helper cells follow `dEH/dt = s1 + rho1 ED r1 EH (1 - b1 EH) - d1 EH`, the cytotoxic cells
    `dE/dt = s2 + rho2 C (A + ED) r2 E (1 - b2 E) - d2 E`, the tumour `dA/dt = r3 A (1 - b3 A) - c1 A E`, the
    dendritic cells `dED/dt = -c2 E ED + u(t)`, and the IL-2 `dC/dt = r4 EH ED - c3 E C - d3 C`. The vaccine `u` is
    `dose` for `width` time units every `period` from `start`, `doses` times.

    The element holds the states and parameters and has no laws of its own: the interactions of `dendritic_cell_vaccine`
    read its parameters.

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
        >>> from skcomplex.models.biology.oncology import VaccinatedTumourElement
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
