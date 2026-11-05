"""Tumour in six compartments under linear-quadratic radiation fractions and a radiosensitizer."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class IrradiatedTumourElement(Element):
    """Tumour in six compartments under linear-quadratic radiation fractions and a radiosensitizer.

    The proliferating cells follow `dA1/dt = r A1 exp(-a_gamma u_tot)`, where `u_tot` is the dose that the fractions of
    an irradiation accumulate. The lethally irradiated cells of `A5` divide once into `A6`, `dA5/dt = -(r + m1) A5` and
    `dA6/dt = 2 r A5 - m1 A6`, and both feed the dying chain `A2 -> A3 -> A4`, which leaves the tumour at the rate
    `m1`. The radiosensitizer is eliminated as `dUs/dt = -du Us`.

    The element holds the states and parameters and has no laws of its own: the interactions of `lq_radiosensitizer`
    read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A1:
            The initial volume of proliferating cells.

        A2:
            The initial volume of the first dying compartment.

        A3:
            The initial volume of the second dying compartment.

        A4:
            The initial volume of the third dying compartment.

        A5:
            The initial volume of lethally irradiated cells that divide once more.

        A6:
            The initial volume of lethally irradiated cells that no longer divide.

        Us:
            The initial plasma concentration of the radiosensitizer.

        u_tot:
            The initial accumulated dose.

        r:
            The growth rate.

        m1:
            The transit rate of the damaged cells.

        a_gamma:
            The inhibition of proliferation per unit of accumulated dose.

        du:
            The elimination rate of the radiosensitizer.

    Examples:
        >>> from skcomplex.systems.biology.oncology import IrradiatedTumourElement
        >>> IrradiatedTumourElement('tumour', A1=10.0).A1
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 27.0,
        A2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        A3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        A4: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        A5: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        A6: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Us: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u_tot: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.04,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.26,
        a_gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.004,
        du: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.544,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A4 = A4
        self.A5 = A5
        self.A6 = A6
        self.Us = Us
        self.u_tot = u_tot
        self.r = r
        self.m1 = m1
        self.a_gamma = a_gamma
        self.du = du
