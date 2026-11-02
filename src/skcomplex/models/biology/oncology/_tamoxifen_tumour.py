"""Breast tumour in three Gompertzian cell-cycle phases under tamoxifen, its metabolite, and the lymphocytes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class TamoxifenTumourElement(Element):
    """Breast tumour in three Gompertzian cell-cycle phases under tamoxifen, its metabolite, and the lymphocytes.

    The phases follow `dAg/dt = -mg Ag L + 2 mm Am L - a1 Ag (UX + delta1 UH)`, `dAs/dt = -ms As + mg Ag L`, and
    `dAm/dt = -mm Am L + ms As`, with `L = log(Asat / (Ag + As + Am))`. Tamoxifen passes from the gut `X0` through
    `X1` to the plasma `X2`, `dX0/dt = -k01 X0 + u(t)`, `dX1/dt = k01 X0 - k12 X1`,
    `dX2/dt = k12 X1 - (du1 + k23) X2`, and the plasma turns it into 4-hydroxytamoxifen,
    `dX3/dt = k23 X2 - du2 X3`, with the concentrations `UX = X2 / V` and `UH = X3 / V`. The lymphocytes follow
    `dE/dt = s - d1 E - a2 E (UX + delta2 UH)`, and tamoxifen enters the gut at the constant rate `u`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `tamoxifen_cell_cycle` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        Ag:
            The initial volume of cells in the growth phase.

        As:
            The initial volume of cells in the synthesis phase.

        Am:
            The initial volume of cells in mitosis.

        X0:
            The initial tamoxifen in the gut.

        X1:
            The initial tamoxifen in the second compartment.

        X2:
            The initial tamoxifen in the plasma.

        X3:
            The initial 4-hydroxytamoxifen.

        E:
            The initial circulating lymphocytes, in millions.

        mg:
            The transit rate out of the growth phase.

        ms:
            The transit rate out of the synthesis phase.

        mm:
            The transit rate out of mitosis.

        Asat:
            The volume at which growth saturates.

        a1:
            The kill rate of the tumour per unit of drug.

        a2:
            The kill rate of the lymphocytes per unit of drug.

        delta1:
            The tumour kill of the metabolite relative to tamoxifen.

        delta2:
            The lymphocyte kill of the metabolite relative to tamoxifen.

        k01:
            The transfer rate from the gut.

        k12:
            The transfer rate to the plasma.

        k23:
            The rate at which the plasma metabolizes tamoxifen.

        du1:
            The elimination rate of tamoxifen.

        du2:
            The elimination rate of the metabolite.

        V:
            The volume of distribution.

        s:
            The influx of lymphocytes, in millions per unit of time.

        d1:
            The death rate of lymphocytes.

        u:
            The infusion rate of tamoxifen.

    Examples:
        >>> from skcomplex.models.biology.oncology import TamoxifenTumourElement
        >>> TamoxifenTumourElement('tumour').Asat
        10000.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Ag: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 900.0,
        As: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        Am: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        X0: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        X1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        X2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        X3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        mg: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0013,
        ms: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.039,
        mm: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0169,
        Asat: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10000.0,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0062,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        delta1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 25.0,
        delta2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 25.0,
        k01: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.048,
        k12: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.993,
        k23: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 35.932,
        du1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.145,
        du2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 39.525,
        V: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 8.592,
        s: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.121,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.012,
        u: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 12.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Ag = Ag
        self.As = As
        self.Am = Am
        self.X0 = X0
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.E = E
        self.mg = mg
        self.ms = ms
        self.mm = mm
        self.Asat = Asat
        self.a1 = a1
        self.a2 = a2
        self.delta1 = delta1
        self.delta2 = delta2
        self.k01 = k01
        self.k12 = k12
        self.k23 = k23
        self.du1 = du1
        self.du2 = du2
        self.V = V
        self.s = s
        self.d1 = d1
        self.u = u
