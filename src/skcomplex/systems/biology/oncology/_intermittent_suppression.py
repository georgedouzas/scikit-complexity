"""Prostate cancer cells under androgen suppression that PSA thresholds switch on and off."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class IntermittentSuppressionElement(Element):
    """Prostate cancer cells under androgen suppression that PSA thresholds switch on and off.

    The dependent cells follow `dAD/dt = r1 (k1 + (1 - k1) U / (U + k2)) AD - d1 (k3 + (1 - k3) U / (U + k4)) AD
    - m1 (1 - U / U0) AD`, the independent cells `dAI/dt = m1 (1 - U / U0) AD + r2 (1 - U / U0) AI - d2 AI`, and
    the serum androgen `du/dt = -du (U - U0) - U0 du u`. The therapy `u` turns 1 when the PSA `x1 AD + x2 AI` rises
    through `h1` and 0 when it falls through `h0`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `ideta_intermittent_androgen` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AD:
            The initial number of androgen-dependent cells.

        AI:
            The initial number of androgen-independent cells.

        U:
            The initial serum androgen concentration.

        u:
            The initial therapy, 1 on and 0 off.

        r1:
            The growth rate of the dependent cells.

        r2:
            The growth rate of the independent cells.

        d1:
            The death rate of the dependent cells.

        d2:
            The death rate of the independent cells.

        m1:
            The maximum rate at which dependent cells turn independent.

        k1:
            The growth of the dependent cells without androgen, relative to `r1`.

        k2:
            The androgen at which the growth of the dependent cells is half way.

        k3:
            The death of the dependent cells without androgen, relative to `d1`.

        k4:
            The androgen at which the death of the dependent cells is half way.

        U0:
            The normal androgen concentration.

        du:
            The elimination rate of the androgen.

        x1:
            The PSA per dependent cell.

        x2:
            The PSA per independent cell.

    Examples:
        >>> from skcomplex.systems.biology.oncology import IntermittentSuppressionElement
        >>> IntermittentSuppressionElement('tumour').U0
        30.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        AI: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 30.0,
        u: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0204,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0242,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0076,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0168,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5e-05,
        k1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        k2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        k3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 8.0,
        k4: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        U0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
        du: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
        x1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        x2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.AD = AD
        self.AI = AI
        self.U = U
        self.u = u
        self.r1 = r1
        self.r2 = r2
        self.d1 = d1
        self.d2 = d2
        self.m1 = m1
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.U0 = U0
        self.du = du
        self.x1 = x1
        self.x2 = x2
