"""Tumour volume, the vascular volume that sets its carrying capacity, and an angiogenesis inhibitor."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class AngiogenicTumourElement(Element):
    """Tumour volume, the vascular volume that sets its carrying capacity, and an angiogenesis inhibitor.

    The tumour follows `dA/dt = -lambda1 A log(A / Vb)`, the vasculature
    `dVb/dt = -lambda2 Vb + xi_s A - xi_i Vb A^(2/3) - a Vb U`, and the inhibitor `dU/dt = -du U + u(t)`, where the
    infusion `u` is `dose` for `width` time units every `period`, from `start` on. A `width` equal to the `period`
    is a continuous infusion.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `hahnfeldt` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial tumour volume.

        Vb:
            The initial vascular volume.

        U:
            The initial concentration of the inhibitor.

        lambda1:
            The rate at which the tumour approaches its capacity.

        lambda2:
            The spontaneous loss rate of the vasculature.

        xi_s:
            The angiogenic stimulation by the tumour.

        xi_i:
            The angiogenic inhibition by the tumour.

        a:
            The vascular loss per unit of inhibitor.

        du:
            The clearance rate of the inhibitor.

        dose:
            The infusion rate during a dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

    Examples:
        >>> from skcomplex.models.biology.oncology import AngiogenicTumourElement
        >>> AngiogenicTumourElement('tumour').xi_s
        5.85
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 17347.0,
        Vb: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 17347.0,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lambda1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.192,
        lambda2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        xi_s: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.85,
        xi_i: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.00873,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.66,
        du: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.3,
        dose: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        period: float = 1.0,
        width: float = 1.0,
        start: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.Vb = Vb
        self.U = U
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.xi_s = xi_s
        self.xi_i = xi_i
        self.a = a
        self.du = du
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start
