"""Hunting immune cells recruited from resting cells after a delay, a tumour, and a drug that kills all three."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class RestingHuntingChemotherapyElement(Element):
    """Hunting immune cells recruited from resting cells after a delay, a tumour, and a drug that kills all three.

    The hunting cells follow `dE/dt = m1 E Q(t - tau) - d1 E - c1 A E - a E / (bd + E) U`, the resting cells
    `dQ/dt = r1 Q (1 - b1 Q) - m1 E Q(t - tau) - a Q / (bd + Q) U`, the tumour
    `dA/dt = r2 A (1 - b2 A) - c2 A E - a A / (bd + A) U`, and the drug
    `dU/dt = u - (k + g (E / (bd + E) + Q / (bd + Q) + A / (bd + A))) U`, which binds to the cells it kills. Before
    the start the resting cells keep their initial value.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `delayed_resting_hunting_chemotherapy` read its parameters.

    Args:
        label:
            The label of the patient.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        E:
            The initial hunting cells.

        Q:
            The initial resting cells.

        A:
            The initial tumour cells.

        U:
            The initial drug concentration.

        m1:
            The rate at which resting cells become hunting cells.

        d1:
            The death rate of hunting cells.

        c1:
            The rate at which the tumour inactivates hunting cells.

        c2:
            The rate at which hunting cells kill the tumour.

        r1:
            The growth rate of resting cells.

        b1:
            The reciprocal carrying capacity of resting cells.

        r2:
            The growth rate of the tumour.

        b2:
            The reciprocal carrying capacity of the tumour.

        a:
            The kill rate of the drug on every population.

        bd:
            The half saturation of the kill.

        g:
            The rate at which the drug combines with every population.

        k:
            The elimination rate of the drug, `dU` in the book.

        tau:
            The delay of the conversion of resting cells to hunting cells.

        u:
            The constant infusion rate of the drug.

    Examples:
        >>> from skcomplex.models.biology.oncology import RestingHuntingChemotherapyElement
        >>> RestingHuntingChemotherapyElement('patient').tau
        45.6
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        Q: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.48,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.18,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.093,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0412,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005133,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.6515,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0245,
        b1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.18,
        b2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        bd: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.0001,
        g: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 45.6,
        u: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.E = E
        self.Q = Q
        self.A = A
        self.U = U
        self.m1 = m1
        self.d1 = d1
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.b1 = b1
        self.r2 = r2
        self.b2 = b2
        self.a = a
        self.bd = bd
        self.g = g
        self.k = k
        self.tau = tau
        self.u = u
