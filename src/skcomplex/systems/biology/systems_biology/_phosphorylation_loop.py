"""Three proteins, each phosphorylated and dephosphorylated by Michaelis-Menten kinetics, that regulate each other."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class PhosphorylationLoopElement(Element):
    """Three proteins, each phosphorylated and dephosphorylated by Michaelis-Menten kinetics, that regulate each other.

    The states are the unphosphorylated amounts, and each total is conserved. With `u_T = A_T - A`, and so on:

    - `dA/dt = kp1 u_T / (K1 + u_T) - kd1 C A / (K1 + A)`,
    - `dB/dt = kp2 A (B_T - B) / (K2 + B_T - B) - kd2 B / (K2 + B)`,
    - `dC/dt = kp3 B (C_T - C) / (K3 + C_T - C) - kd3 C / (K3 + C)`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `phosphorylation_oscillator` read its parameters.

    Args:
        label:
            The label of the loop.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial amount of unphosphorylated A.

        B:
            The initial amount of unphosphorylated B.

        C:
            The initial amount of unphosphorylated C.

        kp1:
            The maximal rate that restores A.

        kd1:
            The catalytic rate of the conversion of A by C.

        kp2:
            The catalytic rate of the conversion of B by A.

        kd2:
            The maximal rate that removes B.

        kp3:
            The catalytic rate of the conversion of C by B.

        kd3:
            The maximal rate that removes C.

        K1:
            The Michaelis constant of both reactions of A.

        K2:
            The Michaelis constant of both reactions of B.

        K3:
            The Michaelis constant of both reactions of C.

        A_T:
            The total amount of A.

        B_T:
            The total amount of B.

        C_T:
            The total amount of C.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import PhosphorylationLoopElement
        >>> PhosphorylationLoopElement('loop', A=0.5).A
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        B: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        C: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        kp1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        kd1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        kp2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        kd2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        kp3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        kd3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        K2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        K3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        A_T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        B_T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        C_T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.B = B
        self.C = C
        self.kp1 = kp1
        self.kd1 = kd1
        self.kp2 = kp2
        self.kd2 = kd2
        self.kp3 = kp3
        self.kd3 = kd3
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.A_T = A_T
        self.B_T = B_T
        self.C_T = C_T
