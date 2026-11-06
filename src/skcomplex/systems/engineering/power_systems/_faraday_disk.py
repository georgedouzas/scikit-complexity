"""Homopolar generator, a spinning conducting disk in a field that charges a load."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import sign
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class FaradayDiskElement(Element):
    """Homopolar generator, a spinning conducting disk in a field that charges a load."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        I: Annotated[object, StateSpec()] = 0.0,
        U: Annotated[object, StateSpec()] = 0.0,
        omega: Annotated[object, StateSpec()] = 0.0,
        B: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        r1: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.025,
        r2: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.55,
        L: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.2,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.75,
        R1: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.3,
        R2: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 109.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        beta: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        J: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        mu: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00012566370614359174,
        N: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 250.0,
        l: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        M: Annotated[object, ParameterSpec()] = 4.8,
        a0: Annotated[object, ParameterSpec()] = 8.0,
        a1: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.75,
        a3: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.I = I
        self.U = U
        self.omega = omega
        self.B = B
        self.r1 = r1
        self.r2 = r2
        self.L = L
        self.C = C
        self.R1 = R1
        self.R2 = R2
        self.alpha = alpha
        self.beta = beta
        self.J = J
        self.mu = mu
        self.N = N
        self.l = l
        self.M = M
        self.a0 = a0
        self.a1 = a1
        self.a3 = a3

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the spin, the current, and the charge of the disk."""
        return [
            RateLaw(
                s.I,
                (
                    -s.R1 * s.I
                    - s.U
                    + (s.B + s.mu * s.N * s.I / s.l) * (s.r2**2 - s.r1**2) * s.omega / 2
                    - (s.a0 + s.a1 * s.I + s.a3 * s.I**3)
                )
                / s.L,
            ),
            RateLaw(s.U, (s.I - s.U / s.R2) / s.C),
            RateLaw(
                s.omega,
                (
                    s.M
                    - s.alpha * sign(s.omega)
                    - s.beta * s.r2 * s.omega
                    - s.I * (s.B + s.mu * s.N * s.I / s.l) * (s.r2**2 - s.r1**2) / 2
                )
                / s.J,
            ),
        ]
