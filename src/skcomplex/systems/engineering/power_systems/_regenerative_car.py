"""Car whose motor recovers braking energy, slowed by dry and viscous friction."""

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


class RegenerativeCarElement(Element):
    """Car whose motor recovers braking energy, slowed by dry and viscous friction."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[object, StateSpec()] = 20.0,
        I: Annotated[object, StateSpec()] = 0.0,
        Vs: Annotated[object, StateSpec()] = 0.0,
        m: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
        A: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.2,
        L: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        R: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 680.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        beta: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.I = I
        self.Vs = Vs
        self.m = m
        self.A = A
        self.L = L
        self.R = R
        self.C = C
        self.alpha = alpha
        self.beta = beta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the speed, the current, and the charge of the car."""
        return [
            RateLaw(s.v, -(s.A * s.I + s.alpha * sign(s.v) + s.beta * s.v) / s.m),
            RateLaw(s.I, (s.A * s.v - s.R * s.I - s.Vs) / s.L),
            RateLaw(s.Vs, s.I / s.C),
        ]
