"""Share the helpers and fixtures of the tests below this directory."""

from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec

PRECISE = ScipyODESolver(rtol=1e-10, atol=1e-12)


class MassElement(Element):
    """Mass element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 1.0,
        x_d1: float = 0.0,
        m: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.m = m
