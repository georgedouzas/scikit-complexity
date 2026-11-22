"""Share the helpers and fixtures of the tests below this directory."""

from collections.abc import Mapping

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import ScipyODESolver

PRECISE = ScipyODESolver(rtol=1e-10, atol=1e-12)


class MassElement(Element):
    """Mass at the position `x` with the mass `m`, whose velocity is the value `x'`."""

    def __init__(
        self,
        label: str,
        *,
        x: float = 1.0,
        m: float = 2.0,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.x = x
        self.m = m
