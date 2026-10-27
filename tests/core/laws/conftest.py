"""Share the helpers and fixtures of the tests below this directory."""

from skcomplex.core.components import Element

from skcomplex.core.base import parameter, state
from skcomplex.engine.solvers import ScipyODESolver

PRECISE = ScipyODESolver(rtol=1e-10, atol=1e-12)


class MassElement(Element):
    """Mass element."""

    x = state(default=1.0, order=2)
    m = parameter(default=2.0)
