"""Share the helpers and fixtures of the tests below this directory."""

from skcomplex.core.base import parameter, state
from skcomplex.core.components import Element
from skcomplex.engine.solvers import ScipyODESolver

PRECISE = ScipyODESolver(rtol=1e-10, atol=1e-12)


MassElement = Element.define(
    'MassElement',
    states={'x': state(default=1.0, order=2)},
    parameters={'m': parameter(default=2.0)},
)
