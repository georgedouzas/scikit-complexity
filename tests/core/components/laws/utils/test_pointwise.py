"""Test the `_pointwise` module of `skcomplex.core.components.laws.utils`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, cosh, log, pi, sin, var

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import clip, hill, logistic, sign
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.specs import StateSpec, VisualSpec

BACKENDS = (NumpyBackend(), NumbaBackend())
VALUES = np.array([-3.0, -1.0, 0.0, 0.25, 1.0, 2.0, 5.0])
FUNCTIONS = {
    'hill': (lambda x: hill(x**2, 2, 3), lambda x: x**6 / (8 + x**6)),
    'logistic': (lambda x: logistic(x, 1, 0.5), lambda x: 1 / (1 + np.exp(-(x - 1) / 0.5))),
    'clip': (lambda x: clip(x, -1, 2), lambda x: np.clip(x, -1, 2)),
    'clip-low': (lambda x: clip(x, low=0), lambda x: np.clip(x, 0, None)),
    'clip-high': (lambda x: clip(x, high=0.5), lambda x: np.clip(x, None, 0.5)),
    'sign': (sign, np.sign),
}


def _rates(function, backend):
    """Return the rate of every member of a group whose rate is a function of its state."""

    class ValueElement(Element):
        """Element whose rate is a function of its value."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the rate of the value."""
            return [RateLaw(s.x, function(s.x))]

    group = GroupElement('g', element=ValueElement, size=VALUES.size, values={'x': VALUES})
    problem = PassCompiler(backend=backend).prepare(System(elements=[group])).run().problem_
    return problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'g.x')


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
@pytest.mark.parametrize('name', list(FUNCTIONS))
def test_pointwise_functions_match_numpy(name, backend):
    """Test that every pointwise function compiles to the values of its NumPy definition."""
    function, expected = FUNCTIONS[name]
    np.testing.assert_allclose(_rates(function, backend), expected(VALUES), rtol=0, atol=1e-12)


def test_hill_is_one_half_at_its_half_value():
    """Test that the Hill function is one half where its argument equals the half value."""
    x, half, n = var('x half n')
    assert float(hill(x, half, n).subs(x=1.7, half=1.7, n=4)) == 0.5


def test_logistic_is_one_half_at_its_center():
    """Test that the logistic function is one half at its center."""
    x = var('x')
    assert float(logistic(x, 3, 2).subs(x=3)) == 0.5


def test_logistic_of_the_logarithm_of_a_constant_does_not_cancel():
    """Test that the tail of the logistic function of a multiple of the logarithm of a constant has no rounding noise.

    Sage writes the exponential of such an argument as `cosh(u) - sinh(u)`, two terms that cancel for a large `u`.
    """
    x = var('x')
    value = logistic(-(5.0 * log(x) - 5.0 * log(0.6 / sin(0.2 * pi))))
    assert not value.has(cosh(SR.wild(0)))
    for point in (1e3, 1e4, 1e6):
        expected = 1 / (1 + np.exp(5.0 * np.log(point) - 5.0 * np.log(0.6 / np.sin(0.2 * np.pi))))
        assert float(value.subs(x=point)) == pytest.approx(expected, rel=1e-6, abs=1e-15)


def test_clip_without_bounds_keeps_the_expression():
    """Test that a clip without bounds returns the expression unchanged."""
    x = var('x')
    assert bool(clip(x) == x)


def test_sign_has_a_zero_derivative():
    """Test that the sign differentiates to zero away from 0, so Jacobians stay finite."""
    x = var('x')
    assert sign(x).diff(x).subs(x=2) == 0
