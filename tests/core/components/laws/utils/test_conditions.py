"""Test the `_conditions` module of `skcomplex.core.components.laws.utils`."""

import importlib.util
import warnings
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin, var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import both, either, mod, where
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.problems._expression_printer import ExpressionPrinter
from skcomplex.core.specs import StateSpec, VisualSpec


def test_functions_stay_unevaluated_on_symbols():
    """Test that conditions and remainders of symbols stay symbolic."""
    x = var('x')
    assert str(where(x < 0, -x, x)) == 'where(x < 0, -x, x)'
    assert str(mod(x, 2)) == 'mod(x, 2)'
    assert str(either(x < 0, x > 1)) == 'either(x < 0, x > 1)'


@pytest.mark.parametrize(
    ('expression', 'value', 'expected'),
    [
        (lambda x: where(x < 0, -x, x), -2, 2),
        (lambda x: where(either(x < 0, x > 1), 1, 0), 2, 1),
        (lambda x: where(both(x > 0, x < 1), 1, 0), 2, 0),
        (lambda x: mod(x, 1), -0.25, 0.75),
    ],
    ids=['where', 'either', 'both', 'mod'],
)
def test_functions_evaluate_on_numbers(expression, value, expected):
    """Test that conditions and remainders evaluate once their arguments are numbers."""
    x = var('x')
    assert float(expression(x).subs({x: value})) == pytest.approx(expected)


def test_functions_differentiate_by_branch():
    """Test that a condition differentiates each branch and keeps the condition."""
    x = var('x')
    assert str(where(x < 0, -(x**2), x**3).diff(x)) == 'where(x < 0, -2*x, 3*x^2)'


def test_functions_lower_to_numpy_and_numba(tmp_path):
    """Test that NumPy code evaluates conditions and remainders like the Python operators."""
    x = var('x')
    expression = where(either(x < 0, x > 2), -x, mod(x, 2))
    values = np.array([-1.5, 0.5, 3.5])
    function = ExpressionPrinter(time=var('t'), compiled=True).prepare([x]).run([expression]).code_
    result = function(0.0, values)[0]
    assert np.allclose(result, [1.5, 0.5, -3.5])
    scalar = ExpressionPrinter('scalar').prepare().run([expression]).code_[1][0]
    (tmp_path / 'printed.py').write_text(f'def value(x):\n    return {scalar}\n')
    specification = importlib.util.spec_from_file_location('printed', tmp_path / 'printed.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    assert module.value(0.5) == 0.5
    assert 'np.where' in ExpressionPrinter().prepare().run([expression]).code_[1][0]


def test_where_with_equal_branches_simplifies():
    """Test that the second derivative of a condition times a symbol is exactly zero."""
    x, y = var('x'), var('y')
    assert where(x < 0, 3, 3) == 3
    assert (where(x < 0, 1, 2) * y).diff(y, 2).is_trivial_zero()


def test_where_branch_not_chosen_raises_no_warning():
    """Test that NumPy evaluates a condition with a division in the discarded branch without warnings."""
    t, x = var('t'), var('x')
    function = ExpressionPrinter(time=t, compiled=True).prepare([x]).run([where(x > 0, 1 / x, 0)]).code_
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert function(0.0, np.array([0.0, 2.0]))[0].tolist() == [0.0, 0.5]


def test_condition_on_a_nested_modulo_of_time_compiles():
    """Test that a condition whose side holds a modulo of time inside another function compiles and switches."""

    class NestedElement(Element):
        """Nested element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            y: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.y = y

        def _get_laws(self, s):
            """Return the laws of the nested element."""
            return [RateLaw(s.y, where(s.y < sin(where(mod(s.t, 2) < 1, 1, 0)), 1, 0))]

    nested = NestedElement
    problem = PassCompiler().prepare(System(elements=[nested('n')])).run().problem_
    assert problem.drift(0.5, problem.initial_state, problem.initial_parameters)[0] == 1.0
    assert problem.drift(1.5, problem.initial_state, problem.initial_parameters)[0] == 0.0
