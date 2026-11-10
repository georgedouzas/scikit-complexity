"""Test the `problem_spec` module."""

from sage.all__sagemath_symbolics import I

from skcomplex.core.engine._utils import symbol, vector
from skcomplex.core.engine.problems import ProblemSpec


def test_problem_spec_get_real_components_of():
    """Test `ProblemSpec._get_real_components_of` on a small case."""
    x, y = symbol('x'), symbol('y')
    assert str(ProblemSpec._get_real_components_of(vector([x + I * y, 2 * x]))) == '[x, y, 2*x]'
