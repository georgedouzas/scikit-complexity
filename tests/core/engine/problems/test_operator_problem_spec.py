"""Test the `_operator_problem_spec` module of `skcomplex.core.engine.problems`."""

from skcomplex.core.engine.problems import BaseProblemSpec, ProblemSpec
from skcomplex.core.engine.problems._operator_problem_spec import OperatorProblemSpec


def test_operator_problem_is_a_base_problem_and_not_the_compilers():
    """Test that an operator problem derives from the problem base alone, and that its run is recorded."""
    assert issubclass(OperatorProblemSpec, BaseProblemSpec)
    assert not issubclass(OperatorProblemSpec, ProblemSpec)
