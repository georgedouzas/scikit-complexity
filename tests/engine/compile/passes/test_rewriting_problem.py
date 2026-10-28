"""Test the `RewritingProblem` class."""

from skcomplex.engine.problems import BaseProblem, Problem, RewritingProblem


def test_rewriting_problem_is_a_base_problem_and_not_the_compilers():
    """Test that a rewriting problem derives from the problem base alone."""
    assert issubclass(RewritingProblem, BaseProblem)
    assert not issubclass(RewritingProblem, Problem)
