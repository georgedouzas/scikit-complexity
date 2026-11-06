"""Test the `RewritingProblemSpec` class."""

from skcomplex.core.engine.problems import BaseProblemSpec, ProblemSpec, RewritingProblemSpec


def test_rewriting_problem_is_a_base_problem_and_not_the_compilers():
    """Test that a rewriting problem derives from the problem base alone."""
    assert issubclass(RewritingProblemSpec, BaseProblemSpec)
    assert not issubclass(RewritingProblemSpec, ProblemSpec)
