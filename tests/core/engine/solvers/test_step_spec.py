"""Test the `_step_spec` module of `skcomplex.core.engine.solvers`."""

from skcomplex.core.engine.solvers import StepSpec
from skcomplex.core.specs import BaseSpec


def test_a_solver_runs_on_a_step():
    """A step holds the state, the parameters, and the time to advance to."""
    assert issubclass(StepSpec, BaseSpec)
