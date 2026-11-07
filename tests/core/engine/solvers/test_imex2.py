"""Test the `IMEX2Solver` class."""

import numpy as np

from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.solvers import IMEX2Solver
from skcomplex.core.specs import StepSpec
from skcomplex.systems.physics.fluids import KuramotoSivashinskySystem


def test_imex2_second_call_restarts_history():
    """Test that IMEX2 gives the same state from the same start twice in a row, as a fresh stepper does."""
    problem = PassCompiler().prepare(KuramotoSivashinskySystem().prepare().system_).run().problem_
    x, p = problem.initial_state, problem.initial_parameters
    stepper, fresh = IMEX2Solver(dt=0.01), IMEX2Solver(dt=0.01)
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    fresh.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    first = stepper.run(StepSpec(0.0, x, p, 0.1)).state_
    second = stepper.run(StepSpec(0.0, x, p, 0.1)).state_
    assert np.array_equal(first, second)
    assert np.array_equal(second, fresh.run(StepSpec(0.0, x, p, 0.1)).state_)
