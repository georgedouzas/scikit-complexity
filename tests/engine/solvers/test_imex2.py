"""Test the `IMEX2Solver` class."""

import numpy as np

from skcomplex.engine.compile import compile_system
from skcomplex.engine.solvers import (
    IMEX2Solver,
)
from skcomplex.models.physics.fluids import (
    kuramoto_sivashinsky,
)


def test_imex2_second_call_restarts_history():
    """Test that IMEX2 gives the same state from the same start twice in a row, as a fresh stepper does."""
    problem = compile_system(kuramoto_sivashinsky())
    x, p = problem.initial_state, problem.initial_parameters
    stepper, fresh = IMEX2Solver(dt=0.01), IMEX2Solver(dt=0.01)
    stepper.prepare(problem, np.random.default_rng(0))
    fresh.prepare(problem, np.random.default_rng(0))
    first = stepper.run(0.0, x, p, 0.1)
    second = stepper.run(0.0, x, p, 0.1)
    assert np.array_equal(first, second)
    assert np.array_equal(second, fresh.run(0.0, x, p, 0.1))
