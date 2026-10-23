"""Test the iteration of discrete maps."""

import numpy as np

from skcomplex.core.compile import compile_system
from skcomplex.core.solvers import Iteration
from skcomplex.core.system import System
from skcomplex.models.economics.markets import Cobweb, Market


def test_iteration_applies_one_step_per_unit_time():
    """Test that the discrete stepper applies the map once per integer step."""
    problem = compile_system(
        System(components=[Market('m', price=1.0)], interactions=[Cobweb('c', target='m', a=10, b=2, c=1, d=1)]),
    )
    stepper = Iteration()
    stepper.initialize(problem, np.random.default_rng(0))
    assert stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 2.0).tolist() == [2.5]
