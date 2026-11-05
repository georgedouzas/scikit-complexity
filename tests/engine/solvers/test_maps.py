"""Test the iteration of discrete maps."""

import numpy as np

from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.solvers import IterationSolver
from skcomplex.models.economics.markets import CobwebInteraction, MarketElement
from skcomplex.specs import StepSpec


def test_iteration_applies_one_step_per_unit_time():
    """Test that the discrete stepper applies the map once per integer step."""
    problem = compile_system(
        System(
            elements=[MarketElement('m', price=1.0)],
            interactions=[CobwebInteraction('c', target='m', a=10, b=2, c=1, d=1)],
        ),
    )
    stepper = IterationSolver()
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    assert stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 2.0)).state_.tolist() == [2.5]
