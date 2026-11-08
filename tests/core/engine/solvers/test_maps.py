"""Test the iteration of discrete maps."""

import numpy as np

from skcomplex.core.components.system import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.solvers import IterationSolver
from skcomplex.core.specs import StepSpec
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement


def test_iteration_applies_one_step_per_unit_time():
    """Test that the discrete stepper applies the map once per integer step."""
    system = System(
        elements=[MarketElement('m', price=1.0)],
        interactions=[CobwebInteraction('c', target='m', a=10, b=2, c=1, d=1)],
    )
    problem = PassCompiler().prepare(system).run().problem_
    stepper = IterationSolver()
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    assert stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 2.0)).state_.tolist() == [2.5]
