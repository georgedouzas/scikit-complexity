"""Test the exact simulation of jump processes."""

import numpy as np

from skcomplex.core.compile import compile_system
from skcomplex.core.solvers import Gillespie
from skcomplex.core.system import System
from skcomplex.models.biology.epidemiology import Compartment, Recovery


def test_gillespie_conserves_individuals():
    """Test that recoveries move individuals without creating or destroying them."""
    problem = compile_system(
        System(
            components=[Compartment('I', n=50), Compartment('R')],
            interactions=[Recovery('rec', infected='I', recovered='R', gamma=0.5, stochastic=True)],
        ),
    )
    stepper = Gillespie()
    stepper.initialize(problem, np.random.default_rng(1))
    state = stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)
    assert state.sum() == 50
    assert 0 < state[0] < 50
    assert np.all(state == np.round(state))
