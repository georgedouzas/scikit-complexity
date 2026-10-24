"""Test the `_compartments` module of `skcomplex.models.biology.epidemiology`."""

import numpy as np

from skcomplex.core.base import Form
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.epidemiology import Compartment, Infection, Recovery


def test_epidemic_stochastic_mode_is_jump_process():
    """Test that stochastic infection and recovery give integer counts of a jump process."""
    system = System(
        components=[Compartment('S', n=990), Compartment('I', n=10), Compartment('R')],
        interactions=[
            Infection('infection', susceptible='S', infected='I', beta=0.0005, stochastic=True),
            Recovery('recovery', infected='I', recovered='R', gamma=0.1, stochastic=True),
        ],
    )
    assert compile_system(system).form is Form.JUMP
    states = Simulation(system, dt=1.0, random_state=3).run(until=50).trajectory_.states
    assert np.all(states >= 0)
    assert np.all(states == np.round(states))
    assert np.allclose(states.sum(axis=1), 1000)


def test_epidemic_deterministic_mode_conserves_population():
    """Test that the deterministic SIR model keeps the total population."""
    system = System(
        components=[Compartment('S', n=990), Compartment('I', n=10), Compartment('R')],
        interactions=[
            Infection('infection', susceptible='S', infected='I', beta=0.0005),
            Recovery('recovery', infected='I', recovered='R', gamma=0.1),
        ],
    )
    states = Simulation(system, dt=1.0).run(until=50).trajectory_.states
    assert np.allclose(states.sum(axis=1), 1000)
