"""Test the `_compartments` module of `skcomplex.models.biology.epidemiology`."""

import numpy as np

from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.problems import Form
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.epidemiology import CompartmentElement, InfectionInteraction, RecoveryInteraction


def test_epidemic_stochastic_mode_is_jump_process():
    """Test that stochastic infection and recovery give integer counts of a jump process."""
    system = System(
        elements=[CompartmentElement('S', n=990), CompartmentElement('I', n=10), CompartmentElement('R')],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=0.0005, stochastic=True),
            RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.1, stochastic=True),
        ],
    )
    assert compile_system(system).form == Form.JUMP
    states = Simulation(system, dt=1.0, random_state=3).check().run(until=50).trajectory_.states
    assert np.all(states >= 0)
    assert np.all(states == np.round(states))
    assert np.allclose(states.sum(axis=1), 1000)


def test_epidemic_deterministic_mode_conserves_population():
    """Test that the deterministic SIR model keeps the total population."""
    system = System(
        elements=[CompartmentElement('S', n=990), CompartmentElement('I', n=10), CompartmentElement('R')],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=0.0005),
            RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.1),
        ],
    )
    states = Simulation(system, dt=1.0).check().run(until=50).trajectory_.states
    assert np.allclose(states.sum(axis=1), 1000)
