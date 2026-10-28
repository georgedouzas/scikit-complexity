"""Test the `GillespieSolver` class."""

import time

import numpy as np
import pytest

from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import GillespieSolver
from skcomplex.models.biology.epidemiology import CompartmentElement, InfectionInteraction, RecoveryInteraction
from skcomplex.specs import StepSpec


def test_gillespie_conserves_individuals():
    """Test that recoveries move individuals without creating or destroying them."""
    problem = compile_system(
        System(
            elements=[CompartmentElement('I', n=50), CompartmentElement('R')],
            interactions=[RecoveryInteraction('rec', infected='I', recovered='R', gamma=0.5, stochastic=True)],
        ),
    )
    stepper = GillespieSolver()
    stepper.set_params(random_state=np.random.default_rng(1)).check().prepare(problem)
    state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0))
    assert state.sum() == 50
    assert 0 < state[0] < 50
    assert np.all(state == np.round(state))


def _epidemic():
    """Return a stochastic epidemic of 1000 individuals."""
    return System(
        elements=[CompartmentElement('S', n=990), CompartmentElement('I', n=10), CompartmentElement('R')],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=0.0003, stochastic=True),
            RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.1, stochastic=True),
        ],
    )


def _gillespie_run(backend):
    """Return the duration and the trajectory of a compiled Gillespie run after a warm-up run."""
    simulation = Simulation(_epidemic(), backend=backend, solver=GillespieSolver(), dt=1.0, random_state=0).check()
    simulation.run(until=50)
    start = time.perf_counter()
    trajectory = simulation.run(until=50).trajectory_
    return time.perf_counter() - start, trajectory


@pytest.mark.benchmark
def test_compiled_gillespie_runs_serial_kernels():
    """Test that the default Numba backend runs a compiled Gillespie epidemic as fast as a serial one."""
    default, first = _gillespie_run(NumbaBackend())
    serial, second = _gillespie_run(NumbaBackend(parallel=False))
    assert np.array_equal(first.states, second.states)
    assert default <= 1.5 * serial + 0.005
