"""Test the `RK4Solver` class."""

import time

import pytest

from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import (
    BaseSolver,
    RK4Solver,
    Solver,
)
from skcomplex.systems import create_system


def test_fixed_step_integrators_step_over_switches_of_the_drift():
    """Test that RK4 steps over the swing-up controller switch instead of firing its hidden event forever."""
    system = create_system('pendulum_swing_up')
    assert any(event.hidden for event in PassCompiler().prepare(system).run().problem_.events)
    start = time.perf_counter()
    simulation = Simulation(system, dt=0.1, solver=RK4Solver(dt=0.01)).prepare().run(until=75)
    assert time.perf_counter() - start < 30
    assert simulation.time_ == pytest.approx(75)


def test_ode_solvers_derive_from_the_solver_base():
    """Test that the solvers of the compiler's problem derive from the solver base through the engine's solver."""
    assert issubclass(RK4Solver, Solver)
    assert issubclass(RK4Solver, BaseSolver)
