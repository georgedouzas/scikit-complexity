"""Test the `RK4Solver` class."""

import time

import pytest

from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import (
    BaseSolver,
    RK4Solver,
    Solver,
)
from skcomplex.systems.engineering.control import PendulumSwingUpSystem


def test_fixed_step_integrators_step_over_switches_of_the_drift():
    """Test that RK4 steps over the swing-up controller switch instead of firing its hidden event forever."""
    system = PendulumSwingUpSystem().prepare().system_
    assert any(event.hidden for event in PassCompiler().prepare(system).run().problem_.events)
    start = time.perf_counter()
    trajectory = simulate(system, until=75, dt=0.1, solver=RK4Solver(dt=0.01))
    assert time.perf_counter() - start < 30
    assert trajectory.times[-1] == pytest.approx(75)


def test_ode_solvers_derive_from_the_solver_base():
    """Test that the solvers of the compiler's problem derive from the solver base through the engine's solver."""
    assert issubclass(RK4Solver, Solver)
    assert issubclass(RK4Solver, BaseSolver)
