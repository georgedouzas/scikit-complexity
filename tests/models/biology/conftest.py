"""Share the run helpers and solvers of the biology model tests."""

from typing import Any

from skcomplex.core.runtime import Simulation, Trajectory
from skcomplex.core.solvers import ScipyODE, Stepper
from skcomplex.core.system import System
from skcomplex.models import create

TOLERANCE = 0.1
PRECISE = ScipyODE(rtol=1e-10, atol=1e-12)
SYSTEMS_PRECISE = ScipyODE(rtol=1e-9, atol=1e-11)


def _run(name: str, until: float, dt: float, **arguments: Any) -> Trajectory:  # noqa: ANN401
    """Return the trajectory of a model created by name and run with a fixed seed."""
    return Simulation(create(name, **arguments), dt=dt, random_state=1).run(until=until).trajectory_


def _run_until(system: System, until: float, **settings: Any) -> Trajectory:  # noqa: ANN401
    """Return the trajectory of a system run until a time."""
    return Simulation(system, **settings).run(until=until).trajectory_


def _run_precise(system: System, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a system integrated tightly."""
    return Simulation(system, dt=dt, solver=PRECISE).run(until=until).trajectory_


def _run_seeded(system: System, until: float, dt: float, solver: str | Stepper = PRECISE) -> Trajectory:
    """Return the trajectory of a system integrated tightly with a fixed seed."""
    return Simulation(system, dt=dt, random_state=0, solver=solver).run(until=until).trajectory_


def _run_with_solver(system: System, until: float, dt: float, solver: str | Stepper = 'auto') -> Trajectory:
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt, random_state=0, solver=solver).run(until=until).trajectory_
