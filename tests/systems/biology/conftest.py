"""Share the run helpers and solvers of the biology system tests."""

from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation, Trajectory
from skcomplex.core.engine.solvers import BaseSolver, ScipyODESolver
from skcomplex.systems import create_system

TOLERANCE = 0.1
PRECISE = ScipyODESolver(rtol=1e-10, atol=1e-12)
SYSTEMS_PRECISE = ScipyODESolver(rtol=1e-9, atol=1e-11)


def _run(name: str, until: float, dt: float, **arguments: object) -> Trajectory:
    """Return the trajectory of a system created by name and run with a fixed seed."""
    return Simulation(create_system(name, **arguments), dt=dt, random_state=1).prepare().run(until=until).trajectory_


def _run_until(
    system: System,
    until: float,
    *,
    dt: float | None = None,
    solver: BaseSolver | None = None,
    random_state: int | None = None,
) -> Trajectory:
    """Return the trajectory of a system run until a time."""
    simulation = Simulation(system, dt=dt, solver=solver, random_state=random_state)
    return simulation.prepare().run(until=until).trajectory_


def _run_precise(system: System, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a system integrated tightly."""
    return Simulation(system, dt=dt, solver=PRECISE).prepare().run(until=until).trajectory_


def _run_seeded(system: System, until: float, dt: float, solver: BaseSolver = PRECISE) -> Trajectory:
    """Return the trajectory of a system integrated tightly with a fixed seed."""
    return Simulation(system, dt=dt, random_state=0, solver=solver).prepare().run(until=until).trajectory_


def _run_with_solver(system: System, until: float, dt: float, solver: BaseSolver | None = None) -> Trajectory:
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt, random_state=0, solver=solver).prepare().run(until=until).trajectory_
