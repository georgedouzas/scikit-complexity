"""Share the run helpers and solvers of the oncology system tests."""

from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation, Trajectory
from skcomplex.core.engine.solvers import BaseSolver, ScipyODESolver

PRECISE = ScipyODESolver(rtol=1e-9, atol=1e-12)
STIFF = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)


def _run(system: System, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a system integrated tightly."""
    return Simulation(system, dt=dt, solver=PRECISE).prepare().run(until=until).trajectory_


def _run_optional_solver(system: System, until: float, dt: float, solver: BaseSolver | None = None) -> Trajectory:
    """Return the trajectory of a system."""
    settings = {} if solver is None else {'solver': solver}
    return Simulation(system, dt=dt, random_state=0, **settings).prepare().run(until=until).trajectory_
