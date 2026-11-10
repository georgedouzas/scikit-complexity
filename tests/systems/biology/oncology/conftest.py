"""Share the run helpers and solvers of the oncology system tests."""

from skcomplex.core.components.systems import BaseSystem
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.engine.solvers import BaseSolver, ScipyODESolver

PRECISE = ScipyODESolver(rtol=1e-9, atol=1e-12)
STIFF = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)


def _run(system: BaseSystem, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a system integrated tightly."""
    return simulate(system, until=until, dt=dt, solver=PRECISE)


def _run_optional_solver(system: BaseSystem, until: float, dt: float, solver: BaseSolver | None = None) -> Trajectory:
    """Return the trajectory of a system, with its tuned solver unless one is given."""
    return simulate(system, until=until, dt=dt, solver=solver, random_state=0)
