"""Share the run helpers and solvers of the biology system tests."""

from skcomplex.core.components.systems import BaseSystem
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.engine.solvers import BaseSolver, ScipyODESolver

TOLERANCE = 0.1
PRECISE = ScipyODESolver(rtol=1e-10, atol=1e-12)
SYSTEMS_PRECISE = ScipyODESolver(rtol=1e-9, atol=1e-11)


def _run(system: BaseSystem, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a system run with its tuned solver and a fixed seed."""
    return simulate(system, until=until, dt=dt, random_state=1)


def _run_until(
    system: BaseSystem,
    until: float,
    *,
    dt: float | None = None,
    solver: BaseSolver | None = None,
    random_state: int | None = None,
) -> Trajectory:
    """Return the trajectory of a system run until a time, with its tuned settings unless others are given."""
    return simulate(system, until=until, dt=dt, solver=solver, random_state=random_state)


def _run_precise(system: BaseSystem, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a system integrated tightly."""
    return simulate(system, until=until, dt=dt, solver=PRECISE)


def _run_seeded(system: BaseSystem, until: float, dt: float, solver: BaseSolver = PRECISE) -> Trajectory:
    """Return the trajectory of a system integrated tightly with a fixed seed."""
    return simulate(system, until=until, dt=dt, solver=solver, random_state=0)


def _run_with_solver(system: BaseSystem, until: float, dt: float, solver: BaseSolver | None = None) -> Trajectory:
    """Return the trajectory of a system, with its tuned solver unless one is given."""
    return simulate(system, until=until, dt=dt, solver=solver, random_state=0)
