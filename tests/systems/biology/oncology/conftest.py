"""Share the run helpers and solvers of the oncology system tests."""

from skcomplex.core.components.systems import BaseSystem as OldBaseSystem
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.engine.solvers import BaseSolver, ScipyODESolver
from skcomplex.core.systems import BaseSystem

PRECISE = ScipyODESolver(rtol=1e-9, atol=1e-12)
STIFF = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)


def _simulate(
    system: BaseSystem | OldBaseSystem,
    until: float,
    dt: float,
    solver: BaseSolver | None,
    random_state: int | None = None,
) -> Trajectory:
    """Return the trajectory of a system, prepared and simulated with a solver if given when it has the new base."""
    if isinstance(system, BaseSystem):
        system = system if solver is None else system.set_params(solver=solver)
        return system.prepare().simulate(until, dt=dt, random_state=random_state)
    return simulate(system, until=until, dt=dt, solver=solver, random_state=random_state)


def _run(system: BaseSystem | OldBaseSystem, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a system integrated tightly."""
    return _simulate(system, until, dt, PRECISE)


def _run_optional_solver(
    system: BaseSystem | OldBaseSystem,
    until: float,
    dt: float,
    solver: BaseSolver | None = None,
) -> Trajectory:
    """Return the trajectory of a system, with its tuned solver unless one is given."""
    return _simulate(system, until, dt, solver, random_state=0)
