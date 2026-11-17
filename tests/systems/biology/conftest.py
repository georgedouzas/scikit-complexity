"""Share the run helpers and solvers of the biology system tests."""

from collections.abc import Callable, Mapping, Sequence

from skcomplex.core.components import Observable
from skcomplex.core.components.systems import BaseSystem as OldBaseSystem
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.engine.solvers import BaseSolver, ScipyODESolver
from skcomplex.core.systems import BaseSystem

TOLERANCE = 0.1
PRECISE = ScipyODESolver(rtol=1e-10, atol=1e-12)
SYSTEMS_PRECISE = ScipyODESolver(rtol=1e-9, atol=1e-11)


def _simulate(
    system: BaseSystem | OldBaseSystem,
    solver: BaseSolver | None = None,
    *,
    until: float | None,
    dt: float | None = None,
    observables: Sequence[Observable] = (),
    schedule: Mapping[str, Callable[[float], float]] | None = None,
    invariants: Mapping[str, float] | None = None,
    random_state: int | None = None,
) -> Trajectory:
    """Return the trajectory of a system, prepared and simulated with a solver if given when it has the new base."""
    if isinstance(system, BaseSystem):
        if solver is not None:
            system.set_params(solver=solver)
        return system.prepare().simulate(
            until,
            dt=dt,
            observables=observables,
            schedule=schedule,
            invariants=invariants,
            random_state=random_state,
        )
    return simulate(
        system,
        until=until,
        dt=dt,
        solver=solver,
        observables=observables,
        schedule=schedule,
        invariants=invariants,
        random_state=random_state,
    )


def _run(system: BaseSystem | OldBaseSystem, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a system run with its tuned solver and a fixed seed."""
    return _simulate(system, until=until, dt=dt, random_state=1)


def _run_until(
    system: BaseSystem | OldBaseSystem,
    until: float,
    *,
    dt: float | None = None,
    solver: BaseSolver | None = None,
    random_state: int | None = None,
) -> Trajectory:
    """Return the trajectory of a system run until a time, with its tuned settings unless others are given."""
    return _simulate(system, solver, until=until, dt=dt, random_state=random_state)


def _run_precise(system: BaseSystem | OldBaseSystem, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a system integrated tightly."""
    return _simulate(system, PRECISE, until=until, dt=dt)


def _run_seeded(
    system: BaseSystem | OldBaseSystem,
    until: float,
    dt: float,
    solver: BaseSolver = PRECISE,
) -> Trajectory:
    """Return the trajectory of a system integrated tightly with a fixed seed."""
    return _simulate(system, solver, until=until, dt=dt, random_state=0)


def _run_with_solver(
    system: BaseSystem | OldBaseSystem,
    until: float,
    dt: float,
    solver: BaseSolver | None = None,
) -> Trajectory:
    """Return the trajectory of a system, with its tuned solver unless one is given."""
    return _simulate(system, solver, until=until, dt=dt, random_state=0)
