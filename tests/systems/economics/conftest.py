"""Share the run helpers of the economics system tests."""

from collections.abc import Callable, Mapping, Sequence

from skcomplex.core.components import Observable
from skcomplex.core.engine.runtime import Trajectory
from skcomplex.core.engine.solvers import BaseSolver
from skcomplex.core.systems import BaseSystem
from skcomplex.core.types import AnyArray


def _simulate(
    system: BaseSystem,
    solver: BaseSolver | None = None,
    *,
    until: float | None,
    dt: float | None = None,
    observables: Sequence[Observable] = (),
    schedule: Mapping[str, Callable[[float], float]] | None = None,
    invariants: Mapping[str, float] | None = None,
    random_state: int | None = None,
) -> Trajectory:
    """Return the trajectory of a system, prepared and simulated with a solver if given."""
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


def _run(
    system: BaseSystem,
    until: float,
    dt: float | None = None,
    observables: Sequence[Observable] = (),
) -> Trajectory:
    """Return the trajectory of a system run with a fixed random state, with its tuned settings for the others."""
    return _simulate(system, until=until, dt=dt, observables=observables, random_state=0)


def _agents(trajectory: Trajectory, group: str, frame: int) -> Mapping[str, AnyArray]:
    """Return the states of a group's agents at a frame."""
    return trajectory.get_agents(group, frame).values


def _run_observed(
    system: BaseSystem,
    until: float,
    observables: Sequence[Observable] = (),
    invariants: Mapping[str, float] | None = None,
) -> Trajectory:
    """Return the trajectory of a system run with a fixed random state."""
    trajectory = _simulate(system, until=until, random_state=0, observables=observables, invariants=invariants)
    return trajectory
