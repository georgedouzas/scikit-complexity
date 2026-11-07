"""Share the run helpers of the economics system tests."""

from collections.abc import Mapping, Sequence

from skcomplex.core.components.system import BaseSystem, System
from skcomplex.core.engine.runtime import Simulation, Trajectory, simulate
from skcomplex.core.specs import ObservableSpec
from skcomplex.core.types import AnyArray


def _run(
    system: BaseSystem,
    until: float,
    dt: float | None = None,
    observables: Sequence[ObservableSpec] = (),
) -> Trajectory:
    """Return the trajectory of a system run with a fixed random state, with its tuned settings for the others."""
    return simulate(system, until=until, dt=dt, observables=observables, random_state=0)


def _agents(trajectory: Trajectory, group: str, frame: int) -> Mapping[str, AnyArray]:
    """Return the states of a group's agents at a frame."""
    return trajectory.agents(group, frame).values


def _run_observed(
    system: System,
    until: float,
    observables: Sequence[ObservableSpec] = (),
    invariants: Mapping[str, float] | None = None,
) -> Trajectory:
    """Return the trajectory of a system run with a fixed random state."""
    simulation = Simulation(system, random_state=0, observables=observables, invariants=invariants).prepare()
    return simulation.run(until=until).trajectory_
