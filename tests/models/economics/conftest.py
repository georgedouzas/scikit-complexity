"""Share the run helpers of the economics model tests."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy.typing as npt

from skcomplex.core.system import Observable, System
from skcomplex.engine.runtime import Simulation, Trajectory


def _run(system: System, until: float, **settings: Any) -> Trajectory:  # noqa: ANN401
    """Return the trajectory of a system run with a fixed random state."""
    return Simulation(system, random_state=0, **settings).run(until=until).trajectory_


def _agents(trajectory: Trajectory, group: str, frame: int) -> Mapping[str, npt.NDArray[Any]]:
    """Return the states of a group's agents at a frame."""
    return trajectory.agents(group, frame).values


def _run_observed(
    system: System,
    until: float,
    observables: Sequence[Observable] = (),
    invariants: Mapping[str, float] | None = None,
) -> Trajectory:
    """Return the trajectory of a model run with a fixed random state."""
    simulation = Simulation(system, random_state=0, observables=observables, invariants=invariants)
    return simulation.run(until=until).trajectory_
