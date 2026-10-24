"""Share the run helpers of the social model tests."""

from typing import Any

import numpy.typing as npt

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory


def _run(system: System, until: float, **settings: Any) -> Trajectory:  # noqa: ANN401
    """Return the trajectory of a model run with a fixed random state."""
    return Simulation(system, random_state=0, **settings).run(until=until).trajectory_


def _values(trajectory: Trajectory, group: str, name: str, frame: int = -1) -> npt.NDArray[Any]:
    """Return the values of an agent state at a frame."""
    frame = len(trajectory.times) - 1 if frame == -1 else frame
    return trajectory.agents(group, frame).values[name]
