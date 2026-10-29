"""Share the run helpers of the social model tests."""

import numpy as np
import numpy.typing as npt

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory


def _run(system: System, until: float, *, dt: float | None = None) -> Trajectory:
    """Return the trajectory of a model run with a fixed random state."""
    return Simulation(system, random_state=0, dt=dt).check().prepare().run(until=until).trajectory_


def _values(trajectory: Trajectory, group: str, name: str, frame: int = -1) -> npt.NDArray[np.float64]:
    """Return the values of an agent state at a frame."""
    frame = len(trajectory.times) - 1 if frame == -1 else frame
    return np.asarray(trajectory.agents(group, frame).values[name], dtype=np.float64)
