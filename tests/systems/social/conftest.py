"""Share the run helpers of the social system tests."""

import numpy as np
import numpy.typing as npt

from skcomplex.core.engine.runtime import Trajectory
from skcomplex.core.systems import BaseSystem


def _run(system: BaseSystem, until: float, *, dt: float | None = None) -> Trajectory:
    """Return the trajectory of a system prepared and run with a fixed random state and its tuned settings."""
    return system.prepare().simulate(until, dt=dt, random_state=0)


def _values(trajectory: Trajectory, group: str, name: str, frame: int = -1) -> npt.NDArray[np.float64]:
    """Return the values of an agent state at a frame."""
    frame = len(trajectory.times) - 1 if frame == -1 else frame
    return np.asarray(trajectory.get_agents(group, frame).values[name], dtype=np.float64)
