"""Share the run helpers of the social system tests."""

import numpy as np
import numpy.typing as npt

from skcomplex.core.components.systems import BaseSystem
from skcomplex.core.engine.runtime import Trajectory, simulate


def _run(system: BaseSystem, until: float, *, dt: float | None = None) -> Trajectory:
    """Return the trajectory of a system run with a fixed random state, with its tuned settings for the others."""
    return simulate(system, until=until, dt=dt, random_state=0)


def _values(trajectory: Trajectory, group: str, name: str, frame: int = -1) -> npt.NDArray[np.float64]:
    """Return the values of an agent state at a frame."""
    frame = len(trajectory.times) - 1 if frame == -1 else frame
    return np.asarray(trajectory.agents(group, frame).values[name], dtype=np.float64)
