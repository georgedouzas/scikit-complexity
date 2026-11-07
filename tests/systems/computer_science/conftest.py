"""Share the run helper of the computer science system tests."""

from skcomplex.core.components.system import BaseSystem
from skcomplex.core.engine.runtime import Trajectory, simulate


def _run(system: BaseSystem, until: float, dt: float = 1, random_state: int = 0) -> Trajectory:
    """Return the trajectory of a system run with a frame duration and a random state, with its tuned solver."""
    return simulate(system, until=until, dt=dt, random_state=random_state)
