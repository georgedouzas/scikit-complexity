"""Share the run helper of the computer science system tests."""

from skcomplex.core.engine.runtime import Trajectory
from skcomplex.core.systems import BaseSystem


def _run(system: BaseSystem, until: float, dt: float = 1, random_state: int = 0) -> Trajectory:
    """Return the trajectory of a prepared system simulated with a frame duration and a random state."""
    return system.prepare().simulate(until=until, dt=dt, random_state=random_state)
