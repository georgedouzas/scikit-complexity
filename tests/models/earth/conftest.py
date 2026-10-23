"""Share the run helper of the earth science model tests."""

from typing import Any

from skcomplex.core.runtime import Simulation, Trajectory
from skcomplex.core.system import System


def _run(system: System, until: float, dt: float, **settings: Any) -> Trajectory:  # noqa: ANN401
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt, random_state=0, **settings).run(until=until).trajectory_
