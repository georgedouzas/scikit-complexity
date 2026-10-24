"""Share the run helper of the chemical kinetics model tests."""

from typing import Any

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory


def _run(system: System, until: float, dt: float, **settings: Any) -> Trajectory:  # noqa: ANN401
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt, random_state=0, **settings).run(until=until).trajectory_
