"""Share the run helper of the chemical kinetics system tests."""

from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation, Trajectory
from skcomplex.core.engine.solvers import BaseSolver


def _run(system: System, until: float, dt: float, *, solver: BaseSolver | None = None) -> Trajectory:
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt, random_state=0, solver=solver).prepare().run(until=until).trajectory_
