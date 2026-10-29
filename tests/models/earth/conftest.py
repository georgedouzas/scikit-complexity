"""Share the run helper of the earth science model tests."""

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.engine.solvers import BaseSolver


def _run(system: System, until: float, dt: float, solver: BaseSolver | None = None) -> Trajectory:
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt, solver=solver, random_state=0).check().run(until=until).trajectory_
