"""Share the run helper of the chemical kinetics model tests."""

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.engine.solvers import BaseSolver


def _run(system: System, until: float, dt: float, *, solver: BaseSolver | None = None) -> Trajectory:
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt, random_state=0, solver=solver).check().prepare().run(until=until).trajectory_
