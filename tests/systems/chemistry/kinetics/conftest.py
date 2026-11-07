"""Share the run helper of the chemical kinetics system tests."""

from skcomplex.core.components.system import BaseSystem
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.engine.solvers import BaseSolver


def _run(system: BaseSystem, until: float, dt: float, *, solver: BaseSolver | None = None) -> Trajectory:
    """Return the trajectory of a system, with its tuned solver unless one is given."""
    return simulate(system, until=until, dt=dt, solver=solver, random_state=0)
