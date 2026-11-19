"""Share the run helper of the chemical kinetics system tests."""

from skcomplex.core.engine.runtime import Trajectory
from skcomplex.core.engine.solvers import BaseSolver
from skcomplex.core.systems import BaseSystem


def _run(
    system: BaseSystem,
    until: float,
    dt: float,
    *,
    solver: BaseSolver | None = None,
) -> Trajectory:
    """Return the trajectory of a system, with its tuned solver unless one is given."""
    prepared = system.prepare() if solver is None else system.set_params(solver=solver).prepare()
    return prepared.simulate(until=until, dt=dt, random_state=0)
