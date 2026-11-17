"""Share the run helper of the chemical kinetics system tests."""

from skcomplex.core.components.systems import BaseSystem
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.engine.solvers import BaseSolver
from skcomplex.core.systems import BaseSystem as PreparedSystem


def _run(
    system: BaseSystem | PreparedSystem,
    until: float,
    dt: float,
    *,
    solver: BaseSolver | None = None,
) -> Trajectory:
    """Return the trajectory of a system, with its tuned solver unless one is given."""
    if not isinstance(system, PreparedSystem):
        return simulate(system, until=until, dt=dt, solver=solver, random_state=0)
    prepared = system.prepare() if solver is None else system.prepare().system_.set_params(solver=solver).prepare()
    return prepared.simulate(until=until, dt=dt, random_state=0)
