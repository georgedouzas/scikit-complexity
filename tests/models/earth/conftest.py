"""Share the run helper of the earth science model tests."""

from collections.abc import Sequence

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.engine.solvers import BaseSolver
from skcomplex.specs import ObservableSpec


def _run(
    system: System,
    until: float,
    dt: float,
    solver: BaseSolver | None = None,
    observables: Sequence[ObservableSpec] = (),
) -> Trajectory:
    """Return the trajectory of a system."""
    simulation = Simulation(system, dt=dt, solver=solver, random_state=0, observables=observables)
    return simulation.check().run(until=until).trajectory_
