"""Share the run helpers of the mathematics system tests."""

from collections.abc import Sequence

from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation, Trajectory
from skcomplex.core.specs import ObservableSpec
from skcomplex.systems import create_system


def _run(
    system: System,
    until: float,
    dt: float = 1,
    random_state: int = 0,
    observables: Sequence[ObservableSpec] = (),
) -> Trajectory:
    """Return the trajectory of a system run with a frame duration and a random state."""
    simulation = Simulation(system, dt=dt, random_state=random_state, observables=observables).prepare()
    return simulation.run(until=until).trajectory_


def _final(name: str, until: float, **arguments: object) -> Simulation:
    """Return the simulation of a system run to `until` with frames at the start and the end only."""
    return Simulation(create_system(name, **arguments), dt=until, random_state=0).prepare().run(until)
