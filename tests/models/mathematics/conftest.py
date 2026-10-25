"""Share the run helpers of the mathematics model tests."""

from collections.abc import Sequence
from typing import Any

from skcomplex.core.system import ObservableQuantity, System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.models import create


def _run(
    system: System,
    until: float,
    dt: float = 1,
    random_state: int = 0,
    observables: Sequence[ObservableQuantity] = (),
) -> Trajectory:
    """Return the trajectory of a system run with a frame duration and a random state."""
    simulation = Simulation(system, dt=dt, random_state=random_state, observables=observables)
    return simulation.run(until=until).trajectory_


def _final(name: str, until: float, **arguments: Any) -> Simulation:  # noqa: ANN401
    """Return the simulation of a model run to `until` with frames at the start and the end only."""
    return Simulation(create(name, **arguments), dt=until, random_state=0).run(until)
