"""Share the run helpers of the mathematics system tests."""

from collections.abc import Sequence

from skcomplex.core.components.systems import BaseSystem
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.specs import ObservableSpec


def _run(
    system: BaseSystem,
    until: float,
    dt: float = 1,
    random_state: int = 0,
    observables: Sequence[ObservableSpec] = (),
) -> Trajectory:
    """Return the trajectory of a system run with a frame duration and a random state, with its tuned solver."""
    return simulate(system, until=until, dt=dt, random_state=random_state, observables=observables)


def _final(system: BaseSystem, until: float) -> Trajectory:
    """Return the trajectory of a system run to `until` with frames at the start and the end only."""
    return simulate(system, until=until, dt=until, random_state=0)
