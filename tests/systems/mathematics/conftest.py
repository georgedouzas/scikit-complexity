"""Share the run helpers of the mathematics system tests."""

from collections.abc import Sequence

from skcomplex.core.components import Observable
from skcomplex.core.components.systems import BaseSystem as OldBaseSystem
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.systems import BaseSystem


def _run(
    system: BaseSystem | OldBaseSystem,
    until: float,
    dt: float = 1,
    random_state: int = 0,
    observables: Sequence[Observable] = (),
) -> Trajectory:
    """Return the trajectory of a system run with a frame duration and a random state, with its tuned solver."""
    if isinstance(system, BaseSystem):
        return system.prepare().simulate(until, dt=dt, random_state=random_state, observables=observables)
    return simulate(system, until=until, dt=dt, random_state=random_state, observables=observables)


def _final(system: BaseSystem | OldBaseSystem, until: float) -> Trajectory:
    """Return the trajectory of a system run to `until` with frames at the start and the end only."""
    return _run(system, until, dt=until)
