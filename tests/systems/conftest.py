"""Share the run helpers and the event observer of the system tests."""

from collections.abc import Callable

import numpy as np

from skcomplex.core.components import Observable
from skcomplex.core.engine.problems import BaseProblemSpec, FrameSpec
from skcomplex.core.engine.runtime import BaseObserver, ConservationObserver, Trajectory
from skcomplex.core.systems import BaseSystem


def _simulate(system: BaseSystem, **arguments: object) -> Trajectory:
    """Return the trajectory of a system, prepared first unless it is a prepared `System`."""
    if getattr(system, 'system_', None) is not system:
        system.prepare()
    return system.simulate(**arguments)


def _run(system: BaseSystem, until: float, dt: float = 1, random_state: int = 0) -> Trajectory:
    """Return the trajectory of a system run with a frame duration and a random state."""
    return _simulate(system, until=until, dt=dt, random_state=random_state)


def _observables(observed: BaseSystem | None) -> list[Observable]:
    """Return the standard observables of a ready-made system, or none without one."""
    return [] if observed is None else list(observed.observables)


def _run_monitored(system: BaseSystem, dt: float, until: float, observed: BaseSystem | None = None) -> Trajectory:
    """Return the trajectory of a system under a conservation monitor, which raises on any leak.

    The observables are the standard observables of the ready-made system `observed`, or none without one.
    """
    observables = _observables(observed)
    return _simulate(system, until=until, dt=dt, observers=[ConservationObserver()], observables=observables)


def _peak(trajectory: Trajectory, name: str) -> float:
    """Return the time at which a recorded series peaks."""
    return float(np.asarray(trajectory.times)[np.argmax(trajectory.get_values(name))])


def _series(system: BaseSystem, until: float) -> Trajectory:
    """Return the trajectory of the composite of a ready-made system with frames at every step and its observables."""
    observables = _observables(system)
    return _simulate(system.prepare().system_, until=until, dt=1, random_state=0, observables=observables)


def _run_seeded(system: BaseSystem, until: float, dt: float, observed: BaseSystem | None = None) -> Trajectory:
    """Return the trajectory of a system run with a fixed random state.

    The observables are the standard observables of the ready-made system `observed`, or none without one.
    """
    return _simulate(system, until=until, dt=dt, random_state=0, observables=_observables(observed))


class EventsObserver(BaseObserver):
    """Observer that keeps the events of every frame and checks a condition on the frames."""

    def __init__(self, condition: Callable[[FrameSpec], object] | None = None) -> None:
        self.condition = condition

    def _prepare(self, problem: BaseProblemSpec) -> None:
        """Start with no events."""
        self.events_: list[object] = []
        self.passed_ = True

    def _run(self, frame: FrameSpec | None = None) -> None:
        """Keep the frame's events and check the condition."""
        if frame is None:
            return
        self.events_ += list(frame.events)
        if self.condition is not None:
            self.passed_ &= bool(self.condition(frame))
