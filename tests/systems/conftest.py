"""Share the run helpers and the event observer of the system tests."""

from collections.abc import Callable

import numpy as np

from skcomplex.core.components.system import System
from skcomplex.core.engine.problems import BaseProblemSpec, FrameSpec
from skcomplex.core.engine.runtime import BaseObserver, ConservationObserver, Simulation, Trajectory
from skcomplex.core.engine.solvers import BaseSolver
from skcomplex.core.specs import ObservableSpec
from skcomplex.systems import create_system, describe


def _run(system: System, until: float, dt: float = 1, random_state: int = 0) -> Trajectory:
    """Return the trajectory of a system run with a frame duration and a random state."""
    return Simulation(system, dt=dt, random_state=random_state).prepare().run(until=until).trajectory_


def _trajectory(
    name: str,
    until: float,
    dt: float,
    solver: BaseSolver | None = None,
    **options: object,
) -> Trajectory:
    """Return the trajectory of a system run with its standard observables."""
    system = create_system(name, **options)
    observables = describe(name).observables_of(system)
    return (
        Simulation(system, dt=dt, observables=observables, solver=solver, random_state=0)
        .prepare()
        .run(until=until)
        .trajectory_
    )


def _observables(system: System, name: str | None) -> list[ObservableSpec]:
    """Return the standard observables of a catalog system, or none without a name."""
    return [] if name is None else describe(name).observables_of(system)


def _run_monitored(system: System, dt: float, until: float, name: str | None = None) -> Trajectory:
    """Return the trajectory of a system under a conservation monitor, which raises on any leak."""
    observables = _observables(system, name)
    simulation = Simulation(system, dt=dt, observers=[ConservationObserver()], observables=observables).prepare()
    return simulation.run(until=until).trajectory_


def _peak(trajectory: Trajectory, name: str) -> float:
    """Return the time at which a recorded series peaks."""
    return float(np.asarray(trajectory.times)[np.argmax(trajectory.values(name))])


def _series(name: str, until: float, **arguments: object) -> Trajectory:
    """Return the trajectory of a system run with frames at every step."""
    system = create_system(name, **arguments)
    observables = describe(name).observables_of(system)
    return Simulation(system, dt=1, random_state=0, observables=observables).prepare().run(until).trajectory_


def _run_seeded(system: System, until: float, dt: float, name: str | None = None) -> Trajectory:
    """Return the trajectory of a system run with a fixed random state."""
    simulation = Simulation(system, dt=dt, random_state=0, observables=_observables(system, name)).prepare()
    return simulation.run(until=until).trajectory_


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
