"""Share the run helpers and the event observer of the model tests."""

from collections.abc import Callable
from typing import Any

import numpy as np

from skcomplex.core.base import BaseProblem
from skcomplex.core.system import System
from skcomplex.engine.runtime import BaseObserver, ConservationObserver, Frame, Simulation, Trajectory
from skcomplex.engine.solvers import Solver
from skcomplex.models import create, describe


def _run(system: System, until: float, dt: float = 1, random_state: int = 0) -> Trajectory:
    """Return the trajectory of a system run with a frame duration and a random state."""
    return Simulation(system, dt=dt, random_state=random_state).run(until=until).trajectory_


def _trajectory(
    name: str,
    until: float,
    dt: float,
    solver: str | Solver = 'auto',
    **options: Any,  # noqa: ANN401
) -> Trajectory:
    """Return the trajectory of a model run with its standard observables."""
    system = create(name, **options)
    observables = describe(name).observables_of(system)
    return (
        Simulation(system, dt=dt, observables=observables, solver=solver, random_state=0).run(until=until).trajectory_
    )


def _run_monitored(system: System, dt: float, until: float) -> Trajectory:
    """Return the trajectory of a system under a conservation monitor, which raises on any leak."""
    return Simulation(system, dt=dt, observers=[ConservationObserver()]).run(until=until).trajectory_


def _peak(trajectory: Trajectory, name: str) -> float:
    """Return the time at which a recorded series peaks."""
    return float(np.asarray(trajectory.times)[np.argmax(trajectory[name])])


def _series(name: str, until: float, **arguments: Any) -> Trajectory:  # noqa: ANN401
    """Return the trajectory of a model run with frames at every step."""
    system = create(name, **arguments)
    observables = describe(name).observables_of(system)
    return Simulation(system, dt=1, random_state=0, observables=observables).run(until).trajectory_


def _run_seeded(system: System, until: float, dt: float) -> Trajectory:
    """Return the trajectory of a system run with a fixed random state."""
    return Simulation(system, dt=dt, random_state=0).run(until=until).trajectory_


class EventsObserver(BaseObserver):
    """Observer that keeps the events of every frame and checks a condition on the frames."""

    def __init__(self, condition: Callable[[Frame], object] | None = None) -> None:
        """Keep the condition."""
        self.condition = condition

    def _prepare(self, problem: BaseProblem) -> None:
        """Start with no events."""
        self.events_: list[object] = []
        self.passed_ = True

    def _run(self, frame: Frame | None = None) -> None:
        """Keep the frame's events and check the condition."""
        if frame is None:
            return
        self.events_ += list(frame.events)
        if self.condition is not None:
            self.passed_ &= bool(self.condition(frame))
