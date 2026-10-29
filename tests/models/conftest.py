"""Share the run helpers and the event observer of the model tests."""

from collections.abc import Callable
from typing import Any

import numpy as np

from skcomplex.core.system import System
from skcomplex.engine.problems import BaseProblemSpec, FrameSpec
from skcomplex.engine.runtime import BaseObserver, ConservationObserver, Simulation, Trajectory
from skcomplex.engine.solvers import BaseSolver
from skcomplex.models import create, describe
from skcomplex.specs import ObservableSpec


def _run(system: System, until: float, dt: float = 1, random_state: int = 0) -> Trajectory:
    """Return the trajectory of a system run with a frame duration and a random state."""
    return Simulation(system, dt=dt, random_state=random_state).check().run(until=until).trajectory_


def _trajectory(
    name: str,
    until: float,
    dt: float,
    solver: BaseSolver | None = None,
    **options: Any,  # noqa: ANN401
) -> Trajectory:
    """Return the trajectory of a model run with its standard observables."""
    system = create(name, **options)
    observables = describe(name).observables_of(system)
    return (

            Simulation(system, dt=dt, observables=observables, solver=solver, random_state=0)
            .check()
            .run(until=until)
            .trajectory_

    )


def _observables(system: System, model: str | None) -> list[ObservableSpec]:
    """Return the standard observables of a system of a model, or none without a model."""
    return [] if model is None else describe(model).observables_of(system)


def _run_monitored(system: System, dt: float, until: float, model: str | None = None) -> Trajectory:
    """Return the trajectory of a system under a conservation monitor, which raises on any leak."""
    observables = _observables(system, model)
    simulation = Simulation(system, dt=dt, observers=[ConservationObserver()], observables=observables).check()
    return simulation.run(until=until).trajectory_


def _peak(trajectory: Trajectory, name: str) -> float:
    """Return the time at which a recorded series peaks."""
    return float(np.asarray(trajectory.times)[np.argmax(trajectory.values(name))])


def _series(name: str, until: float, **arguments: Any) -> Trajectory:  # noqa: ANN401
    """Return the trajectory of a model run with frames at every step."""
    system = create(name, **arguments)
    observables = describe(name).observables_of(system)
    return Simulation(system, dt=1, random_state=0, observables=observables).check().run(until).trajectory_


def _run_seeded(system: System, until: float, dt: float, model: str | None = None) -> Trajectory:
    """Return the trajectory of a system run with a fixed random state."""
    simulation = Simulation(system, dt=dt, random_state=0, observables=_observables(system, model)).check()
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
