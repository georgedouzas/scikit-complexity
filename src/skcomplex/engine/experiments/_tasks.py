"""Run one task of an experiment, one simulation of a design row, scenario, and replicate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.system import Observable, System
from ..compile import Problem, observables_of
from ..runtime import BaseObserver, Frame, Simulation, Trajectory
from ._designs import Row
from ._results import RunResult
from ._scenarios import Scenario
from ._sources import _problem, split_overrides, with_values

type Output = str | Callable[[Trajectory], Any]


@dataclass(frozen=True)
class Task:
    """One run of an experiment.

    Args:
        index:
            The position of the run in the experiment.

        source:
            The factory or system of the model.

        row:
            The design values.

        scenario:
            The scenario.

        replicate:
            The replicate number.

        seed:
            The random stream of the run.

        until:
            The simulated time to run to.

        settings:
            The `Simulation` arguments.

        outputs:
            The outputs to keep.

        warmup:
            The simulated time before which kept series are discarded.

        observables:
            The observables of the run, or the name of the registered model whose standard observables to use.
    """

    index: int
    source: Callable[..., System] | System
    row: Row
    scenario: Scenario
    replicate: int
    seed: np.random.SeedSequence
    until: float
    settings: Mapping[str, Any]
    outputs: Sequence[Output] | None
    warmup: float
    observables: Sequence[Observable] | str | None = None


def _scalars(name: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the scalar columns of a callable output's value."""
    if isinstance(value, Mapping):
        return {f'{name}.{key}': item for key, item in value.items()}
    return {name: value}


def _default_outputs(problem: Problem) -> list[str]:
    """Return every observable of a problem's system, or every state when it declares none."""
    observables = [
        f'{scope}/{observable.label}' if scope else observable.label
        for scope, observable in (observables_of(problem.system) if problem.system is not None else ())
    ]
    return observables or list(problem.states.names)


def run_task(task: Task) -> RunResult:
    """Run one task and return its kept series and scalar outputs.

    Args:
        task:
            The task.

    Returns:
        result:
            The run's design values, scenario, replicate, seed, series, and scalars.
    """
    arguments, paths = split_overrides({**task.row, **(task.scenario.overrides or {})})
    problem = with_values(_problem(task.source, arguments, task.scenario.system, task.observables), paths)
    schedule = _Schedule(task.scenario.schedule)
    simulation = Simulation(
        problem,
        observers=[schedule],
        random_state=np.random.default_rng(task.seed),
        **task.settings,
    )
    schedule.simulation = simulation
    trajectory = simulation.run(until=task.until).trajectory_
    keep = trajectory.times >= task.warmup - 1e-12
    names = list(task.outputs) if task.outputs is not None else _default_outputs(problem)
    series, scalars = {}, {}
    for output in names:
        if callable(output):
            scalars.update(_scalars(getattr(output, '__name__', 'output'), output(trajectory)))
            continue
        values = np.asarray(trajectory[output])
        series[output] = values[keep]
        if values.ndim == 1 and values.size:
            scalars[output] = values[-1]
    return RunResult(
        index=task.index,
        row=dict(task.row),
        scenario=task.scenario.name,
        replicate=task.replicate,
        seed=int(task.seed.spawn_key[-1]) if task.seed.spawn_key else 0,
        times=trajectory.times[keep],
        series=series,
        scalars=scalars,
    )


class _Schedule(BaseObserver):
    """Observer that applies scheduled parameter changes from the first frame at or after their time."""

    def __init__(self, schedule: Sequence[tuple[float, str, Any]]) -> None:
        super().__init__()
        self.pending = sorted(schedule, key=lambda change: change[0])
        self.simulation: Simulation | None = None

    def on_frame(self, frame: Frame) -> None:
        """Queue every change whose time has come.

        Args:
            frame:
                The frame.
        """
        due = [change for change in self.pending if change[0] <= frame.time + 1e-12]
        if due and self.simulation is not None:
            self.simulation.update_parameters({path: value for _, path, value in due})
            self.pending = self.pending[len(due) :]
