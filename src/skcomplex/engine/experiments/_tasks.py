"""Run one task of an experiment, one simulation of a design row, scenario, and replicate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.problems import BaseProblem
from ...core.system import Observable, System
from ..compile import observables_of
from ..runtime import Simulation, Trajectory
from ._designs import Row
from ._results import RunResult
from ._scenarios import Scenario
from ._sources import _problem, split_overrides, with_values

type Output = str | Callable[[Trajectory], Any]


def _scalars(name: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the scalar columns of a callable output's value."""
    if isinstance(value, Mapping):
        return {f'{name}.{key}': item for key, item in value.items()}
    return {name: value}


def _default_outputs(problem: BaseProblem) -> list[str]:
    """Return every observable of a problem's system, or every state when it declares none."""
    system = problem.system
    observables = [
        f'{scope}/{observable.label}' if scope else observable.label
        for scope, observable in (observables_of(system) if system is not None else ())
    ]
    return observables or list(problem.states.names)


def _step_function(initial: Any, changes: list[tuple[float, Any]]) -> Callable[[float], Any]:  # noqa: ANN401
    """Return the function of time that takes the value of the last change at or before a time, else the initial one."""

    def value_at(time: float) -> Any:  # noqa: ANN401
        due = [value for start, value in changes if start <= time + 1e-12]
        return due[-1] if due else initial

    return value_at


def _initial_value(problem: BaseProblem, path: str) -> Any:  # noqa: ANN401
    """Return the initial value of a parameter or an adaptive parameter, or `None` for a name the problem lacks."""
    layout, vector = (
        (problem.states, problem.initial_state)
        if path in problem.adaptive_parameters
        else (problem.parameters, problem.initial_parameters)
    )
    try:
        entry, indices = layout.locate(path)
    except KeyError:
        return None
    return entry.decode(vector[indices]).reshape(entry.shape)


class _Schedule:
    """Scheduled parameter changes of a task, each in force from the first frame at or after its time.

    Args:
        changes:
            The changes, each a time, a parameter path, and a value.
    """

    def __init__(self, changes: Sequence[tuple[float, str, Any]]) -> None:
        self.changes = changes

    def functions(self, problem: BaseProblem) -> dict[str, Callable[[float], Any]]:
        """Return the value of every scheduled parameter as a function of time, which a simulation reads every frame.

        Before its first change a parameter keeps its initial value in the problem.

        Args:
            problem:
                The problem whose parameters change.

        Returns:
            functions:
                The function of time of every scheduled parameter, by path.
        """
        ordered = sorted(self.changes, key=lambda change: change[0])
        paths = dict.fromkeys(path for _, path, _ in ordered)
        return {
            path: _step_function(
                _initial_value(problem, path),
                [(time, value) for time, name, value in ordered if name == path],
            )
            for path in paths
        }


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

    @property
    def schedule(self) -> _Schedule:
        """Scheduled parameter changes of the scenario, which the run reads at every frame."""
        return _Schedule(self.scenario.schedule)


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
    simulation = Simulation(
        problem,
        random_state=np.random.default_rng(task.seed),
        schedule=task.schedule.functions(problem),
        **task.settings,
    )
    trajectory = simulation.run(until=task.until).trajectory_
    keep = trajectory.times >= task.warmup - 1e-12
    names = list(task.outputs) if task.outputs is not None else _default_outputs(problem)
    series, scalars = {}, {}
    for output in names:
        if callable(output):
            scalars.update(_scalars(getattr(output, '__name__', 'output'), output(trajectory)))
            continue
        values = np.asarray(trajectory.values(output))
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
