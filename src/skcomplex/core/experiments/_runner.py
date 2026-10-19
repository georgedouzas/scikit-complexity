"""Run a model over designs, scenarios, and replicates with independent random streams."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from ..base import ValidationError
from ..compile import Problem, compile_system, observables_of
from ..model import System
from ..runtime import MODELS, Frame, Observer, Simulation, Trajectory
from ._designs import Design, Row
from ._results import ExperimentResult, RunResult
from ._scenarios import Scenario

type Source = str | Callable[..., System] | System
type Output = str | Callable[[Trajectory], Any]

RUN_SETTINGS = ('dt', 'solver', 'backend', 'checked')
_PROBLEMS: dict[tuple[Any, ...], Problem] = {}


def resolve_source(source: Source) -> tuple[Callable[..., System] | System, dict[str, Any]]:
    """Return the factory or system of a model source and its default run settings.

    Args:
        source:
            A registered model name, a factory, or a system.

    Returns:
        source:
            The factory or the system, and the default run settings of a registered model.

    Raises:
        ValidationError:
            The source is none of the three.
    """
    if isinstance(source, str):
        entry = MODELS.resolve(source)
        return entry.factory, dict(entry.defaults)
    if isinstance(source, System) or callable(source):
        return source, {}
    error_msg = f'A model source is a registered model name, a factory, or a system, not {source!r}.'
    raise ValidationError(error_msg)


def split_overrides(values: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return overrides split into factory arguments and dotted paths of states and parameters.

    Args:
        values:
            A mapping from factory argument or dotted path to value.

    Returns:
        overrides:
            The factory arguments and the dotted paths.
    """
    arguments = {name: value for name, value in values.items() if '.' not in name}
    paths = {name: value for name, value in values.items() if '.' in name}
    return arguments, paths


def _problem(source: Callable[..., System] | System, arguments: Mapping[str, Any], system: System | None) -> Problem:
    """Return the compiled problem of a source with factory arguments, compiled once per process and arguments."""
    if system is not None:
        key: tuple[Any, ...] = ('system', id(system))
        built = system
    elif isinstance(source, System):
        if arguments:
            error_msg = (
                f'A system source takes dotted paths of states and parameters, not the arguments {sorted(arguments)}.'
            )
            raise ValidationError(error_msg)
        key, built = ('system', id(source)), source
    else:
        key = (source, tuple(sorted((name, repr(value)) for name, value in arguments.items())))
        built = None
    if key not in _PROBLEMS:
        _PROBLEMS[key] = compile_system(built if built is not None else source(**arguments))
    return _PROBLEMS[key]


def with_values(problem: Problem, paths: Mapping[str, Any]) -> Problem:
    """Return a problem whose initial states and parameters take values by dotted path.

    Args:
        problem:
            The compiled problem.

        paths:
            A mapping from dotted state or parameter path to its value.

    Returns:
        problem:
            A copy of the problem with the new initial values.

    Raises:
        ValidationError:
            A path is neither a state nor a parameter, or a value is outside its domain.
    """
    states, parameters = problem.initial_state.copy(), problem.initial_parameters.copy()
    for name, value in paths.items():
        layout, vector = (problem.states, states) if name in problem.states else (problem.parameters, parameters)
        if name not in layout:
            names = ', '.join([*problem.states.names, *problem.parameters.names])
            error_msg = f'{name} is neither a state nor a parameter of the model. Available paths are: {names}.'
            raise ValidationError(error_msg)
        entry = layout.entry(name)
        if not entry.domain.contains(value):
            error_msg = f'{name} = {value!r} is not in the {entry.domain.value} domain.'
            raise ValidationError(error_msg)
        vector[entry.indices] = np.broadcast_to(entry.domain.encode(value), entry.shape).ravel()
    return dataclasses.replace(problem, initial_state=states, initial_parameters=parameters)


class _Schedule(Observer):
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


def _scalars(name: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the scalar columns of a callable output's value."""
    if isinstance(value, Mapping):
        return {f'{name}.{key}': item for key, item in value.items()}
    return {name: value}


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
    problem = with_values(_problem(task.source, arguments, task.scenario.system), paths)
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


def _default_outputs(problem: Problem) -> list[str]:
    """Return every observable of a problem's system, or every state when it declares none."""
    observables = [
        f'{scope}/{observable.label}' if scope else observable.label
        for scope, observable in (observables_of(problem.system) if problem.system is not None else ())
    ]
    return observables or list(problem.states.names)


class Experiment(BaseEstimator):
    """Runs of a model over a design, scenarios, and replicates, with one random stream per design row and replicate.

    The streams are spawned from `random_state` in a fixed order, so the results do not depend on the executor, and
    every scenario of a design row and replicate shares its stream.

    Args:
        source:
            A registered model name, a factory, or a system.

        design:
            The design of factory arguments and dotted paths, or `None` for one run of the defaults.

        scenarios:
            The scenarios to compare, or `None` for the baseline alone.

        replicates:
            The number of runs per design row and scenario.

        until:
            The simulated time of every run, or `None` for the model's default.

        dt:
            The time between frames, or `None` for the model's default.

        solver:
            The solver, or `None` for the model's default.

        outputs:
            The state and observable paths to keep as series, and functions of a trajectory that return a scalar or a
            mapping of scalars. `None` keeps every observable, or every state when the model declares none.

        warmup:
            The simulated time before which kept series are discarded.

        random_state:
            The seed of the random streams.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor.

    Examples:
        >>> from skcomplex.core.experiments import Experiment, Grid
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> result = Experiment('logistic_growth', Grid(r=[0.5, 1.0]), until=5, outputs=['population.n']).run()
        >>> len(result.rows)
        2
    """

    def __init__(
        self,
        source: Source,
        design: Design | None = None,
        scenarios: Sequence[Scenario] | None = None,
        replicates: int = 1,
        *,
        until: float | None = None,
        dt: float | None = None,
        solver: object = None,
        outputs: Sequence[Output] | None = None,
        warmup: float = 0.0,
        random_state: int | None = None,
        executor: str | Executor = 'serial',
    ) -> None:
        self.source = source
        self.design = design
        self.scenarios = scenarios
        self.replicates = replicates
        self.until = until
        self.dt = dt
        self.solver = solver
        self.outputs = outputs
        self.warmup = warmup
        self.random_state = random_state
        self.executor = executor

    def tasks(self) -> list[Task]:
        """Return the runs of the experiment in their fixed order.

        Returns:
            tasks:
                One task per design row, scenario, and replicate.

        Raises:
            ValidationError:
                The number of replicates is not a positive integer.
        """
        if not isinstance(self.replicates, int) or self.replicates < 1:
            error_msg = f'replicates = {self.replicates!r} must be a positive integer.'
            raise ValidationError(error_msg)
        source, defaults = resolve_source(self.source)
        given = {key: value for key, value in {'dt': self.dt, 'solver': self.solver}.items() if value is not None}
        settings = {key: value for key, value in {**defaults, **given}.items() if key in RUN_SETTINGS}
        until = self.until if self.until is not None else defaults.get('until', 10.0)
        rows = self.design.rows() if self.design is not None else [{}]
        scenarios = list(self.scenarios) if self.scenarios else [Scenario()]
        seeds = np.random.SeedSequence(self.random_state).spawn(len(rows) * self.replicates)
        tasks: list[Task] = []
        for row_index, row in enumerate(rows):
            for scenario in scenarios:
                for replicate in range(self.replicates):
                    seed = seeds[row_index * self.replicates + replicate]
                    task = Task(
                        len(tasks),
                        source,
                        row,
                        scenario,
                        replicate,
                        seed,
                        until,
                        settings,
                        self.outputs,
                        self.warmup,
                    )
                    tasks.append(task)
        return tasks

    def run(self) -> ExperimentResult:
        """Run every task and return the results.

        Returns:
            result:
                The results, also stored as `result_`.
        """
        tasks = self.tasks()
        if self.executor in {'serial', 'thread'}:
            for task in tasks:
                arguments, _ = split_overrides({**task.row, **(task.scenario.overrides or {})})
                _problem(task.source, arguments, task.scenario.system)
        if self.executor == 'serial':
            results = [run_task(task) for task in tasks]
        else:
            executor = self.executor
            owned = isinstance(executor, str)
            if executor == 'thread':
                executor = ThreadPoolExecutor()
            elif executor == 'process':
                executor = ProcessPoolExecutor()
            elif owned:
                error_msg = f"executor = {executor!r} must be 'serial', 'thread', 'process', or an executor."
                raise ValidationError(error_msg)
            try:
                results = list(executor.map(run_task, tasks))  # type: ignore[union-attr]
            finally:
                if owned:
                    executor.shutdown()  # type: ignore[union-attr]
        self.result_ = ExperimentResult(results)
        return self.result_
