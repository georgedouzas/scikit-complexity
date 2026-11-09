"""Run a system over rows of values, scenarios, and replicates.

There is one random stream per row and replicate.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import sys
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, MutableMapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import fields, is_dataclass

import numpy as np
import numpy.typing as npt
from sklearn.base import clone

from ..components.specs import ScenarioSpec
from ..components.systems import BaseSystem
from ..engine.compile._pass_compiler import PassCompiler
from ..engine.problems import BaseProblemSpec
from ..engine.runtime import Trajectory
from ..engine.runtime._simulate import _compile_problem, _simulate
from ..engine.solvers import BaseSolver
from ..errors import ExperimentError, ValidationError
from ..specs import ObservableSpec
from ..specs._run_result_spec import _RunResultSpec
from ._results import ExperimentResult
from .specs._task_spec import _TaskSpec
from .types import Output, Row, Source

EXECUTORS = ('serial', 'thread', 'process')
TIME_TOLERANCE = 1e-12

type Problems = MutableMapping[Hashable, tuple[BaseProblemSpec, BaseSolver]]


def _check_source(source: object) -> Source:
    """Return the system of an experiment after checking it.

    It is a system or a factory of systems.

    Args:
        source:
            The system or the factory to check.

    Returns:
        source:
            The system, or the factory of systems.

    Raises:
        ExperimentError:
            The source is a system name, or neither a system nor a factory.
    """
    if isinstance(source, str):
        error_msg = (
            f'An experiment takes a system or a factory of systems, not the system name {source!r}. Pass an instance '
            'of the system class, such as `SIRSystem()`, or the class itself.'
        )
        raise ExperimentError(error_msg)
    if isinstance(source, BaseSystem) or callable(source):
        return source
    error_msg = f'An experiment takes a system or a factory of systems, not {source!r}.'
    raise ExperimentError(error_msg)


def _check_count(name: str, value: object) -> None:
    """Raise an error for a count of an experiment that is not a positive integer.

    An example of a count is `replicates`.

    Args:
        name:
            The name of the setting.

        value:
            The value of the setting.

    Raises:
        ValidationError:
            The value is not a positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        error_msg = f'{name} = {value!r} must be a positive integer.'
        raise ValidationError(error_msg)


def _build(source: Source, arguments: Mapping[str, object]) -> BaseSystem:
    """Return the system of a source with model parameters set.

    It is a copy of a system, or what a factory builds.

    Args:
        source:
            The system, whose copy takes the arguments through `set_params`, or the factory, which takes them as
            keyword arguments.

        arguments:
            The model parameters by name.

    Returns:
        system:
            The system with the model parameters.

    Raises:
        ValidationError:
            A model parameter is not one of the system.

        ExperimentError:
            The factory returns no system.
    """
    if isinstance(source, BaseSystem):
        unknown = sorted(set(arguments) - set(source.get_params(deep=False)))
        if unknown:
            names = ', '.join(sorted(source.get_params(deep=False)))
            error_msg = (
                f'{", ".join(unknown)} is not a model parameter of {type(source).__name__}. Its model parameters are '
                f'{names}, and dotted paths such as `growth.r` name the states and parameters of its parts.'
            )
            raise ValidationError(error_msg)
        return clone(source).set_params(**arguments)
    system = source(**arguments)
    if not isinstance(system, BaseSystem):
        error_msg = f'The factory of an experiment returns a system, not {type(system).__name__}.'
        raise ExperimentError(error_msg)
    return system


def _name_analyses(
    analyses: Sequence[Callable[[Trajectory], object]] | None,
) -> tuple[tuple[str, Callable[[Trajectory], object]], ...]:
    """Return every analysis with its column prefix.

    The prefix is the name of its function, numbered when the function repeats.

    Args:
        analyses:
            The analyses, functions of a trajectory or bound analyses, or `None`.

    Returns:
        named:
            The pairs of prefix and analysis.

    Raises:
        ValidationError:
            An analysis is not callable.
    """
    for analysis in analyses or ():
        if not callable(analysis):
            error_msg = f'An analysis of an experiment is a function of a trajectory, not {analysis!r}.'
            raise ValidationError(error_msg)
    functions = [getattr(analysis, 'func', analysis) for analysis in analyses or ()]
    names = [getattr(function, '__name__', 'analysis') for function in functions]
    counts, seen = Counter(names), Counter[str]()
    prefixes = []
    for name in names:
        seen[name] += 1
        prefixes.append(f'{name}-{seen[name]}' if counts[name] > 1 else name)
    return tuple(zip(prefixes, analyses or (), strict=True))


def _compile(system: BaseSystem) -> BaseProblemSpec:
    """Return the compiled problem of a system.

    Its layouts name the states and parameters of the system.

    Args:
        system:
            The system.

    Returns:
        problem:
            The compiled problem.
    """
    compiler = system.get_run_settings()['compiler'] or PassCompiler()
    problem: BaseProblemSpec = clone(compiler).prepare(system.prepare().system_).run().problem_
    return problem


def _split_paths(problem: BaseProblemSpec, paths: Mapping[str, object]) -> tuple[Row, Row]:
    """Return the values of dotted paths split into initial states and parameters.

    They are checked against their domains.

    Args:
        problem:
            The compiled problem of the system.

        paths:
            The values by dotted path.

    Returns:
        states:
            The initial values by dotted state path.

        parameters:
            The values by dotted parameter path.

    Raises:
        ValidationError:
            A path is neither a state nor a parameter, or a value is not in its domain.
    """
    states: Row = {}
    parameters: Row = {}
    for name, value in paths.items():
        layout, kept = (problem.states, states) if problem.states.has(name) else (problem.parameters, parameters)
        if not layout.has(name):
            names = ', '.join([*problem.states.names, *problem.parameters.names])
            error_msg = f'{name} is neither a state nor a parameter of the system. Available paths are: {names}.'
            raise ValidationError(error_msg)
        if not layout.entry(name).contains(value):
            error_msg = f'{name} = {value!r} is not in its domain {layout.entry(name).domain}.'
            raise ValidationError(error_msg)
        kept[name] = value
    return states, parameters


def _step_function(
    initial: npt.ArrayLike,
    changes: Sequence[tuple[float, npt.ArrayLike]],
) -> Callable[[float], npt.ArrayLike]:
    """Return the step function of time of a series of changes.

    It takes the value of the last change at or before a time, else the initial value.

    Args:
        initial:
            The value before the first change.

        changes:
            The pairs of time and value, sorted by time.

    Returns:
        function:
            The step function of time.
    """

    def value_at(time: float) -> npt.ArrayLike:
        due = [value for start, value in changes if start <= time + TIME_TOLERANCE]
        return due[-1] if due else initial

    return value_at


def _schedule(
    problem: BaseProblemSpec,
    scenario: ScenarioSpec,
    parameters: Mapping[str, object],
) -> dict[str, Callable[[float], npt.ArrayLike]]:
    """Return every parameter a run fixes or a scenario changes as a function of time.

    `simulate` reads these functions. A parameter given by dotted path keeps its value for the whole run. Before its
    first change a scheduled parameter keeps its value, and every change holds from the first frame at or after its
    time.

    Args:
        problem:
            The compiled problem of the system.

        scenario:
            The scenario.

        parameters:
            The values of the run by dotted parameter path.

    Returns:
        schedule:
            The function of time of every parameter by dotted path.
    """
    functions = {path: _step_function(np.asarray(value), []) for path, value in parameters.items()}
    ordered = sorted(scenario.schedule, key=lambda change: change[0])
    for path in dict.fromkeys(name for _, name, _ in ordered):
        if path in parameters:
            initial = np.asarray(parameters[path])
        else:
            layout, vector = (
                (problem.states, problem.initial_state)
                if path in problem.adaptive_parameters
                else (problem.parameters, problem.initial_parameters)
            )
            try:
                entry, indices = layout.locate(path)
                initial = entry.decode(vector[indices]).reshape(entry.shape)
            except KeyError:
                initial = np.full((), np.nan)
        changes = [(time, np.asarray(value)) for time, name, value in ordered if name == path]
        functions[path] = _step_function(initial, changes)
    return functions


def _read_outputs(
    trajectory: Trajectory,
    outputs: Sequence[Output] | None,
    keep: npt.NDArray[np.bool_],
) -> tuple[dict[str, npt.NDArray[np.generic]], dict[str, object]]:
    """Return the kept series and the scalar outputs of a run.

    Args:
        trajectory:
            The trajectory of the run.

        outputs:
            The state and observable paths and the functions of a trajectory to keep, or `None` for every observable,
            or every state when the system declares none.

        keep:
            Whether every frame is kept, after the warmup.

    Returns:
        series:
            The kept series by path.

        scalars:
            The last value of every scalar series and the values of the functions, by name.
    """
    if outputs is None:
        problem = trajectory.problem
        outputs = [
            f'{scope}/{observable.label}' if scope else observable.label for scope, observable in problem.observables
        ] or list(problem.states.names)
    series, scalars = {}, {}
    for output in outputs:
        if callable(output):
            name, value = getattr(output, '__name__', 'output'), output(trajectory)
            if isinstance(value, Mapping):
                scalars.update({f'{name}.{key}': item for key, item in value.items()})
            else:
                scalars[name] = value
            continue
        array = np.asarray(trajectory.get_values(output))
        series[output] = array[keep]
        if array.ndim == 1 and array.size:
            scalars[output] = array[-1]
    return series, scalars


def _get_structure(task: _TaskSpec, arguments: Mapping[str, object]) -> tuple[int, str]:
    """Return the key of the structure of the system of a task.

    The key holds its source or scenario system, and model parameters. Runs that share the key differ only by dotted
    paths of initial states and parameters, and by seeds, so they share one compiled problem. Arrays are written in
    full, so two arrays that differ never share a key.

    Args:
        task:
            The task.

        arguments:
            The model parameters of the task by name.

    Returns:
        key:
            The identity of the source or the scenario system, and the model parameters written out.
    """
    source = task.scenario.system if task.scenario.system is not None else task.source
    with np.printoptions(threshold=sys.maxsize):
        return id(source), repr(sorted(arguments.items()))


def _run_task(task: _TaskSpec, problems: Problems | None = None) -> _RunResultSpec:
    """Run one task as `simulate` does and keep its outputs.

    It keeps its series and scalar outputs, and runs its analyses on its trajectory. Values without a dot are model
    parameters, set on a copy of the system or passed to the factory. Dotted paths of states replace initial values, and
    dotted paths of parameters keep their value for the whole run. The system is compiled once per structure of the
    running experiment, its source and model parameters, and the runs of one structure reuse its problem with their own
    values and seeds. A system that brings its own compiler is compiled for every run.

    Args:
        task:
            The task.

        problems:
            The compiled problems of the running experiment by structure, filled as tasks compile them, or `None` to
            compile for this run alone.

    Returns:
        result:
            The result of the run.

    Raises:
        ExperimentError:
            An analysis returns no result of an analysis.
    """
    values = {**task.row, **(task.scenario.overrides or {})}
    arguments = {name: value for name, value in values.items() if '.' not in name}
    paths = {name: value for name, value in values.items() if '.' in name}
    system = _build(task.scenario.system if task.scenario.system is not None else task.source, arguments)
    settings = system.get_run_settings()
    compiled = None
    if settings['compiler'] is None:
        store = {} if problems is None else problems
        key = _get_structure(task, arguments)
        if key not in store:
            store[key] = _compile_problem(system, solver=task.solver, observables=task.observables)
        compiled = store[key]
    initial_state, schedule = None, None
    if paths or task.scenario.schedule:
        problem = compiled[0] if compiled is not None else _compile(system)
        states, parameters = _split_paths(problem, paths)
        functions = _schedule(problem, task.scenario, parameters)
        initial_state = {**(settings['initial_state'] or {}), **states} if states else None
        schedule = {**(settings['schedule'] or {}), **functions} if functions else None
    trajectory = _simulate(
        system,
        compiled,
        until=task.until,
        dt=task.dt,
        solver=task.solver,
        observables=task.observables,
        initial_state=initial_state,
        schedule=schedule,
        random_state=np.random.default_rng(task.seed),
    )
    keep = trajectory.times >= task.warmup - TIME_TOLERANCE
    series, scalars = _read_outputs(trajectory, task.outputs, keep)
    for prefix, analysis in task.analyses:
        result = analysis(trajectory)
        if not is_dataclass(result):
            error_msg = f'The analysis {prefix} returned {type(result).__name__}, not the result of an analysis.'
            raise ExperimentError(error_msg)
        for field in fields(result):
            value = getattr(result, field.name)
            if value is not None:
                scalars[f'{prefix}.{field.name}'] = value
    return _RunResultSpec(
        index=task.index,
        row=dict(task.row),
        scenario=task.scenario.name,
        replicate=task.replicate,
        seed=int(task.seed.spawn_key[-1]) if task.seed.spawn_key else 0,
        times=trajectory.times[keep],
        series=series,
        scalars=scalars,
    )


def _run_experiment(
    source: object,
    rows: Sequence[Row],
    scenarios: Sequence[ScenarioSpec],
    replicates: int,
    *,
    analyses: Sequence[Callable[[Trajectory], object]] | None,
    until: float | None,
    dt: float | None,
    solver: BaseSolver | None,
    outputs: Sequence[Output] | None,
    warmup: float,
    observables: Sequence[ObservableSpec],
    random_state: int | None,
    executor: str | Executor,
) -> ExperimentResult:
    """Run a system once per row, scenario, and replicate.

    There is one random stream per row and replicate. The streams are spawned from `random_state` in a fixed order, so
    the results do not depend on the executor, and every scenario of a row and replicate shares its stream. The compiled
    problems live for this call alone: serial and thread runs share them, and every worker process compiles its own.

    Args:
        source:
            The system, or the factory of systems.

        rows:
            The values of every row, by model parameter or dotted path.

        scenarios:
            The scenarios.

        replicates:
            The number of runs per row and scenario.

        analyses:
            The analyses of every run, or `None`.

        until:
            The end time of every run, or `None` for the tuned one.

        dt:
            The time between frames, or `None` for the tuned one.

        solver:
            The solver, or `None` for the tuned one.

        outputs:
            The outputs to keep, or `None` for every observable or state.

        warmup:
            The simulated time before which kept series are discarded.

        observables:
            The observables of every run, or none for the tuned ones.

        random_state:
            The seed of the random streams.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor.

    Returns:
        result:
            The results of every run.

    Raises:
        ValidationError:
            The number of replicates is not a positive integer, or the executor is unknown.

        ExperimentError:
            The worker processes stop before finishing their tasks.
    """
    system = _check_source(source)
    _check_count('replicates', replicates)
    if isinstance(executor, str) and executor not in EXECUTORS:
        error_msg = f"executor = {executor!r} must be 'serial', 'thread', 'process', or an executor."
        raise ValidationError(error_msg)
    named = _name_analyses(analyses)
    seeds = np.random.SeedSequence(random_state).spawn(len(rows) * replicates)
    tasks: list[_TaskSpec] = []
    for row_index, row in enumerate(rows):
        for scenario in scenarios:
            for replicate in range(replicates):
                seed = seeds[row_index * replicates + replicate]
                task = _TaskSpec(
                    len(tasks),
                    system,
                    dict(row),
                    scenario,
                    replicate,
                    seed,
                    until,
                    dt,
                    solver,
                    outputs,
                    warmup,
                    tuple(observables),
                    named,
                )
                tasks.append(task)
    problems: Problems = {}
    if executor == 'serial':
        return ExperimentResult(tuple(_run_task(task, problems) for task in tasks))
    if isinstance(executor, Executor):
        pool = executor
    else:
        pool = ThreadPoolExecutor() if executor == 'thread' else ProcessPoolExecutor()
    try:
        results = tuple(pool.map(functools.partial(_run_task, problems=problems), tasks))
    except BrokenProcessPool as error:
        error_msg = (
            'The worker processes of the experiment stopped before finishing their tasks. On macOS and Windows every '
            'worker imports the main script again, so a script that runs an experiment on processes must guard it '
            "with `if __name__ == '__main__':`. Use executor='thread' or executor='serial' to run without processes."
        )
        raise ExperimentError(error_msg) from error
    finally:
        if isinstance(executor, str):
            pool.shutdown()
    return ExperimentResult(results)
