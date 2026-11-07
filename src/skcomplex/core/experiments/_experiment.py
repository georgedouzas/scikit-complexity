"""Run a system over a design, scenarios, and replicates with independent random streams, and analyse every run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import fields, is_dataclass

import numpy as np
import numpy.typing as npt

from ..components.specs import ScenarioSpec
from ..engine.problems import BaseProblemSpec
from ..engine.runtime import Simulation, Trajectory
from ..engine.solvers import BaseSolver
from ..errors import ExperimentError, ValidationError
from ..specs import ObservableSpec
from ..specs._run_result_spec import _RunResultSpec
from ._base import BaseDesign, BaseExperiment, Source
from ._results import ExperimentResult
from .specs._task_spec import Output, _TaskSpec
from .types import Bounds, Row

EXECUTORS = ('serial', 'thread', 'process')


class Experiment(BaseExperiment):
    """Runs of a system over a design, scenarios, and replicates, with one random stream per design row and replicate.

    The streams are spawned from `random_state` in a fixed order, so the results do not depend on the executor, and
    every scenario of a design row and replicate shares its stream.

    Every run applies every analysis to the run's trajectory and keeps the fields of its result that are set as
    outputs named `<prefix>.<field>`. An analysis is a function of a trajectory, such as one that `bind_analysis`
    returns. The prefix is the name of the analysis function, such as `entropy`, numbered `-1`, `-2`, ... when several
    analyses share a function, as scikit-learn names the steps of `make_pipeline`.

    Args:
        source:
            A factory of systems, or a system, such as one that `skcomplex.systems.create_system` returns.

        design:
            The design of factory arguments and dotted paths, or `None` for one run of the defaults.

        bounds:
            What the design prepares on: a pair `(low, high)` or a SciPy distribution per name for a sampled design, a
            sequence of values per name for a grid design, or `None`.

        scenarios:
            The scenarios to compare, or `None` for the baseline alone.

        replicates:
            The number of runs per design row and scenario.

        analyses:
            The analyses of every run, functions of a trajectory that return a frozen result, or `None`.

        until:
            The simulated time of every run, or `None` for 10.

        dt:
            The time between frames, or `None` for the default of the simulation.

        solver:
            The solver, or `None` for the default of the simulation.

        outputs:
            The state and observable paths to keep as series, and functions of a trajectory that return a scalar or a
            mapping of scalars. `None` keeps every observable, or every state when the system declares none.

        warmup:
            The simulated time before which kept series are discarded.

        random_state:
            The seed of the random streams.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor.

        observables:
            The observables of every run, or `None` for none.

    Attributes:
        tasks_:
            The runs of the experiment in their fixed order, set by `prepare`.

        result_:
            The results of every run, set by `run`.

    Examples:
        >>> from skcomplex.core.experiments import Experiment, GridDesign
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> bounds = {'r': [0.5, 1.0]}
        >>> experiment = Experiment(LogisticGrowthSystem, GridDesign(), bounds, until=5, outputs=['population.n'])
        >>> len(experiment.prepare().run().result_.rows)
        2
    """

    def __init__(
        self,
        source: Source,
        design: BaseDesign | None = None,
        bounds: Bounds | None = None,
        scenarios: Sequence[ScenarioSpec] | None = None,
        replicates: int = 1,
        *,
        analyses: Sequence[Callable[[Trajectory], object]] | None = None,
        until: float | None = None,
        dt: float | None = None,
        solver: BaseSolver | None = None,
        outputs: Sequence[Output] | None = None,
        warmup: float = 0.0,
        random_state: int | None = None,
        executor: str | Executor = 'serial',
        observables: Sequence[ObservableSpec] | None = None,
    ) -> None:
        self.source = source
        self.design = design
        self.bounds = bounds
        self.scenarios = scenarios
        self.replicates = replicates
        self.analyses = analyses
        self.until = until
        self.dt = dt
        self.solver = solver
        self.outputs = outputs
        self.warmup = warmup
        self.random_state = random_state
        self.executor = executor
        self.observables = observables

    def _check(self) -> None:
        """Raise an error for a system name as the source, bad replicates, executor, bounds, or analysis."""
        self._check_source(self.source)
        if isinstance(self.replicates, bool) or not isinstance(self.replicates, int) or self.replicates < 1:
            error_msg = f'replicates = {self.replicates!r} must be a positive integer.'
            raise ValidationError(error_msg)
        if isinstance(self.executor, str) and self.executor not in EXECUTORS:
            error_msg = f"executor = {self.executor!r} must be 'serial', 'thread', 'process', or an executor."
            raise ValidationError(error_msg)
        if self.design is None and self.bounds:
            error_msg = f'The bounds of {sorted(self.bounds)} need a design that prepares on them.'
            raise ValidationError(error_msg)
        if self.design is not None:
            self.design._validated()
        for analysis in self.analyses or ():
            if not callable(analysis):
                error_msg = f'An analysis of an experiment is a function of a trajectory, not {analysis!r}.'
                raise ValidationError(error_msg)

    def _name_analyses(self) -> tuple[tuple[str, Callable[[Trajectory], object]], ...]:
        """Return every analysis with its column prefix, the name of its function numbered when the function repeats."""
        functions = [getattr(analysis, 'func', analysis) for analysis in self.analyses or ()]
        names = [getattr(function, '__name__', 'analysis') for function in functions]
        counts, seen = Counter(names), Counter[str]()
        prefixes = []
        for name in names:
            seen[name] += 1
            prefixes.append(f'{name}-{seen[name]}' if counts[name] > 1 else name)
        return tuple(zip(prefixes, self.analyses or (), strict=True))

    def _prepare(self, context: None = None) -> None:
        """Resolve the source and set the runs of the experiment, compiling every problem once before serial runs."""
        source = self._check_source(self.source)
        observables = list(self.observables) if self.observables is not None else None
        until = float(self.until if self.until is not None else 10.0)
        rows: list[Row] = [{}]
        if self.design is not None:
            rows = [dict(row) for row in self.design.prepare(self.bounds).rows_]
        scenarios = list(self.scenarios) if self.scenarios else [ScenarioSpec()]
        analyses = self._name_analyses()
        seeds = np.random.SeedSequence(self.random_state).spawn(len(rows) * self.replicates)
        self.tasks_: list[_TaskSpec] = []
        for row_index, row in enumerate(rows):
            for scenario in scenarios:
                for replicate in range(self.replicates):
                    seed = seeds[row_index * self.replicates + replicate]
                    task = _TaskSpec(
                        len(self.tasks_),
                        source,
                        row,
                        scenario,
                        replicate,
                        seed,
                        until,
                        self.dt,
                        self.solver,
                        self.outputs,
                        self.warmup,
                        observables,
                        analyses,
                    )
                    self.tasks_.append(task)
        if self.executor in {'serial', 'thread'}:
            for task in self.tasks_:
                arguments, _ = self._split_overrides(task)
                self._compile(task.source, arguments, task.scenario.system, task.observables)

    @staticmethod
    def _split_overrides(task: _TaskSpec) -> tuple[dict[str, object], dict[str, object]]:
        """Return the design values and scenario overrides of a task as factory arguments and dotted paths."""
        values = {**task.row, **(task.scenario.overrides or {})}
        arguments = {name: value for name, value in values.items() if '.' not in name}
        paths = {name: value for name, value in values.items() if '.' in name}
        return arguments, paths

    @staticmethod
    def _schedule(problem: BaseProblemSpec, scenario: ScenarioSpec) -> dict[str, Callable[[float], npt.ArrayLike]]:
        """Return every scheduled parameter of a scenario as a step function of time, which a simulation reads.

        Before its first change a parameter keeps its initial value in the problem, and every change holds from the
        first frame at or after its time.
        """
        ordered = sorted(scenario.schedule, key=lambda change: change[0])
        functions = {}
        for path in dict.fromkeys(name for _, name, _ in ordered):
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
            functions[path] = Experiment._step_function(initial, changes)
        return functions

    @staticmethod
    def _step_function(
        initial: npt.ArrayLike,
        changes: Sequence[tuple[float, npt.ArrayLike]],
    ) -> Callable[[float], npt.ArrayLike]:
        """Return the function of time that takes the value of the last change at or before a time, else the initial."""

        def value_at(time: float) -> npt.ArrayLike:
            due = [value for start, value in changes if start <= time + 1e-12]
            return due[-1] if due else initial

        return value_at

    @staticmethod
    def _run_task(task: _TaskSpec) -> _RunResultSpec:
        """Run one task, keep its series and scalar outputs, and run its analyses on its trajectory."""
        arguments, paths = Experiment._split_overrides(task)
        problem = Experiment._compile(task.source, arguments, task.scenario.system, task.observables)
        problem = Experiment._with_values(problem, paths)
        simulation = Simulation(
            problem,
            random_state=np.random.default_rng(task.seed),
            schedule=Experiment._schedule(problem, task.scenario),
            dt=task.dt,
            solver=task.solver,
        ).prepare()
        trajectory = simulation.run(until=task.until).trajectory_
        keep = trajectory.times >= task.warmup - 1e-12
        if task.outputs is not None:
            outputs = list(task.outputs)
        else:
            outputs = [
                f'{scope}/{observable.label}' if scope else observable.label
                for scope, observable in problem.observables
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
            values = np.asarray(trajectory.values(output))
            series[output] = values[keep]
            if values.ndim == 1 and values.size:
                scalars[output] = values[-1]
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

    def _get_executor(self) -> Executor:
        """Return the executor of the runs, a new thread or process pool for its name."""
        if isinstance(self.executor, Executor):
            return self.executor
        return ThreadPoolExecutor() if self.executor == 'thread' else ProcessPoolExecutor()

    def _run(self, inputs: None = None) -> None:
        """Run every task and keep the results in `result_`."""
        if self.executor == 'serial':
            results = [self._run_task(task) for task in self.tasks_]
        else:
            executor = self._get_executor()
            owned = isinstance(self.executor, str)
            try:
                results = list(executor.map(self._run_task, self.tasks_))
            except BrokenProcessPool as error:
                error_msg = (
                    'The worker processes of the experiment stopped before finishing their tasks. On macOS and '
                    'Windows every worker imports the main script again, so a script that runs an experiment on '
                    "processes must guard it with `if __name__ == '__main__':`. Use executor='thread' or "
                    "executor='serial' to run without processes."
                )
                raise ExperimentError(error_msg) from error
            finally:
                if owned:
                    executor.shutdown()
        self.result_ = ExperimentResult(results)
