"""Run a model over designs, scenarios, and replicates with independent random streams."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
from sklearn.base import BaseEstimator

from ...core.base import ExperimentError, ValidationError
from ...core.system import Observable
from ._designs import Design
from ._results import ExperimentResult
from ._scenarios import Scenario
from ._sources import Source, _problem, resolve_source, split_overrides
from ._tasks import Output, Task, run_task

RUN_SETTINGS = ('dt', 'solver', 'backend', 'checked')


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

        observables:
            The observables of every run, or `None` for the standard observables of a registered model.

    Examples:
        >>> from skcomplex.engine.experiments import Experiment, Grid
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
        observables: Sequence[Observable] | None = None,
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
        self.observables = observables

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
        named = self.source if isinstance(self.source, str) else None
        observables = list(self.observables) if self.observables is not None else named
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
                        observables,
                    )
                    tasks.append(task)
        return tasks

    def run(self) -> ExperimentResult:
        """Run every task and return the results.

        Returns:
            result:
                The results, also stored as `result_`.

        Raises:
            ExperimentError:
                The worker processes of a process executor died, as when a script without a main guard runs the
                experiment.
        """
        tasks = self.tasks()
        if self.executor in {'serial', 'thread'}:
            for task in tasks:
                arguments, _ = split_overrides({**task.row, **(task.scenario.overrides or {})})
                _problem(task.source, arguments, task.scenario.system, task.observables)
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
                    executor.shutdown()  # type: ignore[union-attr]
        self.result_ = ExperimentResult(results)
        return self.result_
