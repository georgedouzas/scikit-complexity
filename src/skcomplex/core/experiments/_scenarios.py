"""Run the scenarios of a system with shared random streams.

They then differ only through what they change.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from concurrent.futures import Executor

from ..components.specs import ScenarioSpec
from ..engine.runtime import Trajectory
from ..engine.solvers import BaseSolver
from ..errors import ValidationError
from ..specs import ObservableSpec
from ._results import ExperimentResult
from ._runs import _run_experiment
from .types import Output, Source


def compare_scenarios(
    system: Source,
    scenarios: Sequence[ScenarioSpec],
    *,
    replicates: int = 1,
    analyses: Sequence[Callable[[Trajectory], object]] | None = None,
    until: float | None = None,
    dt: float | None = None,
    solver: BaseSolver | None = None,
    outputs: Sequence[Output] | None = None,
    warmup: float = 0.0,
    observables: Sequence[ObservableSpec] = (),
    random_state: int | None = None,
    executor: str | Executor = 'serial',
) -> ExperimentResult:
    """Return the runs of every scenario of a system.

    The scenarios share their random streams per replicate. A scenario overrides model parameters or dotted paths before
    the run, changes parameters at given times, or replaces the system for structural changes. Every scenario of a
    replicate runs with the same random stream, so the scenarios differ only through what they change.

    Args:
        system:
            The system, whose copies take model parameters through `set_params`, or a factory of systems, such as a
            system class, which takes them as keyword arguments.

        scenarios:
            The scenarios to compare.

        replicates:
            The number of runs per scenario.

        analyses:
            The analyses of every run, functions of a trajectory that return a frozen result, such as analysis
            functions bound with `bind_analysis`, or `None`. Every field of a result that is set becomes the output
            `<function name>.<field>`, numbered `-1`, `-2`, ... when several analyses share a function.

        until:
            The end time of every run, or `None` for the tuned end time of the system.

        dt:
            The time between frames, or `None` for the tuned one.

        solver:
            The solver, or `None` for the tuned one.

        outputs:
            The state and observable paths to keep as series, whose last values are scalar outputs, and functions of
            a trajectory that return a scalar or a mapping of scalars. `None` keeps every observable, or every state
            when the system declares none.

        warmup:
            The simulated time before which kept series are discarded.

        observables:
            The observables of every run, or none for the tuned ones.

        random_state:
            The seed of the random streams, one per row and replicate.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor, in the role of `n_jobs`.

    Returns:
        result:
            The results of every run, one tidy row per run with the name of its scenario.

    Raises:
        ValidationError:
            No scenario is given, a scenario is not a `ScenarioSpec`, or a setting is not valid.

        ExperimentError:
            The system is a name, or the worker processes stop before finishing.

    Examples:
        >>> from skcomplex.core.components.specs import ScenarioSpec
        >>> from skcomplex.core.experiments import compare_scenarios
        >>> from skcomplex.systems.biology.epidemiology import SIRSystem
        >>> scenarios = [ScenarioSpec(), ScenarioSpec('distancing', overrides={'beta': 0.0001})]
        >>> result = compare_scenarios(SIRSystem(), scenarios, until=50, outputs=['I.n'])
        >>> [row['scenario'] for row in result.rows]
        ['baseline', 'distancing']
    """
    if not scenarios or not all(isinstance(scenario, ScenarioSpec) for scenario in scenarios):
        error_msg = f'compare_scenarios takes a nonempty sequence of ScenarioSpec, not {scenarios!r}.'
        raise ValidationError(error_msg)
    return _run_experiment(
        system,
        [{}],
        list(scenarios),
        replicates,
        analyses=analyses,
        until=until,
        dt=dt,
        solver=solver,
        outputs=outputs,
        warmup=warmup,
        observables=observables,
        random_state=random_state,
        executor=executor,
    )
