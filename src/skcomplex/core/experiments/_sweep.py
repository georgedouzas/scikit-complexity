"""Run a system over values of its parameters and initial states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor

from ..components.specs import ScenarioSpec
from ..engine.runtime import Trajectory
from ..engine.solvers import BaseSolver
from ..specs import ObservableSpec
from ._designs import _get_rows
from ._results import ExperimentResult
from ._runs import _run_experiment
from .types import Bounds, Output, Source


def sweep(
    system: Source,
    parameters: Bounds | Sequence[Mapping[str, object]],
    *,
    design: str = 'grid',
    n: int = 64,
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
    """Return the runs of a system over rows of values.

    It is in the role of `validation_curve` of scikit-learn. A name without a dot is a model parameter, set on a copy of
    the system with `set_params`, or passed to the factory. A dotted path names an initial state or a parameter of a
    part, such as `population.x` or `growth.r`, which keeps its value for the whole run. The rows come from the design:

    - `'grid'`: every combination of a sequence of values per name, the last name varying fastest.
    - `'random'`: `n` independent samples, uniform within a pair `(low, high)` or drawn from a SciPy distribution.
    - `'latin_hypercube'`: `n` samples of a Latin hypercube, one in every stratum of every name.
    - `'sobol'`: `n` points of a scrambled Sobol sequence, best with `n` a power of 2.

    A sequence of rows, each a mapping from name to value, runs as it is.

    It appears in many fields. It maps the phase transition of a magnet over temperature, the final size of an epidemic
    over its rate of infection, and the clusters of an opinion model over its confidence bound.

    Args:
        system:
            The system, whose copies take model parameters through `set_params`, or a factory of systems, such as a
            system class, which takes them as keyword arguments.

        parameters:
            What the design reads by name, or the rows to run.

        design:
            The design of the rows, one of the designs above.

        n:
            The number of rows of a sampled design.

        replicates:
            The number of runs per row, each with its own random stream.

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
            The results of every run, one tidy row per run.

    Raises:
        ValidationError:
            The design is unknown, the values do not fit it, a name is no model parameter, state, or parameter, or a
            setting is not valid.

        ExperimentError:
            The system is a name, or the worker processes stop before finishing.

    Examples:
        >>> from skcomplex.core.experiments import sweep
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> result = sweep(LogisticGrowthSystem(), {'r': [0.5, 1.0]}, until=5, outputs=['population.x'])
        >>> len(result.rows)
        2
    """
    rows = _get_rows(parameters, design, n, random_state)
    return _run_experiment(
        system,
        rows,
        [ScenarioSpec()],
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
