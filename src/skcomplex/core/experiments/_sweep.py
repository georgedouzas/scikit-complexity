"""Run a system over values of its parameters and initial states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor

from ..components import Observable
from ..engine.runtime import Trajectory
from ..engine.solvers import BaseSolver
from ._designs import _get_rows
from ._results import ExperimentResult
from ._runs import _run_experiment
from ._scenario_spec import ScenarioSpec
from .types import Bounds, Output, SystemSource


def sweep(
    system: SystemSource,
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
    observables: Sequence[Observable] = (),
    random_state: int | None = None,
    executor: str | Executor = 'serial',
) -> ExperimentResult:
    """Return the runs of a system over rows of values.

    It runs a copy of the system for every row of values, as `validation_curve` of scikit-learn does, such as
    `sweep(system, {'r': [0.5, 1.0]})` for two growth rates. A name without a dot, such as `r` or the nested
    `growth__r`, is set with `set_params` and the copy is prepared again, and a dotted path, such as `population.x`,
    sets a value of a component on one prepared copy. The rows come from a grid or a random design, and the system
    passed in is left unchanged. It maps the change of state of a magnet over temperature, the final size of an epidemic
    over its rate of infection, and the clusters of opinions over the confidence of the voters.

    Read more in the user guide, under Run experiments.

    Args:
        system:
            The system, whose clones take parameters through `set_params`, or a factory of systems, such as a
            system class, which takes them as keyword arguments.

        parameters:
            What the design reads by name, or the rows to run.

        design:
            The design of the rows. `'grid'` runs every combination of a sequence of values per name, the last name
            varying fastest. `'random'` draws `n` samples, uniform within a pair `(low, high)` or from a SciPy
            distribution. `'latin_hypercube'` draws `n` samples with one in every stratum of every name. `'sobol'`
            takes `n` points of a scrambled Sobol sequence, best with `n` a power of 2. A sequence of rows, each a
            mapping from name to value, runs as it is.

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
            The design is unknown, the values do not fit it, a name is no parameter of the system and no state or
            parameter of a component, or a setting is not valid.

        ExperimentError:
            The system is a name, or the worker processes stop before finishing.

    See Also:
        ensemble: Runs a system many times with independent random streams.
        sensitivity: Ranks the inputs of a system by their effect on an output.

    Examples:
        >>> from skcomplex.core.experiments import sweep
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> result = sweep(LogisticGrowthSystem(), {'r': [0.5, 1.0]}, until=5, outputs=['population.x'])
        >>> result.get_column('r').tolist()
        [0.5, 1.0]
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
