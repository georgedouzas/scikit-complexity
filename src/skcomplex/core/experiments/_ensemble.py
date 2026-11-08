"""Run a system many times with independent random streams."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from concurrent.futures import Executor

from ..components.specs import ScenarioSpec
from ..engine.runtime import Trajectory
from ..engine.solvers import BaseSolver
from ..errors import TagError
from ..specs import ObservableSpec
from ._results import ExperimentResult
from ._runs import _build, _check_source, _run_experiment
from .types import Output, Source


def ensemble(
    system: Source,
    *,
    replicates: int = 10,
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
    """Return the runs of a system with independent random streams, in the role of repeated cross-validation.

    The streams are spawned from `random_state`, so the runs of a stochastic system are independent and reproducible,
    and the result reads them as an ensemble, one row per run and one column per common time. The system needs the
    randomness `stochastic`, or a solver that draws, such as `KineticSolver`, since the runs of a deterministic system
    by a deterministic solver repeat one another.

    Args:
        system:
            The system, whose copies take model parameters through `set_params`, or a factory of systems, such as a
            system class, which takes them as keyword arguments.

        replicates:
            The number of runs.

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
            The results of every run, one tidy row per run, whose `ensemble` and `summary` read the runs together.

    Raises:
        ValidationError:
            The number of runs is not a positive integer, or a setting is not valid.

        ExperimentError:
            The system is a name, or the worker processes stop before finishing.

        TagError:
            The system has the randomness `deterministic` and the solver does not draw.

    Examples:
        >>> from skcomplex.core.experiments import ensemble
        >>> from skcomplex.systems.biology.epidemiology import SIRSystem
        >>> result = ensemble(SIRSystem(stochastic=True), replicates=3, until=10, random_state=0)
        >>> result.column('seed').tolist()
        [0, 1, 2]

        A deterministic system has nothing to repeat:

        >>> ensemble(SIRSystem(), replicates=3, until=10)
        Traceback (most recent call last):
        ...
        skcomplex.core.errors.TagError: ensemble needs the randomness stochastic, but SIRSystem has the randomness ...
    """
    built = _build(_check_source(system), {})
    stepper = solver if solver is not None else built.get_run_settings()['solver']
    if built.get_tags().randomness == 'deterministic' and not getattr(stepper, 'draws', False):
        error_msg = (
            f'ensemble needs the randomness stochastic, but {type(built).__name__} has the randomness deterministic. '
            'Its runs repeat one another, so run it once with `simulate`, or give it a part or a solver that draws.'
        )
        raise TagError(error_msg)
    return _run_experiment(
        system,
        [{}],
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
