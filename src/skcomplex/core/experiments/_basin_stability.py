"""Estimate the share of random initial states that end in each final behaviour."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass

import numpy as np

from ..engine.problems import ProblemSpec
from ..engine.runtime import Trajectory
from ..enums import DomainKind
from ..errors import ValidationError
from ._designs import _get_rows
from ._replicate import _run_replicated
from ._runs import _check_count, _check_source, _run_experiment
from ._scenario_spec import ScenarioSpec
from ._utils import build_system, get_problem
from .types import Row, SystemSource

INTEGER_KINDS = (DomainKind.INTEGER, DomainKind.BOOLEAN)


@dataclass(frozen=True)
class BasinStabilityResult:
    """A result of `basin_stability`, the share of initial states per final behaviour.

    It holds the random initial states, the final behaviour each one ended in, and the share of the states that reached
    each behaviour with its standard error, such as `{'capacity': 1.0}` when every population grows to its capacity. It
    reports how often a power grid stays in step, how often a species survives, and how often a genetic switch ends on.

    Read more in the user guide, under Run experiments.

    Args:
        rows:
            The initial states, one mapping from dotted path to value per run.

        labels:
            The attractor label of every initial state.

        fractions:
            The fraction of initial states per attractor label.

        errors:
            The standard error of every fraction.

    See Also:
        basin_stability: Returns this result.

    Examples:
        >>> from skcomplex.core.experiments import BasinStabilityResult
        >>> BasinStabilityResult([], [], {'capacity': 1.0}, {'capacity': 0.0}).fractions
        {'capacity': 1.0}
    """

    rows: list[Row]
    labels: list[object]
    fractions: dict[object, float]
    errors: dict[object, float]


def basin_stability(
    system: SystemSource,
    bounds: Mapping[str, tuple[float, float]],
    classifier: Callable[[Trajectory], object],
    *,
    n: int = 500,
    until: float | None = None,
    dt: float | None = None,
    random_state: int | None = None,
    executor: str | Executor = 'serial',
    vectorized: bool = False,
) -> BasinStabilityResult:
    """Return the share of random initial states that end in each final behaviour.

    It draws initial states at random within `bounds`, runs each one to `until`, and asks `classifier` which final
    behaviour the run settled into, such as a resting state or a cycle. The share of each behaviour says how well it
    withstands large disturbances. With `vectorized`, every initial state runs as one member of a single simulation. It
    measures how often a power grid stays in step after a fault, how often a population survives, and how often a
    genetic switch ends on rather than off.

    Read more in the user guide, under Run experiments.

    Args:
        system:
            The system, or a factory of systems, such as a system class.

        bounds:
            A mapping from dotted state path to the `(low, high)` bounds of its initial value.

        classifier:
            A function of a trajectory that returns the label of the attractor it ends in.

        n:
            The number of initial states.

        until:
            The end time of every run, or `None` for the tuned end time of the system.

        dt:
            The time between frames, or `None` for the tuned one.

        random_state:
            The seed of the initial states and the runs.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor of the separate runs.

        vectorized:
            Whether to run every initial state as a member of one simulation of the replicated system.

    Returns:
        result:
            The initial states, their attractor labels, and the fraction and its error per attractor.

    Raises:
        ValidationError:
            No bounds are given, the number of initial states is not a positive integer, or a vectorised run does not
            compile to a general problem.

        ExperimentError:
            The system is a name.

    See Also:
        orbit_diagram: Shows the values a run visits over a range of a parameter.
        ensemble: Runs a system many times with independent random streams.

    Examples:
        >>> from skcomplex.core.experiments import basin_stability
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> bounds = {'population.x': (0.1, 10)}
        >>> basin_stability(LogisticGrowthSystem(), bounds, lambda run: 'capacity', n=5, until=5).fractions
        {'capacity': 1.0}
    """
    _check_source(system)
    if not bounds:
        error_msg = 'Basin stability needs the bounds of at least one initial state.'
        raise ValidationError(error_msg)
    _check_count('n', n)
    states = get_problem(build_system(system, {})).states
    integers = [path for path in bounds if states.has(path) and states.get_entry(path).kind in INTEGER_KINDS]
    rows = _get_rows(bounds, 'random', n, random_state, integers)
    if vectorized:
        values = {path: np.asarray([row[path] for row in rows], dtype=float) for path in bounds}
        trajectory, problem = _run_replicated(system, values, until, dt, random_state)
        if not isinstance(problem, ProblemSpec):
            error_msg = f'A vectorised basin reads the members of a general problem, not of a {type(problem).__name__}.'
            raise ValidationError(error_msg)
        labels = [classifier(trajectory.get_member(index, problem)) for index in range(n)]
    else:

        def attractor(trajectory: Trajectory) -> object:
            return classifier(trajectory)

        result = _run_experiment(
            system,
            rows,
            [ScenarioSpec()],
            1,
            analyses=None,
            until=until,
            dt=dt,
            solver=None,
            outputs=[attractor],
            warmup=0.0,
            observables=(),
            random_state=random_state,
            executor=executor,
        )
        labels = [row['attractor'] for row in result.rows]
    fractions = {label: count / n for label, count in Counter(labels).items()}
    errors = {label: float(np.sqrt(fraction * (1 - fraction) / n)) for label, fraction in fractions.items()}
    return BasinStabilityResult(rows, labels, fractions, errors)
