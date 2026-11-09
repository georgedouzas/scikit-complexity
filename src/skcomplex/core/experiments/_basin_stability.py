"""Estimate the basin stability of the attractors of a system.

It is estimated over random initial states.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass

import numpy as np

from ..components.specs import ScenarioSpec
from ..engine.problems import ProblemSpec
from ..engine.runtime import Trajectory
from ..errors import ValidationError
from ._designs import _get_rows
from ._replicate import _run_replicated
from ._runs import _check_count, _check_source, _compile, _run_experiment
from .types import Row, Source


@dataclass(frozen=True)
class BasinStabilityResult:
    """Fractions of random initial states that end in each attractor.

    Args:
        rows:
            The initial states, one mapping from dotted path to value per run.

        labels:
            The attractor label of every initial state.

        fractions:
            The fraction of initial states per attractor label.

        errors:
            The standard error of every fraction.
    """

    rows: list[Row]
    labels: list[object]
    fractions: dict[object, float]
    errors: dict[object, float]


def basin_stability(
    system: Source,
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
    """Return the fractions of random initial states that end in each attractor.

    The initial states are drawn uniformly within bounds. Every initial state is run to `until` and its trajectory is
    labelled by the classifier. With `vectorized`, all initial states run as the members of one simulation of the
    replicated system, and the classifier receives every member as a trajectory of the original system.

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

    Examples:
        >>> from skcomplex.core.experiments import basin_stability
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> result = basin_stability(
        ...     LogisticGrowthSystem(), {'population.x': (0.1, 10)}, lambda trajectory: 'capacity', n=5, until=5
        ... )
        >>> result.fractions
        {'capacity': 1.0}
    """
    _check_source(system)
    if not bounds:
        error_msg = 'Basin stability needs the bounds of at least one initial state.'
        raise ValidationError(error_msg)
    _check_count('n', n)
    rows = _get_rows(bounds, 'random', n, random_state)
    if vectorized:
        values = {path: np.asarray([row[path] for row in rows], dtype=float) for path in bounds}
        trajectory, composite = _run_replicated(system, values, until, dt, random_state)
        problem = _compile(composite)
        if not isinstance(problem, ProblemSpec):
            error_msg = f'A vectorised basin reads the members of a general problem, not of a {type(problem).__name__}.'
            raise ValidationError(error_msg)
        labels = [classifier(trajectory.member(index, problem)) for index in range(n)]
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
