"""Build basin stability over initial conditions from experiments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.system import System
from ..engine.compile import compiled_problem
from ..engine.runtime import Simulation, Trajectory
from ..experiments import Experiment, RandomDesign, Source, resolve_source
from ..experiments._ensembles import replicate

RUN_SETTINGS = ('solver', 'backend')


def _ensemble(
    source: Source,
    values: Mapping[str, Sequence[float]],
    until: float | None,
    settings: Mapping[str, Any],
) -> tuple[Trajectory, System]:
    """Return the trajectory of one simulation of a source replicated over per-member values, and its system."""
    resolved, defaults = resolve_source(source)
    system = resolved if isinstance(resolved, System) else resolved()
    size = len(next(iter(values.values())))
    options = {key: value for key, value in defaults.items() if key in RUN_SETTINGS}
    step = settings.get('dt') or defaults.get('dt', 1.0)
    end = until if until is not None else defaults.get('until', 10.0)
    simulation = Simulation(
        replicate(system, size, values),
        dt=step,
        random_state=settings.get('random_state'),
        **options,
    )
    return simulation.run(until=end).trajectory_, system


def _stability(labels: Sequence[Any], n: int) -> 'BasinStability':
    """Return the fraction and the standard error of every attractor label."""
    counts = Counter(labels)
    fractions = {key: count / n for key, count in counts.items()}
    errors = {key: float(np.sqrt(value * (1 - value) / n)) for key, value in fractions.items()}
    return BasinStability(fractions, errors, n)


@dataclass(frozen=True)
class BasinStability:
    """Fractions of random initial conditions that end in each attractor.

    Args:
        fractions:
            The fraction of initial conditions per attractor label.

        errors:
            The standard error of every fraction.

        samples:
            The number of initial conditions.
    """

    fractions: dict[Any, float]
    errors: dict[Any, float]
    samples: int


def basin_stability(
    source: Source,
    bounds: Mapping[str, tuple[float, float]],
    classifier: Callable[[Trajectory], Any],
    *,
    n: int = 500,
    until: float | None = None,
    random_state: int | None = None,
    executor: str | Executor = 'serial',
    vectorized: bool = False,
    **arguments: Any,  # noqa: ANN401
) -> BasinStability:
    """Return the fraction of initial conditions, drawn uniformly within bounds, that end in each attractor.

    Args:
        source:
            A registered model name, a factory, or a system.

        bounds:
            A mapping from dotted state path to the `(low, high)` bounds of its initial value.

        classifier:
            A function of a trajectory that returns the label of the attractor it ends in.

        n:
            The number of initial conditions.

        until:
            The simulated time of every run, or `None` for the model's default.

        random_state:
            The seed of the initial conditions and the runs.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor.

        vectorized:
            Whether to run every initial condition as a member of one simulation of the replicated system, whose
            members the classifier receives one by one as trajectories of the original system.

        **arguments:
            Other `Experiment` arguments, or `dt` for a vectorised run.

    Returns:
        stability:
            The fraction and standard error of each attractor.

    Examples:
        >>> from skcomplex.analysis import basin_stability
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> result = basin_stability('logistic_growth', {'population.n': (0.1, 10)},
        ...                          lambda trajectory: 'capacity', n=5, until=5)
        >>> result.fractions
        {'capacity': 1.0}
    """

    if vectorized:
        rows = RandomDesign(n, random_state=random_state, **bounds).prepare().run()
        values = {path: [row[path] for row in rows] for path in bounds}
        settings = {'dt': arguments.get('dt'), 'random_state': random_state}
        trajectory, system = _ensemble(source, values, until, settings)
        problem = compiled_problem(system, 'A vectorised basin')
        return _stability([classifier(trajectory.member(index, problem)) for index in range(n)], n)

    def label(trajectory: Trajectory) -> Any:  # noqa: ANN401
        return classifier(trajectory)

    label.__name__ = 'attractor'
    experiment = Experiment(
        source,
        RandomDesign(n, random_state=random_state, **bounds),
        until=until,
        outputs=[label],
        random_state=random_state,
        executor=executor,
        **arguments,
    )
    return _stability([run.scalars['attractor'] for run in experiment.run().runs], n)
