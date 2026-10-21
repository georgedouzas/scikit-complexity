"""Build orbit diagrams over parameter ranges and basin stability over initial conditions from experiments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..compile import compile_system
from ..experiments import Experiment, Grid, RandomDesign, Source, resolve_source
from ..model import System, replicate
from ..runtime import Simulation, Trajectory

RUN_SETTINGS = ('solver', 'backend')


@dataclass(frozen=True)
class OrbitDiagram:
    """Values an observable visits after a transient, for every value of a parameter.

    Args:
        parameter:
            The dotted path or factory argument that was swept.

        values:
            The swept values.

        points:
            The kept values of the observable for every swept value.
    """

    parameter: str
    values: npt.NDArray[np.float64]
    points: list[npt.NDArray[np.float64]]

    def periods(self, decimals: int = 6) -> npt.NDArray[np.int64]:
        """Return the number of distinct kept values for every swept value, the period of periodic orbits.

        Args:
            decimals:
                The decimals to round to before counting distinct values.

        Returns:
            periods:
                One count per swept value.
        """
        return np.asarray([np.unique(np.round(points, decimals)).size for points in self.points], dtype=np.int64)


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


def orbit_diagram(
    source: Source,
    parameter: str,
    values: Sequence[float],
    observable: str,
    *,
    transient: float,
    keep: int,
    dt: float | None = None,
    executor: str | Executor = 'serial',
    vectorized: bool = False,
    **arguments: Any,  # noqa: ANN401
) -> OrbitDiagram:
    """Return the values an observable visits after a transient, over a range of a parameter.

    Args:
        source:
            A registered model name, a factory, or a system.

        parameter:
            The dotted path or factory argument to sweep.

        values:
            The values of the parameter.

        observable:
            The state or observable path whose values are kept.

        transient:
            The simulated time discarded before keeping values.

        keep:
            The number of frames kept after the transient.

        dt:
            The time between frames, or `None` for the model's default, 1 for maps.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor.

        vectorized:
            Whether to run the sweep as one simulation of the system replicated over the values, which needs a dotted
            parameter path and a system of single elements.

        **arguments:
            Other `Experiment` arguments, or `random_state` for a vectorised sweep.

    Returns:
        diagram:
            The kept values for every value of the parameter.

    Examples:
        >>> from skcomplex.core.analysis import orbit_diagram
        >>> from skcomplex.models.mathematics.dynamical_systems import logistic_map
        >>> diagram = orbit_diagram('logistic_map', 'map.r', [2.9, 3.2], 'map.x', transient=1000, keep=20, dt=1)
        >>> diagram.periods().tolist()
        [1, 2]
    """
    step = dt if dt is not None else 1.0
    if vectorized:
        settings = {'dt': step, 'random_state': arguments.get('random_state')}
        trajectory, _ = _ensemble(source, {parameter: list(values)}, transient + (keep - 1) * step, settings)
        series = np.asarray(trajectory[observable])[np.asarray(trajectory.times) >= transient - 1e-9]
        points = [series[:, index] for index in range(len(values))]
        return OrbitDiagram(parameter, np.asarray(values, dtype=float), points)
    experiment = Experiment(
        source,
        Grid(**{parameter: list(values)}),
        until=transient + (keep - 1) * step,
        dt=dt,
        warmup=transient,
        outputs=[observable],
        executor=executor,
        **arguments,
    )
    result = experiment.run()
    return OrbitDiagram(parameter, np.asarray(values, dtype=float), [run.series[observable] for run in result.runs])


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
        >>> from skcomplex.core.analysis import basin_stability
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> result = basin_stability('logistic_growth', {'population.n': (0.1, 10)},
        ...                          lambda trajectory: 'capacity', n=5, until=5)
        >>> result.fractions
        {'capacity': 1.0}
    """

    if vectorized:
        rows = RandomDesign(n, random_state=random_state, **bounds).rows()
        values = {path: [row[path] for row in rows] for path in bounds}
        settings = {'dt': arguments.get('dt'), 'random_state': random_state}
        trajectory, system = _ensemble(source, values, until, settings)
        problem = compile_system(system)
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
