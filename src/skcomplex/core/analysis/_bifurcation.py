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

from ..experiments import Experiment, Grid, RandomDesign, Source
from ..runtime import Trajectory


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

        **arguments:
            Other `Experiment` arguments.

    Returns:
        diagram:
            The kept values for every value of the parameter.

    Examples:
        >>> from skcomplex.core.analysis import orbit_diagram
        >>> from skcomplex.models.nonlinear import logistic_map
        >>> diagram = orbit_diagram('logistic_map', 'map.r', [2.9, 3.2], 'map.x', transient=1000, keep=20, dt=1)
        >>> diagram.periods().tolist()
        [1, 2]
    """
    step = dt if dt is not None else 1.0
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

        **arguments:
            Other `Experiment` arguments.

    Returns:
        stability:
            The fraction and standard error of each attractor.

    Examples:
        >>> from skcomplex.core.analysis import basin_stability
        >>> from skcomplex.models.ecology import logistic_growth
        >>> result = basin_stability('logistic_growth', {'population.n': (0.1, 10)},
        ...                          lambda trajectory: 'capacity', n=5, until=5)
        >>> result.fractions
        {'capacity': 1.0}
    """

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
    labels = [run.scalars['attractor'] for run in experiment.run().runs]
    counts = Counter(labels)
    fractions = {key: count / n for key, count in counts.items()}
    errors = {key: float(np.sqrt(value * (1 - value) / n)) for key, value in fractions.items()}
    return BasinStability(fractions, errors, n)
