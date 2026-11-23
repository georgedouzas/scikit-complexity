"""Build orbit diagrams, the values an observable visits over a parameter range."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..errors import ValidationError
from ..types import Array
from ._replicate import _run_replicated
from ._runs import _check_count, _check_source, _run_experiment
from ._scenario_spec import ScenarioSpec
from .types import SystemSource


@dataclass(frozen=True)
class OrbitDiagramResult:
    """A result of `orbit_diagram`, the values visited for every value of a parameter.

    It holds the values a run keeps visiting after it settles, for every value of the parameter, and how many distinct
    values each run visits, so a period of 2 means a cycle between two values. It shows the period doubling of a
    population, the route to chaos of a forced pendulum, and the pulses of a laser.

    Read more in the user guide, under Run experiments.

    Args:
        points:
            The kept values of the observable for every value of the parameter.

        periods:
            The period of the orbit for every value of the parameter, the smallest shift after which the rounded kept
            values repeat at least once in full, or 0 when they never do, as for a chaotic orbit.

    See Also:
        orbit_diagram: Returns this result.

    Examples:
        >>> from skcomplex.core.experiments import orbit_diagram
        >>> from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem
        >>> orbit_diagram(LogisticMapSystem(), 'r', [2.9, 3.2], 'map.x', transient=1000, keep=20).periods.tolist()
        [1, 2]
    """

    points: list[Array]
    periods: npt.NDArray[np.int64]


def orbit_diagram(
    system: SystemSource,
    parameter: str,
    values: Sequence[float],
    observable: str,
    *,
    transient: float,
    keep: int,
    dt: float | None = None,
    decimals: int = 6,
    random_state: int | None = None,
    executor: str | Executor = 'serial',
    vectorized: bool = False,
) -> OrbitDiagramResult:
    """Return the values an observable visits after a transient, per parameter value.

    It runs the system for every value of `parameter` to `transient + (keep - 1) * dt` and keeps the values of
    `observable` from `transient` on, so a settled cycle shows as the few values it repeats. With `vectorized`, every
    value runs as one member of a single simulation. It shows the period doubling of the logistic map of a population,
    the route to chaos of a forced pendulum, and the pulses of a laser.

    Read more in the user guide, under Run experiments.

    Args:
        system:
            The system, or a factory of systems, such as a system class.

        parameter:
            The parameter to sweep, a name `set_params` takes such as `r` or `growth__r`, or a dotted path such as
            `map.r`, which runs one prepared clone.

        values:
            The values of the parameter.

        observable:
            The state or observable path whose values are kept.

        transient:
            The simulated time discarded before keeping values.

        keep:
            The number of frames kept after the transient.

        dt:
            The time between frames, or `None` for the tuned one of the system. The kept window is counted in steps of
            `dt`, or of 1, the step of maps, when it is `None`.

        decimals:
            The decimals the kept values are rounded to before the period of every orbit is found.

        random_state:
            The seed of stochastic runs.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor of the separate runs.

        vectorized:
            Whether to run every value as a member of one simulation of the replicated system, with `parameter` a
            dotted path such as `'map.r'`. It is much faster for many values, such as a scan of a map over 1200
            values in about a second.

    Returns:
        result:
            The kept values and the period of the orbit of every value of the parameter.

    Raises:
        ValidationError:
            No value is given, or the number of kept frames is not a positive integer.

        ExperimentError:
            The system is a name.

    See Also:
        basin_stability: Gives the share of initial states per final behaviour.
        sweep: Runs a system over values of its parameters.

    Examples:
        >>> from skcomplex.core.experiments import orbit_diagram
        >>> from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem
        >>> diagram = orbit_diagram(LogisticMapSystem(), 'r', [2.9, 3.2], 'map.x', transient=1000, keep=20)
        >>> diagram.periods.tolist()
        [1, 2]
    """
    _check_source(system)
    if not len(values):
        error_msg = 'An orbit diagram needs at least one value of the parameter.'
        raise ValidationError(error_msg)
    _check_count('keep', keep)
    step = dt if dt is not None else 1.0
    until = transient + (keep - 1) * step
    if vectorized:
        trajectory, _ = _run_replicated(system, {parameter: list(values)}, until, step, random_state)
        observed = np.asarray(trajectory.get_values(observable), dtype=np.float64)
        series = observed[np.asarray(trajectory.times) >= transient - 1e-9]
        points = [series[:, index] for index in range(len(values))]
    else:
        result = _run_experiment(
            system,
            [{parameter: value} for value in values],
            [ScenarioSpec()],
            1,
            analyses=None,
            until=until,
            dt=dt,
            solver=None,
            outputs=[observable],
            warmup=transient,
            observables=(),
            random_state=random_state,
            executor=executor,
        )
        points = [np.asarray(run.series[observable], dtype=float) for run in result.runs]
    periods = np.asarray([_get_period(np.round(series, decimals)) for series in points], dtype=np.int64)
    return OrbitDiagramResult(points, periods)


def _get_period(series: Array) -> int:
    """Return the smallest shift after which a series repeats at least once in full, or 0 when none does."""
    for shift in range(1, len(series) // 2 + 1):
        if np.array_equal(series[shift:], series[:-shift]):
            return shift
    return 0
