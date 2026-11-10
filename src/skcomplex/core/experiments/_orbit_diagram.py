"""Build orbit diagrams, the values an observable visits over a parameter range."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..components.specs import ScenarioSpec
from ..errors import ValidationError
from ..types import Array
from ._replicate import _run_replicated
from ._runs import _check_count, _check_source, _run_experiment
from .types import Source


@dataclass(frozen=True)
class OrbitDiagramResult:
    """A result of `orbit_diagram`, the values visited for every value of a parameter.

    Args:
        points:
            The kept values of the observable for every value of the parameter.

        periods:
            The number of distinct kept values for every value of the parameter, the period of periodic orbits.
    """

    points: list[Array]
    periods: npt.NDArray[np.int64]


def orbit_diagram(
    system: Source,
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
    """Return the values an observable visits after a transient.

    They are given for every value of a parameter. Every value of the parameter is run to `transient + (keep - 1) * dt`,
    and the frames from `transient` on are kept. With `vectorized`, all values run as the members of one simulation of
    the system replicated over them, which needs a dotted path of a state or parameter of an element, and a system of
    single elements and their interactions.

    It appears in many fields. It shows the period doubling of the logistic map of a population, the routes to chaos of
    a forced pendulum, and the bifurcations of a laser.

    Args:
        system:
            The system, or a factory of systems, such as a system class.

        parameter:
            The model parameter or the dotted path to sweep.

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
            The decimals the kept values are rounded to before counting the period of every orbit.

        random_state:
            The seed of stochastic runs.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor of the separate runs.

        vectorized:
            Whether to run every value as a member of one simulation of the replicated system.

    Returns:
        result:
            The kept values and the period of the orbit of every value of the parameter.

    Raises:
        ValidationError:
            No value is given, or the number of kept frames is not a positive integer.

        ExperimentError:
            The system is a name.

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
    periods = np.asarray([np.unique(np.round(series, decimals)).size for series in points], dtype=np.int64)
    return OrbitDiagramResult(points, periods)
