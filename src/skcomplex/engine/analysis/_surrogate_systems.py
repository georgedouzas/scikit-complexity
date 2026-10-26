"""Share the sampling, the splitting, the errors, and the element of the surrogate systems fitted to series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import keyword
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...core.base import AnalysisError, BaseEstimator, BaseLaw, Expression
from ...core.components import Element
from ...core.laws import Symbols
from ...core.space import BaseTime, DiscreteTime
from ...core.system import System
from ..runtime import Simulation, Trajectory
from ._adapters import Series, series_of

type Array = npt.NDArray[np.float64]
type SurrogateData = Trajectory | Series | Sequence[Trajectory | Series]
STEP_TOLERANCE = 1e-6
RESERVED_NAMES = frozenset({'label', 'laws', 'visual', 'group', 'define'})


@dataclass(frozen=True)
class Samples:
    """Uniformly sampled series that a surrogate is fitted to.

    Args:
        series:
            The values of every series, one row per sample and one column per variable.

        labels:
            The name of every variable.

        dt:
            The time between consecutive samples.
    """

    series: list[Array]
    labels: list[str]
    dt: float


def _series_list(data: SurrogateData, names: Sequence[str] | None) -> list[Series]:
    """Return the series of every trajectory or series given, with the named variables of trajectories."""
    items = [data] if isinstance(data, Trajectory | Series) else list(data)
    return [item if isinstance(item, Series) else series_of(item, names) for item in items]


def _step_of(series: Series) -> float:
    """Return the time between the samples of a series, after checking that it is uniform."""
    steps = np.diff(np.asarray(series.times, dtype=float))
    if steps.size == 0 or not np.allclose(steps, steps[0], rtol=STEP_TOLERANCE, atol=0.0):
        error_msg = (
            f'The series of {series.labels} are not sampled at a uniform step with at least two samples. Record them '
            'at a fixed interval, such as with the `dt` of `Simulation`.'
        )
        raise AnalysisError(error_msg)
    return float(steps[0])


def _identifier(text: str) -> str:
    """Return a valid state name made from a label, which is neither a keyword nor a name an element reserves."""
    cleaned = re.sub(r'\W+', '_', text).strip('_') or 'v'
    cleaned = f'v_{cleaned}' if cleaned[0].isdigit() else cleaned
    return f'{cleaned}_' if keyword.iskeyword(cleaned) or cleaned in RESERVED_NAMES else cleaned


def samples_of(data: SurrogateData, names: Sequence[str] | None = None) -> Samples:
    """Return the uniformly sampled series of trajectories or series, with common variables and step.

    Args:
        data:
            A trajectory, a series, or a sequence of them.

        names:
            The names of the variables of trajectories, or `None` for every scalar state or observable.

    Returns:
        samples:
            The values of every series, the names of the variables, and the step.

    Raises:
        AnalysisError:
            No series is given, or the series differ in variables or step, or are not uniformly sampled.
    """
    found = _series_list(data, names)
    if not found:
        error_msg = 'No series is given. Pass a trajectory, a series, or a sequence of them.'
        raise AnalysisError(error_msg)
    steps = [_step_of(series) for series in found]
    labels = found[0].labels
    if any(series.labels != labels for series in found) or not np.allclose(steps, steps[0], rtol=STEP_TOLERANCE):
        error_msg = (
            f'The series have the variables {[series.labels for series in found]} and the steps {steps}. Give series '
            'of the same variables sampled at the same step.'
        )
        raise AnalysisError(error_msg)
    return Samples([np.asarray(series.values, dtype=float) for series in found], list(labels), steps[0])


def check_validation(validation: float) -> None:
    """Raise an error for a validation share outside `[0, 1)`.

    Args:
        validation:
            The share of the samples of every series held out at its end.

    Raises:
        AnalysisError:
            The share is outside `[0, 1)`.
    """
    if not 0.0 <= validation < 1.0:
        error_msg = f'The validation share must lie in [0, 1), but got {validation}.'
        raise AnalysisError(error_msg)


def split_samples(series: list[Array], validation: float) -> tuple[list[Array], list[Array]]:
    """Return the leading training part and the trailing validation part of every series.

    Args:
        series:
            The values of every series.

        validation:
            The share of the samples of every series held out at its end, from 0 to below 1.

    Returns:
        parts:
            The training parts and the validation parts, each validation part starting at the last training sample.

    Raises:
        AnalysisError:
            The share is outside `[0, 1)`.
    """
    check_validation(validation)
    cuts = [len(values) - math.floor(validation * (len(values) - 1)) for values in series]
    return [values[:cut] for values, cut in zip(series, cuts, strict=True)], [
        values[cut - 1 :] for values, cut in zip(series, cuts, strict=True)
    ]


def relative_error(predicted: Sequence[Array], actual: Sequence[Array]) -> float:
    """Return the relative error `||predicted - actual|| / ||actual||` over every row of every array.

    Args:
        predicted:
            The predicted increments or derivatives.

        actual:
            The measured increments or derivatives.

    Returns:
        error:
            The relative error, or NaN when there is no row.
    """
    rows = [np.asarray(item, dtype=float) for item in actual if len(item)]
    if not rows:
        return math.nan
    estimate = np.concatenate([np.asarray(item, dtype=float) for item in predicted if len(item)])
    measured = np.concatenate(rows)
    return float(np.linalg.norm(estimate - measured) / np.linalg.norm(measured))


def state_names(labels: Sequence[str]) -> list[str]:
    """Return state names for variables: the last part of every dotted label, or the whole label where those clash.

    Args:
        labels:
            The names of the variables, such as `lorenz.x` or `cells.n[0]`.

    Returns:
        names:
            Valid and distinct identifiers, one per variable.
    """
    short = [_identifier(label.rpartition('.')[2]) for label in labels]
    if len(set(short)) == len(short):
        return short
    full = [_identifier(label) for label in labels]
    return full if len(set(full)) == len(full) else [f'{name}_{index}' for index, name in enumerate(full)]


def surrogate_system(
    kind: str,
    label: str,
    initial: Mapping[str, float],
    laws: Callable[[Symbols], list[BaseLaw]],
    time: BaseTime,
) -> System:
    """Return a system of one surrogate element with the given states, laws, and time.

    Args:
        kind:
            The name of the element type, such as `LinearSurrogate`.

        label:
            The label of the element.

        initial:
            The initial value of every state, by state name.

        laws:
            The function that returns the laws from the namespace of the element.

        time:
            The time of the system.

    Returns:
        system:
            The system.
    """
    element_type = Element.define(kind, states=dict(initial), laws=laws)
    return System(components=[element_type(label)], time=time)


def symbols_of(s: Symbols, names: Sequence[str]) -> list[Expression]:
    """Return the symbols of the named states of a namespace.

    Args:
        s:
            The namespace of the element.

        names:
            The state names.

    Returns:
        symbols:
            The symbols, in order.
    """
    return [getattr(s, name) for name in names]


class BaseSurrogate(BaseEstimator):
    """Base class of the surrogate systems fitted to series, which forecast by simulating the fitted system.

    `prepare(data)` fits the surrogate to a trajectory, a series, or a sequence of them, sampled at one uniform step,
    and
    `run(steps)` forecasts the variables over a number of samples after the last sample of the last series. This base
    runs the fitted system from that sample, and a child may forecast faster.

    Args:
        names:
            The variables of trajectories, or `None` for every scalar state or observable.

        validation:
            The share of the samples of every series held out at its end to measure the validation error.

        label:
            The label of the surrogate element.

    Attributes:
        system_:
            The system of one surrogate element, which `Simulation` runs.

        labels_:
            The variables the surrogate was fitted to, in the order of its states.

        states_:
            The names of the states of the surrogate element.

        last_state_:
            The values of the states at the last sample of the last series, where forecasts start.

        dt_:
            The time between the samples it was fitted to.

        training_error_:
            The relative error of the predicted increments or derivatives on the training samples.

        validation_error_:
            The relative error on the held-out samples, NaN when none are held out.
    """

    system_: System
    labels_: list[str]
    states_: list[str]
    last_state_: Array
    dt_: float
    training_error_: float
    validation_error_: float

    def __init__(
        self,
        names: Sequence[str] | None = None,
        *,
        validation: float = 0.2,
        label: str = 'surrogate',
    ) -> None:
        self.names = names
        self.validation = validation
        self.label = label

    def _check(self) -> None:
        """Raise an error for a validation share outside `[0, 1)`."""
        check_validation(self.validation)

    def _run(self, steps: int) -> Array:
        """Return the variables of the fitted system simulated from the last sample, one row per sample."""
        dt = 1.0 if isinstance(self.system_.time, DiscreteTime) else self.dt_
        names = [f'{self.label}.{name}' for name in self.states_]
        simulation = Simulation(
            self.system_,
            dt=dt,
            initial_state=dict(zip(names, map(float, self.last_state_), strict=True)),
        )
        trajectory = simulation.run(until=steps * dt).trajectory_
        return np.column_stack([trajectory.values(name) for name in names[: len(self.labels_)]])[1 : steps + 1]
