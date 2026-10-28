"""Share the sampling, the splitting, the errors, and the element of the surrogate systems fitted to series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
import keyword
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ..core.elements import Element
from ..core.errors import AnalysisError
from ..core.laws import BaseLaw
from ..core.spaces import BaseTime
from ..core.specs import StateSpec
from ..core.system import System
from ..engine.runtime import Trajectory
from ._base import Series

if TYPE_CHECKING:
    from ._base import BaseAnalysis

type Array = npt.NDArray[np.float64]
type SurrogateSource = Trajectory | npt.ArrayLike | list[Trajectory | npt.ArrayLike]
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


def _runs_of(source: SurrogateSource) -> list[Trajectory | npt.ArrayLike]:
    """Return the runs of a source: a list of trajectories or arrays, the first axis of a 3-D array, or the source."""
    if isinstance(source, list) and all(isinstance(item, Trajectory | np.ndarray) for item in source):
        return source
    if not isinstance(source, Trajectory) and np.ndim(source) == 3:  # noqa: PLR2004
        return list(np.asarray(source, dtype=float))
    return [source]


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


def _read_samples(
    analysis: 'BaseAnalysis',
    source: SurrogateSource,
    variables: Sequence[str] | None,
    dt: float,
) -> Samples:
    """Return the uniformly sampled series of the runs of a source, with common variables and step.

    Args:
        analysis:
            The analysis whose reader takes the series of every run.

        source:
            A trajectory, an array with one row per sample, or several runs as a list or a 3-D array.

        variables:
            The variables of trajectories, or `None` for every scalar state or observable.

        dt:
            The time between the rows of an array.

    Returns:
        samples:
            The values of every series, the names of the variables, and the step.

    Raises:
        AnalysisError:
            No run is given, or the runs differ in variables or step, or are not uniformly sampled.
    """
    found = []
    for run in _runs_of(source):
        series = analysis._read_series(run, variables)
        found.append(series if isinstance(run, Trajectory) else Series(series.times * dt, series.values, series.labels))
    if not found:
        error_msg = 'No series is given. Pass a trajectory, an array, or a list of them.'
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


def _check_validation(validation: float) -> None:
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


def _check_horizon(horizon: int) -> None:
    """Raise an error for a forecast horizon that is not a nonnegative integer.

    Args:
        horizon:
            The number of samples to forecast.

    Raises:
        AnalysisError:
            The horizon is not a nonnegative integer.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int | np.integer) or horizon < 0:
        error_msg = f'The horizon must be a nonnegative number of samples, but got {horizon!r}.'
        raise AnalysisError(error_msg)


def _split_samples(series: list[Array], validation: float) -> tuple[list[Array], list[Array]]:
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
    _check_validation(validation)
    cuts = [len(values) - math.floor(validation * (len(values) - 1)) for values in series]
    return [values[:cut] for values, cut in zip(series, cuts, strict=True)], [
        values[cut - 1 :] for values, cut in zip(series, cuts, strict=True)
    ]


def _relative_error(predicted: Sequence[Array], actual: Sequence[Array]) -> float:
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


def _state_names(labels: Sequence[str]) -> list[str]:
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


def _surrogate_system(
    kind: str,
    label: str,
    initial: Mapping[str, float],
    laws: Callable[[SimpleNamespace], list[BaseLaw]],
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
    states = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=value, annotation=Annotated[float, StateSpec()])
        for name, value in initial.items()
    ]

    def initialize(self: Element, label: str, **values: float) -> None:
        Element.__init__(self, label)
        for name, value in {**initial, **values}.items():
            setattr(self, name, value)

    first = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in ('self', 'label')]
    initialize.__signature__ = inspect.Signature([*first, *states])  # type: ignore[attr-defined]
    element_type = type(kind, (Element,), {'__init__': initialize, '_get_laws': lambda self, s: laws(s)})
    return System(elements=[element_type(label)], time=time)


def _symbols_of(s: SimpleNamespace, names: Sequence[str]) -> list[Expression]:
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
