"""Fit sparse symbolic surrogates to series by SINDy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import keyword
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ..components.elements import DeclaredElement
from ..components.laws import BaseLaw, RateLaw
from ..components.spaces import BaseTime, ContinuousTime
from ..components.systems import CompositeSystem
from ..engine.runtime import simulate
from ..errors import AnalysisError
from ..specs import AnalysisTags, DeclaredVariableSpec, ParameterSpec, SamplesSpec, StateSpec, VisualSpec
from ..types import Array, DataKind
from ._base import _read_samples
from ._utils import load_module
from .types import Runs

DIFFERENCE_POINTS = 3
RESERVED_NAMES = frozenset({'label', 'laws', 'visual', 'variables', 'group', 'define'})
TAGS = AnalysisTags(DataKind.SERIES)


class _SurrogateElement(DeclaredElement):
    """Surrogate element, whose laws a function fitted to series returns."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
        laws: Callable[[SimpleNamespace], list[BaseLaw]] | None = None,
    ) -> None:
        super().__init__(label, visual=visual, variables=variables)
        self.laws = laws

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseLaw]:
        """Return the laws of the fitted function for a namespace of symbols.

        Without a function, there are none.
        """
        return [] if self.laws is None else self.laws(namespace)


@dataclass(frozen=True)
class SparseIdentificationResult:
    """A result of `sparse_identification`, a sparse symbolic system.

    SINDy is the sparse identification of nonlinear dynamics.

    Args:
        samples:
            The series of every run, the names of the variables, and the step.

        system:
            The system of one surrogate element under continuous time, with one rate law per variable.

        labels:
            The variables the system was fitted to, in the order of its states.

        states:
            The names of the states of the surrogate element.

        last_state:
            The values of the states at the last sample of the last run, where the forecast starts.

        dt:
            The time between the samples it was fitted to.

        training_error:
            The relative error of the predicted derivatives on the training samples.

        validation_error:
            The relative error on the held-out samples, NaN when none are held out.

        features:
            The candidate monomials of the states, such as `x y`.

        coefficients:
            The coefficient of every monomial in the rate of every state, one row per state and one column per
            monomial, zero where the term is absent.

        forecast:
            The variables at the `horizon` samples after the last one, one row per sample.
    """

    samples: SamplesSpec
    system: CompositeSystem
    labels: list[str]
    states: list[str]
    last_state: Array
    dt: float
    training_error: float
    validation_error: float
    features: list[str]
    coefficients: Array
    forecast: Array


def sparse_identification(
    source: Runs,
    *,
    variables: Sequence[str] | None = None,
    degree: int = 2,
    threshold: float = 0.1,
    alpha: float = 0.05,
    validation: float = 0.2,
    horizon: int = 0,
    dt: float = 1.0,
    label: str = 'surrogate',
) -> SparseIdentificationResult:
    """Return the sparse symbolic system fitted to series by SINDy, with PySINDy.

    SINDy is the sparse identification of nonlinear dynamics. SINDy writes the derivative of every variable as a sparse
    sum of the monomials of the variables up to a degree. It estimates the derivatives by centred finite differences and
    selects the terms by sequentially thresholded least squares. The fitted system holds one rate law per variable under
    continuous time and starts at the first sample of the first run, and the forecast simulates it from the last sample
    of the last run.

    It appears in many fields. It finds the equations of a fluid flow, of a chemical reaction, and of a predator and
    prey cycle from their series.

    Args:
        source:
            A trajectory, an array with one row per sample, or several runs as a list or a 3-D array.

        variables:
            The variables of trajectories, or `None` for every scalar state or observable.

        degree:
            The highest degree of the candidate monomials.

        threshold:
            The smallest magnitude of a kept coefficient.

        alpha:
            The strength of the ridge penalty of every least-squares fit.

        validation:
            The share of the samples of every run held out at its end to measure the validation error.

        horizon:
            The number of samples to forecast after the last sample of the last run.

        dt:
            The time between the rows of an array.

        label:
            The label of the surrogate element.

    Returns:
        result:
            The fitted system, its errors, its coefficients, and the forecast.

    Raises:
        AnalysisError:
            The validation share lies outside `[0, 1)`, the horizon is negative, or the runs differ in variables or
            step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import sparse_identification
        >>> times = np.arange(0.0, 2.0, 0.001)
        >>> result = sparse_identification(np.exp(-2 * times), variables=['x'], degree=1, dt=0.001)
        >>> result.features, result.coefficients.round(3).tolist()
        (['1', 'x'], [[0.0, -2.0]])
    """
    _check_validation(validation)
    _check_horizon(horizon)
    samples = _read_samples(source, variables, dt)
    states = _state_names(samples.labels)
    training, held = _split_samples(samples.series, validation)
    sindy = load_module('pysindy')
    fitted = sindy.SINDy(
        optimizer=sindy.STLSQ(threshold=threshold, alpha=alpha),
        feature_library=sindy.PolynomialLibrary(degree=degree),
    ).fit(training, t=samples.dt, feature_names=states)
    coefficients = np.asarray(fitted.coefficients(), dtype=float)
    powers = np.asarray(fitted.feature_library.powers_, dtype=int)
    system = _surrogate_system(
        label,
        dict(zip(states, map(float, samples.series[0][0]), strict=True)),
        _sparse_laws(states, coefficients, powers),
        ContinuousTime(),
    )
    last_state = np.asarray(samples.series[-1][-1], dtype=float)
    predict, differentiate = fitted.predict, fitted.differentiation_method
    return SparseIdentificationResult(
        samples,
        system,
        samples.labels,
        states,
        last_state,
        samples.dt,
        _relative_error(*_derivatives(predict, differentiate, training, samples.dt)),
        _relative_error(*_derivatives(predict, differentiate, held, samples.dt)),
        [str(name) for name in fitted.get_feature_names()],
        coefficients,
        _forecast(system, label, states, last_state, dt=samples.dt, horizon=horizon),
    )


def _forecast(
    system: CompositeSystem,
    label: str,
    states: list[str],
    last_state: Array,
    *,
    dt: float,
    horizon: int,
) -> Array:
    """Return the variables of the fitted system simulated from the last sample.

    There is one row per sample.
    """
    names = [f'{label}.{name}' for name in states]
    trajectory = simulate(
        system,
        until=horizon * dt,
        dt=dt,
        initial_state=dict(zip(names, map(float, last_state), strict=True)),
    )
    values = np.column_stack([np.asarray(trajectory.get_values(name), dtype=np.float64) for name in names])
    return values[1 : horizon + 1].reshape(horizon, len(names))


def _term(symbols: Sequence[Expression], powers: npt.NDArray[np.int_]) -> Expression:
    """Return the monomial of the symbols with the given powers."""
    return np.prod([item ** int(power) for item, power in zip(symbols, powers, strict=True) if power], initial=1)


def _sparse_laws(
    names: Sequence[str],
    coefficients: Array,
    powers: npt.NDArray[np.int_],
) -> Callable[[SimpleNamespace], list[BaseLaw]]:
    """Return the function that gives one rate law per state.

    Each law is a sum of the monomials with nonzero coefficients.
    """

    def laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the sparse rate laws."""
        symbols = _symbols_of(s, names)
        terms = [_term(symbols, row) for row in powers]
        rates = [
            sum(float(value) * term for value, term in zip(row, terms, strict=True) if value) for row in coefficients
        ]
        return [RateLaw(item, rate) for item, rate in zip(symbols, rates, strict=True)]

    return laws


def _derivatives(
    predict: Callable[[Array], npt.ArrayLike],
    differentiate: Callable[[Array, float], npt.ArrayLike],
    parts: list[Array],
    dt: float,
) -> tuple[list[Array], list[Array]]:
    """Return the derivatives a system predicts and those estimated from the series.

    The estimate comes from its differentiation method.
    """
    usable = [values for values in parts if len(values) >= DIFFERENCE_POINTS]
    predicted = [np.asarray(predict(values), dtype=np.float64) for values in usable]
    measured = [np.asarray(differentiate(values, dt), dtype=np.float64) for values in usable]
    return predicted, measured


def _identifier(text: str) -> str:
    """Return a valid state name made from a label.

    The name is neither a keyword nor a name an element reserves.
    """
    cleaned = re.sub(r'\W+', '_', text).strip('_') or 'v'
    cleaned = f'v_{cleaned}' if cleaned[0].isdigit() else cleaned
    return f'{cleaned}_' if keyword.iskeyword(cleaned) or cleaned in RESERVED_NAMES else cleaned


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
    """Return the leading training part and the trailing validation part of the series.

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
    """Return the relative error `||predicted - actual|| / ||actual||`.

    It is taken over every row of every array.

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
    """Return state names for variables.

    A name is the last part of every dotted label, or the whole label where those clash.

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
    label: str,
    initial: Mapping[str, float],
    laws: Callable[[SimpleNamespace], list[BaseLaw]],
    time: BaseTime,
    *,
    inputs: Mapping[str, tuple[float, tuple[float, float]]] | None = None,
) -> CompositeSystem:
    """Return a system of one surrogate element.

    It has the given states, laws, time, and control inputs.

    Args:
        label:
            The label of the element.

        initial:
            The initial value of every state, by state name.

        laws:
            The function that returns the laws from the namespace of the element.

        time:
            The time of the system.

        inputs:
            The initial value and the bounds of every control input, an adaptive parameter, by name, or `None`.

    Returns:
        system:
            The system.
    """
    variables = {name: DeclaredVariableSpec(value, StateSpec()) for name, value in initial.items()}
    for name, (value, bounds) in (inputs or {}).items():
        variables[name] = DeclaredVariableSpec(value, ParameterSpec(adaptive=True, bounds=bounds))
    return CompositeSystem(elements=[_SurrogateElement(label, variables=variables, laws=laws)], time=time)


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
