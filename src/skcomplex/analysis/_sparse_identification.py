"""Fit sparse symbolic surrogates to series by sparse identification of nonlinear dynamics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ..core.laws import BaseLaw, RateLaw
from ..core.spaces import ContinuousTime
from ._lazy import module
from ._surrogate_systems import (
    BaseSurrogate,
    SurrogateData,
    relative_error,
    samples_of,
    split_samples,
    state_names,
    surrogate_system,
    symbols_of,
)

type Array = npt.NDArray[np.float64]
DIFFERENCE_POINTS = 3


def _term(symbols: Sequence[Expression], powers: npt.NDArray[np.int_]) -> Expression:
    """Return the monomial of the symbols with the given powers."""
    return np.prod([item ** int(power) for item, power in zip(symbols, powers, strict=True) if power], initial=1)


def _sparse_laws(
    names: Sequence[str],
    coefficients: Array,
    powers: npt.NDArray[np.int_],
) -> Callable[[SimpleNamespace], list[BaseLaw]]:
    """Return the function that gives one rate law per state, a sum of the monomials with nonzero coefficients."""

    def laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the sparse rate laws."""
        symbols = symbols_of(s, names)
        terms = [_term(symbols, row) for row in powers]
        rates = [
            sum(float(value) * term for value, term in zip(row, terms, strict=True) if value) for row in coefficients
        ]
        return [RateLaw(item, rate) for item, rate in zip(symbols, rates, strict=True)]

    return laws


def _derivatives(model: Any, parts: list[Array], dt: float) -> tuple[list[Array], list[Array]]:  # noqa: ANN401
    """Return the derivatives a model predicts and those its differentiation method estimates from the series."""
    usable = [values for values in parts if len(values) >= DIFFERENCE_POINTS]
    predicted = [np.asarray(model.predict(values), dtype=float) for values in usable]
    measured = [np.asarray(model.differentiation_method(values, dt), dtype=float) for values in usable]
    return predicted, measured


class SINDySurrogate(BaseSurrogate):
    """Sparse symbolic surrogate fitted by sparse identification of nonlinear dynamics, computed with PySINDy.

    SINDy writes the derivative of every variable as a sparse sum of the monomials of the variables up to a degree. It
    estimates the derivatives by centred finite differences and selects the terms by sequentially thresholded least
    squares. The surrogate holds one rate law per variable under continuous time and starts at the first sample of the
    first series, and a forecast simulates it from the last sample of the last series.

    Args:
        names:
            The variables of trajectories, or `None` for every scalar state or observable.

        degree:
            The highest degree of the candidate monomials.

        threshold:
            The smallest magnitude of a kept coefficient.

        alpha:
            The strength of the ridge penalty of every least-squares fit.

        validation:
            The share of the samples of every series held out at its end to measure the validation error.

        label:
            The label of the surrogate element.

    Attributes:
        system_:
            The system of one surrogate element under continuous time, with one rate law per variable.

        labels_:
            The variables the surrogate was fitted to, in the order of its states.

        states_:
            The names of the states of the surrogate element.

        last_state_:
            The values of the states at the last sample of the last series, where forecasts start.

        dt_:
            The time between the samples it was fitted to.

        training_error_:
            The relative error of the predicted derivatives on the training samples.

        validation_error_:
            The relative error on the held-out samples, NaN when none are held out.

        features_:
            The candidate monomials of the states, such as `x y`.

        coefficients_:
            The coefficient of every monomial in the rate of every state, one row per state and one column per
            monomial, zero where the term is absent.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import SINDySurrogate, Series
        >>> times = np.arange(0.0, 2.0, 0.001)
        >>> surrogate = SINDySurrogate(degree=1).prepare(Series(times, np.exp(-2 * times)[:, None], ['x']))
        >>> surrogate.features_, surrogate.coefficients_.round(3).tolist()
        (['1', 'x'], [[0.0, -2.0]])
    """

    features_: list[str]
    coefficients_: Array

    def __init__(
        self,
        names: Sequence[str] | None = None,
        *,
        degree: int = 2,
        threshold: float = 0.1,
        alpha: float = 0.05,
        validation: float = 0.2,
        label: str = 'surrogate',
    ) -> None:
        self.names = names
        self.degree = degree
        self.threshold = threshold
        self.alpha = alpha
        self.validation = validation
        self.label = label

    def _prepare(self, data: SurrogateData) -> None:
        """Fit the sparse model to the series and set the surrogate system, its errors, and its coefficients."""
        samples = samples_of(data, self.names)
        states = state_names(samples.labels)
        training, held = split_samples(samples.series, self.validation)
        sindy = module('pysindy')
        model = sindy.SINDy(
            optimizer=sindy.STLSQ(threshold=self.threshold, alpha=self.alpha),
            feature_library=sindy.PolynomialLibrary(degree=self.degree),
        ).fit(training, t=samples.dt, feature_names=states)
        coefficients = np.asarray(model.coefficients(), dtype=float)
        powers = np.asarray(model.feature_library.powers_, dtype=int)
        self.system_ = surrogate_system(
            'SparseSurrogate',
            self.label,
            dict(zip(states, map(float, samples.series[0][0]), strict=True)),
            _sparse_laws(states, coefficients, powers),
            ContinuousTime(),
        )
        self.labels_ = samples.labels
        self.states_ = states
        self.last_state_ = np.asarray(samples.series[-1][-1], dtype=float)
        self.dt_ = samples.dt
        self.training_error_ = relative_error(*_derivatives(model, training, samples.dt))
        self.validation_error_ = relative_error(*_derivatives(model, held, samples.dt))
        self.features_ = [str(name) for name in model.get_feature_names()]
        self.coefficients_ = coefficients
