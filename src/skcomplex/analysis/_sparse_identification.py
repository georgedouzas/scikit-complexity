"""Fit sparse symbolic surrogates to series by sparse identification of nonlinear dynamics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Self

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ..core.laws import BaseLaw, RateLaw
from ..core.spaces import ContinuousTime
from ..engine.runtime import Simulation
from ._base import BaseAnalysis
from ._kinds import DataKind
from ._surrogate_systems import (
    SurrogateSource,
    _check_horizon,
    _check_validation,
    _read_samples,
    _relative_error,
    _split_samples,
    _state_names,
    _surrogate_system,
    _symbols_of,
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
        symbols = _symbols_of(s, names)
        terms = [_term(symbols, row) for row in powers]
        rates = [
            sum(float(value) * term for value, term in zip(row, terms, strict=True) if value) for row in coefficients
        ]
        return [RateLaw(item, rate) for item, rate in zip(symbols, rates, strict=True)]

    return laws


def _derivatives(model: object, parts: list[Array], dt: float) -> tuple[list[Array], list[Array]]:
    """Return the derivatives a model predicts and those its differentiation method estimates from the series."""
    usable = [values for values in parts if len(values) >= DIFFERENCE_POINTS]
    predicted = [np.asarray(model.predict(values), dtype=float) for values in usable]
    measured = [np.asarray(model.differentiation_method(values, dt), dtype=float) for values in usable]
    return predicted, measured


class SparseIdentificationAnalysis(BaseAnalysis):
    """Sparse symbolic system fitted to series by sparse identification of nonlinear dynamics, computed with PySINDy.

    SINDy writes the derivative of every variable as a sparse sum of the monomials of the variables up to a degree. It
    estimates the derivatives by centred finite differences and selects the terms by sequentially thresholded least
    squares. The fitted system holds one rate law per variable under continuous time and starts at the first sample of
    the first run, and the forecast simulates it from the last sample of the last run.

    Args:
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

    Attributes:
        samples_:
            The series of every run, the names of the variables, and the step, set by `prepare`.

        system_:
            The system of one surrogate element under continuous time, with one rate law per variable.

        labels_:
            The variables the system was fitted to, in the order of its states.

        states_:
            The names of the states of the surrogate element.

        last_state_:
            The values of the states at the last sample of the last run, where the forecast starts.

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

        forecast_:
            The variables at the `horizon` samples after the last one, one row per sample.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import SparseIdentificationAnalysis
        >>> times = np.arange(0.0, 2.0, 0.001)
        >>> analysis = SparseIdentificationAnalysis(['x'], degree=1, dt=0.001)
        >>> analysis = analysis.check().prepare(np.exp(-2 * times)).run()
        >>> analysis.features_, analysis.coefficients_.round(3).tolist()
        (['1', 'x'], [[0.0, -2.0]])
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        degree: int = 2,
        threshold: float = 0.1,
        alpha: float = 0.05,
        validation: float = 0.2,
        horizon: int = 0,
        dt: float = 1.0,
        label: str = 'surrogate',
    ) -> None:
        self.variables = variables
        self.degree = degree
        self.threshold = threshold
        self.alpha = alpha
        self.validation = validation
        self.horizon = horizon
        self.dt = dt
        self.label = label

    def _check(self) -> Self:
        """Raise an error for a validation share outside `[0, 1)` or a negative horizon."""
        _check_validation(self.validation)
        _check_horizon(self.horizon)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: SurrogateSource) -> Self:
        """Read the uniformly sampled series of every run."""
        self.samples_ = _read_samples(self, source, self.variables, self.dt)
        return self

    def _run(self) -> Self:
        """Fit the sparse model and set the surrogate system, its errors, its coefficients, and the forecast."""
        samples = self.samples_
        states = _state_names(samples.labels)
        training, held = _split_samples(samples.series, self.validation)
        sindy = self._import('pysindy')
        model = sindy.SINDy(
            optimizer=sindy.STLSQ(threshold=self.threshold, alpha=self.alpha),
            feature_library=sindy.PolynomialLibrary(degree=self.degree),
        ).fit(training, t=samples.dt, feature_names=states)
        coefficients = np.asarray(model.coefficients(), dtype=float)
        powers = np.asarray(model.feature_library.powers_, dtype=int)
        self.system_ = _surrogate_system(
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
        self.training_error_ = _relative_error(*_derivatives(model, training, samples.dt))
        self.validation_error_ = _relative_error(*_derivatives(model, held, samples.dt))
        self.features_ = [str(name) for name in model.get_feature_names()]
        self.coefficients_ = coefficients
        self.forecast_ = self._forecast()
        return self

    def _forecast(self) -> Array:
        """Return the variables of the fitted system simulated from the last sample, one row per sample."""
        names = [f'{self.label}.{name}' for name in self.states_]
        simulation = Simulation(
            self.system_,
            dt=self.dt_,
            initial_state=dict(zip(names, map(float, self.last_state_), strict=True)),
        )
        trajectory = simulation.run(until=self.horizon * self.dt_).trajectory_
        values = np.column_stack([trajectory.values(name) for name in names])
        return values[1 : self.horizon + 1].reshape(self.horizon, len(names))
