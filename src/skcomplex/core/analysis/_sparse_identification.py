"""Fit sparse symbolic surrogates to series by sparse identification of nonlinear dynamics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..base import Expression
from ..laws import Law, Rate, Symbols
from ..space import ContinuousTime
from ._kinds import DataKind, analysis
from ._lazy import module
from ._surrogate_systems import (
    Surrogate,
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
) -> Callable[[Symbols], list[Law]]:
    """Return the function that gives one rate law per state, a sum of the monomials with nonzero coefficients."""

    def laws(s: Symbols) -> list[Law]:
        """Return the sparse rate laws."""
        symbols = symbols_of(s, names)
        terms = [_term(symbols, row) for row in powers]
        rates = [
            sum(float(value) * term for value, term in zip(row, terms, strict=True) if value) for row in coefficients
        ]
        return [Rate(item, rate) for item, rate in zip(symbols, rates, strict=True)]

    return laws


def _derivatives(model: Any, parts: list[Array], dt: float) -> tuple[list[Array], list[Array]]:  # noqa: ANN401
    """Return the derivatives a model predicts and those its differentiation method estimates from the series."""
    usable = [values for values in parts if len(values) >= DIFFERENCE_POINTS]
    predicted = [np.asarray(model.predict(values), dtype=float) for values in usable]
    measured = [np.asarray(model.differentiation_method(values, dt), dtype=float) for values in usable]
    return predicted, measured


@dataclass(frozen=True)
class SparseModel(Surrogate):
    """Sparse symbolic surrogate fitted by sparse identification of nonlinear dynamics.

    Args:
        system:
            The system of one surrogate element under continuous time, with one rate law per variable.

        labels:
            The variables the surrogate was fitted to, in the order of its states.

        states:
            The names of the states of the surrogate element.

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
    """

    features: list[str]
    coefficients: Array


@analysis(DataKind.SERIES)
def fit_sindy(
    data: SurrogateData,
    names: Sequence[str] | None = None,
    *,
    degree: int = 2,
    threshold: float = 0.1,
    alpha: float = 0.05,
    validation: float = 0.2,
    label: str = 'surrogate',
) -> SparseModel:
    """Return the sparse symbolic surrogate that SINDy fits to series, computed with PySINDy.

    SINDy writes the derivative of every variable as a sparse sum of the monomials of the variables up to a degree. It
    estimates the derivatives by centred finite differences and selects the terms by sequentially thresholded least
    squares. The surrogate holds one rate law per variable and starts at the first sample of the first series.

    Args:
        data:
            A trajectory, a series, or a sequence of them, sampled at one uniform step.

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

    Returns:
        model:
            The surrogate system, its errors, its candidate monomials, and their coefficients.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import Series, fit_sindy
        >>> times = np.arange(0.0, 2.0, 0.001)
        >>> model = fit_sindy(Series(times, np.exp(-2 * times)[:, None], ['x']), degree=1)
        >>> model.features, model.coefficients.round(3).tolist()
        (['1', 'x'], [[0.0, -2.0]])
    """
    samples = samples_of(data, names)
    states = state_names(samples.labels)
    training, held = split_samples(samples.series, validation)
    sindy = module('pysindy')
    model = sindy.SINDy(
        optimizer=sindy.STLSQ(threshold=threshold, alpha=alpha),
        feature_library=sindy.PolynomialLibrary(degree=degree),
    ).fit(training, t=samples.dt, feature_names=states)
    coefficients = np.asarray(model.coefficients(), dtype=float)
    powers = np.asarray(model.feature_library.powers_, dtype=int)
    system = surrogate_system(
        'SparseSurrogate',
        label,
        dict(zip(states, map(float, samples.series[0][0]), strict=True)),
        _sparse_laws(states, coefficients, powers),
        ContinuousTime(),
    )
    return SparseModel(
        system,
        samples.labels,
        states,
        samples.dt,
        relative_error(*_derivatives(model, training, samples.dt)),
        relative_error(*_derivatives(model, held, samples.dt)),
        [str(name) for name in model.get_feature_names()],
        coefficients,
    )
