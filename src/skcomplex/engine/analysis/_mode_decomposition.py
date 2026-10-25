"""Fit linear surrogates to series by dynamic mode decomposition, on the variables or on a dictionary of observables."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import TransformerMixin, clone

from ...core.base import BaseLaw, vector
from ...core.laws import RateLaw, Symbols
from ...core.space import ContinuousTime
from ._lazy import module
from ._surrogate_systems import (
    BaseSurrogate,
    Samples,
    SurrogateData,
    relative_error,
    samples_of,
    split_samples,
    state_names,
    surrogate_system,
    symbols_of,
)

type Array = npt.NDArray[np.float64]
type ComplexArray = npt.NDArray[np.complex128]


def _lift(samples: Samples, dictionary: TransformerMixin | None) -> tuple[list[Array], list[str], list[str]]:
    """Return the series of the observables, their labels, and their state names, the variables without a dictionary."""
    names = state_names(samples.labels)
    if dictionary is None:
        return samples.series, samples.labels, names
    fitted = clone(dictionary).fit(np.concatenate(samples.series))
    lifted = [np.asarray(fitted.transform(values), dtype=float) for values in samples.series]
    labels = [str(name) for name in fitted.get_feature_names_out(samples.labels)]
    return lifted, labels, state_names([str(name) for name in fitted.get_feature_names_out(names)])


def _increments(decomposition: Any, parts: list[Array]) -> tuple[list[Array], list[Array]]:  # noqa: ANN401
    """Return the increments over one sample that a decomposition predicts and those of the series."""
    predicted = [np.real(decomposition.predict(values[:-1].T)).T - values[:-1] for values in parts]
    return predicted, [np.diff(values, axis=0) for values in parts]


def _linear_laws(names: Sequence[str], generator: Array) -> Callable[[Symbols], list[BaseLaw]]:
    """Return the function that gives the rate law `dg/dt = L g` of the states named."""

    def laws(s: Symbols) -> list[BaseLaw]:
        """Return the matrix rate law."""
        symbols = symbols_of(s, names)
        rows = [
            sum(float(value) * item for value, item in zip(row, symbols, strict=True) if value) for row in generator
        ]
        return [RateLaw(vector(symbols), vector(rows))]

    return laws


class DMDSurrogate(BaseSurrogate):
    """Linear surrogate fitted by dynamic mode decomposition, computed with PyDMD.

    The decomposition fits the exact DMD of the one-sample map `g(t + dt) = A g(t)` to the pairs of consecutive samples
    of every series. With a dictionary, `g` holds the observables the dictionary computes from the variables, which is
    extended DMD, and the dictionary should keep the variables among its observables. The surrogate follows the rate
    law `dg/dt = L g` with `L = Phi diag(ln(mu) / dt) Phi^+` under continuous time and starts at the first sample of the
    first series. A forecast sums the modes from the last sample of the last series.

    Args:
        names:
            The variables of trajectories, or `None` for every scalar state or observable.

        rank:
            The rank of the truncated singular value decomposition: a number of modes, 0 for the optimal hard
            threshold, a share of the energy below 1, or `None` to keep every singular value.

        dictionary:
            A scikit-learn transformer that computes the observables, such as `PolynomialFeatures(2)`, or `None` for
            the variables themselves.

        validation:
            The share of the samples of every series held out at its end to measure the validation error.

        label:
            The label of the surrogate element.

    Attributes:
        system_:
            The system of one surrogate element under continuous time, whose states follow `dg/dt = L g`.

        labels_:
            The variables, or the observables of the dictionary, in the order of the states.

        states_:
            The names of the states of the surrogate element.

        last_state_:
            The values of the states at the last sample of the last series, where forecasts start.

        dt_:
            The time between the samples it was fitted to.

        training_error_:
            The relative error of the predicted increments over one sample on the training samples.

        validation_error_:
            The relative error on the held-out samples, NaN when none are held out.

        eigenvalues_:
            The continuous eigenvalues `ln(mu) / dt` of the modes, from the eigenvalues `mu` of the one-sample map.

        modes_:
            The modes, one column per mode and one row per state.

        amplitudes_:
            The amplitudes of the modes at the last sample of the last series.

        generator_:
            The matrix `L` of the rate law, one row and one column per state.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.analysis import DMDSurrogate, Series
        >>> times = np.arange(0.0, 10.0, 0.1)
        >>> series = Series(times, np.column_stack([np.cos(times), np.sin(times)]), ['x', 'y'])
        >>> surrogate = DMDSurrogate().prepare(series)
        >>> np.sort(surrogate.eigenvalues_.imag).round(6).tolist()
        [-1.0, 1.0]
        >>> bool(np.allclose(surrogate.run(5)[:, 0], np.cos(times[-1] + 0.1 * np.arange(1, 6))))
        True
    """

    eigenvalues_: ComplexArray
    modes_: ComplexArray
    amplitudes_: ComplexArray
    generator_: Array

    def __init__(
        self,
        names: Sequence[str] | None = None,
        *,
        rank: float | None = None,
        dictionary: TransformerMixin | None = None,
        validation: float = 0.2,
        label: str = 'surrogate',
    ) -> None:
        self.names = names
        self.rank = rank
        self.dictionary = dictionary
        self.validation = validation
        self.label = label

    def _prepare(self, data: SurrogateData) -> None:
        """Fit the decomposition to the series and set the surrogate system, its errors, and its modes."""
        samples = samples_of(data, self.names)
        lifted, labels, states = _lift(samples, self.dictionary)
        training, held = split_samples(lifted, self.validation)
        before = np.concatenate([values[:-1] for values in training]).T
        after = np.concatenate([values[1:] for values in training]).T
        rank = -1 if self.rank is None else self.rank
        decomposition = module('pydmd').DMD(svd_rank=rank, exact=True).fit(before, after)
        modes = np.asarray(decomposition.modes, dtype=complex)
        eigenvalues: ComplexArray = np.asarray(
            np.log(np.asarray(decomposition.eigs, dtype=complex)) / samples.dt,
            dtype=complex,
        )
        generator = np.real(modes @ np.diag(eigenvalues) @ np.linalg.pinv(modes))
        self.system_ = surrogate_system(
            'LinearSurrogate',
            self.label,
            dict(zip(states, map(float, lifted[0][0]), strict=True)),
            _linear_laws(states, generator),
            ContinuousTime(),
        )
        self.labels_ = labels
        self.states_ = states
        self.last_state_ = np.asarray(lifted[-1][-1], dtype=float)
        self.dt_ = samples.dt
        self.training_error_ = relative_error(*_increments(decomposition, training))
        self.validation_error_ = relative_error(*_increments(decomposition, held))
        self.eigenvalues_ = eigenvalues
        self.modes_ = modes
        self.amplitudes_ = np.linalg.lstsq(modes, self.last_state_.astype(complex), rcond=None)[0]
        self.generator_ = generator

    def _run(self, steps: int) -> Array:
        """Return the states the modes give at every sample after the last one, one row per sample."""
        growth = np.exp(np.outer(self.dt_ * np.arange(1, steps + 1), self.eigenvalues_))
        return np.real((growth * self.amplitudes_) @ self.modes_.T)
