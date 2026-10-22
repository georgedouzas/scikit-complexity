"""Fit linear surrogates to series by dynamic mode decomposition, on the variables or on a dictionary of observables."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import TransformerMixin, clone

from ..base import vector
from ..laws import Law, Rate, Symbols
from ..space import ContinuousTime
from ._kinds import DataKind, analysis
from ._lazy import module
from ._surrogates import (
    Samples,
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


def _linear_laws(names: Sequence[str], generator: Array) -> Callable[[Symbols], list[Law]]:
    """Return the function that gives the rate law `dg/dt = L g` of the states named."""

    def laws(s: Symbols) -> list[Law]:
        """Return the matrix rate law."""
        symbols = symbols_of(s, names)
        rows = [
            sum(float(value) * item for value, item in zip(row, symbols, strict=True) if value) for row in generator
        ]
        return [Rate(vector(symbols), vector(rows))]

    return laws


@dataclass(frozen=True)
class ModeDecomposition(Surrogate):
    """Linear surrogate fitted by dynamic mode decomposition, with its modes and continuous eigenvalues.

    Args:
        system:
            The system of one surrogate element under continuous time, whose states follow `dg/dt = L g`.

        labels:
            The variables, or the observables of the dictionary, in the order of the states.

        states:
            The names of the states of the surrogate element.

        dt:
            The time between the samples it was fitted to.

        training_error:
            The relative error of the predicted increments over one sample on the training samples.

        validation_error:
            The relative error on the held-out samples, NaN when none are held out.

        eigenvalues:
            The continuous eigenvalues `ln(mu) / dt` of the modes, from the eigenvalues `mu` of the one-sample map.

        modes:
            The modes, one column per mode and one row per state.

        amplitudes:
            The amplitudes of the modes at the first sample of the first series.

        generator:
            The matrix `L` of the rate law, one row and one column per state.
    """

    eigenvalues: ComplexArray
    modes: ComplexArray
    amplitudes: ComplexArray
    generator: Array

    def forecast(self, times: npt.ArrayLike) -> Array:
        """Return the states the modes give at times after the first sample of the first series.

        Args:
            times:
                The times since the first sample.

        Returns:
            states:
                One row per time and one column per state.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.analysis import Series, fit_dmd
            >>> times = np.arange(0.0, 5.0, 0.1)
            >>> series = Series(times, np.exp(-times)[:, None], ['x'])
            >>> fit_dmd(series).forecast([0.0, 1.0]).round(4).tolist()
            [[1.0], [0.3679]]
        """
        growth = np.exp(np.outer(np.asarray(times, dtype=float), self.eigenvalues))
        return np.real((growth * self.amplitudes) @ self.modes.T)


@analysis(DataKind.SERIES)
def fit_dmd(
    data: SurrogateData,
    names: Sequence[str] | None = None,
    *,
    rank: float | None = None,
    dictionary: TransformerMixin | None = None,
    validation: float = 0.2,
    label: str = 'surrogate',
) -> ModeDecomposition:
    """Return the linear surrogate that dynamic mode decomposition fits to series, computed with PyDMD.

    The decomposition fits the exact DMD of the one-sample map `g(t + dt) = A g(t)` to the pairs of consecutive samples
    of every series. With a dictionary, `g` holds the observables the dictionary computes from the variables, which is
    extended DMD, and the dictionary should keep the variables among its observables. The surrogate follows the rate
    law `dg/dt = L g` with `L = Phi diag(ln(mu) / dt) Phi^+` and starts at the first sample of the first series.

    Args:
        data:
            A trajectory, a series, or a sequence of them, sampled at one uniform step.

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

    Returns:
        decomposition:
            The surrogate system, its errors, its continuous eigenvalues, modes, and amplitudes, and its generator.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import Series, fit_dmd
        >>> times = np.arange(0.0, 10.0, 0.1)
        >>> series = Series(times, np.column_stack([np.cos(times), np.sin(times)]), ['x', 'y'])
        >>> np.sort(fit_dmd(series).eigenvalues.imag).round(6).tolist()
        [-1.0, 1.0]
    """
    samples = samples_of(data, names)
    lifted, labels, states = _lift(samples, dictionary)
    training, held = split_samples(lifted, validation)
    before = np.concatenate([values[:-1] for values in training]).T
    after = np.concatenate([values[1:] for values in training]).T
    decomposition = module('pydmd').DMD(svd_rank=-1 if rank is None else rank, exact=True).fit(before, after)
    modes = np.asarray(decomposition.modes, dtype=complex)
    eigenvalues = np.log(np.asarray(decomposition.eigs, dtype=complex)) / samples.dt
    generator = np.real(modes @ np.diag(eigenvalues) @ np.linalg.pinv(modes))
    amplitudes = np.linalg.lstsq(modes, lifted[0][0].astype(complex), rcond=None)[0]

    system = surrogate_system(
        'LinearSurrogate',
        label,
        dict(zip(states, map(float, lifted[0][0]), strict=True)),
        _linear_laws(states, generator),
        ContinuousTime(),
    )
    return ModeDecomposition(
        system,
        labels,
        states,
        samples.dt,
        relative_error(*_increments(decomposition, training)),
        relative_error(*_increments(decomposition, held)),
        eigenvalues,
        modes,
        amplitudes,
        generator,
    )
