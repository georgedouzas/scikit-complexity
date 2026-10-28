"""Fit linear surrogates to series by dynamic mode decomposition, on the variables or on a dictionary of observables."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Self

import numpy as np
import numpy.typing as npt
from sklearn.base import TransformerMixin, clone

from ..core.laws import BaseLaw, RateLaw
from ..core.spaces import ContinuousTime
from ..engine.problems import vector
from ._base import BaseAnalysis
from ._kinds import DataKind
from ._surrogate_systems import (
    Samples,
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
type ComplexArray = npt.NDArray[np.complex128]


def _lift(samples: Samples, dictionary: TransformerMixin | None) -> tuple[list[Array], list[str], list[str]]:
    """Return the series of the observables, their labels, and their state names, the variables without a dictionary."""
    names = _state_names(samples.labels)
    if dictionary is None:
        return samples.series, samples.labels, names
    fitted = clone(dictionary).fit(np.concatenate(samples.series))
    lifted = [np.asarray(fitted.transform(values), dtype=float) for values in samples.series]
    labels = [str(name) for name in fitted.get_feature_names_out(samples.labels)]
    return lifted, labels, _state_names([str(name) for name in fitted.get_feature_names_out(names)])


def _increments(decomposition: object, parts: list[Array]) -> tuple[list[Array], list[Array]]:
    """Return the increments over one sample that a decomposition predicts and those of the series."""
    predicted = [np.real(decomposition.predict(values[:-1].T)).T - values[:-1] for values in parts]
    return predicted, [np.diff(values, axis=0) for values in parts]


def _linear_laws(names: Sequence[str], generator: Array) -> Callable[[SimpleNamespace], list[BaseLaw]]:
    """Return the function that gives the rate law `dg/dt = L g` of the states named."""

    def laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the matrix rate law."""
        symbols = _symbols_of(s, names)
        rows = [
            sum(float(value) * item for value, item in zip(row, symbols, strict=True) if value) for row in generator
        ]
        return [RateLaw(vector(symbols), vector(rows))]

    return laws


class ModeDecompositionAnalysis(BaseAnalysis):
    """Linear system fitted to series by dynamic mode decomposition, computed with PyDMD.

    The decomposition fits the exact DMD of the one-sample map `g(t + dt) = A g(t)` to the pairs of consecutive samples
    of every run. With a dictionary, `g` holds the observables the dictionary computes from the variables, which is
    extended DMD, and the dictionary should keep the variables among its observables. The fitted system follows the
    rate law `dg/dt = L g` with `L = Phi diag(ln(mu) / dt) Phi^+` under continuous time and starts at the first sample
    of the first run. The forecast sums the modes from the last sample of the last run.

    Args:
        variables:
            The variables of trajectories, or `None` for every scalar state or observable.

        rank:
            The rank of the truncated singular value decomposition: a number of modes, 0 for the optimal hard
            threshold, a share of the energy below 1, or `None` to keep every singular value.

        dictionary:
            A scikit-learn transformer that computes the observables, such as `PolynomialFeatures(2)`, or `None` for
            the variables themselves.

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
            The system of one surrogate element under continuous time, whose states follow `dg/dt = L g`.

        labels_:
            The variables, or the observables of the dictionary, in the order of the states.

        states_:
            The names of the states of the surrogate element.

        last_state_:
            The values of the states at the last sample of the last run, where the forecast starts.

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
            The amplitudes of the modes at the last sample of the last run.

        generator_:
            The matrix `L` of the rate law, one row and one column per state.

        forecast_:
            The states at the `horizon` samples after the last one, one row per sample.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import ModeDecompositionAnalysis
        >>> times = np.arange(0.0, 10.0, 0.1)
        >>> values = np.column_stack([np.cos(times), np.sin(times)])
        >>> analysis = ModeDecompositionAnalysis(['x', 'y'], horizon=5, dt=0.1).check().prepare(values).run()
        >>> np.sort(analysis.eigenvalues_.imag).round(6).tolist()
        [-1.0, 1.0]
        >>> bool(np.allclose(analysis.forecast_[:, 0], np.cos(times[-1] + 0.1 * np.arange(1, 6))))
        True
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        rank: float | None = None,
        dictionary: TransformerMixin | None = None,
        validation: float = 0.2,
        horizon: int = 0,
        dt: float = 1.0,
        label: str = 'surrogate',
    ) -> None:
        self.variables = variables
        self.rank = rank
        self.dictionary = dictionary
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
        """Fit the decomposition and set the surrogate system, its errors, its modes, and the forecast."""
        samples = self.samples_
        lifted, labels, states = _lift(samples, self.dictionary)
        training, held = _split_samples(lifted, self.validation)
        before = np.concatenate([values[:-1] for values in training]).T
        after = np.concatenate([values[1:] for values in training]).T
        rank = -1 if self.rank is None else self.rank
        decomposition = self._import('pydmd').DMD(svd_rank=rank, exact=True).fit(before, after)
        modes = np.asarray(decomposition.modes, dtype=complex)
        eigenvalues: ComplexArray = np.asarray(
            np.log(np.asarray(decomposition.eigs, dtype=complex)) / samples.dt,
            dtype=complex,
        )
        generator = np.real(modes @ np.diag(eigenvalues) @ np.linalg.pinv(modes))
        self.system_ = _surrogate_system(
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
        self.training_error_ = _relative_error(*_increments(decomposition, training))
        self.validation_error_ = _relative_error(*_increments(decomposition, held))
        self.eigenvalues_ = eigenvalues
        self.modes_ = modes
        self.amplitudes_ = np.linalg.lstsq(modes, self.last_state_.astype(complex), rcond=None)[0]
        self.generator_ = generator
        growth = np.exp(np.outer(self.dt_ * np.arange(1, self.horizon + 1), self.eigenvalues_))
        self.forecast_ = np.real((growth * self.amplitudes_) @ self.modes_.T).reshape(self.horizon, len(states))
        return self
