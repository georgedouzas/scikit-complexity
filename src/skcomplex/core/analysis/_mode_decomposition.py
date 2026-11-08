"""Fit linear surrogates to series by dynamic mode decomposition.

The fit is on the variables or on a dictionary of observables.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from sklearn.base import TransformerMixin, clone

from ..components.laws import BaseLaw, RateLaw
from ..components.spaces import ContinuousTime
from ..components.system import System
from ..engine._utils import vector
from ..errors import AnalysisError
from ..specs import AnalysisTags, SamplesSpec
from ..types import Array, ComplexArray, DataKind
from ._base import _import, _read_samples
from ._sparse_identification import (
    _check_horizon,
    _check_validation,
    _relative_error,
    _split_samples,
    _state_names,
    _surrogate_system,
    _symbols_of,
)
from .types import Runs

SMALL_EXPONENT = 1e-12

TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class ModeDecompositionResult:
    """Linear system fitted to series by dynamic mode decomposition.

    Args:
        samples:
            The series of every run, the names of the variables, and the step.

        system:
            The system of one surrogate element under continuous time, whose states follow `dg/dt = L g + M u`.

        labels:
            The variables, or the observables of the dictionary, in the order of the states.

        states:
            The names of the states of the surrogate element.

        last_state:
            The values of the states at the last sample of the last run, where the forecast starts.

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
            The amplitudes of the modes at the last sample of the last run.

        generator:
            The matrix `L` of the rate law, one row and one column per state.

        input_generator:
            The matrix `M` of the rate law, one row per state and one column per input.

        operator:
            The matrix `A` of the one-sample map, one row and one column per state.

        input_matrix:
            The matrix `B` of the one-sample map, one row per state and one column per input.

        dictionary:
            The fitted dictionary, which lifts new states to the observables, or `None` without a dictionary.

        inputs:
            The names of the adaptive parameters of the surrogate element that hold the inputs.

        forecast:
            The states at the `horizon` samples after the last one, one row per sample.
    """

    samples: SamplesSpec
    system: System
    labels: list[str]
    states: list[str]
    last_state: Array
    dt: float
    training_error: float
    validation_error: float
    eigenvalues: ComplexArray
    modes: ComplexArray
    amplitudes: ComplexArray
    generator: Array
    input_generator: Array
    operator: Array
    input_matrix: Array
    dictionary: TransformerMixin | None
    inputs: list[str]
    forecast: Array


def mode_decomposition(
    source: Runs,
    *,
    variables: Sequence[str] | None = None,
    rank: float | None = None,
    dictionary: TransformerMixin | None = None,
    inputs: Sequence[str] | None = None,
    validation: float = 0.2,
    horizon: int = 0,
    dt: float = 1.0,
    label: str = 'surrogate',
) -> ModeDecompositionResult:
    """Return the linear system fitted to series by dynamic mode decomposition.

    It is computed with PyDMD. The decomposition fits the exact DMD of the one-sample map `g(t + dt) = A g(t)` to the
    pairs of consecutive samples of every run. With a dictionary, `g` holds the observables the dictionary computes from
    the variables, which is extended DMD, and the dictionary should keep the variables among its observables. The fitted
    system follows the rate law `dg/dt = L g` with `L = Phi diag(ln(mu) / dt) Phi^+` under continuous time and starts at
    the first sample of the first run. The forecast iterates the map from the last sample of the last run.

    With inputs, the decomposition is dynamic mode decomposition with control, `g(t + dt) = A g(t) + B u(t)`, where `g`
    holds the variables that are not inputs or their observables. The fitted system follows `dg/dt = L g + M u`, where
    `M` gives the same one-sample map under an input held over the sample. Every input is a bounded adaptive parameter
    of the surrogate element, bounded by its range in the samples, and the forecast holds the inputs at their last
    values.

    Args:
        source:
            A trajectory, an array with one row per sample, or several runs as a list or a 3-D array.

        variables:
            The variables of trajectories, or `None` for every scalar state or observable. The inputs join them when
            they are not listed.

        rank:
            The rank of the truncated singular value decomposition: a number of modes, 0 for the optimal hard
            threshold, a share of the energy below 1, or `None` to keep every singular value.

        dictionary:
            A scikit-learn transformer that computes the observables, such as `PolynomialFeatures(2)`, or `None` for
            the variables themselves.

        inputs:
            The names of the variables that are control inputs, or `None` for none.

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
            The fitted system, its errors, its modes and matrices, and the forecast.

    Raises:
        AnalysisError:
            The validation share lies outside `[0, 1)`, the horizon is negative, an input is not a variable, or the
            runs differ in variables or step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import mode_decomposition
        >>> times = np.arange(0.0, 10.0, 0.1)
        >>> values = np.column_stack([np.cos(times), np.sin(times)])
        >>> result = mode_decomposition(values, variables=['x', 'y'], horizon=5, dt=0.1)
        >>> np.sort(result.eigenvalues.imag).round(6).tolist()
        [-1.0, 1.0]
        >>> bool(np.allclose(result.forecast[:, 0], np.cos(times[-1] + 0.1 * np.arange(1, 6))))
        True
        >>> u = np.random.default_rng(0).uniform(-1.0, 1.0, 100)
        >>> x = np.zeros(100)
        >>> for k in range(99):
        ...     x[k + 1] = 0.9 * x[k] + 0.5 * u[k]
        >>> result = mode_decomposition(np.column_stack([x, u]), variables=['x', 'u'], inputs=['u'], validation=0.0)
        >>> result.operator.round(6).tolist(), result.input_matrix.round(6).tolist()
        ([[0.9]], [[0.5]])
    """
    _check_validation(validation)
    _check_horizon(horizon)
    added = [name for name in inputs or () if name not in (variables or ())]
    read = None if variables is None else [*variables, *added]
    all_samples = _read_samples(source, read, dt)
    samples, input_labels, controls = _split_inputs(all_samples, inputs)
    fitted = None if dictionary is None else clone(dictionary).fit(np.concatenate(samples.series))
    lifted, labels, states = _lift(samples, fitted)
    training, held = _split_samples(lifted, validation)
    driven, driven_held = _split_samples(controls, validation)
    before = np.concatenate([values[:-1] for values in training]).T
    after = np.concatenate([values[1:] for values in training]).T
    applied = np.concatenate([values[:-1] for values in driven]).T
    operator, input_matrix, modes, multipliers = _fit_operators(before, after, applied, rank)
    eigenvalues: ComplexArray = np.asarray(np.log(multipliers) / samples.dt, dtype=complex)
    generator = np.real(modes @ np.diag(eigenvalues) @ np.linalg.pinv(modes))
    input_generator = _input_generator(modes, multipliers, eigenvalues, input_matrix, samples.dt)
    names = _input_names(input_labels, states)
    system = _surrogate_system(
        label,
        dict(zip(states, map(float, lifted[0][0]), strict=True)),
        _linear_laws(states, generator, names, input_generator),
        ContinuousTime(),
        inputs=_input_declarations(names, controls),
    )
    last_state = np.asarray(lifted[-1][-1], dtype=float)
    return ModeDecompositionResult(
        samples=all_samples,
        system=system,
        labels=labels,
        states=states,
        last_state=last_state,
        dt=samples.dt,
        training_error=_relative_error(*_increments(operator, input_matrix, training, driven)),
        validation_error=_relative_error(*_increments(operator, input_matrix, held, driven_held)),
        eigenvalues=eigenvalues,
        modes=modes,
        amplitudes=np.linalg.lstsq(modes, last_state.astype(complex), rcond=None)[0],
        generator=generator,
        input_generator=input_generator,
        operator=operator,
        input_matrix=input_matrix,
        dictionary=fitted,
        inputs=names,
        forecast=_forecast(last_state, operator, input_matrix, controls[-1][-1], horizon),
    )


def _split_inputs(samples: SamplesSpec, inputs: Sequence[str] | None) -> tuple[SamplesSpec, list[str], list[Array]]:
    """Return the samples without the inputs, and the labels and series of the inputs.

    Args:
        samples:
            The series of every run with the inputs among the variables.

        inputs:
            The names of the variables that are control inputs, or `None` for none.

    Returns:
        parts:
            The samples of the other variables, the labels of the inputs, and one array per run, with one column per
            input.

    Raises:
        AnalysisError:
            An input is not a variable of the series.
    """
    names = list(inputs or ())
    missing = sorted(set(names) - set(samples.labels))
    if missing:
        error_msg = f'The inputs {missing} are not variables of the series, which are {samples.labels}.'
        raise AnalysisError(error_msg)
    others = [index for index, label in enumerate(samples.labels) if label not in names]
    columns = [samples.labels.index(name) for name in names]
    labels = [samples.labels[index] for index in others]
    return (
        SamplesSpec([values[:, others] for values in samples.series], labels, samples.dt),
        names,
        [values[:, columns] for values in samples.series],
    )


def _fit_operators(
    before: Array,
    after: Array,
    controls: Array,
    rank: float | None,
) -> tuple[Array, Array, ComplexArray, ComplexArray]:
    """Return the matrices `A` and `B`, modes, and eigenvalues of the one-sample map.

    They are fitted with PyDMD.

    Args:
        before:
            The states before every pair of samples, one column per pair.

        after:
            The states after every pair of samples, one column per pair.

        controls:
            The inputs of every pair of samples, one column per pair and one row per input.

        rank:
            The rank of the truncated singular value decomposition, or `None` to keep every singular value.

    Returns:
        operators:
            The matrices `A` and `B`, the modes, and the eigenvalues of `A`.
    """
    svd_rank = -1 if rank is None else rank
    pydmd = _import('pydmd')
    if not len(controls):
        fitted = pydmd.DMD(svd_rank=svd_rank, exact=True).fit(before, after)
        modes = np.asarray(fitted.modes, dtype=complex)
        eigenvalues = np.asarray(fitted.eigs, dtype=complex)
        operator = np.real(modes @ np.diag(eigenvalues) @ np.linalg.pinv(modes))
        return operator, np.zeros((len(before), 0)), modes, eigenvalues
    rank_with_inputs = svd_rank + len(controls) if svd_rank >= 1 else svd_rank
    fitted = pydmd.DMDc(svd_rank=svd_rank, svd_rank_omega=rank_with_inputs, lag=before.shape[1])
    fitted.fit(np.hstack([before, after]), controls)
    basis = np.asarray(fitted.basis)
    operator = np.real(basis @ np.asarray(fitted.operator.as_numpy_array) @ basis.conj().T)
    modes = np.asarray(fitted.modes, dtype=complex)
    return operator, np.real(np.asarray(fitted.B)), modes, np.asarray(fitted.eigs, dtype=complex)


def _forecast(last_state: Array, operator: Array, input_matrix: Array, last_input: Array, horizon: int) -> Array:
    """Return the states at the samples of the horizon.

    The inputs are held at their last values.
    """
    current = last_state
    rows = []
    for _ in range(horizon):
        current = operator @ current + input_matrix @ last_input
        rows.append(current)
    return np.asarray(rows, dtype=float).reshape(horizon, len(current))


def _input_generator(
    modes: ComplexArray,
    multipliers: ComplexArray,
    eigenvalues: ComplexArray,
    input_matrix: Array,
    dt: float,
) -> Array:
    """Return the matrix `M` of the rate law that gives the matrix `B`.

    The flow of the rate law over a sample under a held input gives `B`.
    """
    slow = np.abs(eigenvalues) * dt < SMALL_EXPONENT
    safe = np.where(slow, 1.0, eigenvalues)
    weights = np.where(slow, dt, (multipliers - 1.0) / safe)
    integral = np.real(modes @ np.diag(weights) @ np.linalg.pinv(modes))
    return np.linalg.lstsq(integral, input_matrix, rcond=None)[0]


def _input_names(labels: Sequence[str], states: Sequence[str]) -> list[str]:
    """Return the parameter names of the inputs.

    They differ from the names of the states.
    """
    names = _state_names(labels)
    return [f'{name}_input' if name in states else name for name in names]


def _input_declarations(
    names: Sequence[str],
    controls: Sequence[Array],
) -> dict[str, tuple[float, tuple[float, float]]]:
    """Return the first value and the range in the samples of every input, by name."""
    joined = np.concatenate(controls)
    return {
        name: (float(controls[0][0, index]), (float(joined[:, index].min()), float(joined[:, index].max())))
        for index, name in enumerate(names)
    }


def _lift(samples: SamplesSpec, fitted: TransformerMixin | None) -> tuple[list[Array], list[str], list[str]]:
    """Return the series, labels, and state names of the observables.

    Without a dictionary, the observables are the variables.
    """
    names = _state_names(samples.labels)
    if fitted is None:
        return samples.series, samples.labels, names
    lifted = [np.asarray(fitted.transform(values), dtype=float) for values in samples.series]
    labels = [str(name) for name in fitted.get_feature_names_out(samples.labels)]
    return (
        lifted,
        labels,
        _state_names([str(name) for name in fitted.get_feature_names_out(names)]),
    )


def _increments(
    operator: Array,
    input_matrix: Array,
    parts: list[Array],
    controls: list[Array],
) -> tuple[list[Array], list[Array]]:
    """Return the increments over one sample that the map predicts and the series has.

    The map is the one-sample map.
    """
    predicted = [
        values[:-1] @ (operator - np.eye(len(operator))).T + driven[:-1] @ input_matrix.T
        for values, driven in zip(parts, controls, strict=True)
    ]
    return predicted, [np.diff(values, axis=0) for values in parts]


def _linear_laws(
    names: Sequence[str],
    generator: Array,
    inputs: Sequence[str],
    input_generator: Array,
) -> Callable[[SimpleNamespace], list[BaseLaw]]:
    """Return the function that gives the rate law `dg/dt = L g + M u`.

    The rate law is that of the states named.
    """
    matrix = np.hstack([generator, input_generator])

    def laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the matrix rate law."""
        symbols = _symbols_of(s, names)
        read = [*symbols, *_symbols_of(s, inputs)]
        rows = [sum(float(value) * item for value, item in zip(row, read, strict=True) if value) for row in matrix]
        return [RateLaw(vector(symbols), vector(rows))]

    return laws
