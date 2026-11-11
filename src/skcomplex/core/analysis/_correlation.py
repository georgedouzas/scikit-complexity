"""Measure correlations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import optimize

from ..engine.runtime import Trajectory
from ..enums import DataKind
from ..errors import AnalysisError
from ..specs import AnalysisTags, SeriesSpec
from ..types import Array
from ._base import STEP_TOLERANCE, _check_choice, _check_settings, _read_fields, _read_real, _read_series, _step_of
from .types import Source

MIN_LINE_POINTS = 2
MIN_FIT_POINTS = 3
KINDS = ('auto', 'two-time', 'function', 'length', 'overlap')
METHODS = ('moment', 'fit')
SETTINGS = {
    'auto': {'max_lag', 'dt'},
    'two-time': {'waiting', 'lags', 'connected'},
    'function': {'max_distance'},
    'length': {'method', 'spacing', 'frames'},
    'overlap': set(),
}
MOMENT_ITERATIONS = 8
REPLICAS = 2
TAGS = AnalysisTags(DataKind.FIELD)
KIND_TAGS = {
    'auto': AnalysisTags(DataKind.SERIES),
    'two-time': AnalysisTags(DataKind.ENSEMBLE),
}


@dataclass(frozen=True)
class CorrelationResult:
    """A result of `correlation`, the correlation of series or of a field.

    It covers every series of a source.

    Args:
        values:
            The series, one per column, the ensemble, the field, or the two configurations.

        labels:
            The label of every series, of the ensemble or field, or of the two configurations.

        lags:
            The lags in time, for `'auto'`.

        correlation:
            One autocorrelation per series for `'auto'`, one row per waiting time and one column per lag for
            `'two-time'`, and the correlation at every distance for `'function'`.

        time:
            The integrated autocorrelation time of every series, for `'auto'`.

        distances:
            The distances, for `'function'`.

        length:
            The correlation length, for `'length'`.

        overlap:
            The overlap of the two configurations, one per frame for frames, for `'overlap'`.
    """

    values: Array
    labels: list[str]
    lags: Array | None = None
    correlation: Array | None = None
    time: Array | None = None
    distances: Array | None = None
    length: float | Array | None = None
    overlap: float | Array | None = None


def correlation(
    source: Source,
    *,
    kind: str = 'auto',
    variables: Sequence[str] | None = None,
    max_lag: int | None = None,
    dt: float | None = None,
    waiting: Sequence[int] | None = None,
    lags: Sequence[int] | None = None,
    connected: bool = False,
    max_distance: int | None = None,
    method: str = 'moment',
    spacing: float = 1.0,
    frames: bool = False,
) -> CorrelationResult:
    """Return the correlation of series, of an ensemble at two times, or of a field.

    The correlation of a field is in space. It covers every series of a source.

    The kinds read series, ensembles, or fields:

    - `'auto'`: the autocorrelation of every series at lags 0 to `max_lag`, computed with the FFT, and its integrated
      autocorrelation time `dt (1/2 + Σ r(k))`, summed to the first negative correlation.
    - `'two-time'`: the two-time correlation `C(t_w + tau, t_w) = <x(t_w) x(t_w + tau)>` of an ensemble with one row
      per run and one column per frame, any further axes, such as sites, averaged over.
    - `'function'`: the radially averaged spatial correlation of a field on a periodic grid, 1 at distance 0.
    - `'length'`: the correlation length of a field. The `'moment'` method fits the Ornstein-Zernike form
      `1 / S(q) ∝ 1 + xi² Σ 4 sin²(q_i / 2)` over the modes with `q̂² xi² <= 1`, and the `'fit'` method fits
      `C(r) ~ r^-a exp(-r / xi)`. Several frames are taken as fluctuations about their common mean.
    - `'overlap'`: the overlap `q = <s_i t_i>` of two configurations `s` and `t` of the same sites, ±1 for identical or
      reversed configurations of ±1 values and near 0 for independent ones, per frame for frames. It reads a pair of
      trajectories, whose fields it compares at every frame, or two arrays stacked on the first axis.

    The kinds `'function'` and `'length'` read the last frame of a trajectory, or every frame when `frames` is true.

    It appears in many fields. It finds how far order reaches in a magnet, how long a stock price remembers its past,
    and how two regions of the brain move together.

    Args:
        source:
            The trajectory, an array of series, an ensemble, a field, or a pair of replicas for `'overlap'`.

        kind:
            The correlation to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series, or the first field for a field.

        max_lag:
            The largest lag in steps, or `None` for half the series length.

        dt:
            The time between the samples of an array, or `None` for 1. A trajectory gives its own step.

        waiting:
            The waiting times `t_w` in frames.

        lags:
            The lags `tau` in frames.

        connected:
            Whether to subtract the product of the means at the two times.

        max_distance:
            The largest distance in cells, or `None` for half the smallest side.

        method:
            The method of the correlation length, `'moment'` or `'fit'`.

        spacing:
            The distance between neighbouring sites.

        frames:
            Whether the first axis of an array indexes frames, and whether to read every frame of a trajectory.

    Returns:
        result:
            The data read and its correlation.

    Raises:
        AnalysisError:
            The kind or method is unknown, a setting the kind does not read is changed, a setting the kind needs is
            missing, a setting is not positive, or `dt` differs from the step of a trajectory.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import correlation
        >>> correlation([1.0, -1.0, 1.0, -1.0], max_lag=1).correlation.round(2).tolist()
        [[1.0, -0.75]]
        >>> correlation(np.ones((4, 10)), kind='two-time', waiting=[2], lags=[0, 3]).correlation.tolist()
        [[1.0, 1.0]]
        >>> correlation([[1, -1, 1, 1], [1, -1, -1, 1]], kind='overlap').overlap
        0.5
    """
    _check_choice('correlation', 'kind', kind, KINDS)
    _check_choice('correlation', 'method', method, METHODS)
    _check_settings(
        correlation,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        max_lag=max_lag,
        dt=dt,
        waiting=waiting,
        lags=lags,
        connected=connected,
        max_distance=max_distance,
        method=method,
        spacing=spacing,
        frames=frames,
    )
    if kind == 'two-time' and (waiting is None or lags is None):
        error_msg = 'correlation of kind two-time needs waiting times and lags. Give both in frames.'
        raise AnalysisError(error_msg)
    if (dt is not None and dt <= 0) or spacing <= 0:
        error_msg = f'correlation has dt={dt!r} and spacing={spacing!r}. Use positive values.'
        raise AnalysisError(error_msg)
    if kind == 'auto':
        series = _read_series(source, variables)
        step = _read_step(source, series, dt)
        values = series.values
        autocorrelation = np.asarray([_autocorrelation(column, max_lag) for column in values.T])
        return CorrelationResult(
            values,
            series.labels,
            lags=np.asarray(np.arange(autocorrelation.shape[1]) * step, dtype=float),
            correlation=autocorrelation,
            time=np.asarray([_autocorrelation_time(column, step) for column in values.T]),
        )
    name = variables[0] if variables else None
    if kind == 'two-time' and waiting is not None and lags is not None:
        ensemble = np.asarray(source, dtype=float)
        two_time = _two_time_correlation(ensemble, waiting, lags, connected=connected)
        return CorrelationResult(ensemble, [name or 'ensemble'], correlation=two_time)
    if kind == 'overlap':
        replicas = _read_replicas(source, name)
        product = replicas[0] * replicas[1]
        overlap = float(product.mean()) if product.ndim == 1 else product.reshape(product.shape[0], -1).mean(axis=1)
        return CorrelationResult(replicas, ['first', 'second'], overlap=overlap)
    field = _read_fields(source, name)
    if isinstance(source, Trajectory) and not frames:
        field = field[-1]
    labels = [name or 'field']
    if kind == 'function':
        distances, curve = _correlation_function(field, max_distance)
        return CorrelationResult(field, labels, correlation=curve, distances=distances)
    grids = field if frames else field[None]
    return CorrelationResult(field, labels, length=_correlation_length(grids, method, spacing))


def _read_replicas(source: Source, name: str | None) -> Array:
    """Return the fields of two trajectories, or two configurations of an array.

    The two configurations are stacked on the first axis of the array.
    """
    replicas: list[Source] = list(source) if isinstance(source, list | tuple | np.ndarray) else [source]
    if len(replicas) != REPLICAS:
        error_msg = (
            f'An overlap reads {REPLICAS} replicas, a pair of trajectories or two arrays stacked on the first '
            f'axis, but the source has {len(replicas)}. Give two replicas.'
        )
        raise AnalysisError(error_msg)
    return np.stack([_read_fields(replica, name) for replica in replicas])


def _read_step(source: Source, series: SeriesSpec, dt: float | None) -> float:
    """Return the time between samples, read from a trajectory or given for an array.

    Args:
        source:
            The trajectory or the array of series.

        series:
            The series read from the source.

        dt:
            The time between the samples of an array, or `None` for 1.

    Returns:
        step:
            The time between samples.

    Raises:
        AnalysisError:
            `dt` differs from the step of a trajectory.
    """
    if not isinstance(source, Trajectory):
        return 1.0 if dt is None else float(dt)
    step = _step_of(series)
    if dt is not None and not np.isclose(dt, step, rtol=STEP_TOLERANCE, atol=0.0):
        error_msg = (
            f'correlation has dt={dt!r}, but the trajectory is sampled every {step!r}. Leave dt unset for a '
            'trajectory, which gives its own step.'
        )
        raise AnalysisError(error_msg)
    return step


def _autocorrelation(values: npt.ArrayLike, max_lag: int | None = None) -> Array:
    """Return the autocorrelation of a series by the FFT.

    The lags run from 0 to `max_lag`, or to half its length.
    """
    array = _read_real(values, 'values')
    array = array - array.mean()
    size = array.size
    spectrum = np.fft.rfft(array, n=2 * size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[:size]
    correlation /= correlation[0] if correlation[0] else 1.0
    return correlation[: (size // 2 if max_lag is None else max_lag) + 1]


def _autocorrelation_time(values: npt.ArrayLike, dt: float = 1.0) -> float:
    """Return the integrated autocorrelation time `dt (1/2 + Σ r(k))`.

    The sum runs to the first negative correlation.
    """
    correlation = _autocorrelation(values)
    negative = np.flatnonzero(correlation <= 0)
    end = int(negative[0]) if negative.size else correlation.size
    return float(dt * (correlation[:end].sum() - 0.5))


def _two_time_correlation(
    ensemble: Array,
    waiting: Sequence[int],
    lags: Sequence[int],
    *,
    connected: bool,
) -> Array:
    """Return `<x(t_w) x(t_w + tau)>` of an ensemble.

    There is one row per waiting time and one column per lag.
    """
    array = ensemble.reshape(ensemble.shape[0], ensemble.shape[1], -1)
    if max(waiting) + max(lags) >= array.shape[1]:
        error_msg = (
            f'A waiting time plus a lag reaches frame {max(waiting) + max(lags)}, beyond {array.shape[1] - 1}. '
            'Use shorter waiting times or lags.'
        )
        raise AnalysisError(error_msg)
    offsets = np.asarray(lags, dtype=int)
    rows = []
    for start in waiting:
        first = array[:, start][:, None, :]
        later = array[:, start + offsets]
        row = np.mean(first * later, axis=(0, 2))
        rows.append(row - first.mean() * later.mean(axis=(0, 2)) if connected else row)
    return np.asarray(rows, dtype=float)


def _correlation_function(field: npt.ArrayLike, max_distance: int | None = None) -> tuple[Array, Array]:
    """Return the distances and the radially averaged spatial correlation of a field.

    The field lies on a periodic grid.
    """
    grid = _read_real(field, 'field')
    grid = grid - grid.mean()
    spectrum = np.fft.fftn(grid)
    correlation = np.real(np.fft.ifftn(spectrum * np.conj(spectrum))) / grid.size
    if correlation.flat[0] == 0:
        error_msg = 'The field is constant, so it has no correlation function. Pass a field that varies.'
        raise AnalysisError(error_msg)
    correlation /= correlation.flat[0]
    axes = np.meshgrid(*(np.minimum(np.arange(size), size - np.arange(size)) for size in grid.shape), indexing='ij')
    distance = np.rint(np.sqrt(sum(axis**2 for axis in axes))).astype(int)
    largest = min(grid.shape) // 2 if max_distance is None else max_distance
    sums = np.bincount(distance.ravel(), correlation.ravel())[: largest + 1]
    counts = np.bincount(distance.ravel())[: largest + 1]
    return np.arange(largest + 1, dtype=float), sums / counts


def _ornstein_zernike_length(hat: Array, power: Array, limit: float) -> float:
    """Return `sqrt(slope / intercept)` of a fit of `1 / S` against `q̂²`.

    The fit runs over the modes of `q̂²` to `limit`.
    """
    inside = (hat <= limit) & (power > 0)
    if inside.sum() < MIN_LINE_POINTS:
        return float('nan')
    slope, intercept = np.polyfit(hat[inside], 1 / power[inside], 1)
    if slope <= 0:
        return 0.0
    if intercept <= 0:
        return float('inf')
    return float(np.sqrt(slope / intercept))


def _moment_length(grids: Array, spacing: float) -> float:
    """Return the second-moment length from fits of `1 / S(q) ∝ 1 + xi² q̂²`.

    The fits run over the modes with `q̂² xi² <= 1`.
    """
    shape = grids.shape[1:]
    axes = tuple(range(1, grids.ndim))
    fluctuations = grids - (grids.mean() if grids.shape[0] > 1 else grids.mean(axis=axes, keepdims=True))
    power = (np.mean(np.abs(np.fft.fftn(fluctuations, axes=axes)) ** 2, axis=0) / np.prod(shape)).ravel()
    lattice = np.meshgrid(*(2 * np.pi * np.fft.fftfreq(size) for size in shape), indexing='ij')
    hat = np.sum([4 * np.sin(axis / 2) ** 2 for axis in lattice], axis=0).ravel()
    if grids.shape[0] == 1:
        hat, power = hat[1:], power[1:]
    smallest = 4 * np.sin(np.pi / max(shape)) ** 2
    limit = 1.0
    length = float('nan')
    for _ in range(MOMENT_ITERATIONS):
        length = _ornstein_zernike_length(hat, power, limit)
        if not np.isfinite(length) or length <= 0:
            break
        limit = float(np.clip(1 / length**2, 2.5 * smallest, 1.0))
    return length * spacing


def _fitted_length(grids: Array, spacing: float) -> float:
    """Return the length `xi` of a fit of `C(r) = r^-a exp(-r / xi)`.

    The fit is to the correlation function.
    """
    curves = [_correlation_function(grid) for grid in grids]
    distances = curves[0][0][1:] * spacing
    values = np.mean([curve[1][1:] for curve in curves], axis=0)
    positive = values > 0
    if positive.sum() < MIN_FIT_POINTS:
        error_msg = (
            f'The correlation function is positive at fewer than {MIN_FIT_POINTS} distances, so no length can be '
            "fitted. Use the method 'moment'."
        )
        raise AnalysisError(error_msg)

    def logarithm(distance: Array, power: float, length: float, scale: float) -> Array:
        """Return the logarithm of the fitted form."""
        return scale - power * np.log(distance) - distance / length

    guess = (0.0, float(distances[positive].mean()), 0.0)
    parameters, _ = optimize.curve_fit(
        logarithm,
        distances[positive],
        np.log(values[positive]),
        p0=guess,
        bounds=([-5.0, 1e-6, -50.0], [5.0, 1e6, 50.0]),
    )
    return float(parameters[1])


def _correlation_length(grids: Array, method: str, spacing: float) -> float:
    """Return the correlation length of frames of a field.

    It is found by the second moment or by a fit.
    """
    if method == 'fit':
        return _fitted_length(grids, spacing)
    return _moment_length(grids, spacing)
