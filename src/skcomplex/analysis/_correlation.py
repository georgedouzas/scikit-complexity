"""Measure correlations: autocorrelations of series, two-time correlations, spatial correlations, and overlaps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy import optimize

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis, Source

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
"""Settings every kind reads, besides `kind` and `variables`."""
MOMENT_ITERATIONS = 8
REPLICAS = 2


class CorrelationAnalysis(BaseAnalysis):
    """Correlation of every series of a source, of an ensemble at two times, or of a field in space, of one kind.

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

    Args:
        kind:
            The correlation to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series, or the first field for a field.

        max_lag:
            The largest lag in steps, or `None` for half the series length.

        dt:
            The time between samples.

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

    Attributes:
        values_:
            The series, one per column, the ensemble, the field, or the two configurations.

        labels_:
            The label of every series, of the ensemble or field, or of the two configurations.

        lags_:
            The lags in time of `'auto'`.

        correlation_:
            One autocorrelation per series for `'auto'`, one row per waiting time and one column per lag for
            `'two-time'`, and the correlation at every distance for `'function'`.

        time_:
            The integrated autocorrelation time of every series, for `'auto'`.

        distances_:
            The distances of `'function'`.

        length_:
            The correlation length, for `'length'`.

        overlap_:
            The overlap of the two configurations, one per frame for frames.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import CorrelationAnalysis
        >>> analysis = CorrelationAnalysis(max_lag=1).check().prepare([1.0, -1.0, 1.0, -1.0]).run()
        >>> analysis.correlation_.round(2).tolist()
        [[1.0, -0.75]]
        >>> two_time = CorrelationAnalysis('two-time', waiting=[2], lags=[0, 3])
        >>> two_time.check().prepare(np.ones((4, 10))).run().correlation_.tolist()
        [[1.0, 1.0]]
        >>> CorrelationAnalysis('overlap').check().prepare([[1, -1, 1, 1], [1, -1, -1, 1]]).run().overlap_
        0.5
    """

    def __init__(
        self,
        kind: str = 'auto',
        variables: Sequence[str] | None = None,
        *,
        max_lag: int | None = None,
        dt: float = 1.0,
        waiting: Sequence[int] | None = None,
        lags: Sequence[int] | None = None,
        connected: bool = False,
        max_distance: int | None = None,
        method: str = 'moment',
        spacing: float = 1.0,
        frames: bool = False,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.max_lag = max_lag
        self.dt = dt
        self.waiting = waiting
        self.lags = lags
        self.connected = connected
        self.max_distance = max_distance
        self.method = method
        self.spacing = spacing
        self.frames = frames

    def _check(self) -> Self:
        """Validate the kind, the method, the settings the kind needs, and that the kind reads every changed one."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_choice('method', self.method, METHODS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.kind == 'two-time' and (self.waiting is None or self.lags is None):
            error_msg = f'{type(self).__name__} of kind two-time needs waiting times and lags. Give both in frames.'
            raise AnalysisError(error_msg)
        if self.dt <= 0 or self.spacing <= 0:
            error_msg = f'{type(self).__name__} has dt={self.dt!r} and spacing={self.spacing!r}. Use positive values.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the series kind for autocorrelations, the ensemble kind for two times, and the field kind else."""
        if self.kind == 'auto':
            return DataKind.SERIES
        if self.kind == 'two-time':
            return DataKind.ENSEMBLE
        return DataKind.FIELD

    def _prepare_replicas(self, source: Source, name: str | None) -> Self:
        """Read the fields of two trajectories, or two configurations stacked on the first axis of an array."""
        replicas: list[Source] = list(source) if isinstance(source, list | tuple | np.ndarray) else [source]
        if len(replicas) != REPLICAS:
            error_msg = (
                f'An overlap reads {REPLICAS} replicas, a pair of trajectories or two arrays stacked on the first '
                f'axis, but the source has {len(replicas)}. Give two replicas.'
            )
            raise AnalysisError(error_msg)
        self.values_ = np.stack([self._read_fields(replica, name) for replica in replicas])
        self.labels_ = ['first', 'second']
        return self

    def _prepare(self, source: Source) -> Self:
        """Read the series as columns, the ensemble as an array, or the frames of a field."""
        if self.kind == 'auto':
            series = self._read_series(source, self.variables)
            self.values_ = series.values
            self.labels_ = series.labels
            return self
        name = self.variables[0] if self.variables else None
        if self.kind == 'two-time':
            self.values_ = np.asarray(source, dtype=float)
            self.labels_ = [name or 'ensemble']
            return self
        if self.kind == 'overlap':
            return self._prepare_replicas(source, name)
        field = self._read_fields(source, name)
        if isinstance(source, Trajectory) and not self.frames:
            field = field[-1]
        self.values_ = field
        self.labels_ = [name or 'field']
        return self

    def _run(self, inputs: object = None) -> Self:
        """Compute the correlation of the kind."""
        if self.kind == 'auto':
            self.correlation_ = np.asarray([self._autocorrelation(column, self.max_lag) for column in self.values_.T])
            self.lags_ = np.arange(self.correlation_.shape[1]) * self.dt
            self.time_ = np.asarray([self._autocorrelation_time(column, self.dt) for column in self.values_.T])
        elif self.kind == 'two-time':
            if self.waiting is None or self.lags is None:
                error_msg = f'{type(self).__name__} of kind two-time needs waiting times and lags. Give both.'
                raise AnalysisError(error_msg)
            self.correlation_ = self._two_time_correlation(
                self.values_,
                self.waiting,
                self.lags,
                connected=self.connected,
            )
        elif self.kind == 'function':
            self.distances_, self.correlation_ = self._correlation_function(self.values_, self.max_distance)
        elif self.kind == 'length':
            grids = self.values_ if self.frames else self.values_[None]
            self.length_ = self._correlation_length(grids, self.method, self.spacing)
        else:
            product = self.values_[0] * self.values_[1]
            self.overlap_ = (
                float(product.mean()) if product.ndim == 1 else product.reshape(product.shape[0], -1).mean(axis=1)
            )
        return self

    @staticmethod
    def _autocorrelation(values: npt.ArrayLike, max_lag: int | None = None) -> Array:
        """Return the autocorrelation of a series at lags 0 to `max_lag`, or to half its length, by the FFT."""
        array = BaseAnalysis._read_real(values, 'values')
        array = array - array.mean()
        size = array.size
        spectrum = np.fft.rfft(array, n=2 * size)
        correlation = np.fft.irfft(spectrum * np.conj(spectrum))[:size]
        correlation /= correlation[0] if correlation[0] else 1.0
        return correlation[: (size // 2 if max_lag is None else max_lag) + 1]

    @staticmethod
    def _autocorrelation_time(values: npt.ArrayLike, dt: float = 1.0) -> float:
        """Return the integrated autocorrelation time `dt (1/2 + Σ r(k))`, summed to the first negative correlation."""
        correlation = CorrelationAnalysis._autocorrelation(values)
        negative = np.flatnonzero(correlation <= 0)
        end = int(negative[0]) if negative.size else correlation.size
        return float(dt * (correlation[:end].sum() - 0.5))

    @staticmethod
    def _two_time_correlation(
        ensemble: Array,
        waiting: Sequence[int],
        lags: Sequence[int],
        *,
        connected: bool,
    ) -> Array:
        """Return `<x(t_w) x(t_w + tau)>` of an ensemble, one row per waiting time and one column per lag."""
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

    @staticmethod
    def _correlation_function(field: npt.ArrayLike, max_distance: int | None = None) -> tuple[Array, Array]:
        """Return the distances and the radially averaged spatial correlation of a field on a periodic grid."""
        grid = BaseAnalysis._read_real(field, 'field')
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

    @staticmethod
    def _ornstein_zernike_length(hat: Array, power: Array, limit: float) -> float:
        """Return `sqrt(slope / intercept)` of a fit of `1 / S` against `q̂²` over the modes with `q̂²` up to a limit."""
        inside = (hat <= limit) & (power > 0)
        if inside.sum() < MIN_LINE_POINTS:
            return float('nan')
        slope, intercept = np.polyfit(hat[inside], 1 / power[inside], 1)
        if slope <= 0:
            return 0.0
        if intercept <= 0:
            return float('inf')
        return float(np.sqrt(slope / intercept))

    @staticmethod
    def _moment_length(grids: Array, spacing: float) -> float:
        """Return the second-moment length from fits of `1 / S(q) ∝ 1 + xi² q̂²` over the modes with `q̂² xi² <= 1`."""
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
            length = CorrelationAnalysis._ornstein_zernike_length(hat, power, limit)
            if not np.isfinite(length) or length <= 0:
                break
            limit = float(np.clip(1 / length**2, 2.5 * smallest, 1.0))
        return length * spacing

    @staticmethod
    def _fitted_length(grids: Array, spacing: float) -> float:
        """Return the length `xi` of a fit of `C(r) = r^-a exp(-r / xi)` to the correlation function."""
        curves = [CorrelationAnalysis._correlation_function(grid) for grid in grids]
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

    @staticmethod
    def _correlation_length(grids: Array, method: str, spacing: float) -> float:
        """Return the correlation length of frames of a field by the second moment or by a fit."""
        if method == 'fit':
            return CorrelationAnalysis._fitted_length(grids, spacing)
        return CorrelationAnalysis._moment_length(grids, spacing)
