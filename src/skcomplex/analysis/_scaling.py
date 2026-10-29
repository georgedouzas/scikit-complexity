"""Measure scaling: fluctuation exponents of series, fractal dimensions, variance-mean laws, and power laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

import nolds
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis
from .types import Source

MIN_LINE_POINTS = 2

KINDS = ('detrended-fluctuation', 'rescaled-range', 'box-counting', 'correlation-dimension', 'variance-mean', 'power')
AGAINST = ('mean', 'size')
SETTINGS = {
    'detrended-fluctuation': {'scales', 'order'},
    'rescaled-range': {'scales'},
    'box-counting': {'scales'},
    'correlation-dimension': {'scales', 'fit_sums', 'random_state'},
    'variance-mean': {'scales', 'against'},
    'power': {'parameters', 'critical', 'above'},
}
"""Settings every kind reads, besides `kind` and `variables`."""
MIN_FLUCTUATION_VALUES = 100
RADII = 30
PAIR_SAMPLE = 200_000
MIN_POINTS = 2
FRAME_DIMENSIONS = 3


class ScalingAnalysis(BaseAnalysis):
    """Scaling exponent of every series of a source, fractal dimension of a set of cells or points, or scaling law.

    The kinds read series, fields, points, or samples at several sizes:

    - `'detrended-fluctuation'`: the slope of the log fluctuation of the integrated series, detrended in windows by
      a polynomial of degree `order`, against the log window size. It is 0.5 for white noise, the Hurst exponent `H`
      for fractional Gaussian noise, and `H + 1` for its integral, such as 1.5 for a random walk.
    - `'rescaled-range'`: the Hurst exponent by the rescaled range, 0.5 for uncorrelated noise, above 0.5 for
      persistent and below 0.5 for anti-persistent noise.
    - `'box-counting'`: the box-counting dimension of the occupied cells of a field, the slope of the log box count
      against the log box size, read from the last frame of a trajectory.
    - `'correlation-dimension'`: the Grassberger-Procaccia correlation dimension of a point set, one row per point,
      such as the columns of an embedded series, from tree counts of close pairs.
    - `'variance-mean'`: the mean and the standard deviation of a quantity at every scale, and the slope of
      `log deviation` against `log mean` with `against='mean'`, or of `log (deviation / mean)` against `log size` with
      `against='size'`. It reads the counts of a density field in square boxes of growing side over every frame, where
      the slope against the mean is 1/2 for independent particles and above for giant fluctuations, or a mapping from
      size to samples, where the slope against the size is -1/2 for the sums of independent parts.
    - `'power'`: the exponent `b` of the power law `y ~ |p - p_c|^b` of every series against the distance of the
      control `parameters` to the `critical` value, fitted on the side above or below it where `y` is positive, such as
      the exponent `beta` of an order parameter.

    Args:
        kind:
            The scaling to measure, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series, or the first field for a field.

        scales:
            The window sizes, the box sides in cells, or the radii, or `None` for windows spaced logarithmically up
            to a tenth of the series, powers of 2 up to an eighth of the smallest side, or a quarter of it for
            `'variance-mean'`, or 30 radii spaced logarithmically over the range of pair distances.

        order:
            The order of the detrending polynomial.

        fit_sums:
            The smallest and largest correlation sums of the scaling region the slope is fitted in.

        random_state:
            The seed of the sample of pairs that sets the default radii.

        against:
            What the variance-mean law is fitted against, `'mean'` or `'size'`.

        parameters:
            The control parameter value of every row of the series of `'power'`.

        critical:
            The value of the control parameter the distance is measured to.

        above:
            Whether the fit uses the values above the critical value, as for couplings, rather than below it, as for
            temperatures.

    Attributes:
        values_:
            The series, one per column, the field or its frames, or the points, and an empty array for samples by size.

        samples_:
            The samples by size, or `None` for a source of series, points, or a field.

        labels_:
            The label of every series, of the field, or of every size.

        scales_:
            The window sizes, the box sides, the radii, or the sizes.

        fluctuations_:
            The fluctuation, or the rescaled range, of every series at every window size.

        counts_:
            The number of occupied boxes of every side.

        sums_:
            The share of pairs of points closer than every radius.

        means_:
            The mean of the quantity at every scale.

        deviations_:
            The standard deviation of the quantity at every scale.

        fit_range_:
            The smallest and largest radii of the fit of the correlation dimension.

        exponent_:
            The exponent of every series for the fluctuation kinds and `'power'`, the slope of the variance-mean law,
            and the dimension for the other kinds.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import ScalingAnalysis
        >>> walk = np.cumsum(np.random.default_rng(0).standard_normal(5000))
        >>> ScalingAnalysis().check().prepare(walk).run().exponent_.round(1).tolist()
        [1.5]
        >>> round(ScalingAnalysis('box-counting').check().prepare(np.ones((256, 256))).run().exponent_, 2)
        2.0
        >>> points = np.random.default_rng(0).random((5000, 2))
        >>> round(ScalingAnalysis('correlation-dimension').check().prepare(points).run().exponent_, 1)
        2.0
        >>> counts = np.random.default_rng(0).poisson(5.0, (20, 64, 64))
        >>> round(ScalingAnalysis('variance-mean').check().prepare(counts).run().exponent_, 1)
        0.5
        >>> power = ScalingAnalysis('power', parameters=[0.9, 0.99], critical=1.0)
        >>> power.check().prepare([0.1**0.5, 0.01**0.5]).run().exponent_.round(6).tolist()
        [0.5]
    """

    exponent_: float | Array

    def __init__(
        self,
        kind: str = 'detrended-fluctuation',
        variables: Sequence[str] | None = None,
        *,
        scales: Sequence[float] | None = None,
        order: int = 1,
        fit_sums: tuple[float, float] = (1e-4, 1e-2),
        random_state: int | None = 0,
        against: str = 'mean',
        parameters: Sequence[float] | None = None,
        critical: float | None = None,
        above: bool = False,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.scales = scales
        self.order = order
        self.fit_sums = fit_sums
        self.random_state = random_state
        self.against = against
        self.parameters = parameters
        self.critical = critical
        self.above = above

    def _check(self) -> None:
        """Validate the kind, the order, the settings the kind needs, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_choice('against', self.against, AGAINST)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.order < 0:
            error_msg = f'{type(self).__name__} has order={self.order!r}. Use 0 or more.'
            raise AnalysisError(error_msg)
        if self.kind == 'power' and (self.parameters is None or self.critical is None):
            error_msg = (
                f'{type(self).__name__} of kind power needs the control parameter values and the critical value. Set '
                '`parameters` and `critical`.'
            )
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the field kind for box counting and variance-mean, the ensemble kind for power, and series else."""
        if self.kind in {'box-counting', 'variance-mean'}:
            return DataKind.FIELD
        return DataKind.ENSEMBLE if self.kind == 'power' else DataKind.SERIES

    def _prepare(self, source: Source | Mapping[float, npt.ArrayLike]) -> None:
        """Read the series or the points as columns, the last frame or every frame of a field, or samples by size."""
        self.samples_: dict[float, Array] | None = None
        if self.kind == 'variance-mean':
            if isinstance(source, Mapping):
                self.samples_ = {float(size): np.asarray(samples, dtype=np.float64) for size, samples in source.items()}
                self.values_: Array = np.zeros(0)
                self.labels_ = [f'{size:g}' for size in self.samples_]
                return
            name = self.variables[0] if self.variables else None
            frames = self._read_fields(source, name)
            self.values_ = frames if frames.ndim == FRAME_DIMENSIONS else frames[None]
            self.labels_ = [name or 'field']
            return
        if self.kind == 'box-counting':
            name = self.variables[0] if self.variables else None
            field = self._read_fields(source, name)
            self.values_ = field[-1] if isinstance(source, Trajectory) else field
            self.labels_ = [name or 'field']
            return
        series = self._read_series(source, self.variables)
        self.values_ = series.values
        self.labels_ = series.labels

    def _run(self, inputs: object = None) -> None:
        """Compute the exponents of the series, or the dimension of the cells or points."""
        if self.kind == 'box-counting':
            self.scales_, self.counts_, self.exponent_ = self._box_counting(self.values_, self.scales)
        elif self.kind == 'variance-mean':
            if self.samples_ is not None:
                self.scales_, self.means_, self.deviations_ = self._size_moments(self.samples_)
            else:
                self.scales_, self.means_, self.deviations_ = self._box_moments(self.values_, self.scales)
            self.exponent_ = self._variance_mean_exponent(self.scales_, self.means_, self.deviations_, self.against)
        elif self.kind == 'power':
            self.exponent_ = np.asarray([self._power_exponent(column) for column in self.values_.T])
        elif self.kind == 'correlation-dimension':
            self.scales_, self.sums_, self.exponent_, self.fit_range_ = self._correlation_dimension(
                self.values_,
                self.scales,
                fit_sums=self.fit_sums,
                random_state=self.random_state,
            )
        else:
            method = 'rs' if self.kind == 'rescaled-range' else 'dfa'
            curves = [self._fluctuation(column, self.scales, self.order, method=method) for column in self.values_.T]
            self.scales_ = curves[0][0]
            self.fluctuations_ = np.asarray([fluctuations for _, fluctuations, _ in curves])
            self.exponent_ = np.asarray([exponent for _, _, exponent in curves])

    @staticmethod
    def _fluctuation(
        values: npt.ArrayLike,
        scales: npt.ArrayLike | None = None,
        order: int = 1,
        *,
        method: str = 'dfa',
    ) -> tuple[Array, Array, float]:
        """Return the window sizes, the fluctuations, and the exponent of a series by DFA or by the rescaled range."""
        array = BaseAnalysis._read_real(values, 'values')
        if array.size < MIN_FLUCTUATION_VALUES:
            error_msg = (
                f'A fluctuation analysis needs at least {MIN_FLUCTUATION_VALUES} values, not {array.size}. '
                'Pass a longer series.'
            )
            raise AnalysisError(error_msg)
        sizes = None if scales is None else np.asarray(scales, dtype=int)
        if method == 'rs':
            exponent, (log_scales, log_values, _) = nolds.hurst_rs(array, nvals=sizes, fit='poly', debug_data=True)
        else:
            exponent, (log_scales, log_values, _) = nolds.dfa(
                array,
                nvals=sizes,
                order=order,
                fit_exp='poly',
                debug_data=True,
            )
        return np.exp(log_scales), np.exp(log_values), float(exponent)

    @staticmethod
    def _box_moments(grids: Array, boxes: npt.ArrayLike | None) -> tuple[Array, Array, Array]:
        """Return the box sides and the mean and deviation of the counts in square boxes of every side."""
        smallest = min(grids.shape[1:])
        sides = (
            2 ** np.arange(int(np.log2(max(smallest // 4, 1))) + 1) if boxes is None else np.asarray(boxes, dtype=int)
        )
        means, deviations = [], []
        for side in sides:
            rows, columns = grids.shape[1] // side, grids.shape[2] // side
            trimmed = grids[:, : rows * side, : columns * side]
            counts = trimmed.reshape(grids.shape[0], rows, side, columns, side).sum(axis=(2, 4)).ravel()
            means.append(counts.mean())
            deviations.append(counts.std())
        return np.asarray(sides, dtype=float), np.asarray(means), np.asarray(deviations)

    @staticmethod
    def _size_moments(samples: Mapping[float, Array]) -> tuple[Array, Array, Array]:
        """Return the sizes in increasing order and the mean and deviation of the samples at every size."""
        sizes = np.asarray(sorted(samples), dtype=np.float64)
        means = np.asarray([samples[size].mean() for size in sorted(samples)], dtype=np.float64)
        deviations = np.asarray([samples[size].std() for size in sorted(samples)], dtype=np.float64)
        return sizes, means, deviations

    @staticmethod
    def _variance_mean_exponent(scales: Array, means: Array, deviations: Array, against: str) -> float:
        """Return the slope of the log deviation against the log mean, or of the log ratio against the log size."""
        if against == 'mean':
            return float(np.polyfit(np.log(means), np.log(deviations), 1)[0])
        return float(np.polyfit(np.log(scales), np.log(deviations / means), 1)[0])

    def _power_exponent(self, values: Array) -> float:
        """Return the slope of the log values against the log distance to the critical value on the chosen side."""
        grid = np.asarray(self.parameters, dtype=float)
        distance = grid - self.critical if self.above else self.critical - grid
        inside = (distance > 0) & (values > 0)
        if inside.sum() < MIN_POINTS:
            error_msg = (
                f'Fewer than {MIN_POINTS} positive values lie on the chosen side of {self.critical}. Give values on '
                'that side, or set `above`.'
            )
            raise AnalysisError(error_msg)
        return float(np.polyfit(np.log(distance[inside]), np.log(values[inside]), 1)[0])

    @staticmethod
    def _box_counting(occupied: npt.ArrayLike, sizes: npt.ArrayLike | None = None) -> tuple[Array, Array, float]:
        """Return the box sides, the box counts, and the box-counting dimension of a set of cells."""
        grid = np.asarray(occupied, dtype=bool)
        sides = (
            2 ** np.arange(int(np.log2(min(grid.shape) // 8)) + 1) if sizes is None else np.asarray(sizes, dtype=int)
        )
        counts = []
        for side in sides:
            trimmed = grid[tuple(slice(0, size - size % side) for size in grid.shape)]
            shape = [item for size in trimmed.shape for item in (size // side, side)]
            boxes = trimmed.reshape(shape).any(axis=tuple(range(1, 2 * grid.ndim, 2)))
            counts.append(boxes.sum())
        dimension = float(-np.polyfit(np.log(sides), np.log(counts), 1)[0])
        return np.asarray(sides, dtype=float), np.asarray(counts, dtype=float), dimension

    @staticmethod
    def _pair_distances(points: Array, rng: np.random.Generator) -> Array:
        """Return the distances of a random sample of pairs of distinct points."""
        first = rng.integers(0, points.shape[0], PAIR_SAMPLE)
        second = rng.integers(0, points.shape[0], PAIR_SAMPLE)
        distinct = first != second
        return np.linalg.norm(points[first[distinct]] - points[second[distinct]], axis=1)

    @staticmethod
    def _correlation_dimension(
        points: npt.ArrayLike,
        radii: npt.ArrayLike | None = None,
        *,
        fit_sums: tuple[float, float] = (1e-4, 1e-2),
        random_state: int | None = 0,
    ) -> tuple[Array, Array, float, tuple[float, float]]:
        """Return the radii, the correlation sums, the Grassberger-Procaccia dimension, and the fit range of points."""
        array = np.asarray(points, dtype=float)
        array = array.reshape(array.shape[0], -1)
        size = array.shape[0]
        if radii is None:
            distances = ScalingAnalysis._pair_distances(array, np.random.default_rng(random_state))
            low, high = np.quantile(distances, [fit_sums[0] / 10, 0.5])
            radii = np.geomspace(max(low, 1e-12), high, RADII)
        grid = np.asarray(radii, dtype=float)
        tree = cKDTree(array)
        sums = (tree.count_neighbors(tree, grid) - size) / (size * (size - 1))
        inside = np.flatnonzero((sums >= fit_sums[0]) & (sums <= fit_sums[1]))
        if inside.size < MIN_LINE_POINTS:
            error_msg = f'Fewer than 2 radii have correlation sums within {fit_sums}. Give more radii or points.'
            raise AnalysisError(error_msg)
        slope = float(np.polyfit(np.log(grid[inside]), np.log(sums[inside]), 1)[0])
        return grid, sums, slope, (float(grid[inside[0]]), float(grid[inside[-1]]))
