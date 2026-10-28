"""Measure scaling: fluctuation exponents of series and fractal dimensions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from collections.abc import Sequence
from typing import Self

import nolds
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ..core.errors import AnalysisError
from ..engine.runtime import Trajectory
from ._base import BaseAnalysis, Source
from ._kinds import DataKind
from ._utils import is_changed, real_array

type Array = npt.NDArray[np.float64]
KINDS = ('detrended-fluctuation', 'rescaled-range', 'box-counting', 'correlation-dimension')
SETTINGS = {
    'detrended-fluctuation': {'scales', 'order'},
    'rescaled-range': {'scales'},
    'box-counting': {'scales'},
    'correlation-dimension': {'scales', 'fit_sums', 'random_state'},
}
"""Settings every kind reads, besides `kind` and `variables`."""
MIN_FLUCTUATION_VALUES = 100
RADII = 30
PAIR_SAMPLE = 200_000


def _fluctuation(
    values: npt.ArrayLike,
    scales: npt.ArrayLike | None = None,
    order: int = 1,
    *,
    method: str = 'dfa',
) -> tuple[Array, Array, float]:
    """Return the window sizes, the fluctuations, and the exponent of a series by DFA or by the rescaled range."""
    array = real_array(values, 'values')
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


def _box_counting(occupied: npt.ArrayLike, sizes: npt.ArrayLike | None = None) -> tuple[Array, Array, float]:
    """Return the box sides, the box counts, and the box-counting dimension of a set of cells."""
    grid = np.asarray(occupied, dtype=bool)
    sides = 2 ** np.arange(int(np.log2(min(grid.shape) // 8)) + 1) if sizes is None else np.asarray(sizes, dtype=int)
    counts = []
    for side in sides:
        trimmed = grid[tuple(slice(0, size - size % side) for size in grid.shape)]
        shape = [item for size in trimmed.shape for item in (size // side, side)]
        boxes = trimmed.reshape(shape).any(axis=tuple(range(1, 2 * grid.ndim, 2)))
        counts.append(boxes.sum())
    dimension = float(-np.polyfit(np.log(sides), np.log(counts), 1)[0])
    return np.asarray(sides, dtype=float), np.asarray(counts, dtype=float), dimension


def _pair_distances(points: Array, rng: np.random.Generator) -> Array:
    """Return the distances of a random sample of pairs of distinct points."""
    first = rng.integers(0, points.shape[0], PAIR_SAMPLE)
    second = rng.integers(0, points.shape[0], PAIR_SAMPLE)
    distinct = first != second
    return np.linalg.norm(points[first[distinct]] - points[second[distinct]], axis=1)


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
        distances = _pair_distances(array, np.random.default_rng(random_state))
        low, high = np.quantile(distances, [fit_sums[0] / 10, 0.5])
        radii = np.geomspace(max(low, 1e-12), high, RADII)
    grid = np.asarray(radii, dtype=float)
    tree = cKDTree(array)
    sums = (tree.count_neighbors(tree, grid) - size) / (size * (size - 1))
    inside = np.flatnonzero((sums >= fit_sums[0]) & (sums <= fit_sums[1]))
    if inside.size < 2:  # noqa: PLR2004
        error_msg = f'Fewer than 2 radii have correlation sums within {fit_sums}. Give more radii or points.'
        raise AnalysisError(error_msg)
    slope = float(np.polyfit(np.log(grid[inside]), np.log(sums[inside]), 1)[0])
    return grid, sums, slope, (float(grid[inside[0]]), float(grid[inside[-1]]))


class ScalingAnalysis(BaseAnalysis):
    """Scaling exponent of every series of a source, or fractal dimension of a set of cells or points, of one kind.

    The kinds read series, fields, or points:

    - `'detrended-fluctuation'`: the slope of the log fluctuation of the integrated series, detrended in windows by
      a polynomial of degree `order`, against the log window size. It is 0.5 for white noise, the Hurst exponent `H`
      for fractional Gaussian noise, and `H + 1` for its integral, such as 1.5 for a random walk.
    - `'rescaled-range'`: the Hurst exponent by the rescaled range, 0.5 for uncorrelated noise, above 0.5 for
      persistent and below 0.5 for anti-persistent noise.
    - `'box-counting'`: the box-counting dimension of the occupied cells of a field, the slope of the log box count
      against the log box size, read from the last frame of a trajectory.
    - `'correlation-dimension'`: the Grassberger-Procaccia correlation dimension of a point set, one row per point,
      such as the columns of an embedded series, from tree counts of close pairs.

    Args:
        kind:
            The scaling to measure, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series, or the first field for a field.

        scales:
            The window sizes, the box sides in cells, or the radii, or `None` for windows spaced logarithmically up
            to a tenth of the series, powers of 2 up to an eighth of the smallest side, or 30 radii spaced
            logarithmically over the range of pair distances.

        order:
            The order of the detrending polynomial.

        fit_sums:
            The smallest and largest correlation sums of the scaling region the slope is fitted in.

        random_state:
            The seed of the sample of pairs that sets the default radii.

    Attributes:
        values_:
            The series, one per column, the field, or the points.

        labels_:
            The label of every series, or of the field.

        scales_:
            The window sizes, the box sides, or the radii.

        fluctuations_:
            The fluctuation, or the rescaled range, of every series at every window size.

        counts_:
            The number of occupied boxes of every side.

        sums_:
            The share of pairs of points closer than every radius.

        fit_range_:
            The smallest and largest radii of the fit of the correlation dimension.

        exponent_:
            The exponent of every series for the fluctuation kinds, and the dimension for the other kinds.

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
    """

    def __init__(
        self,
        kind: str = 'detrended-fluctuation',
        variables: Sequence[str] | None = None,
        *,
        scales: Sequence[float] | None = None,
        order: int = 1,
        fit_sums: tuple[float, float] = (1e-4, 1e-2),
        random_state: int | None = 0,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.scales = scales
        self.order = order
        self.fit_sums = fit_sums
        self.random_state = random_state

    def _check_choice(self, name: str, value: str, choices: tuple[str, ...]) -> None:
        """Raise an error for a setting that is not one of its choices."""
        if value not in choices:
            error_msg = f'{type(self).__name__} has {name}={value!r}. Use one of {", ".join(choices)}.'
            raise AnalysisError(error_msg)

    def _check_ignored(self) -> None:
        """Raise an error for a setting changed from its default that the kind does not read."""
        defaults = {
            name: parameter.default for name, parameter in inspect.signature(type(self).__init__).parameters.items()
        }
        read = SETTINGS[self.kind] | {'kind', 'variables'}
        ignored = [
            name for name, value in self.get_params().items() if name not in read and is_changed(value, defaults[name])
        ]
        if ignored:
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} does not read {", ".join(ignored)}. It reads '
                f'{", ".join(sorted(SETTINGS[self.kind]))}.'
            )
            raise AnalysisError(error_msg)

    def _check(self) -> Self:
        """Validate the kind, the order, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_ignored()
        if self.order < 0:
            error_msg = f'{type(self).__name__} has order={self.order!r}. Use 0 or more.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the field kind for box counting, and the series kind otherwise."""
        return DataKind.FIELD if self.kind == 'box-counting' else DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the series or the points as columns, or the last frame of a field."""
        if self.kind == 'box-counting':
            name = self.variables[0] if self.variables else None
            field = self._read_fields(source, name)
            self.values_ = field[-1] if isinstance(source, Trajectory) else field
            self.labels_ = [name or 'field']
            return self
        series = self._read_series(source, self.variables)
        self.values_ = series.values
        self.labels_ = series.labels
        return self

    def _run(self) -> Self:
        """Compute the exponents of the series, or the dimension of the cells or points."""
        if self.kind == 'box-counting':
            self.scales_, self.counts_, self.exponent_ = _box_counting(self.values_, self.scales)
        elif self.kind == 'correlation-dimension':
            self.scales_, self.sums_, self.exponent_, self.fit_range_ = _correlation_dimension(
                self.values_,
                self.scales,
                fit_sums=self.fit_sums,
                random_state=self.random_state,
            )
        else:
            method = 'rs' if self.kind == 'rescaled-range' else 'dfa'
            curves = [_fluctuation(column, self.scales, self.order, method=method) for column in self.values_.T]
            self.scales_ = curves[0][0]
            self.fluctuations_ = np.asarray([fluctuations for _, fluctuations, _ in curves])
            self.exponent_ = np.asarray([exponent for _, _, exponent in curves])
        return self
