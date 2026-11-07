"""Measure scaling: fluctuation exponents, fractal dimensions, variance-mean laws, power laws, and allometry."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import nolds
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree
from scipy.stats import bootstrap, linregress

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array
from ._base import _check_choice, _check_settings, _read_fields, _read_real, _read_series
from .types import Source

MIN_LINE_POINTS = 2

KINDS = (
    'detrended-fluctuation',
    'rescaled-range',
    'box-counting',
    'correlation-dimension',
    'variance-mean',
    'power',
    'allometric',
)
AGAINST = ('mean', 'size')
SETTINGS = {
    'detrended-fluctuation': {'scales', 'order'},
    'rescaled-range': {'scales'},
    'box-counting': {'scales'},
    'correlation-dimension': {'scales', 'fit_sums', 'random_state'},
    'variance-mean': {'scales', 'against'},
    'power': {'parameters', 'critical', 'above'},
    'allometric': {'confidence', 'resamples', 'random_state'},
}
"""Settings every kind reads, besides `kind` and `variables`."""

MIN_FLUCTUATION_VALUES = 100
RADII = 30
PAIR_SAMPLE = 200_000
MIN_POINTS = 2
FRAME_DIMENSIONS = 3
PAIR_COLUMNS = 2


@dataclass(frozen=True)
class ScalingResult:
    """Scaling exponent of every series, fractal dimension of a set of cells or points, or scaling law.

    Args:
        values:
            The series, one per column, the field or its frames, or the points, an empty array for samples by size,
            and the pairs of sizes and samples of `'allometric'`, one per row.

        samples:
            The samples by size, or `None` for a source of series, points, or a field.

        labels:
            The label of every series, of the field, or of every size.

        exponent:
            The exponent of every series for the fluctuation kinds and `'power'`, the slope of the variance-mean law,
            the exponent of the allometric law, and the dimension for the other kinds.

        scales:
            The window sizes, the box sides, the radii, or the sizes, or `None` for `'power'`.

        fluctuations:
            The fluctuation, or the rescaled range, of every series at every window size, for the fluctuation kinds.

        counts:
            The number of occupied boxes of every side, for `'box-counting'`.

        sums:
            The share of pairs of points closer than every radius, for `'correlation-dimension'`.

        means:
            The mean of the quantity at every scale, for `'variance-mean'`.

        deviations:
            The standard deviation of the quantity at every scale, for `'variance-mean'`.

        fit_range:
            The smallest and largest radii of the fit of the correlation dimension, for `'correlation-dimension'`.

        interval:
            The lower and upper ends of the confidence interval of the allometric exponent, for `'allometric'`.
    """

    values: Array
    samples: dict[float, Array] | None
    labels: list[str]
    exponent: float | Array
    scales: Array | None = None
    fluctuations: Array | None = None
    counts: Array | None = None
    sums: Array | None = None
    means: Array | None = None
    deviations: Array | None = None
    fit_range: tuple[float, float] | None = None
    interval: tuple[float, float] | None = None


def scaling(
    source: Source | Mapping[float, npt.ArrayLike],
    *,
    kind: str = 'detrended-fluctuation',
    variables: Sequence[str] | None = None,
    scales: Sequence[float] | None = None,
    order: int = 1,
    fit_sums: tuple[float, float] = (1e-4, 1e-2),
    random_state: int | None = 0,
    against: str = 'mean',
    parameters: Sequence[float] | None = None,
    critical: float | None = None,
    above: bool = False,
    confidence: float = 0.95,
    resamples: int = 1000,
) -> ScalingResult:
    """Return the scaling exponent of every series, the fractal dimension of cells or points, or a scaling law.

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
    - `'allometric'`: the slope of the logarithm of every sample of a quantity against the logarithm of its size, the
      exponent `b` of `y ~ N^b` across a set of runs, with a bootstrap confidence interval over the pairs. It reads a
      mapping from size to samples, or two columns of sizes and samples.

    Args:
        source:
            The trajectory, the series, the field, the points, or a mapping from size to samples.

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
            The seed of the sample of pairs that sets the default radii, or of the bootstrap.

        against:
            What the variance-mean law is fitted against, `'mean'` or `'size'`.

        parameters:
            The control parameter value of every row of the series of `'power'`.

        critical:
            The value of the control parameter the distance is measured to.

        above:
            Whether the fit uses the values above the critical value, as for couplings, rather than below it, as for
            temperatures.

        confidence:
            The confidence level of the interval of `'allometric'`.

        resamples:
            The number of bootstrap resamples of `'allometric'`.

    Returns:
        result:
            What the source was read as, the exponent, and the curves it is fitted on.

    Raises:
        AnalysisError:
            The kind or a setting does not fit, or `'allometric'` reads a sample or a size that is not positive, or
            fewer than two sizes.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import scaling
        >>> walk = np.cumsum(np.random.default_rng(0).standard_normal(5000))
        >>> scaling(walk).exponent.round(1).tolist()
        [1.5]
        >>> round(scaling(np.ones((256, 256)), kind='box-counting').exponent, 2)
        2.0
        >>> points = np.random.default_rng(0).random((5000, 2))
        >>> round(scaling(points, kind='correlation-dimension').exponent, 1)
        2.0
        >>> counts = np.random.default_rng(0).poisson(5.0, (20, 64, 64))
        >>> round(scaling(counts, kind='variance-mean').exponent, 1)
        0.5
        >>> power = scaling([0.1**0.5, 0.01**0.5], kind='power', parameters=[0.9, 0.99], critical=1.0)
        >>> power.exponent.round(6).tolist()
        [0.5]
        >>> rng = np.random.default_rng(0)
        >>> samples = {size: size**1.15 * rng.lognormal(0.0, 0.1, 5) for size in (100, 200, 400, 800, 1600)}
        >>> allometric = scaling(samples, kind='allometric', resamples=200)
        >>> round(allometric.exponent, 1)
        1.2
        >>> bool(allometric.interval[0] < 1.15 < allometric.interval[1])
        True
    """
    _check_choice('scaling', 'kind', kind, KINDS)
    _check_choice('scaling', 'against', against, AGAINST)
    _check_settings(
        scaling,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        scales=scales,
        order=order,
        fit_sums=fit_sums,
        random_state=random_state,
        against=against,
        parameters=parameters,
        critical=critical,
        above=above,
        confidence=confidence,
        resamples=resamples,
    )
    _check(kind, order, parameters=parameters, critical=critical, confidence=confidence, resamples=resamples)
    values, samples, labels = _read(source, kind, variables)
    if kind == 'allometric':
        x, y = np.log(values).T
        result = bootstrap(
            (x, y),
            _slope,
            paired=True,
            vectorized=True,
            n_resamples=resamples,
            confidence_level=confidence,
            rng=random_state,
        )
        return ScalingResult(
            values,
            samples,
            labels,
            float(linregress(x, y).slope),
            scales=np.unique(values[:, 0]),
            interval=(float(result.confidence_interval.low), float(result.confidence_interval.high)),
        )
    if kind == 'box-counting':
        sides, counts, dimension = _box_counting(values, scales)
        return ScalingResult(values, samples, labels, dimension, scales=sides, counts=counts)
    if kind == 'variance-mean':
        moments = _size_moments(samples) if samples is not None else _box_moments(values, scales)
        exponent = _variance_mean_exponent(*moments, against)
        return ScalingResult(
            values,
            samples,
            labels,
            exponent,
            scales=moments[0],
            means=moments[1],
            deviations=moments[2],
        )
    if kind == 'power':
        exponents = np.asarray(
            [_power_exponent(column, parameters, float(critical or 0.0), above=above) for column in values.T],
        )
        return ScalingResult(values, samples, labels, exponents)
    if kind == 'correlation-dimension':
        radii, sums, dimension, fit_range = _correlation_dimension(
            values,
            scales,
            fit_sums=fit_sums,
            random_state=random_state,
        )
        return ScalingResult(values, samples, labels, dimension, scales=radii, sums=sums, fit_range=fit_range)
    method = 'rs' if kind == 'rescaled-range' else 'dfa'
    curves = [_fluctuation(column, scales, order, method=method) for column in values.T]
    return ScalingResult(
        values,
        samples,
        labels,
        np.asarray([exponent for _, _, exponent in curves]),
        scales=curves[0][0],
        fluctuations=np.asarray([fluctuations for _, fluctuations, _ in curves]),
    )


def _check(
    kind: str,
    order: int,
    *,
    parameters: Sequence[float] | None,
    critical: float | None,
    confidence: float,
    resamples: int,
) -> None:
    """Validate the order, the settings the kind needs, the confidence level, and the number of resamples."""
    if order < 0:
        error_msg = f'scaling has order={order!r}. Use 0 or more.'
        raise AnalysisError(error_msg)
    if kind == 'power' and (parameters is None or critical is None):
        error_msg = (
            'scaling of kind power needs the control parameter values and the critical value. Set `parameters` and '
            '`critical`.'
        )
        raise AnalysisError(error_msg)
    if not 0 < confidence < 1:
        error_msg = f'scaling has confidence={confidence!r}. Use a level between 0 and 1.'
        raise AnalysisError(error_msg)
    if resamples < 1:
        error_msg = f'scaling has resamples={resamples!r}. Use 1 or more.'
        raise AnalysisError(error_msg)


def _read(
    source: Source | Mapping[float, npt.ArrayLike],
    kind: str,
    variables: Sequence[str] | None,
) -> tuple[Array, dict[float, Array] | None, list[str]]:
    """Return the series or the points as columns, the last frame or every frame of a field, or samples by size."""
    if kind == 'allometric':
        return _read_pairs(source), None, ['size', 'samples']
    name = variables[0] if variables else None
    if kind == 'variance-mean':
        if isinstance(source, Mapping):
            samples = {float(size): np.asarray(values, dtype=np.float64) for size, values in source.items()}
            return np.zeros(0), samples, [f'{size:g}' for size in samples]
        frames = _read_fields(source, name)
        return (frames if frames.ndim == FRAME_DIMENSIONS else frames[None]), None, [name or 'field']
    if kind == 'box-counting':
        field = _read_fields(source, name)
        return (field[-1] if isinstance(source, Trajectory) else field), None, [name or 'field']
    series = _read_series(source, variables)
    return series.values, None, series.labels


def _read_pairs(source: object) -> Array:
    """Return the pairs of sizes and samples of a mapping from size to samples or of two columns."""
    if isinstance(source, Mapping):
        rows = [
            (float(size), float(sample))
            for size, samples in source.items()
            for sample in np.ravel(np.asarray(samples, dtype=np.float64))
        ]
        pairs = np.asarray(rows, dtype=np.float64).reshape(-1, PAIR_COLUMNS)
    else:
        pairs = np.asarray(source, dtype=np.float64)
    if pairs.ndim != PAIR_COLUMNS or pairs.shape[1] != PAIR_COLUMNS:
        error_msg = 'An allometric fit reads a mapping from size to samples, or two columns of sizes and samples.'
        raise AnalysisError(error_msg)
    if np.any(pairs <= 0):
        error_msg = 'An allometric fit takes logarithms, so its sizes and samples must be positive. Drop the others.'
        raise AnalysisError(error_msg)
    if np.unique(pairs[:, 0]).size < MIN_LINE_POINTS:
        error_msg = f'An allometric fit needs at least {MIN_LINE_POINTS} sizes. Give more sizes.'
        raise AnalysisError(error_msg)
    return pairs


def _slope(x: Array, y: Array, axis: int = -1) -> Array:
    """Return the slope of the regression of `y` on `x` along an axis."""
    x_centered = x - x.mean(axis=axis, keepdims=True)
    y_centered = y - y.mean(axis=axis, keepdims=True)
    return np.sum(x_centered * y_centered, axis=axis) / np.sum(x_centered**2, axis=axis)


def _fluctuation(
    values: npt.ArrayLike,
    scales: npt.ArrayLike | None = None,
    order: int = 1,
    *,
    method: str = 'dfa',
) -> tuple[Array, Array, float]:
    """Return the window sizes, the fluctuations, and the exponent of a series by DFA or by the rescaled range."""
    array = _read_real(values, 'values')
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


def _box_moments(grids: Array, boxes: npt.ArrayLike | None) -> tuple[Array, Array, Array]:
    """Return the box sides and the mean and deviation of the counts in square boxes of every side."""
    smallest = min(grids.shape[1:])
    sides = 2 ** np.arange(int(np.log2(max(smallest // 4, 1))) + 1) if boxes is None else np.asarray(boxes, dtype=int)
    means, deviations = [], []
    for side in sides:
        rows, columns = grids.shape[1] // side, grids.shape[2] // side
        trimmed = grids[:, : rows * side, : columns * side]
        counts = trimmed.reshape(grids.shape[0], rows, side, columns, side).sum(axis=(2, 4)).ravel()
        means.append(counts.mean())
        deviations.append(counts.std())
    return np.asarray(sides, dtype=float), np.asarray(means), np.asarray(deviations)


def _size_moments(samples: Mapping[float, Array]) -> tuple[Array, Array, Array]:
    """Return the sizes in increasing order and the mean and deviation of the samples at every size."""
    sizes = np.asarray(sorted(samples), dtype=np.float64)
    means = np.asarray([samples[size].mean() for size in sorted(samples)], dtype=np.float64)
    deviations = np.asarray([samples[size].std() for size in sorted(samples)], dtype=np.float64)
    return sizes, means, deviations


def _variance_mean_exponent(scales: Array, means: Array, deviations: Array, against: str) -> float:
    """Return the slope of the log deviation against the log mean, or of the log ratio against the log size."""
    if against == 'mean':
        return float(np.polyfit(np.log(means), np.log(deviations), 1)[0])
    return float(np.polyfit(np.log(scales), np.log(deviations / means), 1)[0])


def _power_exponent(values: Array, parameters: Sequence[float] | None, critical: float, *, above: bool) -> float:
    """Return the slope of the log values against the log distance to the critical value on the chosen side."""
    grid = np.asarray(parameters, dtype=float)
    distance = grid - critical if above else critical - grid
    inside = (distance > 0) & (values > 0)
    if inside.sum() < MIN_POINTS:
        error_msg = (
            f'Fewer than {MIN_POINTS} positive values lie on the chosen side of {critical}. Give values on '
            'that side, or set `above`.'
        )
        raise AnalysisError(error_msg)
    return float(np.polyfit(np.log(distance[inside]), np.log(values[inside]), 1)[0])


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
    if inside.size < MIN_LINE_POINTS:
        error_msg = f'Fewer than 2 radii have correlation sums within {fit_sums}. Give more radii or points.'
        raise AnalysisError(error_msg)
    slope = float(np.polyfit(np.log(grid[inside]), np.log(sums[inside]), 1)[0])
    return grid, sums, slope, (float(grid[inside[0]]), float(grid[inside[-1]]))
