"""Describe the distribution of a sample: histograms, survival functions, moments, inequality, tails, and records."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import _check_choice, _check_settings, _read_real, _read_samples_of
from .types import Source

ORDER_COUNT = 2

KINDS = (
    'histogram',
    'ccdf',
    'moments',
    'moment-ratio',
    'dispersion-index',
    'gini',
    'tail-index',
    'records',
    'total-variation',
)
SETTINGS = {
    'histogram': {'bins', 'log'},
    'ccdf': set(),
    'moments': {'absolute'},
    'moment-ratio': {'orders'},
    'dispersion-index': set(),
    'gini': set(),
    'tail-index': {'tail'},
    'records': set(),
    'total-variation': set(),
}
"""Settings every kind reads, besides `kind`, `variables`, and `group`."""

POOLED = ('histogram', 'ccdf', 'gini', 'tail-index')
"""The kinds that pool every series into one sample."""

GROUPS = 2
UNIT_COLUMNS = 2

TAGS = AnalysisTags(DataKind.SERIES)
KIND_TAGS = {
    'records': AnalysisTags(DataKind.ENSEMBLE),
}


@dataclass(frozen=True)
class DistributionResult:
    """Distribution of the values of a source.

    Args:
        values:
            The pooled sample, or the samples, one per column.

        labels:
            The label of every sample.

        edges:
            The bin edges of the histogram, for `'histogram'`.

        density:
            The probability density in every bin of the histogram, for `'histogram'`.

        support:
            The sorted distinct values of the complementary cumulative distribution, for `'ccdf'`.

        survival:
            The share of samples at or above every distinct value, `P(X >= x)`, for `'ccdf'`.

        mean:
            The mean of every sample, for `'moments'`.

        variance:
            The variance of every sample, for `'moments'`.

        skewness:
            The skewness of every sample, for `'moments'`.

        kurtosis:
            The excess kurtosis of every sample, 0 for Gaussian samples, for `'moments'`.

        ratio:
            The moment ratio of every sample, for `'moment-ratio'`.

        cumulant:
            The normalized cumulant of every sample, for `'moment-ratio'`.

        index:
            The dispersion index of every sample, for `'dispersion-index'`.

        gini:
            The Gini coefficient, for `'gini'`.

        tail_index:
            The Hill estimate of the tail index, for `'tail-index'`.

        records:
            The number of records of every series, the first value included, for `'records'`.

        distance:
            The total variation distance between the two groups, for `'total-variation'`.
    """

    values: Array
    labels: list[str]
    edges: Array | None = None
    density: Array | None = None
    support: Array | None = None
    survival: Array | None = None
    mean: Array | None = None
    variance: Array | None = None
    skewness: Array | None = None
    kurtosis: Array | None = None
    ratio: Array | None = None
    cumulant: Array | None = None
    index: Array | None = None
    gini: float | None = None
    tail_index: float | None = None
    records: npt.NDArray[np.int64] | None = None
    distance: float | None = None


def distribution(
    source: Source,
    *,
    kind: str = 'histogram',
    variables: Sequence[str] | None = None,
    group: str | None = None,
    bins: int = 30,
    log: bool = False,
    absolute: bool = False,
    orders: tuple[int, int] = (4, 2),
    tail: int = 10,
) -> DistributionResult:
    """Return the distribution of the values of a source, of one kind.

    The source gives samples as columns: the series of a trajectory or an array, or the speeds of the agents of a
    trajectory, one sample per frame. The first kinds pool every sample into one:

    - `'histogram'`: the edges and the probability density of `bins` bins, equally spaced in the logarithm of the
      values with `log=True` for heavy tails.
    - `'ccdf'`: the complementary cumulative distribution, the share of samples at or above every distinct value.
    - `'gini'`: the Gini coefficient of nonnegative values, 0 for equality and close to 1 when one holds everything.
    - `'tail-index'`: the Hill estimate of the tail index from the `tail` largest positive samples, the exponent
      `alpha - 1` of the survival function of a density `p(x) ∝ x^-alpha`.

    The other kinds describe every sample:

    - `'moments'`: the mean, the variance, the skewness, and the excess kurtosis of every sample, or of its absolute
      values with `absolute=True`.
    - `'moment-ratio'`: the ratio `<x^p> / <x^q>^(p / q)` of the moments of the even `orders` `(p, q)`, and the
      cumulant `1 - ratio / ratio_gauss`, normalized by the ratio of a centred Gaussian so that it is 0 for Gaussian
      samples. For the orders 4 and 2 the ratio is the kurtosis about 0 and the cumulant `1 - <x^4> / (3 <x^2>^2)`, 2/3
      for samples at ±1.
    - `'dispersion-index'`: the variance over the mean of every sample of counts, 1 for Poisson counts, above 1 for
      overdispersed and below 1 for underdispersed counts.
    - `'records'`: the number of records of every series, the values larger than every value before them, about the
      harmonic number of the length for independent values.
    - `'total-variation'`: the total variation distance `1/2 Σ_u |a_u / A - b_u / B|` between the distributions of
      two groups across units, from one row per item with its group and its unit, 0 for identical distributions and
      1 for disjoint ones.

    Args:
        source:
            The trajectory, or an array with one column per sample.

        kind:
            The description to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        group:
            The path of the agent or particle group whose speeds are read, or `None` for the series of a trajectory, or
            the first group of a trajectory without series.

        bins:
            The number of bins of the histogram.

        log:
            Whether the bins of the histogram are equally spaced in the logarithm of the values, which needs positive
            values.

        absolute:
            Whether the moments are those of the absolute values.

        orders:
            The positive even orders `(p, q)` of the moment ratio.

        tail:
            The number of largest samples of the Hill estimate.

    Returns:
        result:
            The samples and their description.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, a setting is out of range, or the
            samples do not suit the kind.

    Examples:
        >>> from skcomplex.core.analysis import distribution
        >>> distribution([1, 2, 4, 8], kind='histogram', bins=3, log=True).edges.round(2).tolist()
        [1.0, 2.0, 4.0, 8.0]
        >>> distribution([1, 1, 2, 4], kind='ccdf').survival.tolist()
        [1.0, 0.5, 0.25]
        >>> distribution([0, 0, 0, 4], kind='gini').gini
        0.75
        >>> distribution([1.0, -1.0, 1.0], kind='moment-ratio').cumulant.round(4).tolist()
        [0.6667]
        >>> distribution([1, 3, 1, 3], kind='dispersion-index').index.tolist()
        [0.5]
        >>> distribution([[0, 0], [0, 0], [1, 1], [1, 1]], kind='total-variation').distance
        1.0
    """
    _check_choice('distribution', 'kind', kind, KINDS)
    _check_settings(
        distribution,
        'kind',
        SETTINGS[kind] | {'variables', 'group'},
        kind=kind,
        variables=variables,
        group=group,
        bins=bins,
        log=log,
        absolute=absolute,
        orders=orders,
        tail=tail,
    )
    if bins < 1 or tail < 1:
        error_msg = f'distribution has bins={bins!r} and tail={tail!r}. Use 1 or more.'
        raise AnalysisError(error_msg)
    if len(orders) != ORDER_COUNT or any(order <= 0 or order % 2 for order in orders):
        error_msg = f'distribution has orders={orders!r}. Use two positive even orders.'
        raise AnalysisError(error_msg)
    series = _read_samples_of(source, variables, group)
    samples = _read_real(series.values, 'values')
    values = samples.ravel() if kind in POOLED else samples
    labels = series.labels
    if kind == 'total-variation' and values.shape[1] != UNIT_COLUMNS:
        error_msg = (
            'distribution of kind total-variation reads one row per item with its group and its unit, not an array '
            f'of shape {values.shape}.'
        )
        raise AnalysisError(error_msg)
    if kind in POOLED:
        return _describe_pooled(values, labels, kind, bins, tail, log=log)
    return _describe_samples(values, labels, kind, orders, absolute=absolute)


def _describe_pooled(
    values: Array,
    labels: list[str],
    kind: str,
    bins: int,
    tail: int,
    *,
    log: bool,
) -> DistributionResult:
    """Return the description of a kind that pools every sample into one."""
    if kind == 'histogram':
        edges, density = _histogram(values, bins, log=log)
        return DistributionResult(values, labels, edges=edges, density=density)
    if kind == 'ccdf':
        support, survival = _ccdf(values)
        return DistributionResult(values, labels, support=support, survival=survival)
    if kind == 'gini':
        return DistributionResult(values, labels, gini=_gini(values))
    return DistributionResult(values, labels, tail_index=_tail_index(values, tail))


def _describe_samples(
    values: Array,
    labels: list[str],
    kind: str,
    orders: tuple[int, int],
    *,
    absolute: bool,
) -> DistributionResult:
    """Return the description of a kind that describes every sample."""
    if kind == 'moments':
        measured = np.abs(values) if absolute else values
        return DistributionResult(
            values,
            labels,
            mean=np.asarray([column.mean() for column in measured.T]),
            variance=np.asarray([column.var() for column in measured.T]),
            skewness=stats.skew(measured, axis=0),
            kurtosis=stats.kurtosis(measured, axis=0),
        )
    if kind == 'moment-ratio':
        ratio, cumulant = _moment_ratio(values, orders)
        return DistributionResult(values, labels, ratio=ratio, cumulant=cumulant)
    if kind == 'dispersion-index':
        return DistributionResult(values, labels, index=np.asarray([_dispersion_index(column) for column in values.T]))
    if kind == 'records':
        return DistributionResult(values, labels, records=_record_counts(values.T))
    return DistributionResult(values, labels, distance=_total_variation(values[:, 0], values[:, 1]))


def _histogram(array: Array, bins: int, *, log: bool) -> tuple[Array, Array]:
    """Return the edges and the probability density of a histogram, with bins equally spaced in the logarithm."""
    if log:
        if np.any(array <= 0):
            error_msg = 'Logarithmic bins need positive values. Drop zeros or use linear bins with `log=False`.'
            raise AnalysisError(error_msg)
        edges = np.logspace(np.log10(array.min()), np.log10(array.max()), bins + 1)
    else:
        edges = np.linspace(array.min(), array.max(), bins + 1)
    density, edges = np.histogram(array, bins=edges, density=True)
    return edges, density


def _ccdf(values: npt.ArrayLike) -> tuple[Array, Array]:
    """Return the sorted distinct values of a sample and the share of samples at or above each of them."""
    array = np.sort(_read_real(values, 'values'))
    distinct, first = np.unique(array, return_index=True)
    return distinct, 1 - first / array.size


def _moment_ratio(samples: Array, orders: tuple[int, int]) -> tuple[Array, Array]:
    """Return the ratio `<x^p> / <x^q>^(p / q)` of every column and one minus it over its Gaussian value."""
    high, low = orders
    power = high / low
    numerator = np.asarray([np.mean(column**high) for column in samples.T])
    denominator = np.asarray([np.mean(column**low) for column in samples.T]) ** power
    gaussian = math.prod(range(high - 1, 0, -2)) / math.prod(range(low - 1, 0, -2)) ** power
    return numerator / denominator, 1 - numerator / (gaussian * denominator)


def _dispersion_index(values: npt.ArrayLike) -> float:
    """Return the variance over the mean of counts, raising an error when the mean is 0."""
    array = np.asarray(values, dtype=float)
    if array.mean() == 0:
        error_msg = 'The counts have mean 0, so their dispersion index is undefined. Use counts with a positive mean.'
        raise AnalysisError(error_msg)
    return float(array.var() / array.mean())


def _gini(array: Array) -> float:
    """Return the Gini coefficient of nonnegative values."""
    array = np.sort(array)
    if array.size == 0 or array.sum() == 0:
        return 0.0
    ranks = np.arange(1, array.size + 1)
    return float(np.sum((2 * ranks - array.size - 1) * array) / (array.size * array.sum()))


def _tail_index(array: Array, tail: int) -> float:
    """Return the Hill estimate of the tail index from the `tail` largest samples."""
    array = np.sort(array)[::-1]
    if not 0 < tail < array.size:
        error_msg = f'tail = {tail} must lie between 1 and the number of samples minus 1, {array.size - 1}.'
        raise AnalysisError(error_msg)
    return float(1 / np.mean(np.log(array[:tail] / array[tail])))


def _record_counts(sequences: Array) -> npt.NDArray[np.int64]:
    """Return the number of records of every row, the values larger than every value before them."""
    previous = np.maximum.accumulate(sequences, axis=1)
    first = np.ones((sequences.shape[0], 1), dtype=bool)
    records = np.concatenate([first, sequences[:, 1:] > previous[:, :-1]], axis=1)
    return records.sum(axis=1).astype(np.int64)


def _total_variation(groups: Array, units: Array) -> float:
    """Return half the L1 distance between the distributions of two groups across units."""
    labels = np.unique(groups)
    if labels.size != GROUPS:
        error_msg = f'The total variation compares two groups, not {labels.size}. Give items of two groups.'
        raise AnalysisError(error_msg)
    _, places = np.unique(units, return_inverse=True)
    first = np.bincount(places[groups == labels[0]], minlength=places.max() + 1)
    second = np.bincount(places[groups == labels[1]], minlength=places.max() + 1)
    return float(0.5 * np.sum(np.abs(first / first.sum() - second / second.sum())))
