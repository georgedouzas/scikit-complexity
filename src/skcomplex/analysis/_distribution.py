"""Describe the distribution of a sample: histograms, survival functions, moments, inequality, tails, and records."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis, Source

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


class DistributionAnalysis(BaseAnalysis):
    """Distribution of the values of a source, of one kind.

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

    Attributes:
        values_:
            The pooled sample, or the samples, one per column.

        labels_:
            The label of every sample.

        edges_:
            The bin edges of the histogram.

        density_:
            The probability density in every bin of the histogram.

        support_:
            The sorted distinct values of the complementary cumulative distribution.

        survival_:
            The share of samples at or above every distinct value, `P(X >= x)`.

        mean_:
            The mean of every sample.

        variance_:
            The variance of every sample.

        skewness_:
            The skewness of every sample.

        kurtosis_:
            The excess kurtosis of every sample, 0 for Gaussian samples.

        ratio_:
            The moment ratio of every sample.

        cumulant_:
            The normalized cumulant of every sample.

        index_:
            The dispersion index of every sample.

        gini_:
            The Gini coefficient.

        tail_index_:
            The Hill estimate of the tail index.

        records_:
            The number of records of every series, the first value included.

        distance_:
            The total variation distance between the two groups.

    Examples:
        >>> from skcomplex.analysis import DistributionAnalysis
        >>> histogram = DistributionAnalysis('histogram', bins=3, log=True).check().prepare([1, 2, 4, 8]).run()
        >>> histogram.edges_.round(2).tolist()
        [1.0, 2.0, 4.0, 8.0]
        >>> DistributionAnalysis('ccdf').check().prepare([1, 1, 2, 4]).run().survival_.tolist()
        [1.0, 0.5, 0.25]
        >>> DistributionAnalysis('gini').check().prepare([0, 0, 0, 4]).run().gini_
        0.75
        >>> DistributionAnalysis('moment-ratio').check().prepare([1.0, -1.0, 1.0]).run().cumulant_.round(4).tolist()
        [0.6667]
        >>> DistributionAnalysis('dispersion-index').check().prepare([1, 3, 1, 3]).run().index_.tolist()
        [0.5]
        >>> DistributionAnalysis('total-variation').check().prepare([[0, 0], [0, 0], [1, 1], [1, 1]]).run().distance_
        1.0
    """

    def __init__(
        self,
        kind: str = 'histogram',
        variables: Sequence[str] | None = None,
        *,
        group: str | None = None,
        bins: int = 30,
        log: bool = False,
        absolute: bool = False,
        orders: tuple[int, int] = (4, 2),
        tail: int = 10,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.group = group
        self.bins = bins
        self.log = log
        self.absolute = absolute
        self.orders = orders
        self.tail = tail

    def _check(self) -> Self:
        """Validate the kind, that the kind reads every changed setting, the bins, the tail, and the orders."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables', 'group'})
        if self.bins < 1 or self.tail < 1:
            error_msg = f'{type(self).__name__} has bins={self.bins!r} and tail={self.tail!r}. Use 1 or more.'
            raise AnalysisError(error_msg)
        if len(self.orders) != 2 or any(order <= 0 or order % 2 for order in self.orders):  # noqa: PLR2004
            error_msg = f'{type(self).__name__} has orders={self.orders!r}. Use two positive even orders.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the ensemble kind for records, and the series kind otherwise."""
        return DataKind.ENSEMBLE if self.kind == 'records' else DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the samples of the source, pooled into one sample for the pooled kinds."""
        series = self._read_samples_of(source, self.variables, self.group)
        values = self._read_real(series.values, 'values')
        self.values_ = values.ravel() if self.kind in POOLED else values
        self.labels_ = series.labels
        if self.kind == 'total-variation' and self.values_.shape[1] != UNIT_COLUMNS:
            error_msg = (
                f'{type(self).__name__} of kind total-variation reads one row per item with its group and its unit, '
                f'not an array of shape {self.values_.shape}.'
            )
            raise AnalysisError(error_msg)
        return self

    def _run(self, inputs: object = None) -> Self:
        """Compute the description of the samples."""
        if self.kind == 'histogram':
            self.edges_, self.density_ = self._histogram(self.values_, self.bins, log=self.log)
        elif self.kind == 'ccdf':
            self.support_, self.survival_ = self._ccdf(self.values_)
        elif self.kind == 'moments':
            samples = np.abs(self.values_) if self.absolute else self.values_
            self.mean_ = np.asarray([column.mean() for column in samples.T])
            self.variance_ = np.asarray([column.var() for column in samples.T])
            self.skewness_ = stats.skew(samples, axis=0)
            self.kurtosis_ = stats.kurtosis(samples, axis=0)
        elif self.kind == 'moment-ratio':
            self.ratio_, self.cumulant_ = self._moment_ratio(self.values_, self.orders)
        elif self.kind == 'dispersion-index':
            self.index_ = np.asarray([self._dispersion_index(column) for column in self.values_.T])
        elif self.kind == 'gini':
            self.gini_ = self._gini(self.values_)
        elif self.kind == 'tail-index':
            self.tail_index_ = self._tail_index(self.values_, self.tail)
        elif self.kind == 'records':
            self.records_ = self._record_counts(self.values_.T)
        else:
            self.distance_ = self._total_variation(self.values_[:, 0], self.values_[:, 1])
        return self

    @staticmethod
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

    @staticmethod
    def _ccdf(values: npt.ArrayLike) -> tuple[Array, Array]:
        """Return the sorted distinct values of a sample and the share of samples at or above each of them."""
        array = np.sort(BaseAnalysis._read_real(values, 'values'))
        distinct, first = np.unique(array, return_index=True)
        return distinct, 1 - first / array.size

    @staticmethod
    def _moment_ratio(samples: Array, orders: tuple[int, int]) -> tuple[Array, Array]:
        """Return the ratio `<x^p> / <x^q>^(p / q)` of every column and one minus it over its Gaussian value."""
        high, low = orders
        power = high / low
        numerator = np.asarray([np.mean(column**high) for column in samples.T])
        denominator = np.asarray([np.mean(column**low) for column in samples.T]) ** power
        gaussian = math.prod(range(high - 1, 0, -2)) / math.prod(range(low - 1, 0, -2)) ** power
        return numerator / denominator, 1 - numerator / (gaussian * denominator)

    @staticmethod
    def _dispersion_index(values: npt.ArrayLike) -> float:
        """Return the variance over the mean of counts, raising an error when the mean is 0."""
        array = np.asarray(values, dtype=float)
        if array.mean() == 0:
            error_msg = (
                'The counts have mean 0, so their dispersion index is undefined. Use counts with a positive mean.'
            )
            raise AnalysisError(error_msg)
        return float(array.var() / array.mean())

    @staticmethod
    def _gini(array: Array) -> float:
        """Return the Gini coefficient of nonnegative values."""
        array = np.sort(array)
        if array.size == 0 or array.sum() == 0:
            return 0.0
        ranks = np.arange(1, array.size + 1)
        return float(np.sum((2 * ranks - array.size - 1) * array) / (array.size * array.sum()))

    @staticmethod
    def _tail_index(array: Array, tail: int) -> float:
        """Return the Hill estimate of the tail index from the `tail` largest samples."""
        array = np.sort(array)[::-1]
        if not 0 < tail < array.size:
            error_msg = f'tail = {tail} must lie between 1 and the number of samples minus 1, {array.size - 1}.'
            raise AnalysisError(error_msg)
        return float(1 / np.mean(np.log(array[:tail] / array[tail])))

    @staticmethod
    def _record_counts(sequences: Array) -> npt.NDArray[np.int64]:
        """Return the number of records of every row, the values larger than every value before them."""
        previous = np.maximum.accumulate(sequences, axis=1)
        first = np.ones((sequences.shape[0], 1), dtype=bool)
        records = np.concatenate([first, sequences[:, 1:] > previous[:, :-1]], axis=1)
        return records.sum(axis=1).astype(np.int64)

    @staticmethod
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
