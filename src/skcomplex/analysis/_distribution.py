"""Describe the distribution of a sample: histograms, survival functions, Gini coefficients, tail indices, records."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

type Array = npt.NDArray[np.float64]
KINDS = ('histogram', 'ccdf', 'gini', 'tail-index', 'records')
SETTINGS = {
    'histogram': {'bins', 'log'},
    'ccdf': set(),
    'gini': set(),
    'tail-index': {'tail'},
    'records': set(),
}
"""Settings every kind reads, besides `kind` and `variables`."""


class DistributionAnalysis(BaseAnalysis):
    """Distribution of the values of a source, of one kind.

    Every kind but `'records'` reads the values of the chosen series pooled into one sample:

    - `'histogram'`: the edges and the probability density of `bins` bins, equally spaced in the logarithm of the
      values with `log=True` for heavy tails.
    - `'ccdf'`: the complementary cumulative distribution, the share of samples at or above every distinct value.
    - `'gini'`: the Gini coefficient of nonnegative values, 0 for equality and close to 1 when one holds everything.
    - `'tail-index'`: the Hill estimate of the tail index from the `tail` largest positive samples, the exponent
      `alpha - 1` of the survival function of a density `p(x) ∝ x^-alpha`.
    - `'records'`: the number of records of every series, the values larger than every value before them, about the
      harmonic number of the length for independent values.

    Args:
        kind:
            The description to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        bins:
            The number of bins of the histogram.

        log:
            Whether the bins of the histogram are equally spaced in the logarithm of the values, which needs positive
            values.

        tail:
            The number of largest samples of the Hill estimate.

    Attributes:
        values_:
            The pooled sample, or the series, one per column, for `'records'`.

        labels_:
            The label of every series.

        edges_:
            The bin edges of the histogram.

        density_:
            The probability density in every bin of the histogram.

        support_:
            The sorted distinct values of the complementary cumulative distribution.

        survival_:
            The share of samples at or above every distinct value, `P(X >= x)`.

        gini_:
            The Gini coefficient.

        tail_index_:
            The Hill estimate of the tail index.

        records_:
            The number of records of every series, the first value included.

    Examples:
        >>> from skcomplex.analysis import DistributionAnalysis
        >>> histogram = DistributionAnalysis('histogram', bins=3, log=True).check().prepare([1, 2, 4, 8]).run()
        >>> histogram.edges_.round(2).tolist()
        [1.0, 2.0, 4.0, 8.0]
        >>> DistributionAnalysis('ccdf').check().prepare([1, 1, 2, 4]).run().survival_.tolist()
        [1.0, 0.5, 0.25]
        >>> DistributionAnalysis('gini').check().prepare([0, 0, 0, 4]).run().gini_
        0.75
    """

    def __init__(
        self,
        kind: str = 'histogram',
        variables: Sequence[str] | None = None,
        *,
        bins: int = 30,
        log: bool = False,
        tail: int = 10,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.bins = bins
        self.log = log
        self.tail = tail

    def _check(self) -> Self:
        """Validate the kind, that the kind reads every changed setting, and the numbers of bins and tail samples."""
        if self.kind not in KINDS:
            error_msg = f'{type(self).__name__} has kind={self.kind!r}. Use one of {", ".join(KINDS)}.'
            raise AnalysisError(error_msg)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.bins < 1 or self.tail < 1:
            error_msg = f'{type(self).__name__} has bins={self.bins!r} and tail={self.tail!r}. Use 1 or more.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the ensemble kind for records, and the series kind otherwise."""
        return DataKind.ENSEMBLE if self.kind == 'records' else DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the series of the source, pooled into one sample except for records."""
        series = self._read_series(source, self.variables)
        values = self._read_real(series.values, 'values')
        self.values_ = values if self.kind == 'records' else values.ravel()
        self.labels_ = series.labels
        return self

    def _run(self) -> Self:
        """Compute the description of the sample."""
        if self.kind == 'histogram':
            self.edges_, self.density_ = self._histogram(self.values_, self.bins, log=self.log)
        elif self.kind == 'ccdf':
            self.support_, self.survival_ = self._ccdf(self.values_)
        elif self.kind == 'gini':
            self.gini_ = self._gini(self.values_)
        elif self.kind == 'tail-index':
            self.tail_index_ = self._tail_index(self.values_, self.tail)
        else:
            self.records_ = self._record_counts(self.values_.T)
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
