"""Measure information sample by sample: local entropy, local mutual information, and local transfer entropy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats

from ..errors import AnalysisError
from ..types import AnyArray, Array, DataKind
from ._base import BaseAnalysis
from ._information import InformationAnalysis
from .types import Source

KINDS = ('entropy', 'mutual', 'transfer')
ESTIMATORS = ('plugin', 'gaussian')
SETTINGS = {
    'entropy': {'estimator', 'base'},
    'mutual': {'estimator', 'base'},
    'transfer': {'history', 'lag', 'estimator', 'base'},
}
"""Settings every kind reads, besides `kind` and `variables`."""


class LocalInformationAnalysis(BaseAnalysis):
    """Information of the series of a source sample by sample, whose mean over the samples is the global measure.

    The kinds measure every series or every pair of series:

    - `'entropy'`: the local entropy `-log p(x)` of every sample of every series.
    - `'mutual'`: the local mutual information `log p(x, y) / (p(x) p(y))` of every sample of every pair of series,
      negative where the series mislead about each other.
    - `'transfer'`: the local transfer entropy `log p(y_t | y_past, x_{t-lag}) / p(y_t | y_past)` from every series
      to every series at every time from `max(history, lag)` on.

    The probabilities are the frequencies of symbols, or the densities of a fitted Gaussian.

    Args:
        kind:
            The local information to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        history:
            The number of past target values of the transfer entropy.

        lag:
            The delay of the source of the transfer entropy.

        estimator:
            `'plugin'` for the frequencies of symbols, or `'gaussian'` for the densities of a fitted Gaussian.

        base:
            The base of the logarithm, 2 for bits.

    Attributes:
        values_:
            The series, one per column.

        labels_:
            The label of every series.

        information_:
            One row of local entropies per series for `'entropy'`, and for `'mutual'` and `'transfer'` one row of
            local values from the series of every first index to the series of every second index, `nan` from a series
            to itself.

    Examples:
        >>> from skcomplex.core.analysis import LocalInformationAnalysis
        >>> local = LocalInformationAnalysis('mutual').prepare([[0, 0], [0, 0], [1, 1], [1, 0]]).run()
        >>> local.information_[0, 1].round(3).tolist()
        [0.415, 0.415, 1.0, -0.585]
    """

    def __init__(
        self,
        kind: str = 'entropy',
        variables: Sequence[str] | None = None,
        *,
        history: int = 1,
        lag: int = 1,
        estimator: str = 'plugin',
        base: float = 2.0,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.history = history
        self.lag = lag
        self.estimator = estimator
        self.base = base

    def _check(self) -> None:
        """Validate the kind, the estimator, the numeric settings, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_choice('estimator', self.estimator, ESTIMATORS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.history < 1 or self.lag < 1 or self.base <= 0:
            error_msg = (
                f'{type(self).__name__} has history={self.history!r}, lag={self.lag!r}, and base={self.base!r}. Use '
                'a history and a lag of 1 or more and a positive base.'
            )
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: Source) -> None:
        """Read the series of the source as columns."""
        series = self._read_series(source, self.variables)
        self.values_ = series.values
        self.labels_ = series.labels

    def _compute_pair(self, first: AnyArray, second: AnyArray, *, same: bool) -> Array:
        """Return the local mutual information or the local transfer entropy from one series to another in nats."""
        if same:
            size = first.size - max(self.history, self.lag) if self.kind == 'transfer' else first.size
            return np.full(size, np.nan)
        if self.kind == 'transfer':
            return self._local_transfer_entropy(first, second, self.history, self.lag, self.estimator)
        x, y = first[:, None], second[:, None]
        joint = self._log_probabilities(np.column_stack([x, y]), self.estimator)
        return joint - self._log_probabilities(x, self.estimator) - self._log_probabilities(y, self.estimator)

    def _run(self, inputs: object = None) -> None:
        """Compute the local information of every series or of every pair of series."""
        columns = self.values_.T
        local: list[Array] | list[list[Array]]
        if self.kind == 'entropy':
            local = [-self._log_probabilities(column[:, None], self.estimator) for column in columns]
        else:
            local = [
                [self._compute_pair(first, second, same=row == column) for column, second in enumerate(columns)]
                for row, first in enumerate(columns)
            ]
        self.information_ = np.asarray(local, dtype=float) / math.log(self.base)

    @staticmethod
    def _log_probabilities(columns: AnyArray, estimator: str) -> Array:
        """Return the logarithm of the probability, or of the fitted Gaussian density, of every row in nats."""
        if estimator == 'plugin':
            codes = InformationAnalysis._row_codes(columns)
            counts = np.bincount(codes).astype(float)
            return np.log(counts[codes] / codes.size)
        values = BaseAnalysis._read_real(columns, 'values')
        covariance = np.atleast_2d(np.cov(values, rowvar=False, bias=True))
        density = stats.multivariate_normal(values.mean(axis=0), covariance)
        return np.asarray(density.logpdf(values), dtype=float).reshape(-1)

    @staticmethod
    def _local_transfer_entropy(x: AnyArray, y: AnyArray, history: int, lag: int, estimator: str) -> Array:
        """Return `log p(y_t | y_past, x_{t-lag}) / p(y_t | y_past)` in nats at each time from `max(history, lag)`."""
        start = max(history, lag)
        future = y[start:, None]
        past = np.column_stack([y[start - step : y.size - step] for step in range(1, history + 1)])
        driver = x[start - lag : x.size - lag, None]
        return (
            LocalInformationAnalysis._log_probabilities(np.column_stack([future, past, driver]), estimator)
            - LocalInformationAnalysis._log_probabilities(np.column_stack([past, driver]), estimator)
            - LocalInformationAnalysis._log_probabilities(np.column_stack([future, past]), estimator)
            + LocalInformationAnalysis._log_probabilities(past, estimator)
        )
