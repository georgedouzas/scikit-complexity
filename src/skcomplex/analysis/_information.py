"""Measure information among series: mutual information, transfer, storage, total correlations, and divergences."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Self

import ennemi
import numpy as np
import numpy.typing as npt
import ordpy
from scipy import stats

from ..core.errors import AnalysisError
from ._base import BaseAnalysis, Source
from ._entropy import EntropyAnalysis
from ._kinds import DataKind

type Array = npt.NDArray[np.float64]
type Columns = npt.NDArray[np.generic]
KINDS = (
    'mutual',
    'auto-mutual',
    'transfer',
    'active-storage',
    'correlation-information',
    'total-correlation',
    'dual-total-correlation',
    'o-information',
    'o-information-gradients',
    'kl-divergence',
)
PAIRWISE = ('mutual', 'transfer', 'kl-divergence')
JOINT = ('total-correlation', 'dual-total-correlation', 'o-information', 'o-information-gradients')
SYMBOLS = ('bins', 'quantiles', 'ordinal')
TRANSFER_ESTIMATORS = ('plugin', 'knn')
JOINT_ESTIMATORS = ('plugin', 'gaussian', 'knn')
SETTINGS = {
    'mutual': {'symbols', 'base'},
    'auto-mutual': {'symbols', 'max_delay', 'base'},
    'transfer': {'symbols', 'history', 'lag', 'estimator', 'k', 'base'},
    'active-storage': {'symbols', 'history', 'base'},
    'correlation-information': {'symbols', 'dimension', 'base'},
    'total-correlation': {'symbols', 'estimator', 'k', 'base'},
    'dual-total-correlation': {'symbols', 'estimator', 'k', 'base'},
    'o-information': {'symbols', 'estimator', 'k', 'base'},
    'o-information-gradients': {'symbols', 'estimator', 'k', 'base'},
    'kl-divergence': {'base'},
}
"""Settings every kind reads, besides `kind` and `variables`."""
SYMBOL_SETTINGS = {None: set(), 'bins': {'bins'}, 'quantiles': {'bins'}, 'ordinal': {'order', 'delay'}}
"""Settings every symbolization reads."""


class InformationAnalysis(BaseAnalysis):
    """Information among the series of a source, of one kind.

    The kinds measure pairs of series, every series, or all series together:

    - `'mutual'`: the mutual information `H(X) + H(Y) - H(X, Y)` of every pair of series, by plug-in counts.
    - `'transfer'`: the transfer entropy `H(Y_t | Y_past) - H(Y_t | Y_past, X_{t-lag})` from every series to every
      series, the target conditioned on `history` past values, by plug-in counts or by the Kraskov-Stögbauer-Grassberger
      nearest-neighbour estimate of `k` neighbours.
    - `'kl-divergence'`: the Kullback-Leibler divergence of every series, a probability vector, from every series.
    - `'auto-mutual'`: the mutual information of every series with itself delayed by 0 to `max_delay` steps.
    - `'active-storage'`: the information the past `history` values of every series hold about its next value.
    - `'correlation-information'`: Lindgren's correlation information `k_m = ΔS_{m-1} - ΔS_m` of every series for
      block lengths 2 to `dimension`.
    - `'total-correlation'`, `'dual-total-correlation'`, `'o-information'`: the information all series share at any
      order, the information two or more of them share, and their difference, positive for redundancy and negative for
      synergy, by plug-in counts with `dit`, the closed form of a fitted Gaussian, or chains of neighbour estimates.
    - `'o-information-gradients'`: the O-information every series adds to the others.

    Every kind but `'kl-divergence'` reads the series as symbols, or first turns them into symbols with `symbols`:
    `'bins'` for `bins` bins of equal width, `'quantiles'` for `bins` equally populated bins, or `'ordinal'` for the
    rank pattern of every `order` values spaced by `delay` steps.

    Args:
        kind:
            The information to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        symbols:
            `None` to read the series as symbols, or `'bins'`, `'quantiles'`, or `'ordinal'` to symbolize them.

        bins:
            The number of symbols of bins and quantiles.

        order:
            The length of ordinal patterns.

        delay:
            The number of steps between the values of an ordinal pattern.

        history:
            The number of past values of the target of the transfer entropy, or of the series of the storage.

        lag:
            The delay of the source of the transfer entropy.

        max_delay:
            The largest delay of the auto mutual information.

        dimension:
            The largest block length of the correlation information.

        estimator:
            `'plugin'` for counts of symbols, `'gaussian'` for a fitted Gaussian, or `'knn'` for nearest neighbours.

        k:
            The number of neighbours of the `'knn'` estimator.

        base:
            The base of the logarithm, 2 for bits.

    Attributes:
        values_:
            The series, one per column.

        labels_:
            The label of every series.

        symbols_:
            The symbols the kinds measure, one column per series, the series themselves without `symbols`.

        information_:
            A matrix with the measure from the series of every row to the series of every column for `'mutual'`,
            `'transfer'`, and `'kl-divergence'`, `nan` from a series to itself for `'transfer'`; one value per series,
            with one column per delay or block length for `'auto-mutual'` and `'correlation-information'`; one value
            per series for `'o-information-gradients'`; and a single value for the other kinds of all series together.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import InformationAnalysis
        >>> x = np.random.default_rng(0).integers(0, 2, 10000)
        >>> transfer = InformationAnalysis('transfer').check().prepare(np.column_stack([x, np.roll(x, 1)])).run()
        >>> transfer.information_.round(1).tolist()
        [[nan, 1.0], [0.0, nan]]
        >>> xor = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
        >>> round(InformationAnalysis('o-information').check().prepare(xor).run().information_, 6)
        -1.0
    """

    def __init__(
        self,
        kind: str = 'mutual',
        variables: Sequence[str] | None = None,
        *,
        symbols: str | None = None,
        bins: int = 2,
        order: int = 3,
        delay: int = 1,
        history: int = 1,
        lag: int = 1,
        max_delay: int = 100,
        dimension: int = 8,
        estimator: str = 'plugin',
        k: int = 3,
        base: float = 2.0,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.symbols = symbols
        self.bins = bins
        self.order = order
        self.delay = delay
        self.history = history
        self.lag = lag
        self.max_delay = max_delay
        self.dimension = dimension
        self.estimator = estimator
        self.k = k
        self.base = base

    def _check(self) -> Self:
        """Validate the kind, the choices, the numeric settings, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_choice('symbols', self.symbols, (None, *SYMBOLS))
        self._check_settings('kind', SETTINGS[self.kind] | SYMBOL_SETTINGS[self.symbols] | {'variables'})
        estimators = TRANSFER_ESTIMATORS if self.kind == 'transfer' else JOINT_ESTIMATORS
        self._check_choice('estimator', self.estimator, estimators)
        counts = {
            'bins': self.bins,
            'order': self.order,
            'delay': self.delay,
            'history': self.history,
            'lag': self.lag,
            'k': self.k,
        }
        small = [f'{name}={value!r}' for name, value in counts.items() if value < 1]
        if small:
            error_msg = f'{type(self).__name__} has {", ".join(small)}. Use 1 or more.'
            raise AnalysisError(error_msg)
        if self.max_delay < 0 or self.dimension < 2 or self.base <= 0:  # noqa: PLR2004
            error_msg = (
                f'{type(self).__name__} has max_delay={self.max_delay!r}, dimension={self.dimension!r}, and '
                f'base={self.base!r}. Use a max_delay of 0 or more, a dimension of 2 or more, and a positive base.'
            )
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the series of the source as columns and turn them into symbols."""
        series = self._read_series(source, self.variables)
        self.values_ = series.values
        self.labels_ = series.labels
        if self.symbols is None:
            self.symbols_ = self.values_
            return self
        self.symbols_ = np.column_stack(
            [
                self._symbolize(column, self.symbols, self.bins, order=self.order, delay=self.delay)
                for column in self.values_.T
            ],
        )
        return self

    def _compute_pair(self, first: Columns, second: Columns, *, same: bool) -> float:
        """Return the mutual information, the transfer entropy, or the divergence from one series to another."""
        if self.kind == 'transfer' and same:
            return math.nan
        if self.kind == 'mutual':
            return self._mutual_information(first, second, base=self.base)
        if self.kind == 'transfer':
            return self._transfer_entropy(
                first,
                second,
                history=self.history,
                lag=self.lag,
                base=self.base,
                estimator=self.estimator,
                k=self.k,
            )
        return self._kl_divergence(first, second, base=self.base)

    def _compute_series(self, symbols: Columns) -> float | Array:
        """Return the auto mutual information, the storage, or the correlation information of one series."""
        if self.kind == 'auto-mutual':
            return self._delayed_information(symbols, self.max_delay, base=self.base)
        if self.kind == 'active-storage':
            return self._active_information_storage(symbols, self.history, base=self.base)
        return self._correlation_information(symbols, self.dimension, base=self.base)

    def _compute_joint(self, columns: Columns) -> float | Array:
        """Return the total correlations, the O-information, or its gradients of all series together."""
        scale = math.log(self.base)
        if self.kind == 'total-correlation':
            return self._total(columns, self.estimator, self.k) / scale
        if self.kind == 'dual-total-correlation':
            return self._dual(columns, self.estimator, self.k) / scale
        whole = self._o_information(columns, self.estimator, self.k, self.base)
        if self.kind == 'o-information':
            return whole
        parts = [
            self._o_information(np.delete(columns, index, axis=1), self.estimator, self.k, self.base)
            for index in range(columns.shape[1])
        ]
        return whole - np.asarray(parts, dtype=float)

    def _run(self) -> Self:
        """Compute the information of every pair of series, of every series, or of all series together."""
        columns = self.symbols_
        if self.kind in PAIRWISE:
            self.information_ = np.asarray(
                [
                    [self._compute_pair(first, second, same=row == column) for column, second in enumerate(columns.T)]
                    for row, first in enumerate(columns.T)
                ],
                dtype=float,
            )
        elif self.kind in JOINT:
            self.information_ = self._compute_joint(columns)
        else:
            self.information_ = np.asarray([self._compute_series(column) for column in columns.T], dtype=float)
        return self

    @staticmethod
    def _rank_index(patterns: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Return the lexicographic rank of every permutation, one per row, as its Lehmer code."""
        order = patterns.shape[1]
        later_smaller = np.triu(patterns[:, None, :] < patterns[:, :, None], k=1).sum(axis=2)
        weights = np.asarray([math.factorial(order - 1 - position) for position in range(order)], dtype=np.int64)
        return later_smaller.astype(np.int64) @ weights

    @staticmethod
    def _ordinal_patterns(values: npt.ArrayLike, order: int = 3, *, delay: int = 1) -> npt.NDArray[np.int64]:
        """Return the ordinal pattern of every window of a series as the lexicographic rank of its permutation."""
        array = np.asarray(values, dtype=float)
        patterns = np.asarray(ordpy.ordinal_sequence(array, dx=order, taux=delay), dtype=np.int64)
        return InformationAnalysis._rank_index(patterns.reshape(-1, order))

    @staticmethod
    def _symbolize(
        values: npt.ArrayLike,
        method: str = 'quantiles',
        n: int = 2,
        *,
        order: int = 3,
        delay: int = 1,
    ) -> npt.NDArray[np.int64]:
        """Return a series as `n` equal-width bins, `n` equally filled bins, or ordinal patterns of length `order`."""
        array = BaseAnalysis._read_real(values, 'values')
        if method == 'bins':
            edges = np.linspace(array.min(), array.max(), n + 1)[1:-1]
            return np.digitize(array, edges).astype(np.int64)
        if method == 'quantiles':
            edges = np.quantile(array, np.linspace(0, 1, n + 1)[1:-1])
            return np.digitize(array, edges).astype(np.int64)
        return InformationAnalysis._ordinal_patterns(array, order, delay=delay)

    @staticmethod
    def _codes(items: Sequence[Hashable]) -> list[int]:
        """Return one integer code per distinct item."""
        table: dict[Hashable, int] = {}
        return [table.setdefault(item, len(table)) for item in items]

    @staticmethod
    def _joint_entropy(*columns: Sequence[Hashable], base: float) -> float:
        """Return the entropy of aligned columns taken together."""
        return EntropyAnalysis._shannon(InformationAnalysis._codes(list(zip(*columns, strict=True))), base=base)

    @staticmethod
    def _histories(values: npt.NDArray[np.generic], history: int, start: int) -> list[tuple[object, ...]]:
        """Return the `history` values before every time from `start` on."""
        return [tuple(values[t - history : t]) for t in range(start, values.size)]

    @staticmethod
    def _kl_divergence(p: npt.ArrayLike, q: npt.ArrayLike, *, base: float = 2.0) -> float:
        """Return the Kullback-Leibler divergence of a distribution `p` from a reference `q`, infinite off support."""
        return float(stats.entropy(np.asarray(p, dtype=float), np.asarray(q, dtype=float), base=base))

    @staticmethod
    def _correlation_information(symbols: npt.ArrayLike, max_m: int = 8, *, base: float = 2.0) -> Array:
        """Return Lindgren's correlation information `k_m = ΔS_{m-1} - ΔS_m` for block lengths 2 to `max_m`."""
        differences = np.diff(EntropyAnalysis._block_entropies(symbols, max_m, base=base))
        return -np.diff(differences)

    @staticmethod
    def _mutual_information(first: npt.ArrayLike, second: npt.ArrayLike, *, base: float = 2.0) -> float:
        """Return the mutual information `H(X) + H(Y) - H(X, Y)` of two symbol sequences by plug-in counts."""
        x, y = list(np.asarray(first).ravel()), list(np.asarray(second).ravel())
        joint = list(zip(x, y, strict=True))
        return (
            EntropyAnalysis._shannon(x, base=base)
            + EntropyAnalysis._shannon(y, base=base)
            - EntropyAnalysis._shannon(InformationAnalysis._codes(joint), base=base)
        )

    @staticmethod
    def _delayed_information(symbols: npt.NDArray[np.generic], max_delay: int, *, base: float) -> Array:
        """Return the mutual information of symbols with themselves delayed by 0 to `max_delay` steps."""
        return np.asarray(
            [
                InformationAnalysis._mutual_information(symbols[: symbols.size - delay], symbols[delay:], base=base)
                for delay in range(max_delay + 1)
            ],
        )

    @staticmethod
    def _active_information_storage(symbols: npt.ArrayLike, history: int = 1, *, base: float = 2.0) -> float:
        """Return `I(X_t ; X_{t-history}, ..., X_{t-1})`, the information the past symbols hold about the next one."""
        x = np.asarray(symbols).ravel()
        future = [(value,) for value in x[history:]]
        past = InformationAnalysis._histories(x, history, history)
        return float(
            InformationAnalysis._joint_entropy(future, base=base)
            + InformationAnalysis._joint_entropy(past, base=base)
            - InformationAnalysis._joint_entropy(future, past, base=base),
        )

    @staticmethod
    def _knn_transfer(x: Array, y: Array, history: int, lag: int, k: int) -> float:
        """Return the transfer entropy in nats by the nearest-neighbour conditional mutual information."""
        start = max(history, lag)
        past = np.column_stack([y[start - step : y.size - step] for step in range(1, history + 1)])
        estimate = ennemi.estimate_mi(y[start:], x[start - lag : x.size - lag], cond=past, k=k)
        return float(np.asarray(estimate).item())

    @staticmethod
    def _transfer_entropy(
        source: npt.ArrayLike,
        target: npt.ArrayLike,
        *,
        history: int = 1,
        lag: int = 1,
        base: float = 2.0,
        estimator: str = 'plugin',
        k: int = 3,
    ) -> float:
        """Return `H(Y_t | Y_past) - H(Y_t | Y_past, X_{t-lag})` by plug-in counts or nearest neighbours."""
        if estimator == 'knn':
            x, y = np.asarray(source, dtype=float).ravel(), np.asarray(target, dtype=float).ravel()
            return InformationAnalysis._knn_transfer(x, y, history, lag, k) / math.log(base)
        x, y = np.asarray(source).ravel(), np.asarray(target).ravel()
        start = max(history, lag)
        future = [(value,) for value in y[start:]]
        past = InformationAnalysis._histories(y, history, start)
        driver = [(x[t - lag],) for t in range(start, y.size)]
        return float(
            InformationAnalysis._joint_entropy(future, past, base=base)
            - InformationAnalysis._joint_entropy(past, base=base)
            - InformationAnalysis._joint_entropy(future, past, driver, base=base)
            + InformationAnalysis._joint_entropy(past, driver, base=base),
        )

    @staticmethod
    def _row_codes(columns: Columns) -> npt.NDArray[np.int64]:
        """Return one integer code per distinct row."""
        _, codes = np.unique(columns, axis=0, return_inverse=True)
        return np.asarray(codes, dtype=np.int64).ravel()

    @staticmethod
    def _log_determinant(columns: Array) -> float:
        """Return the logarithm of the determinant of the sample covariance of the columns."""
        if columns.shape[1] == 0:
            return 0.0
        _, value = np.linalg.slogdet(np.atleast_2d(np.cov(columns, rowvar=False, bias=True)))
        return float(value)

    @staticmethod
    def _knn_chain(target: Array, others: Array, k: int, given: Array | None = None) -> float:
        """Return `I(target ; others | given)` in nats as a chain of k-nearest-neighbour conditional informations."""
        condition = np.empty((target.size, 0)) if given is None else given
        total = 0.0
        for column in range(others.shape[1]):
            cond = np.column_stack([condition, others[:, :column]])
            estimate = ennemi.estimate_mi(target, others[:, column], cond=cond if cond.shape[1] else None, k=k)
            total += float(np.asarray(estimate).item())
        return total

    @staticmethod
    def _dit_distribution(columns: Columns) -> object:
        """Return the `dit` distribution of the empirical frequencies of the rows."""
        counts = Counter(tuple(row) for row in columns.tolist())
        total = sum(counts.values())
        return BaseAnalysis._import('dit').Distribution(list(counts), [count / total for count in counts.values()])

    @staticmethod
    def _total(columns: Columns, estimator: str, k: int) -> float:
        """Return the total correlation `Σ H(X_i) - H(X)` of the columns in nats."""
        if estimator == 'plugin':
            return float(
                BaseAnalysis._import('dit').multivariate.total_correlation(
                    InformationAnalysis._dit_distribution(columns),
                ),
            ) * math.log(
                2,
            )
        values = BaseAnalysis._read_real(columns, 'data')
        if estimator == 'gaussian':
            return -InformationAnalysis._log_determinant(values / values.std(axis=0)) / 2
        return sum(
            InformationAnalysis._knn_chain(values[:, index], values[:, :index], k)
            for index in range(1, values.shape[1])
        )

    @staticmethod
    def _dual(columns: Columns, estimator: str, k: int) -> float:
        """Return the dual total correlation `H(X) - Σ H(X_i | X_-i)` of the columns in nats."""
        if estimator == 'plugin':
            return float(
                BaseAnalysis._import('dit').multivariate.dual_total_correlation(
                    InformationAnalysis._dit_distribution(columns),
                ),
            ) * math.log(2)
        values = BaseAnalysis._read_real(columns, 'data')
        if estimator == 'gaussian':
            precision = np.linalg.inv(np.atleast_2d(np.cov(values, rowvar=False, bias=True)))
            return (InformationAnalysis._log_determinant(values) + float(np.sum(np.log(np.diag(precision))))) / 2
        parts = sum(
            InformationAnalysis._knn_chain(values[:, index], np.delete(values, index, axis=1), k)
            for index in range(values.shape[1])
        )
        return parts - InformationAnalysis._total(values, estimator, k)

    @staticmethod
    def _o_information(columns: Columns, estimator: str, k: int, base: float) -> float:
        """Return the O-information, total minus dual total correlation, in the base of the logarithm."""
        return (
            InformationAnalysis._total(columns, estimator, k) - InformationAnalysis._dual(columns, estimator, k)
        ) / math.log(base)
