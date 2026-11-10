"""Measure information among series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import ennemi
import numpy as np
import numpy.typing as npt
import ordpy
from scipy import stats

from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import AnyArray, Array, DataKind
from ._base import _check_choice, _check_settings, _read_real, _read_series
from ._entropy import _block_entropies, _shannon
from ._utils import load_module
from .types import Source

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
SYMBOL_SETTINGS = {None: set(), 'bins': {'bins'}, 'quantiles': {'bins'}, 'ordinal': {'order', 'delay'}}
MIN_DIMENSION = 2
TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class InformationResult:
    """A result of `information`, the information among series.

    Args:
        values:
            The series, one per column.

        labels:
            The label of every series.

        symbols:
            The symbols the kinds measure, one column per series, the series themselves without `symbols`.

        information:
            A matrix with the measure from the series of every row to the series of every column for `'mutual'`,
            `'transfer'`, and `'kl-divergence'`, `nan` from a series to itself for `'transfer'`. One value per series,
            with one column per delay or block length for `'auto-mutual'` and `'correlation-information'`. One value
            per series for `'o-information-gradients'`. A single value for the other kinds of all series together.
    """

    values: Array
    labels: list[str]
    symbols: AnyArray
    information: float | Array


def information(
    source: Source,
    *,
    kind: str = 'mutual',
    variables: Sequence[str] | None = None,
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
) -> InformationResult:
    """Return the information among the series of a source, of one kind.

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

    It appears in many fields. It measures the flow of information between regions of the brain, between markets, and
    between the members of a flock.

    Args:
        source:
            The trajectory, or an array with one row per time and one column per series.

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

    Returns:
        result:
            The series, their symbols, and the information among them.

    Raises:
        AnalysisError:
            A choice is unknown, a setting the kind does not read is changed, or a setting is out of range.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import information
        >>> x = np.random.default_rng(0).integers(0, 2, 10000)
        >>> transfer = information(np.column_stack([x, np.roll(x, 1)]), kind='transfer')
        >>> transfer.information.round(1).tolist()
        [[nan, 1.0], [0.0, nan]]
        >>> xor = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
        >>> round(information(xor, kind='o-information').information, 6)
        -1.0
    """
    _check_choice('information', 'kind', kind, KINDS)
    _check_choice('information', 'symbols', symbols, (None, *SYMBOLS))
    _check_settings(
        information,
        'kind',
        SETTINGS[kind] | SYMBOL_SETTINGS[symbols] | {'variables'},
        kind=kind,
        variables=variables,
        symbols=symbols,
        bins=bins,
        order=order,
        delay=delay,
        history=history,
        lag=lag,
        max_delay=max_delay,
        dimension=dimension,
        estimator=estimator,
        k=k,
        base=base,
    )
    estimators = TRANSFER_ESTIMATORS if kind == 'transfer' else JOINT_ESTIMATORS
    _check_choice('information', 'estimator', estimator, estimators)
    counts = {'bins': bins, 'order': order, 'delay': delay, 'history': history, 'lag': lag, 'k': k}
    small = [f'{name}={value!r}' for name, value in counts.items() if value < 1]
    if small:
        error_msg = f'information has {", ".join(small)}. Use 1 or more.'
        raise AnalysisError(error_msg)
    if max_delay < 0 or dimension < MIN_DIMENSION or base <= 0:
        error_msg = (
            f'information has max_delay={max_delay!r}, dimension={dimension!r}, and base={base!r}. Use a max_delay '
            'of 0 or more, a dimension of 2 or more, and a positive base.'
        )
        raise AnalysisError(error_msg)
    series = _read_series(source, variables)
    columns: AnyArray = series.values
    if symbols is not None:
        columns = np.column_stack(
            [_symbolize(column, symbols, bins, order=order, delay=delay) for column in series.values.T],
        )
    measure: float | Array
    if kind in PAIRWISE:
        measure = np.asarray(
            [
                [
                    _compute_pair(
                        kind,
                        first,
                        second,
                        history=history,
                        lag=lag,
                        estimator=estimator,
                        k=k,
                        base=base,
                        same=row == column,
                    )
                    for column, second in enumerate(columns.T)
                ]
                for row, first in enumerate(columns.T)
            ],
            dtype=float,
        )
    elif kind in JOINT:
        measure = _compute_joint(kind, columns, estimator, k, base)
    else:
        measure = np.asarray(
            [
                _compute_series(kind, column, max_delay=max_delay, history=history, dimension=dimension, base=base)
                for column in columns.T
            ],
            dtype=float,
        )
    return InformationResult(series.values, series.labels, columns, measure)


def _compute_pair(
    kind: str,
    first: AnyArray,
    second: AnyArray,
    *,
    history: int,
    lag: int,
    estimator: str,
    k: int,
    base: float,
    same: bool,
) -> float:
    """Return the mutual information, transfer entropy, or divergence of two series.

    They go from one series to another.
    """
    if kind == 'transfer' and same:
        return math.nan
    if kind == 'mutual':
        return _mutual_information(first, second, base=base)
    if kind == 'transfer':
        return _transfer_entropy(first, second, history=history, lag=lag, base=base, estimator=estimator, k=k)
    return _kl_divergence(first, second, base=base)


def _compute_series(
    kind: str,
    symbols: AnyArray,
    *,
    max_delay: int,
    history: int,
    dimension: int,
    base: float,
) -> float | Array:
    """Return the auto mutual information, storage, or correlation information.

    They are those of one series.
    """
    if kind == 'auto-mutual':
        return _delayed_information(symbols, max_delay, base=base)
    if kind == 'active-storage':
        return _active_information_storage(symbols, history, base=base)
    return _correlation_information(symbols, dimension, base=base)


def _compute_joint(kind: str, columns: AnyArray, estimator: str, k: int, base: float) -> float | Array:
    """Return the total correlations, the O-information, or its gradients.

    They are those of all series together.
    """
    scale = math.log(base)
    if kind == 'total-correlation':
        return _total(columns, estimator, k) / scale
    if kind == 'dual-total-correlation':
        return _dual(columns, estimator, k) / scale
    whole = _o_information(columns, estimator, k, base)
    if kind == 'o-information':
        return whole
    parts = [_o_information(np.delete(columns, index, axis=1), estimator, k, base) for index in range(columns.shape[1])]
    return whole - np.asarray(parts, dtype=float)


def _rank_index(patterns: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Return the lexicographic rank of every permutation, one per row.

    The rank is its Lehmer code.
    """
    order = patterns.shape[1]
    later_smaller = np.triu(patterns[:, None, :] < patterns[:, :, None], k=1).sum(axis=2)
    weights = np.asarray([math.factorial(order - 1 - position) for position in range(order)], dtype=np.int64)
    return later_smaller.astype(np.int64) @ weights


def _ordinal_patterns(values: npt.ArrayLike, order: int = 3, *, delay: int = 1) -> npt.NDArray[np.int64]:
    """Return the ordinal pattern of every window of a series.

    It is the lexicographic rank of its permutation.
    """
    array = np.asarray(values, dtype=float)
    patterns = np.asarray(ordpy.ordinal_sequence(array, dx=order, taux=delay), dtype=np.int64)
    return _rank_index(patterns.reshape(-1, order))


def _symbolize(
    values: npt.ArrayLike,
    method: str = 'quantiles',
    n: int = 2,
    *,
    order: int = 3,
    delay: int = 1,
) -> npt.NDArray[np.int64]:
    """Return a series as bins or ordinal patterns.

    The bins are `n` equal-width or `n` equally filled bins, and the patterns have length `order`.
    """
    array = _read_real(values, 'values')
    if method == 'bins':
        edges = np.linspace(array.min(), array.max(), n + 1)[1:-1]
        return np.digitize(array, edges).astype(np.int64)
    if method == 'quantiles':
        edges = np.quantile(array, np.linspace(0, 1, n + 1)[1:-1])
        return np.digitize(array, edges).astype(np.int64)
    return _ordinal_patterns(array, order, delay=delay)


def _codes(items: Sequence[Hashable]) -> list[int]:
    """Return one integer code per distinct item."""
    table: dict[Hashable, int] = {}
    return [table.setdefault(item, len(table)) for item in items]


def _joint_entropy(*columns: Sequence[Hashable], base: float) -> float:
    """Return the entropy of aligned columns taken together."""
    return _shannon(_codes(list(zip(*columns, strict=True))), base=base)


def _histories(values: npt.NDArray[np.generic], history: int, start: int) -> list[tuple[object, ...]]:
    """Return the `history` values before every time from `start` on."""
    return [tuple(values[t - history : t]) for t in range(start, values.size)]


def _kl_divergence(p: npt.ArrayLike, q: npt.ArrayLike, *, base: float = 2.0) -> float:
    """Return the Kullback-Leibler divergence of `p` from a reference `q`.

    Here `p` is a distribution. It is infinite off support.
    """
    return float(stats.entropy(np.asarray(p, dtype=float), np.asarray(q, dtype=float), base=base))


def _correlation_information(symbols: npt.ArrayLike, max_m: int = 8, *, base: float = 2.0) -> Array:
    """Return Lindgren's correlation information `k_m = ΔS_{m-1} - ΔS_m`.

    The block lengths run from 2 to `max_m`.
    """
    differences = np.diff(_block_entropies(symbols, max_m, base=base))
    return -np.diff(differences)


def _mutual_information(first: npt.ArrayLike, second: npt.ArrayLike, *, base: float = 2.0) -> float:
    """Return the mutual information `H(X) + H(Y) - H(X, Y)` of two symbol sequences.

    It uses plug-in counts.
    """
    x, y = list(np.asarray(first).ravel()), list(np.asarray(second).ravel())
    joint = list(zip(x, y, strict=True))
    return _shannon(x, base=base) + _shannon(y, base=base) - _shannon(_codes(joint), base=base)


def _delayed_information(symbols: npt.NDArray[np.generic], max_delay: int, *, base: float) -> Array:
    """Return the mutual information of symbols with themselves delayed.

    The delays run from 0 to `max_delay` steps.
    """
    return np.asarray(
        [
            _mutual_information(symbols[: symbols.size - delay], symbols[delay:], base=base)
            for delay in range(max_delay + 1)
        ],
    )


def _active_information_storage(symbols: npt.ArrayLike, history: int = 1, *, base: float = 2.0) -> float:
    """Return `I(X_t ; X_{t-history}, ..., X_{t-1})`.

    It is the information the past symbols hold about the next one.
    """
    x = np.asarray(symbols).ravel()
    future = [(value,) for value in x[history:]]
    past = _histories(x, history, history)
    return float(
        _joint_entropy(future, base=base) + _joint_entropy(past, base=base) - _joint_entropy(future, past, base=base),
    )


def _knn_transfer(x: Array, y: Array, history: int, lag: int, k: int) -> float:
    """Return the transfer entropy in nats.

    It uses the nearest-neighbour conditional mutual information.
    """
    start = max(history, lag)
    past = np.column_stack([y[start - step : y.size - step] for step in range(1, history + 1)])
    estimate = ennemi.estimate_mi(y[start:], x[start - lag : x.size - lag], cond=past, k=k)
    return float(np.asarray(estimate).item())


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
    """Return the transfer entropy by plug-in counts or nearest neighbours.

    It is `H(Y_t | Y_past) - H(Y_t | Y_past, X_{t-lag})`.
    """
    if estimator == 'knn':
        x, y = np.asarray(source, dtype=float).ravel(), np.asarray(target, dtype=float).ravel()
        return _knn_transfer(x, y, history, lag, k) / math.log(base)
    x, y = np.asarray(source).ravel(), np.asarray(target).ravel()
    start = max(history, lag)
    future = [(value,) for value in y[start:]]
    past = _histories(y, history, start)
    driver = [(x[t - lag],) for t in range(start, y.size)]
    return float(
        _joint_entropy(future, past, base=base)
        - _joint_entropy(past, base=base)
        - _joint_entropy(future, past, driver, base=base)
        + _joint_entropy(past, driver, base=base),
    )


def _row_codes(columns: AnyArray) -> npt.NDArray[np.int64]:
    """Return one integer code per distinct row."""
    _, codes = np.unique(columns, axis=0, return_inverse=True)
    return np.asarray(codes, dtype=np.int64).ravel()


def _log_determinant(columns: Array) -> float:
    """Return the logarithm of the determinant of the sample covariance of the columns."""
    if columns.shape[1] == 0:
        return 0.0
    _, value = np.linalg.slogdet(np.atleast_2d(np.cov(columns, rowvar=False, bias=True)))
    return float(value)


def _knn_chain(target: Array, others: Array, k: int, given: Array | None = None) -> float:
    """Return `I(target ; others | given)` in nats.

    It is a chain of k-nearest-neighbour conditional informations.
    """
    condition = np.empty((target.size, 0)) if given is None else given
    total = 0.0
    for column in range(others.shape[1]):
        cond = np.column_stack([condition, others[:, :column]])
        estimate = ennemi.estimate_mi(target, others[:, column], cond=cond if cond.shape[1] else None, k=k)
        total += float(np.asarray(estimate).item())
    return total


def _dit_distribution(columns: AnyArray) -> object:
    """Return the `dit` distribution of the empirical frequencies of the rows."""
    counts = Counter(tuple(row) for row in columns.tolist())
    total = sum(counts.values())
    return load_module('dit').Distribution(list(counts), [count / total for count in counts.values()])


def _total(columns: AnyArray, estimator: str, k: int) -> float:
    """Return the total correlation `Σ H(X_i) - H(X)` of the columns in nats."""
    if estimator == 'plugin':
        return float(
            load_module('dit').multivariate.total_correlation(
                _dit_distribution(columns),
            ),
        ) * math.log(
            2,
        )
    values = _read_real(columns, 'data')
    if estimator == 'gaussian':
        return -_log_determinant(values / values.std(axis=0)) / 2
    return sum(_knn_chain(values[:, index], values[:, :index], k) for index in range(1, values.shape[1]))


def _dual(columns: AnyArray, estimator: str, k: int) -> float:
    """Return the dual total correlation of the columns in nats.

    It is `H(X) - Σ H(X_i | X_-i)`.
    """
    if estimator == 'plugin':
        return float(
            load_module('dit').multivariate.dual_total_correlation(
                _dit_distribution(columns),
            ),
        ) * math.log(2)
    values = _read_real(columns, 'data')
    if estimator == 'gaussian':
        precision = np.linalg.inv(np.atleast_2d(np.cov(values, rowvar=False, bias=True)))
        return (_log_determinant(values) + float(np.sum(np.log(np.diag(precision))))) / 2
    parts = sum(_knn_chain(values[:, index], np.delete(values, index, axis=1), k) for index in range(values.shape[1]))
    return parts - _total(values, estimator, k)


def _o_information(columns: AnyArray, estimator: str, k: int, base: float) -> float:
    """Return the O-information in the base of the logarithm.

    It is the total minus the dual total correlation.
    """
    return (_total(columns, estimator, k) - _dual(columns, estimator, k)) / math.log(base)
