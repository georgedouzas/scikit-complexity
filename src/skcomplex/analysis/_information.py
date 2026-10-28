"""Quantify structure with information measures: entropies, entropy rates, excess entropy, and information flows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Hashable, Sequence

import ennemi
import numpy as np
import numpy.typing as npt
import ordpy
from scipy import stats

from ..core.errors import AnalysisError
from ._entropy import _block_entropies, _shannon
from ._kinds import DataKind, analysis
from ._utils import real_array

type Array = npt.NDArray[np.float64]
METHODS = ('bins', 'quantiles', 'ordinal')
ESTIMATORS = ('plugin', 'knn')
def _rank_index(patterns: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Return the lexicographic rank of every permutation, one per row, as its Lehmer code."""
    order = patterns.shape[1]
    later_smaller = np.triu(patterns[:, None, :] < patterns[:, :, None], k=1).sum(axis=2)
    weights = np.asarray([math.factorial(order - 1 - position) for position in range(order)], dtype=np.int64)
    return later_smaller.astype(np.int64) @ weights


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


def _knn_transfer(x: Array, y: Array, history: int, lag: int, k: int) -> float:
    """Return the transfer entropy in nats by the nearest-neighbour conditional mutual information."""
    start = max(history, lag)
    past = np.column_stack([y[start - step : y.size - step] for step in range(1, history + 1)])
    estimate = ennemi.estimate_mi(y[start:], x[start - lag : x.size - lag], cond=past, k=k)
    return float(np.asarray(estimate).item())


def ordinal_patterns(values: npt.ArrayLike, order: int = 3, *, delay: int = 1) -> npt.NDArray[np.int64]:
    """Return the ordinal pattern of every window of a series as the lexicographic rank of its permutation.

    Args:
        values:
            The series.

        order:
            The number of values in every pattern.

        delay:
            The number of steps between the values of a pattern.

    Returns:
        symbols:
            One symbol per window, from 0 to `order! - 1`, distinct for distinct rank patterns.

    Examples:
        >>> from skcomplex.analysis import ordinal_patterns
        >>> ordinal_patterns([1.0, 2.0, 3.0, 2.5, 0.5], order=3).tolist()
        [0, 1, 5]
    """
    array = np.asarray(values, dtype=float)
    patterns = np.asarray(ordpy.ordinal_sequence(array, dx=order, taux=delay), dtype=np.int64)
    return _rank_index(patterns.reshape(-1, order))


@analysis(DataKind.SERIES)
def symbolize(
    values: npt.ArrayLike,
    method: str = 'quantiles',
    n: int = 2,
    *,
    order: int = 3,
    delay: int = 1,
) -> npt.NDArray[np.int64]:
    """Return a series as symbols: equal-width bins, equally populated bins, or ordinal patterns.

    Args:
        values:
            The series.

        method:
            `'bins'` for `n` bins of equal width, `'quantiles'` for `n` bins of equal population, or `'ordinal'` for
            the rank pattern of every `order` values spaced by `delay` steps.

        n:
            The number of symbols for bins and quantiles.

        order:
            The length of ordinal patterns.

        delay:
            The number of steps between the values of an ordinal pattern.

    Returns:
        symbols:
            One integer symbol per value, or per pattern for ordinal symbols.

    Raises:
        AnalysisError:
            The method is unknown.

    Examples:
        >>> from skcomplex.analysis import symbolize
        >>> symbolize([0.1, 0.9, 0.4, 0.6], 'quantiles', 2).tolist()
        [0, 1, 0, 1]
    """
    array = real_array(values, 'values')
    if method == 'bins':
        edges = np.linspace(array.min(), array.max(), n + 1)[1:-1]
        return np.digitize(array, edges).astype(np.int64)
    if method == 'quantiles':
        edges = np.quantile(array, np.linspace(0, 1, n + 1)[1:-1])
        return np.digitize(array, edges).astype(np.int64)
    if method == 'ordinal':
        return ordinal_patterns(array, order, delay=delay)
    error_msg = f'Unknown symbolisation {method!r}. Use one of {", ".join(METHODS)}.'
    raise AnalysisError(error_msg)


@analysis(DataKind.SERIES)
def kl_divergence(p: npt.ArrayLike, q: npt.ArrayLike, *, base: float = 2.0) -> float:
    """Return the Kullback-Leibler divergence of a distribution `p` from a reference distribution `q`.

    Args:
        p:
            The probabilities.

        q:
            The reference probabilities over the same outcomes.

        base:
            The base of the logarithm.

    Returns:
        divergence:
            The divergence, infinite when `q` is 0 where `p` is not.

    Examples:
        >>> from skcomplex.analysis import kl_divergence
        >>> kl_divergence([0.5, 0.5], [0.5, 0.5])
        0.0
    """
    return float(stats.entropy(np.asarray(p, dtype=float), np.asarray(q, dtype=float), base=base))


@analysis(DataKind.SERIES)
def correlation_information(symbols: npt.ArrayLike, max_m: int = 8, *, base: float = 2.0) -> Array:
    """Return Lindgren's correlation information `k_m = ΔS_{m-1} - ΔS_m` for block lengths 2 to `max_m`.

    Args:
        symbols:
            A sequence of symbols.

        max_m:
            The largest block length.

        base:
            The base of the logarithm.

    Returns:
        information:
            The information in correlations of every length from 2 to `max_m`.
    """
    differences = np.diff(_block_entropies(symbols, max_m, base=base))
    return -np.diff(differences)


@analysis(DataKind.SERIES)
def mutual_information(first: npt.ArrayLike, second: npt.ArrayLike, *, base: float = 2.0) -> float:
    """Return the mutual information of two symbol sequences.

    Args:
        first:
            The first symbols.

        second:
            The second symbols, aligned with the first.

        base:
            The base of the logarithm.

    Returns:
        information:
            `H(X) + H(Y) - H(X, Y)`.

    Examples:
        >>> from skcomplex.analysis import mutual_information
        >>> mutual_information([0, 1, 0, 1], [0, 1, 0, 1])
        1.0
    """
    x, y = list(np.asarray(first).ravel()), list(np.asarray(second).ravel())
    joint = list(zip(x, y, strict=True))
    return _shannon(x, base=base) + _shannon(y, base=base) - _shannon(_codes(joint), base=base)


@analysis(DataKind.SERIES)
def active_information_storage(symbols: npt.ArrayLike, history: int = 1, *, base: float = 2.0) -> float:
    """Return the active information storage, the information the past `history` symbols hold about the next one.

    Args:
        symbols:
            The symbols.

        history:
            The number of past symbols.

        base:
            The base of the logarithm.

    Returns:
        storage:
            `I(X_t ; X_{t-history}, ..., X_{t-1})`, estimated by plug-in counts.

    Examples:
        >>> from skcomplex.analysis import active_information_storage
        >>> round(active_information_storage([0, 0, 1, 1] * 100, 2), 3)
        1.0
    """
    x = np.asarray(symbols).ravel()
    future = [(value,) for value in x[history:]]
    past = _histories(x, history, history)
    return float(
        _joint_entropy(future, base=base) + _joint_entropy(past, base=base) - _joint_entropy(future, past, base=base),
    )


@analysis(DataKind.SERIES)
def transfer_entropy(
    source: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    history: int = 1,
    lag: int = 1,
    base: float = 2.0,
    estimator: str = 'plugin',
    k: int = 3,
) -> float:
    """Return the transfer entropy from a source to a target, the information the source's past adds about the target.

    It is `H(Y_t | Y_past) - H(Y_t | Y_past, X_{t-lag})`, estimated by plug-in counts on symbols, or by the
    Kraskov-Stögbauer-Grassberger nearest-neighbour conditional mutual information on continuous values.

    Args:
        source:
            The source symbols or values.

        target:
            The target symbols or values, aligned with the source.

        history:
            The number of past target values conditioned on.

        lag:
            The delay of the source.

        base:
            The base of the logarithm.

        estimator:
            `'plugin'` for counts of symbols, or `'knn'` for nearest neighbours of continuous values.

        k:
            The number of neighbours of the `'knn'` estimator.

    Returns:
        transfer:
            The transfer entropy, 0 when the source tells nothing new.

    Raises:
        AnalysisError:
            The estimator is unknown.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import transfer_entropy
        >>> x = np.random.default_rng(0).integers(0, 2, 10000)
        >>> y = np.roll(x, 1)
        >>> round(transfer_entropy(x, y), 1), round(transfer_entropy(y, x), 1)
        (1.0, 0.0)
    """
    if estimator not in ESTIMATORS:
        error_msg = f'Unknown estimator {estimator!r}. Use one of {", ".join(ESTIMATORS)}.'
        raise AnalysisError(error_msg)
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


