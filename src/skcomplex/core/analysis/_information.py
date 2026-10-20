"""Quantify structure with information measures: entropies, entropy rates, excess entropy, and information flows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections import Counter
from collections.abc import Hashable, Sequence

import ennemi
import numpy as np
import numpy.typing as npt
import ordpy

from ..base import AnalysisError
from ._kinds import DataKind, analysis
from ._lazy import module
from ._utils import real_array

type Array = npt.NDArray[np.float64]
METHODS = ('bins', 'quantiles', 'ordinal')
ENTROPIES = ('shannon', 'renyi', 'tsallis')
ESTIMATORS = ('plugin', 'knn')
TIES = ('order', 'random')


def _rank_index(patterns: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Return the lexicographic rank of every permutation, one per row, as its Lehmer code."""
    order = patterns.shape[1]
    later_smaller = np.triu(patterns[:, None, :] < patterns[:, :, None], k=1).sum(axis=2)
    weights = np.asarray([math.factorial(order - 1 - position) for position in range(order)], dtype=np.int64)
    return later_smaller.astype(np.int64) @ weights


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
        >>> from skcomplex.core.analysis import ordinal_patterns
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
        >>> from skcomplex.core.analysis import symbolize
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


def _probabilities(items: Sequence[Hashable]) -> Array:
    """Return the empirical probabilities of distinct items."""
    counts = np.asarray(list(Counter(items).values()), dtype=float)
    return counts / counts.sum()


def _distribution(values: npt.ArrayLike, *, probabilities: bool) -> Array:
    """Return the nonzero probabilities of symbols, or of a probability vector."""
    p = np.asarray(values, dtype=float) if probabilities else _probabilities(list(np.asarray(values).ravel()))
    return p[p > 0]


@analysis(DataKind.SERIES)
def shannon_entropy(values: npt.ArrayLike, *, base: float = 2.0, probabilities: bool = False) -> float:
    """Return the Shannon entropy of symbols, or of a probability vector.

    Args:
        values:
            The symbols, or the probabilities when `probabilities` is true.

        base:
            The base of the logarithm, 2 for bits.

        probabilities:
            Whether `values` is already a probability vector.

    Returns:
        entropy:
            The entropy.

    Examples:
        >>> from skcomplex.core.analysis import shannon_entropy
        >>> shannon_entropy([0, 1, 0, 1]), shannon_entropy([0.25] * 4, probabilities=True)
        (1.0, 2.0)
    """
    p = _distribution(values, probabilities=probabilities)
    return float(-np.sum(p * np.log(p)) / np.log(base))


@analysis(DataKind.SERIES)
def renyi_entropy(
    values: npt.ArrayLike,
    order: float = 2.0,
    *,
    base: float = 2.0,
    probabilities: bool = False,
) -> float:
    """Return the Rényi entropy `log(Σ p^a) / (1 - a)` of symbols or probabilities, the Shannon entropy at order 1.

    Args:
        values:
            The symbols, or the probabilities when `probabilities` is true.

        order:
            The order `a`, any nonnegative number or `inf`, where the entropy is `-log max p`.

        base:
            The base of the logarithm, 2 for bits.

        probabilities:
            Whether `values` is already a probability vector.

    Returns:
        entropy:
            The entropy.

    Examples:
        >>> from skcomplex.core.analysis import renyi_entropy
        >>> [round(renyi_entropy([0.25] * 4, order, probabilities=True), 12) for order in (0, 1, 2, float('inf'))]
        [2.0, 2.0, 2.0, 2.0]
    """
    p = _distribution(values, probabilities=probabilities)
    if order == 1:
        return shannon_entropy(p, base=base, probabilities=True)
    if math.isinf(order):
        return float(-np.log(p.max()) / np.log(base))
    return float(np.log(np.sum(p**order)) / (1 - order) / np.log(base))


@analysis(DataKind.SERIES)
def tsallis_entropy(values: npt.ArrayLike, index: float = 2.0, *, probabilities: bool = False) -> float:
    """Return the Tsallis entropy `(1 - Σ p^q) / (q - 1)` of symbols or probabilities, in nats at index 1.

    Args:
        values:
            The symbols, or the probabilities when `probabilities` is true.

        index:
            The entropic index `q`.

        probabilities:
            Whether `values` is already a probability vector.

    Returns:
        entropy:
            The entropy, the Shannon entropy in nats at index 1.

    Examples:
        >>> from skcomplex.core.analysis import tsallis_entropy
        >>> tsallis_entropy([0.5, 0.5], 2, probabilities=True)
        0.5
    """
    p = _distribution(values, probabilities=probabilities)
    if index == 1:
        return shannon_entropy(p, base=math.e, probabilities=True)
    return float((1 - np.sum(p**index)) / (index - 1))


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
        >>> from skcomplex.core.analysis import kl_divergence
        >>> kl_divergence([0.5, 0.5], [0.5, 0.5])
        0.0
    """
    first, second = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = first > 0
    if np.any(second[mask] == 0):
        return float('inf')
    return float(np.sum(first[mask] * np.log(first[mask] / second[mask])) / np.log(base))


@analysis(DataKind.SERIES)
def block_entropy(symbols: npt.ArrayLike, m: int, *, base: float = 2.0) -> float:
    """Return the entropy of the blocks of `m` consecutive symbols, or of square patches of side `m` of a lattice.

    Args:
        symbols:
            A sequence of symbols, or a two-dimensional lattice of symbols.

        m:
            The block length, or the side of square patches.

        base:
            The base of the logarithm.

    Returns:
        entropy:
            The block entropy `S_m`.

    Examples:
        >>> from skcomplex.core.analysis import block_entropy
        >>> round(block_entropy([0, 0, 1, 1] * 50, 2), 3)
        2.0
    """
    array = np.asarray(symbols)
    if m == 0:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(array, (m,) * array.ndim)
    blocks = windows.reshape(-1, m**array.ndim)
    return shannon_entropy(_probabilities([block.tobytes() for block in blocks]), base=base, probabilities=True)


@analysis(DataKind.SERIES)
def block_entropies(symbols: npt.ArrayLike, max_m: int, *, base: float = 2.0) -> Array:
    """Return the block entropies `S_0` to `S_max_m`.

    Args:
        symbols:
            A sequence of symbols, or a two-dimensional lattice of symbols.

        max_m:
            The largest block length.

        base:
            The base of the logarithm.

    Returns:
        entropies:
            One entropy per block length from 0.
    """
    return np.asarray([block_entropy(symbols, m, base=base) for m in range(max_m + 1)])


@analysis(DataKind.SERIES)
def entropy_rate(symbols: npt.ArrayLike, max_m: int = 8, *, base: float = 2.0) -> float:
    """Return the entropy rate, the limit of `S_m - S_{m-1}`, estimated at the largest block length.

    Args:
        symbols:
            A sequence of symbols.

        max_m:
            The largest block length, kept well below the logarithm of the sequence length in the base.

        base:
            The base of the logarithm.

    Returns:
        rate:
            The entropy rate per symbol.

    Examples:
        >>> from skcomplex.core.analysis import entropy_rate
        >>> round(entropy_rate([0, 0, 1, 1] * 100, 4), 3)
        0.0
    """
    entropies = block_entropies(symbols, max_m, base=base)
    return float(max(entropies[-1] - entropies[-2], 0.0))


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
    differences = np.diff(block_entropies(symbols, max_m, base=base))
    return -np.diff(differences)


@analysis(DataKind.SERIES)
def excess_entropy(symbols: npt.ArrayLike, max_m: int = 8, *, base: float = 2.0) -> float:
    """Return the excess entropy, or effective measure complexity, `S_m - m s` at the largest block length.

    Args:
        symbols:
            A sequence of symbols.

        max_m:
            The largest block length.

        base:
            The base of the logarithm.

    Returns:
        complexity:
            The information that the past holds about the future.

    Examples:
        >>> from skcomplex.core.analysis import excess_entropy
        >>> round(excess_entropy([0, 0, 1, 1] * 100, 4), 3)
        2.0
    """
    entropies = block_entropies(symbols, max_m, base=base)
    rate = max(entropies[-1] - entropies[-2], 0.0)
    return float(entropies[-1] - max_m * rate)


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
        >>> from skcomplex.core.analysis import mutual_information
        >>> mutual_information([0, 1, 0, 1], [0, 1, 0, 1])
        1.0
    """
    x, y = list(np.asarray(first).ravel()), list(np.asarray(second).ravel())
    joint = list(zip(x, y, strict=True))
    return shannon_entropy(x, base=base) + shannon_entropy(y, base=base) - shannon_entropy(_codes(joint), base=base)


def _codes(items: Sequence[Hashable]) -> list[int]:
    """Return one integer code per distinct item."""
    table: dict[Hashable, int] = {}
    return [table.setdefault(item, len(table)) for item in items]


def _joint_entropy(*columns: Sequence[Hashable], base: float) -> float:
    """Return the entropy of aligned columns taken together."""
    return shannon_entropy(_codes(list(zip(*columns, strict=True))), base=base)


def _histories(values: npt.NDArray[np.generic], history: int, start: int) -> list[tuple[object, ...]]:
    """Return the `history` values before every time from `start` on."""
    return [tuple(values[t - history : t]) for t in range(start, values.size)]


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
        >>> from skcomplex.core.analysis import active_information_storage
        >>> round(active_information_storage([0, 0, 1, 1] * 100, 2), 3)
        1.0
    """
    x = np.asarray(symbols).ravel()
    future = [(value,) for value in x[history:]]
    past = _histories(x, history, history)
    return float(
        _joint_entropy(future, base=base) + _joint_entropy(past, base=base) - _joint_entropy(future, past, base=base),
    )


def _knn_transfer(x: Array, y: Array, history: int, lag: int, k: int) -> float:
    """Return the transfer entropy in nats by the nearest-neighbour conditional mutual information."""
    start = max(history, lag)
    past = np.column_stack([y[start - step : y.size - step] for step in range(1, history + 1)])
    estimate = ennemi.estimate_mi(y[start:], x[start - lag : x.size - lag], cond=past, k=k)
    return float(np.asarray(estimate).item())


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
        >>> from skcomplex.core.analysis import transfer_entropy
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


def break_ties(values: npt.ArrayLike, ties: str = 'random', random_state: int | None = None) -> Array:
    """Return values with equal values made distinct at random, keeping the order of distinct values.

    Args:
        values:
            The values.

        ties:
            `'random'` to add a random offset smaller than a quarter of the smallest gap between distinct values, or
            `'order'` to return the values unchanged.

        random_state:
            The seed of the offsets.

    Returns:
        values:
            The values with ties broken.

    Raises:
        AnalysisError:
            The tie rule is unknown.

    Examples:
        >>> from skcomplex.core.analysis import break_ties
        >>> broken = break_ties([1.0, 1.0, 2.0], random_state=0)
        >>> len(set(broken.tolist())), bool(broken[2] > broken[:2].max())
        (3, True)
    """
    array = np.asarray(values, dtype=float)
    if ties not in TIES:
        error_msg = f'Unknown tie rule {ties!r}. Use one of {", ".join(TIES)}.'
        raise AnalysisError(error_msg)
    if ties == 'order':
        return array
    gaps = np.diff(np.unique(array))
    scale = gaps.min() / 4 if gaps.size else 1.0
    return array + np.random.default_rng(random_state).uniform(0, scale, array.shape)


def _entropy_of(probabilities: Array, entropy: str, index: float) -> float:
    """Return the Shannon or Rényi entropy in bits, or the Tsallis entropy, of probabilities."""
    if entropy == 'renyi':
        return renyi_entropy(probabilities, index, probabilities=True)
    if entropy == 'tsallis':
        return tsallis_entropy(probabilities, index, probabilities=True)
    return shannon_entropy(probabilities, probabilities=True)


def _largest_entropy(outcomes: int, entropy: str, index: float, base: float) -> float:
    """Return the entropy of the uniform law on a number of outcomes."""
    if entropy == 'tsallis':
        return tsallis_entropy(np.full(outcomes, 1 / outcomes), index, probabilities=True)
    return math.log(outcomes) / math.log(base)


@analysis(DataKind.SERIES)
def permutation_entropy(
    values: npt.ArrayLike,
    order: int | tuple[int, int] = 3,
    *,
    delay: int = 1,
    normalize: bool = True,
    entropy: str = 'shannon',
    index: float = 2.0,
    ties: str = 'order',
    random_state: int | None = None,
) -> float:
    """Return the permutation entropy of a series or a two-dimensional field, the entropy of its ordinal patterns.

    Args:
        values:
            The series, or a two-dimensional field when `order` is a window.

        order:
            The length of the patterns of a series, or the window `(columns, rows)` of the patterns of a field.

        delay:
            The number of steps between the values of a pattern, along both axes of a field.

        normalize:
            Whether to divide by the entropy of uniform patterns, so the result lies between 0 and 1.

        entropy:
            `'shannon'`, `'renyi'`, or `'tsallis'`.

        index:
            The order of the Rényi entropy or the index of the Tsallis entropy.

        ties:
            `'order'` to rank equal values by position, or `'random'` to rank them at random, as suits discrete
            states such as spins.

        random_state:
            The seed of the random ranks of equal values.

    Returns:
        entropy:
            The permutation entropy, in bits for Shannon and Rényi and in nats for Tsallis without normalization.

    Raises:
        AnalysisError:
            The entropy is unknown.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import permutation_entropy
        >>> permutation_entropy(np.arange(100.0))
        0.0
    """
    if entropy not in ENTROPIES:
        error_msg = f'Unknown entropy {entropy!r}. Use one of {", ".join(ENTROPIES)}.'
        raise AnalysisError(error_msg)
    dx, dy = (order, 1) if isinstance(order, int) else order
    array = break_ties(values, ties, random_state)
    _, probabilities = ordpy.ordinal_distribution(array, dx=dx, dy=dy, taux=delay, tauy=delay)
    value = _entropy_of(probabilities, entropy, index)
    if normalize:
        value /= _largest_entropy(math.factorial(dx * dy), entropy, index, 2.0)
    return float(abs(value))


@analysis(DataKind.SERIES)
def lempel_ziv_complexity(symbols: npt.ArrayLike, *, normalize: bool = True) -> float:
    """Return the Lempel-Ziv complexity of a sequence, the number of new phrases in its LZ76 parsing.

    Args:
        symbols:
            The symbols.

        normalize:
            Whether to divide by `n / log_k(n)` with `k` distinct symbols, close to 1 for random sequences.

    Returns:
        complexity:
            The complexity.

    Examples:
        >>> from skcomplex.core.analysis import lempel_ziv_complexity
        >>> lempel_ziv_complexity([0] * 64, normalize=False)
        2.0
    """
    _, codes = np.unique(np.asarray(symbols).ravel(), return_inverse=True)
    return float(module('antropy').lziv_complexity(codes.astype(np.int64), normalize=normalize))
