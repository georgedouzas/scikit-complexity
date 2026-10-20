"""Quantify structure with information measures: entropies, entropy rates, excess entropy, and information flows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np
import numpy.typing as npt

from ..base import AnalysisError
from ._utils import real_array

type Array = npt.NDArray[np.float64]
METHODS = ('bins', 'quantiles', 'ordinal')


def symbolize(values: npt.ArrayLike, method: str = 'quantiles', n: int = 2, *, order: int = 3) -> npt.NDArray[np.int64]:
    """Return a series as symbols: equal-width bins, equally populated bins, or ordinal patterns.

    Args:
        values:
            The series.

        method:
            `'bins'` for `n` bins of equal width, `'quantiles'` for `n` bins of equal population, or `'ordinal'` for
            the rank pattern of every `order` consecutive values.

        n:
            The number of symbols for bins and quantiles.

        order:
            The length of ordinal patterns.

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
        windows = np.lib.stride_tricks.sliding_window_view(array, order)
        patterns = np.argsort(windows, axis=1, kind='stable')
        return np.asarray([int(''.join(map(str, pattern)), order) for pattern in patterns], dtype=np.int64)
    error_msg = f'Unknown symbolisation {method!r}. Use one of {", ".join(METHODS)}.'
    raise AnalysisError(error_msg)


def _probabilities(items: Sequence[Hashable]) -> Array:
    """Return the empirical probabilities of distinct items."""
    counts = np.asarray(list(Counter(items).values()), dtype=float)
    return counts / counts.sum()


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
    p = np.asarray(values, dtype=float) if probabilities else _probabilities(list(np.asarray(values).ravel()))
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(base))


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


def transfer_entropy(
    source: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    history: int = 1,
    lag: int = 1,
    base: float = 2.0,
) -> float:
    """Return the transfer entropy from a source to a target, the information the source's past adds about the target.

    It is `H(Y_t | Y_past) - H(Y_t | Y_past, X_{t-lag})`, estimated by plug-in counts on symbols.

    Args:
        source:
            The source symbols.

        target:
            The target symbols, aligned with the source.

        history:
            The number of past target symbols conditioned on.

        lag:
            The delay of the source.

        base:
            The base of the logarithm.

    Returns:
        transfer:
            The transfer entropy, 0 when the source tells nothing new.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import transfer_entropy
        >>> x = np.random.default_rng(0).integers(0, 2, 10000)
        >>> y = np.roll(x, 1)
        >>> round(transfer_entropy(x, y), 1), round(transfer_entropy(y, x), 1)
        (1.0, 0.0)
    """
    x, y = np.asarray(source).ravel(), np.asarray(target).ravel()
    start = max(history, lag)
    future = [(value,) for value in y[start:]]
    past = [tuple(y[t - history : t]) for t in range(start, y.size)]
    driver = [(x[t - lag],) for t in range(start, y.size)]
    return float(
        _joint_entropy(future, past, base=base)
        - _joint_entropy(past, base=base)
        - _joint_entropy(future, past, driver, base=base)
        + _joint_entropy(past, driver, base=base),
    )


def permutation_entropy(values: npt.ArrayLike, order: int = 3, *, normalize: bool = True) -> float:
    """Return the permutation entropy of a series, the entropy of its ordinal patterns.

    Args:
        values:
            The series.

        order:
            The length of the patterns.

        normalize:
            Whether to divide by the largest entropy, `log2(order!)`, so the result lies between 0 and 1.

    Returns:
        entropy:
            The permutation entropy.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import permutation_entropy
        >>> permutation_entropy(np.arange(100.0))
        0.0
    """
    entropy = shannon_entropy(symbolize(values, 'ordinal', order=order))
    value = entropy / np.log2(float(np.prod(np.arange(1, order + 1)))) if normalize else entropy
    return float(abs(value))


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
    sequence = list(np.asarray(symbols).ravel())
    size = len(sequence)
    phrases, start, length = 1, 0, 1
    while start + length < size:
        window = sequence[: start + length - 1]
        phrase = sequence[start : start + length]
        found = any(window[index : index + length] == phrase for index in range(len(window) - length + 2))
        if found and start + length < size:
            length += 1
            continue
        phrases += 1
        start += length
        length = 1
    if not normalize:
        return float(phrases)
    alphabet = max(len(set(sequence)), 2)
    return float(phrases * np.log(size) / np.log(alphabet) / size)
