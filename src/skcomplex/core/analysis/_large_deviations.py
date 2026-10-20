"""Measure large deviations: generating functions, rate functions, and exponential reweighting."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import linalg, special

from ..base import AnalysisError
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]
MIN_SIZES = 2


@analysis(DataKind.PATH)
def tilted_generating_function(rates: npt.ArrayLike, weights: npt.ArrayLike, k: npt.ArrayLike) -> Array:
    """Return the scaled cumulant generating function of a counting observable of a jump process, exactly.

    The observable adds `weights[i, j]` at every jump from `i` to `j`, and the function at `k` is the largest
    eigenvalue of the tilted generator `rates * exp(k weights) - diag(escape rates)`. A diagonal rate counts jumps that
    keep the state, such as the events of a Poisson process.

    Args:
        rates:
            The rate of every jump, one row per origin.

        weights:
            The increment of the observable at every jump.

        k:
            The values of the tilt.

    Returns:
        function:
            The generating function at every tilt.

    Examples:
        >>> from skcomplex.core.analysis import tilted_generating_function
        >>> tilted_generating_function([[2.0]], [[1.0]], [0.0, 1.0]).round(4).tolist()
        [0.0, 3.4366]
    """
    matrix = np.asarray(rates, dtype=float)
    increments = np.asarray(weights, dtype=float)
    escape = np.diag(matrix.sum(axis=1))
    tilts = np.atleast_1d(np.asarray(k, dtype=float))
    return np.asarray([linalg.eigvals(matrix * np.exp(tilt * increments) - escape).real.max() for tilt in tilts])


@analysis(DataKind.SERIES)
def block_generating_function(
    increments: npt.ArrayLike,
    k: npt.ArrayLike,
    block: int,
    dt: float = 1.0,
) -> Array:
    """Return the scaled cumulant generating function of a time-additive observable from blocks of one long run.

    Args:
        increments:
            The increment of the observable in every step, such as the count of events.

        k:
            The values of the tilt.

        block:
            The number of steps of every block.

        dt:
            The time of one step.

    Returns:
        function:
            `ln <exp(k A)> / (block dt)` over the blocks at every tilt.

    Raises:
        AnalysisError:
            The run holds fewer than 2 blocks.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import block_generating_function
        >>> counts = np.random.default_rng(0).poisson(0.2, 1_000_000)
        >>> bool(abs(block_generating_function(counts, [1.0], 10, 0.1)[0] - 2 * (np.e - 1)) < 0.2)
        True
    """
    array = np.asarray(increments, dtype=float)
    count = array.size // block
    if count < MIN_SIZES:
        error_msg = f'The run of {array.size} steps holds fewer than 2 blocks of {block} steps.'
        raise AnalysisError(error_msg)
    sums = array[: count * block].reshape(count, block).sum(axis=1)
    tilts = np.atleast_1d(np.asarray(k, dtype=float))
    return (special.logsumexp(np.outer(tilts, sums), axis=1) - np.log(count)) / (block * dt)


@analysis(DataKind.SERIES)
def legendre_transform(k: npt.ArrayLike, function: npt.ArrayLike, values: npt.ArrayLike) -> Array:
    """Return the rate function `I(a) = max_k (k a - lambda(k))` of a generating function sampled on a grid.

    Args:
        k:
            The grid of tilts.

        function:
            The generating function at every tilt.

        values:
            The values `a` of the observable per unit time.

    Returns:
        rates:
            The rate function at every value.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import legendre_transform
        >>> k = np.linspace(-3, 3, 6001)
        >>> round(float(legendre_transform(k, 2 * (np.exp(k) - 1), [3.0])[0]), 4)
        0.2164
    """
    tilts, function_values = np.asarray(k, dtype=float), np.asarray(function, dtype=float)
    points = np.atleast_1d(np.asarray(values, dtype=float))
    return np.max(np.outer(points, tilts) - function_values[None, :], axis=1)


@dataclass(frozen=True)
class RateFunction:
    """Rate function of an order parameter estimated from its histograms at several sizes.

    Args:
        values:
            The centres of the bins.

        rates:
            The rate function at every centre, 0 at its minimum, `nan` where a size left the bin empty.
    """

    values: Array
    rates: Array


@analysis(DataKind.ENSEMBLE)
def histogram_rate_function(
    samples: Mapping[int, npt.ArrayLike],
    bins: int | npt.ArrayLike = 41,
    value_range: tuple[float, float] | None = None,
) -> RateFunction:
    """Return the rate function `I` of `P_N(m) ≍ exp(-N I(m))` fitted across the histograms of several sizes `N`.

    In every bin `-ln P_N` is fitted linearly in `N`, so prefactors that do not grow exponentially with the size
    drop out.

    Args:
        samples:
            The samples of the order parameter at every size.

        bins:
            The number of bins or their edges.

        value_range:
            The range of the bins, or `None` for the range of all samples.

    Returns:
        rate:
            The bin centres and the rate function.

    Raises:
        AnalysisError:
            Fewer than 2 sizes are given.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import histogram_rate_function
        >>> rng = np.random.default_rng(0)
        >>> samples = {n: rng.binomial(n, 0.5, 200_000) / n for n in (50, 100, 200)}
        >>> rate = histogram_rate_function(samples, 21, (0.0, 1.0))
        >>> round(float(np.nanmin(rate.rates)), 6)
        0.0
    """
    sizes = sorted(samples)
    if len(sizes) < MIN_SIZES:
        error_msg = f'A rate function from histograms needs at least {MIN_SIZES} sizes, not {len(sizes)}.'
        raise AnalysisError(error_msg)
    pooled = np.concatenate([np.asarray(samples[size], dtype=float) for size in sizes])
    edges = np.histogram_bin_edges(pooled, bins=bins, range=value_range)
    arrays = [np.asarray(samples[size], dtype=np.float64) for size in sizes]
    densities = np.asarray([np.histogram(array, bins=edges, density=True)[0] for array in arrays])
    with np.errstate(divide='ignore'):
        logs = -np.log(densities)
    rates = np.full(edges.size - 1, np.nan)
    for index in range(rates.size):
        column = logs[:, index]
        finite = np.isfinite(column)
        if finite.sum() >= MIN_SIZES:
            rates[index] = np.polyfit(np.asarray(sizes, dtype=float)[finite], column[finite], 1)[0]
    return RateFunction((edges[1:] + edges[:-1]) / 2, rates - np.nanmin(rates))


@dataclass(frozen=True)
class Reweighted:
    """Samples reweighted by an exponential tilt.

    Args:
        weights:
            The normalized weight of every sample.

        mean:
            The weighted mean of the samples.

        effective_size:
            The effective number of samples, `1 / Σ w²`.
    """

    weights: Array
    mean: float
    effective_size: float


@analysis(DataKind.ENSEMBLE)
def reweight(values: npt.ArrayLike, log_weights: npt.ArrayLike) -> Reweighted:
    """Return samples reweighted to another law by the logarithms of the likelihood ratios, such as `beta dh M`.

    Args:
        values:
            The samples of the observable.

        log_weights:
            The logarithm of the ratio of the new to the old probability of every sample.

    Returns:
        reweighted:
            The weights, the reweighted mean, and the effective sample size.

    Examples:
        >>> from skcomplex.core.analysis import reweight
        >>> round(reweight([-1.0, 1.0], [0.0, 0.0]).mean, 6)
        0.0
    """
    array = np.asarray(values, dtype=float)
    logs = np.asarray(log_weights, dtype=float)
    weights = np.exp(logs - special.logsumexp(logs))
    return Reweighted(weights, float(np.sum(weights * array)), float(1 / np.sum(weights**2)))
