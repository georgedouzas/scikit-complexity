"""Describe distributions with heavy tails: histograms, survival functions, power-law fits, Hill estimates, and Gini."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import optimize, special, stats

from ..base import AnalysisError
from ._utils import real_array

type Array = npt.NDArray[np.float64]
MIN_TAIL = 10


def histogram(values: npt.ArrayLike, bins: int = 30, *, log: bool = False) -> tuple[Array, Array]:
    """Return the edges and the probability density of a histogram, with logarithmic bins for heavy tails.

    Args:
        values:
            The samples.

        bins:
            The number of bins.

        log:
            Whether the bins are equally spaced in the logarithm of the values, which needs positive values.

    Returns:
        histogram:
            The bin edges and the density in every bin.

    Examples:
        >>> from skcomplex.core.analysis import histogram
        >>> edges, density = histogram([1, 2, 4, 8], bins=3, log=True)
        >>> edges.round(2).tolist()
        [1.0, 2.0, 4.0, 8.0]
    """
    array = real_array(values, 'values')
    if log:
        if np.any(array <= 0):
            error_msg = 'Logarithmic bins need positive values. Drop zeros or use linear bins.'
            raise AnalysisError(error_msg)
        edges = np.logspace(np.log10(array.min()), np.log10(array.max()), bins + 1)
    else:
        edges = np.linspace(array.min(), array.max(), bins + 1)
    density, edges = np.histogram(array, bins=edges, density=True)
    return edges, density


def ccdf(values: npt.ArrayLike) -> tuple[Array, Array]:
    """Return the complementary cumulative distribution, the share of samples at or above each distinct value.

    Args:
        values:
            The samples.

    Returns:
        ccdf:
            The sorted distinct values and `P(X >= x)` at each of them.

    Examples:
        >>> from skcomplex.core.analysis import ccdf
        >>> ccdf([1, 1, 2, 4])[1].tolist()
        [1.0, 0.5, 0.25]
    """
    array = np.sort(real_array(values, 'values'))
    distinct, first = np.unique(array, return_index=True)
    return distinct, 1 - first / array.size


def _alpha(tail: Array, xmin: float, *, discrete: bool) -> float:
    """Return the maximum-likelihood exponent of a power law above `xmin`."""
    if not discrete:
        return 1 + tail.size / np.sum(np.log(tail / xmin))
    total = np.sum(np.log(tail))

    def negative(alpha: float) -> float:
        return tail.size * np.log(special.zeta(alpha, xmin)) + alpha * total

    return float(optimize.minimize_scalar(negative, bounds=(1.0001, 8), method='bounded').x)


def _model_ccdf(points: Array, alpha: float, xmin: float, *, discrete: bool) -> Array:
    """Return the survival function of a power law above `xmin` at points."""
    if discrete:
        return special.zeta(alpha, points) / special.zeta(alpha, xmin)
    return (points / xmin) ** (1 - alpha)


def _ks(tail: Array, alpha: float, xmin: float, *, discrete: bool) -> float:
    """Return the Kolmogorov-Smirnov distance between the tail and the fitted power law."""
    distinct, empirical = ccdf(tail)
    return float(np.abs(empirical - _model_ccdf(distinct, alpha, xmin, discrete=discrete)).max())


def _fit(array: Array, *, discrete: bool, candidates: int) -> tuple[float, float, float]:
    """Return the exponent, the lower cutoff, and the distance that minimise the distance over cutoffs."""
    ordered = np.sort(array)
    distinct = np.unique(ordered)
    above = ordered.size - np.searchsorted(ordered, distinct)
    distinct = distinct[above >= MIN_TAIL]
    if distinct.size > candidates:
        distinct = distinct[np.unique(np.linspace(0, distinct.size - 1, candidates).astype(int))]
    choices = distinct
    best = (np.nan, np.nan, np.inf)
    for xmin in choices:
        tail = array[array >= xmin]
        if tail.size < MIN_TAIL:
            continue
        alpha = _alpha(tail, float(xmin), discrete=discrete)
        distance = _ks(tail, alpha, float(xmin), discrete=discrete)
        if distance < best[2]:
            best = (alpha, float(xmin), distance)
    return best


def _sample_power_law(size: int, alpha: float, xmin: float, rng: np.random.Generator, *, discrete: bool) -> Array:
    """Return samples of a power law above `xmin`, rounded with Clauset's approximation when discrete."""
    uniform = rng.random(size)
    if discrete:
        return np.floor((xmin - 0.5) * (1 - uniform) ** (-1 / (alpha - 1)) + 0.5)
    return xmin * (1 - uniform) ** (-1 / (alpha - 1))


def _vuong(first: Array, second: Array) -> tuple[float, float]:
    """Return the log-likelihood ratio of two fits and the two-sided p-value of its sign."""
    difference = first - second
    ratio = float(difference.sum())
    spread = float(difference.std())
    p_value = float(special.erfc(abs(ratio) / (np.sqrt(2 * difference.size) * spread))) if spread > 0 else 1.0
    return ratio, p_value


@dataclass(frozen=True)
class PowerLawFit:
    """Power law fitted to the tail of a sample.

    Args:
        alpha:
            The exponent of the density, `p(x) ∝ x^-alpha`.

        xmin:
            The lower cutoff of the tail.

        ks:
            The Kolmogorov-Smirnov distance between the tail and the fit.

        p_value:
            The share of bootstrap samples from the fit whose distance exceeds `ks`, or `None` without bootstrap.
            Values above 0.1 do not reject the power law.

        tail:
            The number of samples at or above `xmin`.

        comparisons:
            For every alternative, the log-likelihood ratio of the power law against it, positive when the power law
            fits better, and its p-value.
    """

    alpha: float
    xmin: float
    ks: float
    p_value: float | None
    tail: int
    comparisons: dict[str, tuple[float, float]] = field(default_factory=dict)


def fit_power_law(
    values: npt.ArrayLike,
    *,
    xmin: float | None = None,
    discrete: bool = False,
    bootstrap: int = 100,
    candidates: int = 50,
    random_state: int | None = None,
) -> PowerLawFit:
    """Fit a power law to the tail of a sample by maximum likelihood, with Clauset, Shalizi, and Newman's method.

    The lower cutoff minimises the Kolmogorov-Smirnov distance between the tail and the fit. The goodness of fit is the
    share of bootstrap samples whose distance exceeds the data's, and the power law is compared with exponential and
    log-normal tails by Vuong's likelihood-ratio test.

    Args:
        values:
            The positive samples.

        xmin:
            A fixed lower cutoff, or `None` to choose it.

        discrete:
            Whether the samples are integers, such as avalanche sizes.

        bootstrap:
            The number of bootstrap samples for the p-value, 0 for none.

        candidates:
            The largest number of cutoffs tried.

        random_state:
            The seed of the bootstrap.

    Returns:
        fit:
            The exponent, the cutoff, the distance, the p-value, the tail size, and the comparisons.

    Raises:
        AnalysisError:
            The sample has fewer than 10 positive values at or above a cutoff.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import fit_power_law
        >>> samples = (1 - np.random.default_rng(0).random(5000)) ** (-1 / 1.5)
        >>> round(float(fit_power_law(samples, xmin=1.0, bootstrap=0).alpha), 1)
        2.5
    """
    array = real_array(values, 'values')
    array = array[array > 0]
    if xmin is None:
        alpha, cutoff, distance = _fit(array, discrete=discrete, candidates=candidates)
    else:
        tail = array[array >= xmin]
        if tail.size < MIN_TAIL:
            error_msg = f'Only {tail.size} samples lie at or above xmin = {xmin}. A fit needs at least {MIN_TAIL}.'
            raise AnalysisError(error_msg)
        cutoff = float(xmin)
        alpha = _alpha(tail, cutoff, discrete=discrete)
        distance = _ks(tail, alpha, cutoff, discrete=discrete)
    if not np.isfinite(alpha):
        error_msg = f'No cutoff leaves at least {MIN_TAIL} samples in the tail. Give more samples.'
        raise AnalysisError(error_msg)
    tail = array[array >= cutoff]
    p_value = None
    if bootstrap:
        rng = np.random.default_rng(random_state)
        body = array[array < cutoff]
        exceed = 0
        for _ in range(bootstrap):
            in_tail = rng.random(array.size) < tail.size / array.size
            synthetic = np.concatenate(
                [
                    _sample_power_law(int(in_tail.sum()), alpha, cutoff, rng, discrete=discrete),
                    rng.choice(body, int((~in_tail).sum())) if body.size else np.zeros(0),
                ],
            )
            if xmin is None:
                distance_synthetic = _fit(synthetic, discrete=discrete, candidates=candidates)[2]
            else:
                synthetic_tail = synthetic[synthetic >= cutoff]
                synthetic_alpha = _alpha(synthetic_tail, cutoff, discrete=discrete)
                distance_synthetic = _ks(synthetic_tail, synthetic_alpha, cutoff, discrete=discrete)
            exceed += distance_synthetic >= distance
        p_value = exceed / bootstrap
    return PowerLawFit(
        alpha,
        cutoff,
        distance,
        p_value,
        int(tail.size),
        _compare(tail, alpha, cutoff, discrete=discrete),
    )


def _compare(tail: Array, alpha: float, xmin: float, *, discrete: bool) -> dict[str, tuple[float, float]]:
    """Return the likelihood-ratio comparisons of the power law against exponential and log-normal tails."""
    if discrete:
        power = -alpha * np.log(tail) - np.log(special.zeta(alpha, xmin))
    else:
        power = np.log((alpha - 1) / xmin) - alpha * np.log(tail / xmin)
    rate = 1 / max(tail.mean() - xmin, 1e-12)
    exponential = np.log(rate) - rate * (tail - xmin)
    logs = np.log(tail)

    def lognormal(parameters: Array) -> Array:
        mu, sigma = parameters[0], abs(parameters[1]) + 1e-12
        survival = stats.norm.sf((np.log(xmin) - mu) / sigma)
        return stats.norm.logpdf(logs, mu, sigma) - logs - np.log(max(survival, 1e-300))

    fit = optimize.minimize(
        lambda parameters: -lognormal(parameters).sum(),
        [logs.mean(), logs.std() + 1e-6],
        method='Nelder-Mead',
    )
    return {'exponential': _vuong(power, exponential), 'lognormal': _vuong(power, lognormal(fit.x))}


def hill(values: npt.ArrayLike, k: int) -> float:
    """Return the Hill estimate of the tail index from the `k` largest samples, the exponent of the survival function.

    Args:
        values:
            The positive samples.

        k:
            The number of largest samples used.

    Returns:
        index:
            The tail index, `alpha - 1` for a density `p(x) ∝ x^-alpha`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import hill
        >>> samples = (1 - np.random.default_rng(0).random(20000)) ** (-1 / 1.5)
        >>> round(hill(samples, 2000), 1)
        1.5
    """
    array = np.sort(real_array(values, 'values'))[::-1]
    if not 0 < k < array.size:
        error_msg = f'k = {k} must lie between 1 and the number of samples minus 1, {array.size - 1}.'
        raise AnalysisError(error_msg)
    return float(1 / np.mean(np.log(array[:k] / array[k])))


def gini(values: npt.ArrayLike) -> float:
    """Return the Gini coefficient of nonnegative values, 0 for equality and close to 1 when one holds everything.

    Args:
        values:
            The nonnegative values, such as wealth.

    Returns:
        gini:
            The Gini coefficient.

    Examples:
        >>> from skcomplex.core.analysis import gini
        >>> gini([1, 1, 1, 1]), gini([0, 0, 0, 4])
        (0.0, 0.75)
    """
    array = np.sort(real_array(values, 'values'))
    if array.size == 0 or array.sum() == 0:
        return 0.0
    ranks = np.arange(1, array.size + 1)
    return float(np.sum((2 * ranks - array.size - 1) * array) / (array.size * array.sum()))
