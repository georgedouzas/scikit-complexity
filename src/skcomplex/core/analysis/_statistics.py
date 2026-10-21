"""Describe distributions with heavy tails: histograms, survival functions, power-law fits, Hill estimates, and Gini."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._lazy import module

if TYPE_CHECKING:
    import powerlaw
from ..base import AnalysisError
from ._kinds import DataKind, analysis
from ._utils import real_array

type Array = npt.NDArray[np.float64]
MIN_TAIL = 10
ALTERNATIVES = ('exponential', 'lognormal', 'stretched_exponential', 'truncated_power_law')


@analysis(DataKind.SERIES)
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

    Raises:
        AnalysisError:
            Logarithmic bins are asked for samples that are not all positive.

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


@analysis(DataKind.SERIES)
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


def _fit_at(array: Array, xmin: float, xmax: float | None, *, discrete: bool) -> 'powerlaw.Fit':
    """Return the maximum-likelihood power law of the samples between the cutoffs."""
    return module('powerlaw').Fit(
        array,
        xmin=xmin,
        xmax=xmax,
        discrete=discrete,
        estimate_discrete=False,
        verbose=False,
    )


def _cutoffs(array: Array, xmax: float | None, candidates: int) -> Array:
    """Return up to `candidates` lower cutoffs that leave at least `MIN_TAIL` samples in the tail."""
    inside = np.sort(array if xmax is None else array[array <= xmax])
    distinct = np.unique(inside)
    above = inside.size - np.searchsorted(inside, distinct)
    distinct = distinct[above >= MIN_TAIL]
    if distinct.size > candidates:
        distinct = distinct[np.unique(np.linspace(0, distinct.size - 1, candidates).astype(int))]
    return distinct


def _best_fit(array: Array, xmax: float | None, *, discrete: bool, candidates: int) -> 'powerlaw.Fit':
    """Return the power law whose lower cutoff minimises the Kolmogorov-Smirnov distance over the candidates."""
    cutoffs = _cutoffs(array, xmax, candidates)
    if cutoffs.size == 0:
        error_msg = f'No cutoff leaves at least {MIN_TAIL} samples in the tail. Give more samples.'
        raise AnalysisError(error_msg)
    fits = [_fit_at(array, float(cutoff), xmax, discrete=discrete) for cutoff in cutoffs]
    return min(fits, key=lambda fit: float(fit.power_law.D))


def _sample_power_law(size: int, alpha: float, xmin: float, rng: np.random.Generator, *, discrete: bool) -> Array:
    """Return samples of a power law above `xmin`, rounded with Clauset's approximation when discrete."""
    uniform = rng.random(size)
    if discrete:
        return np.floor((xmin - 0.5) * (1 - uniform) ** (-1 / (alpha - 1)) + 0.5)
    return xmin * (1 - uniform) ** (-1 / (alpha - 1))


def _bootstrap_distance(
    synthetic: Array,
    xmin: float | None,
    xmax: float | None,
    *,
    discrete: bool,
    candidates: int,
) -> float:
    """Return the distance of the fit to a synthetic sample, refitting its cutoff unless the cutoff is fixed."""
    if xmin is None:
        return float(_best_fit(synthetic, xmax, discrete=discrete, candidates=candidates).power_law.D)
    return float(_fit_at(synthetic, xmin, xmax, discrete=discrete).power_law.D)


def _comparison(fit: 'powerlaw.Fit', alternative: str) -> tuple[float, float]:
    """Return the log-likelihood ratio of the power law against an alternative and its p-value."""
    ratio, p_value = fit.distribution_compare('power_law', alternative)
    return float(ratio), float(p_value)


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
            For every alternative, `'exponential'`, `'lognormal'`, `'stretched_exponential'`, and
            `'truncated_power_law'`, the log-likelihood ratio of the power law against it, positive when the power law
            fits better, and its p-value.

        xmax:
            The upper cutoff of the tail, or `None` without one.
    """

    alpha: float
    xmin: float
    ks: float
    p_value: float | None
    tail: int
    comparisons: dict[str, tuple[float, float]] = field(default_factory=dict)
    xmax: float | None = None


def _fit_power_law(
    array: Array,
    xmin: float | None,
    xmax: float | None,
    *,
    discrete: bool,
    bootstrap: int,
    candidates: int,
    random_state: int | None,
) -> PowerLawFit:
    """Fit a power law to the positive samples, as `fit_power_law` describes, with the warnings of `powerlaw` shown."""
    if xmin is None:
        fit = _best_fit(array, xmax, discrete=discrete, candidates=candidates)
    else:
        inside = array[(array >= xmin) & (array <= (np.inf if xmax is None else xmax))]
        if inside.size < MIN_TAIL:
            error_msg = f'Only {inside.size} samples lie at or above xmin = {xmin}. A fit needs at least {MIN_TAIL}.'
            raise AnalysisError(error_msg)
        fit = _fit_at(array, float(xmin), xmax, discrete=discrete)
    alpha, cutoff, distance = float(fit.power_law.alpha), float(fit.xmin), float(fit.power_law.D)
    tail = array[(array >= cutoff) & (array <= (np.inf if xmax is None else xmax))]
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
            exceed += _bootstrap_distance(synthetic, xmin, xmax, discrete=discrete, candidates=candidates) >= distance
        p_value = exceed / bootstrap
    comparisons = {name: _comparison(fit, name) for name in ALTERNATIVES}
    return PowerLawFit(alpha, cutoff, distance, p_value, int(tail.size), comparisons, xmax)


@analysis(DataKind.SERIES)
def fit_power_law(
    values: npt.ArrayLike,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    discrete: bool = False,
    bootstrap: int = 100,
    candidates: int = 50,
    random_state: int | None = None,
) -> PowerLawFit:
    """Fit a power law to the tail of a sample by maximum likelihood, with Clauset, Shalizi, and Newman's method.

    The lower cutoff minimises the Kolmogorov-Smirnov distance between the tail and the fit. The goodness of fit is the
    share of bootstrap samples whose distance exceeds the data's, and the power law is compared with exponential,
    log-normal, stretched exponential, and truncated power-law tails by likelihood-ratio tests. Discrete samples use
    the exact discrete likelihood. The warnings of `powerlaw` about its optimiser, such as an alternative without a
    valid fit, are silenced during the call.

    Args:
        values:
            The positive samples.

        xmin:
            A fixed lower cutoff, or `None` to choose it.

        xmax:
            A fixed upper cutoff, or `None` for none.

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
            The exponent, the cutoffs, the distance, the p-value, the tail size, and the comparisons.

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
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module=r'powerlaw\.')
        return _fit_power_law(
            array[array > 0],
            xmin,
            xmax,
            discrete=discrete,
            bootstrap=bootstrap,
            candidates=candidates,
            random_state=random_state,
        )


@analysis(DataKind.SERIES)
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

    Raises:
        AnalysisError:
            `k` is not between 1 and the number of samples minus 1.

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


@analysis(DataKind.SERIES)
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
