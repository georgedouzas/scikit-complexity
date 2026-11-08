"""Fit laws to samples.

The laws are power-law tails, stable laws, extreme-value laws, and named laws.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import powerlaw
from scipy import stats

from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import _check_settings, _read_real, _read_samples_of
from .types import Source

NAMED_LAWS = {
    'exponential': stats.expon,
    'normal': stats.norm,
    'lognormal': stats.lognorm,
    'gamma': stats.gamma,
    'maxwell': stats.maxwell,
    'rayleigh': stats.rayleigh,
    'uniform': stats.uniform,
}
LAWS = ('power', 'stable', 'extreme-value', *NAMED_LAWS)
ALTERNATIVES = ('exponential', 'lognormal', 'stretched_exponential', 'truncated_power_law')
SETTINGS = {
    'power': {'minimum', 'maximum', 'discrete', 'bootstrap', 'candidates', 'compare_with', 'random_state'},
    'stable': set(),
    'extreme-value': {'block'},
    **{law: {'fixed', 'simulations', 'random_state'} for law in NAMED_LAWS},
}
"""Settings every law reads, besides `law`, `variables`, and `group`."""

MIN_TAIL = 10
MIN_BLOCKS = 10

TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class FitLawResult:
    """Law fitted to the values of a source, pooled into one sample.

    Args:
        values:
            The pooled sample.

        labels:
            The label of every series.

        exponent:
            The exponent `alpha` of the density of the power law, for `'power'`.

        minimum:
            The lower cutoff of the power law, for `'power'`.

        maximum:
            The upper cutoff of the power law, or `None` without one.

        statistic:
            The Kolmogorov-Smirnov distance between the tail and the power law, or between the sample and a named law.

        p_value:
            For the power law, the share of bootstrap samples from the fit whose distance exceeds `statistic`, or
            `None` without bootstrap, values above 0.1 not rejecting it. For a named law, the Monte Carlo p-value,
            values below the significance level rejecting it.

        tail:
            The number of samples between the cutoffs of the power law, for `'power'`.

        comparisons:
            For every alternative of `compare_with`, the log-likelihood ratio of the power law against it, positive
            when the power law fits better, and its p-value, for `'power'`.

        index:
            The stability index of the stable law, 2 for the normal law, for `'stable'`.

        skewness:
            The skewness of the stable law, for `'stable'`.

        shape:
            The shape `xi` of the extreme-value law, 0 for the Gumbel class, positive for the Fréchet class of heavy
            tails, and negative for the Weibull class of bounded tails, for `'extreme-value'`.

        location:
            The location of the stable or the extreme-value law.

        scale:
            The scale of the stable or the extreme-value law.

        maxima:
            The maximum of every block of the extreme-value law, for `'extreme-value'`.

        parameters:
            The fitted and fixed parameters of a named law.
    """

    values: Array
    labels: list[str]
    exponent: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    statistic: float | None = None
    p_value: float | None = None
    tail: int | None = None
    comparisons: dict[str, tuple[float, float]] | None = None
    index: float | None = None
    skewness: float | None = None
    shape: float | None = None
    location: float | None = None
    scale: float | None = None
    maxima: Array | None = None
    parameters: dict[str, float] | None = None


def fit_law(
    source: Source,
    *,
    law: str = 'power',
    variables: Sequence[str] | None = None,
    group: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    discrete: bool = False,
    bootstrap: int = 100,
    candidates: int = 50,
    compare_with: Sequence[str] = ALTERNATIVES,
    block: int | None = None,
    fixed: Mapping[str, float] | None = None,
    simulations: int = 199,
    random_state: int | None = 0,
) -> FitLawResult:
    """Return the law fitted to the values of a source, pooled into one sample.

    The source gives the series of a trajectory or an array, or the speeds of the agents of a trajectory at every
    frame.

    The laws are:

    - `'power'`: a power law `p(x) ∝ x^-alpha` fitted to the tail of the positive samples by maximum likelihood, with
      Clauset, Shalizi, and Newman's method. The lower cutoff minimises the Kolmogorov-Smirnov distance between the
      tail and the fit, unless `minimum` fixes it. The goodness of fit is the share of `bootstrap` samples whose
      distance exceeds the data's, and the power law is compared with every alternative of `compare_with` by a
      likelihood-ratio test. Discrete samples use the exact discrete likelihood. The warnings of `powerlaw` about its
      optimiser are silenced during the run.
    - `'stable'`: the Lévy stable law fitted by McCulloch's quantile method, in SciPy's `S1` parameterization.
    - `'extreme-value'`: the generalized extreme-value law fitted by maximum likelihood to the maxima of blocks of
      `block` samples.
    - `'exponential'`, `'normal'`, `'lognormal'`, `'gamma'`, `'maxwell'`, `'rayleigh'`, and `'uniform'`: the named law
      with its free parameters fitted, tested with a Monte Carlo Kolmogorov-Smirnov test that accounts for the fit.

    Args:
        source:
            The trajectory, or an array of samples.

        law:
            The law to fit, one of the laws above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        group:
            The path of the agent or particle group whose speeds are read, or `None` for the series of a trajectory, or
            the first group of a trajectory without series.

        minimum:
            A fixed lower cutoff of the power law, or `None` to choose it.

        maximum:
            A fixed upper cutoff of the power law, or `None` for none.

        discrete:
            Whether the samples of the power law are integers, such as avalanche sizes.

        bootstrap:
            The number of bootstrap samples for the p-value of the power law, 0 for none.

        candidates:
            The largest number of lower cutoffs tried.

        compare_with:
            The alternatives the power law is compared with, among `'exponential'`, `'lognormal'`,
            `'stretched_exponential'`, and `'truncated_power_law'`.

        block:
            The number of samples per block of the extreme-value law.

        fixed:
            The parameters of a named law held fixed by SciPy name, such as `{'loc': 0}`, or `None` to fit all.

        simulations:
            The number of Monte Carlo samples of the null distribution of a named law.

        random_state:
            The seed of the bootstrap of the power law and of the Monte Carlo samples of a named law.

    Returns:
        result:
            The pooled sample and the fitted law.

    Raises:
        AnalysisError:
            The law or an alternative is unknown, a setting the law does not read is changed, a needed setting is
            missing, or the sample is too small for the fit.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import fit_law
        >>> samples = (1 - np.random.default_rng(0).random(5000)) ** (-1 / 1.5)
        >>> round(fit_law(samples, law='power', minimum=1.0, bootstrap=0).exponent, 1)
        2.5
        >>> fit = fit_law(np.random.default_rng(0).exponential(2.0, 1000), law='exponential', fixed={'loc': 0})
        >>> round(fit.parameters['scale'], 0), fit.p_value > 0.05
        (2.0, True)
    """
    if law not in LAWS:
        error_msg = f'fit_law has law={law!r}. Use one of {", ".join(LAWS)}.'
        raise AnalysisError(error_msg)
    unknown = [name for name in compare_with if name not in ALTERNATIVES]
    if unknown:
        error_msg = f'fit_law has compare_with={compare_with!r}. Use some of {", ".join(ALTERNATIVES)}.'
        raise AnalysisError(error_msg)
    _check_settings(
        fit_law,
        'law',
        SETTINGS[law] | {'variables', 'group'},
        law=law,
        variables=variables,
        group=group,
        minimum=minimum,
        maximum=maximum,
        discrete=discrete,
        bootstrap=bootstrap,
        candidates=candidates,
        compare_with=compare_with,
        block=block,
        fixed=fixed,
        simulations=simulations,
        random_state=random_state,
    )
    if law == 'extreme-value' and block is None:
        error_msg = f'fit_law of law {law!r} needs the number of samples per block. Set `block`.'
        raise AnalysisError(error_msg)
    series = _read_samples_of(source, variables, group)
    values = _read_real(series.values, 'values').ravel()
    labels = series.labels
    if law == 'power':
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module=r'powerlaw\.')
            return _fit_power(
                values,
                labels,
                minimum,
                maximum,
                discrete=discrete,
                bootstrap=bootstrap,
                candidates=candidates,
                compare_with=compare_with,
                random_state=random_state,
            )
    if law == 'stable':
        index, skewness, location, scale = stats.levy_stable.fit(values, optimizer=_starting_point)
        return FitLawResult(
            values,
            labels,
            index=float(index),
            skewness=float(skewness),
            location=float(location),
            scale=float(scale),
        )
    if law == 'extreme-value':
        return _fit_extreme_value(values, labels, block)
    result = stats.goodness_of_fit(
        NAMED_LAWS[law],
        values,
        known_params=dict(fixed or {}),
        statistic='ks',
        n_mc_samples=simulations,
        rng=np.random.default_rng(random_state),
    )
    return FitLawResult(
        values,
        labels,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        parameters={name: float(value) for name, value in result.fit_result.params._asdict().items()},
    )


def _fit_bootstrap(
    array: Array,
    fit: powerlaw.Fit,
    tail: int,
    minimum: float | None,
    maximum: float | None,
    *,
    discrete: bool,
    bootstrap: int,
    candidates: int,
    random_state: int | None,
) -> float:
    """Return the share of synthetic samples farther from their fit.

    A synthetic sample is a power-law tail and a resampled body.
    """
    alpha, cutoff, distance = float(fit.power_law.alpha), float(fit.xmin), float(fit.power_law.D)
    rng = np.random.default_rng(random_state)
    body = array[array < cutoff]
    exceed = 0
    for _ in range(bootstrap):
        in_tail = rng.random(array.size) < tail / array.size
        synthetic = np.concatenate(
            [
                _sample_power_law(int(in_tail.sum()), alpha, cutoff, rng, discrete=discrete),
                rng.choice(body, int((~in_tail).sum())) if body.size else np.zeros(0),
            ],
        )
        if minimum is None:
            refit = _best_fit(synthetic, maximum, discrete=discrete, candidates=candidates)
        else:
            refit = _fit_at(synthetic, minimum, maximum, discrete=discrete)
        exceed += float(refit.power_law.D) >= distance
    return exceed / bootstrap


def _fit_power(
    values: Array,
    labels: list[str],
    minimum: float | None,
    maximum: float | None,
    *,
    discrete: bool,
    bootstrap: int,
    candidates: int,
    compare_with: Sequence[str],
    random_state: int | None,
) -> FitLawResult:
    """Fit a power law to the tail of the positive samples, and test and compare it.

    It is compared with the alternatives.
    """
    array = values[values > 0]
    upper = np.inf if maximum is None else maximum
    if minimum is None:
        fit = _best_fit(array, maximum, discrete=discrete, candidates=candidates)
    else:
        inside = array[(array >= minimum) & (array <= upper)]
        if inside.size < MIN_TAIL:
            error_msg = (
                f'Only {inside.size} samples lie at or above minimum = {minimum}. A fit needs at least '
                f'{MIN_TAIL}. Lower the minimum or give more samples.'
            )
            raise AnalysisError(error_msg)
        fit = _fit_at(array, float(minimum), maximum, discrete=discrete)
    cutoff = float(fit.xmin)
    tail = int(np.sum((array >= cutoff) & (array <= upper)))
    p_value = (
        _fit_bootstrap(
            array,
            fit,
            tail,
            minimum,
            maximum,
            discrete=discrete,
            bootstrap=bootstrap,
            candidates=candidates,
            random_state=random_state,
        )
        if bootstrap
        else None
    )
    comparisons = {}
    for name in compare_with:
        ratio, compared = fit.distribution_compare('power_law', name)
        comparisons[name] = (float(ratio), float(compared))
    return FitLawResult(
        values,
        labels,
        exponent=float(fit.power_law.alpha),
        minimum=cutoff,
        maximum=maximum,
        statistic=float(fit.power_law.D),
        p_value=p_value,
        tail=tail,
        comparisons=comparisons,
    )


def _fit_extreme_value(values: Array, labels: list[str], block: int | None) -> FitLawResult:
    """Fit the generalized extreme-value law to the maxima of blocks of samples."""
    if block is None:
        error_msg = 'fit_law of law extreme-value needs the size of a block. Set `block`.'
        raise AnalysisError(error_msg)
    count = values.size // block
    if count < MIN_BLOCKS:
        error_msg = (
            f'{values.size} samples fill {count} blocks of {block}. A fit needs at least {MIN_BLOCKS} '
            'blocks. Use smaller blocks or give more samples.'
        )
        raise AnalysisError(error_msg)
    maxima = values[: count * block].reshape(count, block).max(axis=1)
    shape, location, scale = stats.genextreme.fit(maxima)
    return FitLawResult(
        values,
        labels,
        shape=float(-shape),
        location=float(location),
        scale=float(scale),
        maxima=maxima,
    )


def _fit_at(array: Array, minimum: float, maximum: float | None, *, discrete: bool) -> powerlaw.Fit:
    """Return the maximum-likelihood power law of the samples between the cutoffs."""
    return powerlaw.Fit(
        array,
        xmin=minimum,
        xmax=maximum,
        discrete=discrete,
        estimate_discrete=False,
        verbose=False,
    )


def _cutoffs(array: Array, maximum: float | None, candidates: int) -> Array:
    """Return up to `candidates` lower cutoffs.

    Every cutoff leaves at least `MIN_TAIL` samples in the tail.
    """
    inside = np.sort(array if maximum is None else array[array <= maximum])
    distinct = np.unique(inside)
    above = inside.size - np.searchsorted(inside, distinct)
    distinct = distinct[above >= MIN_TAIL]
    if distinct.size > candidates:
        distinct = distinct[np.unique(np.linspace(0, distinct.size - 1, candidates).astype(int))]
    return distinct


def _best_fit(array: Array, maximum: float | None, *, discrete: bool, candidates: int) -> powerlaw.Fit:
    """Return the power law of the candidate cutoff with the least KS distance.

    The lower cutoff minimises the Kolmogorov-Smirnov distance over the candidates.
    """
    cutoffs = _cutoffs(array, maximum, candidates)
    if cutoffs.size == 0:
        error_msg = f'No cutoff leaves at least {MIN_TAIL} samples in the tail. Give more samples.'
        raise AnalysisError(error_msg)
    fits = [_fit_at(array, float(cutoff), maximum, discrete=discrete) for cutoff in cutoffs]
    return min(fits, key=lambda fit: float(fit.power_law.D))


def _sample_power_law(
    size: int,
    alpha: float,
    minimum: float,
    rng: np.random.Generator,
    *,
    discrete: bool,
) -> Array:
    """Return samples of a power law above `minimum`.

    They are rounded with Clauset's approximation when discrete.
    """
    uniform = rng.random(size)
    if discrete:
        return np.floor((minimum - 0.5) * (1 - uniform) ** (-1 / (alpha - 1)) + 0.5)
    return minimum * (1 - uniform) ** (-1 / (alpha - 1))


def _starting_point(_: object, start: Array, *__: object, **___: object) -> Array:
    """Return SciPy's starting point of a fit.

    For stable laws it is McCulloch's quantile estimate, unchanged.
    """
    return start
