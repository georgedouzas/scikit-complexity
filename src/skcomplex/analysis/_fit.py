"""Fit laws to samples: power-law tails, stable laws, extreme-value laws, and named laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import powerlaw
from scipy import stats

from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis
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


class FitAnalysis(BaseAnalysis):
    """Law fitted to the values of a source, pooled into one sample.

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

    Attributes:
        values_:
            The pooled sample.

        labels_:
            The label of every series.

        exponent_:
            The exponent `alpha` of the density of the power law.

        minimum_:
            The lower cutoff of the power law.

        maximum_:
            The upper cutoff of the power law, or `None` without one.

        statistic_:
            The Kolmogorov-Smirnov distance between the tail and the power law, or between the sample and a named law.

        p_value_:
            For the power law, the share of bootstrap samples from the fit whose distance exceeds `statistic_`, or
            `None` without bootstrap, values above 0.1 not rejecting it. For a named law, the Monte Carlo p-value,
            values below the significance level rejecting it.

        tail_:
            The number of samples between the cutoffs of the power law.

        comparisons_:
            For every alternative of `compare_with`, the log-likelihood ratio of the power law against it, positive
            when the power law fits better, and its p-value.

        index_:
            The stability index of the stable law, 2 for the normal law.

        skewness_:
            The skewness of the stable law.

        shape_:
            The shape `xi` of the extreme-value law, 0 for the Gumbel class, positive for the Fréchet class of heavy
            tails, and negative for the Weibull class of bounded tails.

        location_:
            The location of the stable or the extreme-value law.

        scale_:
            The scale of the stable or the extreme-value law.

        maxima_:
            The maximum of every block of the extreme-value law.

        parameters_:
            The fitted and fixed parameters of a named law.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import FitAnalysis
        >>> samples = (1 - np.random.default_rng(0).random(5000)) ** (-1 / 1.5)
        >>> round(FitAnalysis('power', minimum=1.0, bootstrap=0).check().prepare(samples).run().exponent_, 1)
        2.5
        >>> fit = FitAnalysis('exponential', fixed={'loc': 0}).check()
        >>> fit = fit.prepare(np.random.default_rng(0).exponential(2.0, 1000)).run()
        >>> round(fit.parameters_['scale'], 0), fit.p_value_ > 0.05
        (2.0, True)
    """

    def __init__(
        self,
        law: str = 'power',
        variables: Sequence[str] | None = None,
        *,
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
    ) -> None:
        self.law = law
        self.variables = variables
        self.group = group
        self.minimum = minimum
        self.maximum = maximum
        self.discrete = discrete
        self.bootstrap = bootstrap
        self.candidates = candidates
        self.compare_with = compare_with
        self.block = block
        self.fixed = fixed
        self.simulations = simulations
        self.random_state = random_state

    def _check(self) -> None:
        """Validate the law, the alternatives, that the law reads every changed setting, and its needed settings."""
        if self.law not in LAWS:
            error_msg = f'{type(self).__name__} has law={self.law!r}. Use one of {", ".join(LAWS)}.'
            raise AnalysisError(error_msg)
        unknown = [name for name in self.compare_with if name not in ALTERNATIVES]
        if unknown:
            error_msg = (
                f'{type(self).__name__} has compare_with={self.compare_with!r}. Use some of {", ".join(ALTERNATIVES)}.'
            )
            raise AnalysisError(error_msg)
        self._check_settings('law', SETTINGS[self.law] | {'variables', 'group'})
        if self.law == 'extreme-value' and self.block is None:
            error_msg = f'{type(self).__name__} of law {self.law!r} needs the number of samples per block. Set `block`.'
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: Source) -> None:
        """Read the samples of the source pooled into one sample."""
        series = self._read_samples_of(source, self.variables, self.group)
        self.values_ = self._read_real(series.values, 'values').ravel()
        self.labels_ = series.labels

    def _bootstrap(self, array: Array, fit: powerlaw.Fit) -> float:
        """Return the share of synthetic samples, a power-law tail and a resampled body, farther from their fit."""
        alpha, cutoff, distance = float(fit.power_law.alpha), float(fit.xmin), float(fit.power_law.D)
        rng = np.random.default_rng(self.random_state)
        body = array[array < cutoff]
        exceed = 0
        for _ in range(self.bootstrap):
            in_tail = rng.random(array.size) < self.tail_ / array.size
            synthetic = np.concatenate(
                [
                    self._sample_power_law(int(in_tail.sum()), alpha, cutoff, rng, discrete=self.discrete),
                    rng.choice(body, int((~in_tail).sum())) if body.size else np.zeros(0),
                ],
            )
            if self.minimum is None:
                refit = self._best_fit(synthetic, self.maximum, discrete=self.discrete, candidates=self.candidates)
            else:
                refit = self._fit_at(synthetic, self.minimum, self.maximum, discrete=self.discrete)
            exceed += float(refit.power_law.D) >= distance
        return exceed / self.bootstrap

    def _run_power(self) -> None:
        """Fit a power law to the tail of the positive samples, test it, and compare it with the alternatives."""
        array = self.values_[self.values_ > 0]
        upper = np.inf if self.maximum is None else self.maximum
        if self.minimum is None:
            fit = self._best_fit(array, self.maximum, discrete=self.discrete, candidates=self.candidates)
        else:
            inside = array[(array >= self.minimum) & (array <= upper)]
            if inside.size < MIN_TAIL:
                error_msg = (
                    f'Only {inside.size} samples lie at or above minimum = {self.minimum}. A fit needs at least '
                    f'{MIN_TAIL}. Lower the minimum or give more samples.'
                )
                raise AnalysisError(error_msg)
            fit = self._fit_at(array, float(self.minimum), self.maximum, discrete=self.discrete)
        self.exponent_ = float(fit.power_law.alpha)
        self.minimum_ = float(fit.xmin)
        self.maximum_ = self.maximum
        self.statistic_ = float(fit.power_law.D)
        self.tail_ = int(np.sum((array >= self.minimum_) & (array <= upper)))
        self.p_value_ = self._bootstrap(array, fit) if self.bootstrap else None
        self.comparisons_ = {}
        for name in self.compare_with:
            ratio, p_value = fit.distribution_compare('power_law', name)
            self.comparisons_[name] = (float(ratio), float(p_value))

    def _run_extreme_value(self) -> None:
        """Fit the generalized extreme-value law to the maxima of blocks of samples."""
        if self.block is None:
            error_msg = f'{type(self).__name__} of law extreme-value needs the size of a block. Set `block`.'
            raise AnalysisError(error_msg)
        count = self.values_.size // self.block
        if count < MIN_BLOCKS:
            error_msg = (
                f'{self.values_.size} samples fill {count} blocks of {self.block}. A fit needs at least {MIN_BLOCKS} '
                'blocks. Use smaller blocks or give more samples.'
            )
            raise AnalysisError(error_msg)
        self.maxima_ = self.values_[: count * self.block].reshape(count, self.block).max(axis=1)
        shape, location, scale = stats.genextreme.fit(self.maxima_)
        self.shape_, self.location_, self.scale_ = float(-shape), float(location), float(scale)

    def _run_named_law(self) -> None:
        """Fit the free parameters of a named law and test the sample against it."""
        result = stats.goodness_of_fit(
            NAMED_LAWS[self.law],
            self.values_,
            known_params=dict(self.fixed or {}),
            statistic='ks',
            n_mc_samples=self.simulations,
            rng=np.random.default_rng(self.random_state),
        )
        self.parameters_ = {name: float(value) for name, value in result.fit_result.params._asdict().items()}
        self.statistic_ = float(result.statistic)
        self.p_value_ = float(result.pvalue)

    def _run(self, inputs: object = None) -> None:
        """Fit the law to the sample."""
        if self.law == 'power':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', module=r'powerlaw\.')
                self._run_power()
        elif self.law == 'stable':
            index, skewness, location, scale = stats.levy_stable.fit(self.values_, optimizer=self._starting_point)
            self.index_, self.skewness_ = float(index), float(skewness)
            self.location_, self.scale_ = float(location), float(scale)
        elif self.law == 'extreme-value':
            self._run_extreme_value()
        else:
            self._run_named_law()

    @staticmethod
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

    @staticmethod
    def _cutoffs(array: Array, maximum: float | None, candidates: int) -> Array:
        """Return up to `candidates` lower cutoffs that leave at least `MIN_TAIL` samples in the tail."""
        inside = np.sort(array if maximum is None else array[array <= maximum])
        distinct = np.unique(inside)
        above = inside.size - np.searchsorted(inside, distinct)
        distinct = distinct[above >= MIN_TAIL]
        if distinct.size > candidates:
            distinct = distinct[np.unique(np.linspace(0, distinct.size - 1, candidates).astype(int))]
        return distinct

    @staticmethod
    def _best_fit(array: Array, maximum: float | None, *, discrete: bool, candidates: int) -> powerlaw.Fit:
        """Return the power law whose lower cutoff minimises the Kolmogorov-Smirnov distance over the candidates."""
        cutoffs = FitAnalysis._cutoffs(array, maximum, candidates)
        if cutoffs.size == 0:
            error_msg = f'No cutoff leaves at least {MIN_TAIL} samples in the tail. Give more samples.'
            raise AnalysisError(error_msg)
        fits = [FitAnalysis._fit_at(array, float(cutoff), maximum, discrete=discrete) for cutoff in cutoffs]
        return min(fits, key=lambda fit: float(fit.power_law.D))

    @staticmethod
    def _sample_power_law(
        size: int,
        alpha: float,
        minimum: float,
        rng: np.random.Generator,
        *,
        discrete: bool,
    ) -> Array:
        """Return samples of a power law above `minimum`, rounded with Clauset's approximation when discrete."""
        uniform = rng.random(size)
        if discrete:
            return np.floor((minimum - 0.5) * (1 - uniform) ** (-1 / (alpha - 1)) + 0.5)
        return minimum * (1 - uniform) ** (-1 / (alpha - 1))

    @staticmethod
    def _starting_point(_: object, start: Array, *__: object, **___: object) -> Array:
        """Return SciPy's starting point of a fit, McCulloch's quantile estimate for stable laws, unchanged."""
        return start
