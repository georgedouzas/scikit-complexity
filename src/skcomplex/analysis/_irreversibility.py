"""Measure irreversibility: entropy production, detailed balance, fluctuation theorems, and responses."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array
from ._base import BaseAnalysis, Source
from ._kinds import DataKind
from ._markov import MarkovAnalysis

MIN_PAIRS = 3
MIN_PATHS = 2
TUR_BOUND = 2.0
KINDS = (
    'entropy-production',
    'jump-entropy-production',
    'langevin-entropy-production',
    'count-entropy-production',
    'detailed-balance',
    'fluctuation-theorem',
    'uncertainty-relation',
    'response',
    'fluctuation-dissipation',
)
SETTINGS = {
    'entropy-production': set(),
    'jump-entropy-production': {'rates', 'duration'},
    'langevin-entropy-production': {'force', 'temperature', 'dt'},
    'count-entropy-production': {'n_states', 'dt'},
    'detailed-balance': {'n_states', 'significance'},
    'fluctuation-theorem': {'bins'},
    'uncertainty-relation': {'entropy', 'resamples', 'random_state'},
    'response': {'unperturbed', 'field'},
    'fluctuation-dissipation': {'unperturbed', 'field', 'correlation', 'temperature'},
}
"""Settings every kind reads, besides `kind` and `variables`."""
PATH_KINDS = ('jump-entropy-production', 'count-entropy-production', 'detailed-balance')
ENSEMBLE_KINDS = (
    'langevin-entropy-production',
    'fluctuation-theorem',
    'uncertainty-relation',
    'response',
    'fluctuation-dissipation',
)


class IrreversibilityAnalysis(BaseAnalysis):
    """Irreversibility of jump and Langevin dynamics, of one kind.

    The kinds read a rate matrix, jump paths, Langevin paths, samples, or ensembles:

    - `'entropy-production'`: the stationary entropy production rate of a jump process from its rate matrix,
      `Σ_{i<j} (p_i k_ij - p_j k_ji) ln(p_i k_ij / (p_j k_ji))`, 0 under detailed balance.
    - `'jump-entropy-production'`: the entropy every jump path produces in the medium, `Σ ln(k_ij / k_ji)` over its
      jumps with the `rates`, and the mean rate over the `duration` of the paths.
    - `'langevin-entropy-production'`: the heat every overdamped Langevin path releases over the temperature,
      `∫ F(x) ∘ dx / T`, and the mean rate.
    - `'count-entropy-production'`: the entropy production rate of a discrete path from its transition counts alone,
      `Σ_{i<j} (N_ij - N_ji) ln(N_ij / N_ji)` per unit time.
    - `'detailed-balance'`: the chi-squared test of detailed balance on the transition counts of discrete paths.
    - `'fluctuation-theorem'`: the log ratio of the probabilities of producing `s` and `-s` from samples of the
      entropy, and its slope against `s`, 1 when the theorem holds.
    - `'uncertainty-relation'`: the squared relative fluctuation `Var(J) / ⟨J⟩²` of samples of an integrated current
      times the mean `entropy` produced over the same duration, at least 2 (Barato and Seifert 2015).
    - `'response'`: the integrated response `chi(t) = (<x(t)>_h - <x(t)>_0) / h` of an ensemble with the field
      against the `unperturbed` ensemble.
    - `'fluctuation-dissipation'`: the response, and the slope of `T chi` against `C(t_w, t_w) - C(t, t_w)` of the
      two-time `correlation` over the same lags, 1 in equilibrium and below 1 in an aging system.

    Args:
        kind:
            The measure to compute, one of the kinds above.

        variables:
            The discrete state or the position of a trajectory to read, or `None` for the first one.

        rates:
            The rate of every jump, one row per origin.

        duration:
            The duration of every jump path, or `None` for the duration of a trajectory.

        force:
            The force as a function of position, or a constant force.

        temperature:
            The temperature of the bath.

        dt:
            The time between frames of Langevin paths, or between records of a discrete path.

        n_states:
            The number of states, or `None` for the largest state plus one.

        significance:
            The level below which the p-value rejects detailed balance.

        bins:
            The number of bins of entropy, symmetric about 0.

        entropy:
            The mean entropy produced along a path over the duration of the currents.

        resamples:
            The number of bootstrap resamples of the standard error of the uncertainty product.

        random_state:
            The seed of the bootstrap.

        unperturbed:
            The ensemble without the field, one row per run and one column per frame.

        field:
            The strength of the field.

        correlation:
            The two-time correlation over lags from 0.

    Attributes:
        values_:
            The rate matrix, the paths one row per path, or the samples the kind reads.

        duration_:
            The duration of the jump paths, the setting or the duration of a trajectory.

        entropy_production_:
            The entropy production per unit time in units of `k_B`, of the entropy production kinds.

        entropies_:
            The entropy produced along every path, of `'jump-entropy-production'` and
            `'langevin-entropy-production'`.

        currents_:
            The net number of transitions from every state to every other per step, of `'detailed-balance'`.

        statistic_:
            The statistic `Σ (N_ij - N_ji)² / (N_ij + N_ji)` over pairs of states, of `'detailed-balance'`.

        p_value_:
            The chi-squared p-value of the statistic under detailed balance, of `'detailed-balance'`.

        balanced_:
            Whether the p-value exceeds the significance level, of `'detailed-balance'`.

        centres_:
            The centres of the bins of entropy whose mirror bin is also populated, of `'fluctuation-theorem'`.

        log_ratios_:
            `ln P(s) / P(-s)` at every centre, of `'fluctuation-theorem'`.

        slope_:
            The slope of the log ratio against the entropy, of `'fluctuation-theorem'`.

        mean_:
            The mean of the current over the paths, of `'uncertainty-relation'`.

        variance_:
            The variance of the current over the paths, of `'uncertainty-relation'`.

        product_:
            The squared relative fluctuation of the current times the entropy, of `'uncertainty-relation'`.

        error_:
            The standard error of the product from bootstrap resamples of the paths, of `'uncertainty-relation'`.

        holds_:
            Whether the product is at least 2 within two standard errors, of `'uncertainty-relation'`.

        response_:
            The integrated response at every frame, of `'response'` and `'fluctuation-dissipation'`.

        fluctuation_dissipation_ratio_:
            The fluctuation-dissipation ratio, of `'fluctuation-dissipation'`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import IrreversibilityAnalysis
        >>> ring = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
        >>> production = IrreversibilityAnalysis().check().prepare(ring).run().entropy_production_
        >>> bool(abs(production - np.log(2)) < 1e-12)
        True
        >>> jumps = IrreversibilityAnalysis('jump-entropy-production', rates=[[0, 2], [1, 0]], duration=3.0)
        >>> jumps.check().prepare([0, 1, 0, 1]).run().entropies_.round(3).tolist()
        [0.693]
        >>> IrreversibilityAnalysis('detailed-balance').check().prepare([0, 1, 2] * 1000).run().balanced_
        False
    """

    def __init__(
        self,
        kind: str = 'entropy-production',
        variables: Sequence[str] | None = None,
        *,
        rates: npt.ArrayLike | None = None,
        duration: float | None = None,
        force: Callable[[Array], Array] | float = 1.0,
        temperature: float = 1.0,
        dt: float = 1.0,
        n_states: int | None = None,
        significance: float = 0.05,
        bins: int = 40,
        entropy: float | None = None,
        resamples: int = 200,
        random_state: int | None = None,
        unperturbed: npt.ArrayLike | None = None,
        field: float = 1.0,
        correlation: npt.ArrayLike | None = None,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.rates = rates
        self.duration = duration
        self.force = force
        self.temperature = temperature
        self.dt = dt
        self.n_states = n_states
        self.significance = significance
        self.bins = bins
        self.entropy = entropy
        self.resamples = resamples
        self.random_state = random_state
        self.unperturbed = unperturbed
        self.field = field
        self.correlation = correlation

    def _check_needed(self, names: Sequence[str]) -> None:
        """Raise an error for a setting the kind needs that is not given."""
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            error_msg = f'{type(self).__name__} of kind {self.kind!r} needs {", ".join(missing)}. Give them.'
            raise AnalysisError(error_msg)

    def _check(self) -> Self:
        """Validate the kind, the settings each kind needs, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.kind == 'jump-entropy-production':
            self._check_needed(['rates'])
        if self.kind in ('response', 'fluctuation-dissipation'):
            self._check_needed(['unperturbed'])
        if self.kind == 'fluctuation-dissipation':
            self._check_needed(['correlation'])
        if self.kind == 'uncertainty-relation':
            self._check_needed(['entropy'])
        if self.temperature <= 0 or self.dt <= 0 or self.field == 0:
            error_msg = (
                f'{type(self).__name__} has temperature={self.temperature!r}, dt={self.dt!r}, and '
                f'field={self.field!r}. Use a positive temperature and step and a nonzero field.'
            )
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the path kind for rates and discrete paths, and the ensemble kind for samples and ensembles."""
        return DataKind.ENSEMBLE if self.kind in ENSEMBLE_KINDS else DataKind.PATH

    def _prepare(self, source: Source) -> Self:
        """Read the rate matrix, the paths as rows, the samples, or the ensemble the kind measures."""
        name = self.variables[0] if self.variables else None
        self.duration_ = self.duration
        if isinstance(source, Trajectory) and self.kind in PATH_KINDS:
            path = self._read_path(source, name)
            self.values_ = np.asarray(path.states, dtype=np.int64).reshape(1, -1)
            if self.duration is None:
                self.duration_ = float(path.times[-1] - path.times[0])
            return self
        if isinstance(source, Trajectory):
            self.values_ = self._read_series(source, self.variables).values[:, 0].reshape(1, -1)
            return self
        dtype = np.int64 if self.kind in PATH_KINDS else float
        array = np.asarray(source, dtype=dtype)
        paths = self.kind in PATH_KINDS or self.kind == 'langevin-entropy-production'
        self.values_ = array.reshape(1, -1) if paths and array.ndim == 1 else array
        return self

    def _run_jumps(self) -> None:
        """Compute the entropy every jump path produces in the medium."""
        duration = self.duration_
        if duration is None:
            error_msg = f'{type(self).__name__} reads paths of an array without their duration. Give the duration.'
            raise AnalysisError(error_msg)
        matrix = np.asarray(self.rates, dtype=float)
        origins, destinations = self.values_[:, :-1], self.values_[:, 1:]
        moved = origins != destinations
        steps = np.where(
            moved,
            np.log(matrix[origins, destinations] / np.where(moved, matrix[destinations, origins], 1)),
            0,
        )
        self.entropies_ = steps.sum(axis=1)
        self.entropy_production_ = float(self.entropies_.mean() / duration)

    def _run_langevin(self) -> None:
        """Compute the heat every Langevin path releases over the temperature."""
        array = self.values_
        midpoints = (array[:, 1:] + array[:, :-1]) / 2
        forces = self.force(midpoints) if callable(self.force) else np.full_like(midpoints, self.force)
        self.entropies_ = np.sum(forces * np.diff(array, axis=1), axis=1) / self.temperature
        self.entropy_production_ = float(self.entropies_.mean() / (self.dt * (array.shape[1] - 1)))

    def _run_detailed_balance(self) -> None:
        """Test detailed balance on the transition counts."""
        counts = MarkovAnalysis._transition_counts(self.values_, self.n_states).astype(float)
        steps = counts.sum()
        upper = np.triu_indices_from(counts, k=1)
        forward, backward = counts[upper], counts.T[upper]
        visited = (forward + backward) > 0
        self.statistic_ = float(np.sum((forward - backward)[visited] ** 2 / (forward + backward)[visited]))
        self.p_value_ = float(stats.chi2.sf(self.statistic_, max(int(visited.sum()), 1)))
        self.currents_ = (counts - counts.T) / steps
        self.balanced_ = self.p_value_ > self.significance

    def _run_fluctuation_theorem(self) -> None:
        """Compute the log ratios of producing `s` and `-s` and their slope."""
        array = self.values_.ravel()
        reach = float(np.abs(array).max())
        counts, edges = np.histogram(array, bins=2 * (self.bins // 2), range=(-reach, reach))
        centres = (edges[1:] + edges[:-1]) / 2
        mirrored = counts[::-1]
        usable = (centres > 0) & (counts > 0) & (mirrored > 0)
        if usable.sum() < MIN_PAIRS:
            error_msg = f'Fewer than {MIN_PAIRS} bins hold both s and -s. Give more samples or fewer bins.'
            raise AnalysisError(error_msg)
        self.log_ratios_ = np.log(counts[usable] / mirrored[usable])
        self.centres_ = centres[usable]
        self.slope_ = float(np.sum(self.centres_ * self.log_ratios_) / np.sum(self.centres_**2))

    def _get_product(self, sample: Array) -> float:
        """Return the squared relative fluctuation of a sample of currents times the entropy."""
        return float(sample.var(ddof=1) / sample.mean() ** 2 * float(self.entropy or 0.0))

    def _run_uncertainty_relation(self) -> None:
        """Compute the precision of the current against the entropy production."""
        values = self.values_.ravel()
        entropy = float(self.entropy or 0.0)
        if values.size < MIN_PATHS or entropy < 0:
            error_msg = (
                f'The uncertainty relation needs at least {MIN_PATHS} paths and a nonnegative entropy production, but '
                f'got {values.size} paths and {self.entropy}.'
            )
            raise AnalysisError(error_msg)
        self.mean_, self.variance_ = float(values.mean()), float(values.var(ddof=1))
        if self.mean_ == 0:
            error_msg = (
                'The mean current is zero, so its relative fluctuation is undefined. Measure a current that flows.'
            )
            raise AnalysisError(error_msg)
        rng = np.random.default_rng(self.random_state)
        boots = [self._get_product(values[rng.integers(values.size, size=values.size)]) for _ in range(self.resamples)]
        self.product_, self.error_ = self._get_product(values), float(np.std(boots, ddof=1))
        self.holds_ = self.product_ + 2 * self.error_ >= TUR_BOUND

    def _run(self, inputs: object = None) -> Self:
        """Compute the measure of the kind."""
        if self.kind == 'entropy-production':
            self.entropy_production_ = self._entropy_production(self.values_)
        elif self.kind == 'jump-entropy-production':
            self._run_jumps()
        elif self.kind == 'langevin-entropy-production':
            self._run_langevin()
        elif self.kind == 'count-entropy-production':
            self.entropy_production_ = self._count_entropy_production(self.values_, self.n_states, self.dt)
        elif self.kind == 'detailed-balance':
            self._run_detailed_balance()
        elif self.kind == 'fluctuation-theorem':
            self._run_fluctuation_theorem()
        elif self.kind == 'uncertainty-relation':
            self._run_uncertainty_relation()
        else:
            self.response_ = self._response(self.values_, np.asarray(self.unperturbed, dtype=float), self.field)
            if self.kind == 'fluctuation-dissipation':
                self.fluctuation_dissipation_ratio_ = self._fluctuation_dissipation_ratio(
                    np.asarray(self.correlation, dtype=float),
                    self.response_,
                    self.temperature,
                )
        return self

    @staticmethod
    def _entropy_production(rates: npt.ArrayLike) -> float:
        """Return `Σ_{i<j} (p_i k_ij - p_j k_ji) ln(p_i k_ij / (p_j k_ji))` with `p` the stationary law of the rates."""
        matrix = np.asarray(rates, dtype=float)
        matrix = matrix - np.diag(np.diag(matrix))
        if np.any((matrix > 0) != (matrix.T > 0)):
            error_msg = (
                'A jump has no reverse jump, so the entropy production is infinite. Give every jump its reverse.'
            )
            raise AnalysisError(error_msg)
        flux = MarkovAnalysis._stationary_distribution(matrix, rates=True)[:, None] * matrix
        positive = flux > 0
        return float(0.5 * np.sum((flux - flux.T)[positive] * np.log(flux[positive] / flux.T[positive])))

    @staticmethod
    def _count_entropy_production(states: npt.ArrayLike, n_states: int | None, dt: float) -> float:
        """Return `Σ_{i<j} (N_ij - N_ji) ln(N_ij / N_ji)` over the pairs visited both ways, per unit time."""
        counts = MarkovAnalysis._transition_counts(states, n_states).astype(float)
        upper = np.triu(np.ones_like(counts, dtype=bool), k=1)
        both = upper & (counts > 0) & (counts.T > 0)
        forward, backward = counts[both], counts.T[both]
        return float(np.sum((forward - backward) * np.log(forward / backward)) / (counts.sum() * dt))

    @staticmethod
    def _response(perturbed: npt.ArrayLike, unperturbed: npt.ArrayLike, field: float) -> Array:
        """Return the integrated response `chi(t) = (<x(t)>_h - <x(t)>_0) / h` of ensembles with and without a field."""
        first, second = np.asarray(perturbed, dtype=float), np.asarray(unperturbed, dtype=float)
        return (first.mean(axis=0) - second.mean(axis=0)) / field

    @staticmethod
    def _fluctuation_dissipation_ratio(correlation: npt.ArrayLike, susceptibility: Array, temperature: float) -> float:
        """Return the slope of `T chi` against `C(t_w, t_w) - C(t, t_w)`."""
        values = np.asarray(correlation, dtype=float)
        decay = values[0] - values
        scaled = temperature * susceptibility
        return float(np.sum(decay * scaled) / np.sum(decay**2))
