"""Measure irreversibility."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..engine.runtime import Trajectory
from ..enums import DataKind
from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array
from ._base import _check_choice, _check_settings, _read_path, _read_series
from ._markov import _stationary_distribution, _transition_counts
from .types import Source

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
    'langevin-entropy-production': {'drive', 'intensity', 'dt'},
    'count-entropy-production': {'n_states', 'dt'},
    'detailed-balance': {'n_states', 'significance'},
    'fluctuation-theorem': {'bins'},
    'uncertainty-relation': {'entropy', 'resamples', 'random_state'},
    'response': {'unperturbed', 'field'},
    'fluctuation-dissipation': {'unperturbed', 'field', 'correlation', 'intensity'},
}
PATH_KINDS = ('jump-entropy-production', 'count-entropy-production', 'detailed-balance')
TAGS = AnalysisTags(DataKind.PATH)
KIND_TAGS = {
    'langevin-entropy-production': AnalysisTags(DataKind.ENSEMBLE),
    'fluctuation-theorem': AnalysisTags(DataKind.ENSEMBLE),
    'uncertainty-relation': AnalysisTags(DataKind.ENSEMBLE),
    'response': AnalysisTags(DataKind.ENSEMBLE),
    'fluctuation-dissipation': AnalysisTags(DataKind.ENSEMBLE),
}


@dataclass(frozen=True)
class IrreversibilityResult:
    """A result of `irreversibility`, the irreversibility of a process.

    It is its entropy production, detailed balance, fluctuation relations, or response.

    Args:
        values:
            The rate matrix, the paths one row per path, or the samples the kind reads.

        duration:
            The duration of the jump paths, the setting or the duration of a trajectory.

        entropy_production:
            The entropy production per unit time in units of `k_B`, of the entropy production kinds.

        entropies:
            The entropy produced along every path, of `'jump-entropy-production'` and
            `'langevin-entropy-production'`.

        currents:
            The net number of transitions from every state to every other per step, of `'detailed-balance'`.

        statistic:
            The statistic `Σ (N_ij - N_ji)² / (N_ij + N_ji)` over pairs of states, of `'detailed-balance'`.

        p_value:
            The chi-squared p-value of the statistic under detailed balance, of `'detailed-balance'`.

        balanced:
            Whether the p-value exceeds the significance level, of `'detailed-balance'`.

        centres:
            The centres of the bins of entropy whose mirror bin is also populated, of `'fluctuation-theorem'`.

        log_ratios:
            `ln P(s) / P(-s)` at every centre, of `'fluctuation-theorem'`.

        slope:
            The slope of the log ratio against the entropy, of `'fluctuation-theorem'`.

        mean:
            The mean of the current over the paths, of `'uncertainty-relation'`.

        variance:
            The variance of the current over the paths, of `'uncertainty-relation'`.

        product:
            The squared relative fluctuation of the current times the entropy, of `'uncertainty-relation'`.

        error:
            The standard error of the product from bootstrap resamples of the paths, of `'uncertainty-relation'`.

        holds:
            Whether the product is at least 2 within two standard errors, of `'uncertainty-relation'`.

        response:
            The integrated response at every frame, of `'response'` and `'fluctuation-dissipation'`.

        fluctuation_dissipation_ratio:
            The fluctuation-dissipation ratio, of `'fluctuation-dissipation'`.
    """

    values: npt.NDArray[np.int64] | Array
    duration: float | None = None
    entropy_production: float | None = None
    entropies: Array | None = None
    currents: Array | None = None
    statistic: float | None = None
    p_value: float | None = None
    balanced: bool | None = None
    centres: Array | None = None
    log_ratios: Array | None = None
    slope: float | None = None
    mean: float | None = None
    variance: float | None = None
    product: float | None = None
    error: float | None = None
    holds: bool | None = None
    response: Array | None = None
    fluctuation_dissipation_ratio: float | None = None


def irreversibility(
    source: Source,
    *,
    kind: str = 'entropy-production',
    variables: Sequence[str] | None = None,
    rates: npt.ArrayLike | None = None,
    duration: float | None = None,
    drive: Callable[[Array], Array] | float = 1.0,
    intensity: float = 1.0,
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
) -> IrreversibilityResult:
    """Return the irreversibility of a process, of one kind.

    The kinds measure its entropy production, detailed balance, fluctuation relations, or response.

    The kinds read a rate matrix, jump paths, Langevin paths, samples, or ensembles:

    - `'entropy-production'`: the stationary entropy production rate of a jump process from its rate matrix,
      `Σ_{i<j} (p_i k_ij - p_j k_ji) ln(p_i k_ij / (p_j k_ji))`, 0 under detailed balance.
    - `'jump-entropy-production'`: the entropy every jump path produces in the medium, `Σ ln(k_ij / k_ji)` over its
      jumps with the `rates`, and the mean rate over the `duration` of the paths.
    - `'langevin-entropy-production'`: the entropy every overdamped Langevin path produces, the work of the `drive`
      times the `intensity`, `b ∫ F(x) ∘ dx`, and the mean rate.
    - `'count-entropy-production'`: the entropy production rate of a discrete path from its transition counts alone,
      `Σ_{i<j} (N_ij - N_ji) ln(N_ij / N_ji)` per unit time.
    - `'detailed-balance'`: the chi-squared test of detailed balance on the transition counts of discrete paths.
    - `'fluctuation-theorem'`: the log ratio of the probabilities of producing `s` and `-s` from samples of the
      entropy, and its slope against `s`, 1 when the theorem holds.
    - `'uncertainty-relation'`: the squared relative fluctuation `Var(J) / ⟨J⟩²` of samples of an integrated current
      times the mean `entropy` produced over the same duration, at least 2 (Barato and Seifert 2015).
    - `'response'`: the integrated response `chi(t) = (<x(t)>_h - <x(t)>_0) / h` of an ensemble with the field
      against the `unperturbed` ensemble.
    - `'fluctuation-dissipation'`: the response, and the slope of `chi / b` against `C(t_w, t_w) - C(t, t_w)` of the
      two-time `correlation` over the same lags, 1 in equilibrium and below 1 in an aging system.

    It appears in many fields. It measures how far a process is from equilibrium, in molecular motors, in heart beats,
    and in the prices of a market.

    Args:
        source:
            The rate matrix, the trajectory or an array of paths, one row per path, the samples, or the ensemble the
            kind measures.

        kind:
            The measure to compute, one of the kinds above.

        variables:
            The discrete state or the position of a trajectory to read, or `None` for the first one.

        rates:
            The rate of every jump, one row per origin.

        duration:
            The duration of every jump path, or `None` for the duration of a trajectory.

        drive:
            The drift along a Langevin path, a function of position or a constant.

        intensity:
            The factor `b` of the work of the drive in the entropy, and one over the factor of the response in the
            fluctuation-dissipation ratio, as in the acceptance `exp(-intensity * change)` of sampling.

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

    Returns:
        result:
            The values read and the measures of the kind.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, a setting the kind needs is missing, a
            setting is out of range, or the data do not suit the kind.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import irreversibility
        >>> ring = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
        >>> production = irreversibility(ring).entropy_production
        >>> bool(abs(production - np.log(2)) < 1e-12)
        True
        >>> jumps = irreversibility([0, 1, 0, 1], kind='jump-entropy-production', rates=[[0, 2], [1, 0]], duration=3.0)
        >>> jumps.entropies.round(3).tolist()
        [0.693]
        >>> irreversibility([0, 1, 2] * 1000, kind='detailed-balance').balanced
        False
    """
    _check_choice('irreversibility', 'kind', kind, KINDS)
    _check_settings(
        irreversibility,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        rates=rates,
        duration=duration,
        drive=drive,
        intensity=intensity,
        dt=dt,
        n_states=n_states,
        significance=significance,
        bins=bins,
        entropy=entropy,
        resamples=resamples,
        random_state=random_state,
        unperturbed=unperturbed,
        field=field,
        correlation=correlation,
    )
    _check(
        kind,
        rates=rates,
        unperturbed=unperturbed,
        correlation=correlation,
        entropy=entropy,
        intensity=intensity,
        dt=dt,
        field=field,
    )
    values, read_duration = _read_values(source, kind, variables, duration)
    if kind == 'entropy-production':
        return IrreversibilityResult(values, read_duration, entropy_production=_entropy_production(values))
    if kind == 'jump-entropy-production':
        return _jumps(values, read_duration, np.asarray(rates, dtype=float))
    if kind == 'count-entropy-production':
        production = _count_entropy_production(values, n_states, dt)
        return IrreversibilityResult(values, read_duration, entropy_production=production)
    if kind == 'detailed-balance':
        return _detailed_balance(values, read_duration, n_states, significance)
    return _of_ensemble(
        values,
        read_duration,
        kind,
        drive=drive,
        intensity=intensity,
        dt=dt,
        bins=bins,
        entropy=entropy,
        resamples=resamples,
        random_state=random_state,
        unperturbed=unperturbed,
        field=field,
        correlation=correlation,
    )


def _check(
    kind: str,
    *,
    rates: npt.ArrayLike | None,
    unperturbed: npt.ArrayLike | None,
    correlation: npt.ArrayLike | None,
    entropy: float | None,
    intensity: float,
    dt: float,
    field: float,
) -> None:
    """Raise an error for a missing setting of the kind, or for a value out of range.

    The values checked are an intensity, a step, or a field.
    """
    needed = {
        'rates': kind == 'jump-entropy-production',
        'unperturbed': kind in ('response', 'fluctuation-dissipation'),
        'correlation': kind == 'fluctuation-dissipation',
        'entropy': kind == 'uncertainty-relation',
    }
    given = {'rates': rates, 'unperturbed': unperturbed, 'correlation': correlation, 'entropy': entropy}
    missing = [name for name, needs in needed.items() if needs and given[name] is None]
    if missing:
        error_msg = f'irreversibility of kind {kind!r} needs {", ".join(missing)}. Give them.'
        raise AnalysisError(error_msg)
    if intensity <= 0 or dt <= 0 or field == 0:
        error_msg = (
            f'irreversibility has intensity={intensity!r}, dt={dt!r}, and field={field!r}. Use a positive intensity '
            'and step and a nonzero field.'
        )
        raise AnalysisError(error_msg)


def _of_ensemble(
    values: npt.NDArray[np.int64] | Array,
    duration: float | None,
    kind: str,
    *,
    drive: Callable[[Array], Array] | float,
    intensity: float,
    dt: float,
    bins: int,
    entropy: float | None,
    resamples: int,
    random_state: int | None,
    unperturbed: npt.ArrayLike | None,
    field: float,
    correlation: npt.ArrayLike | None,
) -> IrreversibilityResult:
    """Return the measure of a kind that reads samples or an ensemble."""
    if kind == 'langevin-entropy-production':
        midpoints = (values[:, 1:] + values[:, :-1]) / 2
        drifts = drive(midpoints) if callable(drive) else np.full_like(midpoints, drive)
        entropies = intensity * np.sum(drifts * np.diff(values, axis=1), axis=1)
        production = float(entropies.mean() / (dt * (values.shape[1] - 1)))
        return IrreversibilityResult(values, duration, entropy_production=production, entropies=entropies)
    if kind == 'fluctuation-theorem':
        return _fluctuation_theorem(values, duration, bins)
    if kind == 'uncertainty-relation':
        return _uncertainty_relation(values, duration, float(entropy or 0.0), resamples, random_state)
    response = _response(values, np.asarray(unperturbed, dtype=float), field)
    ratio = None
    if kind == 'fluctuation-dissipation':
        ratio = _fluctuation_dissipation_ratio(np.asarray(correlation, dtype=float), response, intensity)
    return IrreversibilityResult(values, duration, response=response, fluctuation_dissipation_ratio=ratio)


def _read_values(
    source: Source,
    kind: str,
    variables: Sequence[str] | None,
    duration: float | None,
) -> tuple[npt.NDArray[np.int64] | Array, float | None]:
    """Return the data the kind measures and the duration.

    The data are the rate matrix, the paths as rows, the samples, or the ensemble.
    """
    name = variables[0] if variables else None
    if isinstance(source, Trajectory) and kind in PATH_KINDS:
        path = _read_path(source, name)
        values = np.asarray(path.states, dtype=np.int64).reshape(1, -1)
        return values, float(path.times[-1] - path.times[0]) if duration is None else duration
    if isinstance(source, Trajectory):
        return _read_series(source, variables).values[:, 0].reshape(1, -1), duration
    dtype = np.int64 if kind in PATH_KINDS else float
    array = np.asarray(source, dtype=dtype)
    paths = kind in PATH_KINDS or kind == 'langevin-entropy-production'
    return (array.reshape(1, -1) if paths and array.ndim == 1 else array), duration


def _jumps(values: npt.NDArray[np.int64] | Array, duration: float | None, matrix: Array) -> IrreversibilityResult:
    """Return the entropy every jump path produces in the medium."""
    if duration is None:
        error_msg = 'irreversibility reads paths of an array without their duration. Give the duration.'
        raise AnalysisError(error_msg)
    origins, destinations = values[:, :-1], values[:, 1:]
    moved = origins != destinations
    steps = np.where(
        moved,
        np.log(matrix[origins, destinations] / np.where(moved, matrix[destinations, origins], 1)),
        0,
    )
    entropies = steps.sum(axis=1)
    return IrreversibilityResult(
        values,
        duration,
        entropy_production=float(entropies.mean() / duration),
        entropies=entropies,
    )


def _detailed_balance(
    values: npt.NDArray[np.int64] | Array,
    duration: float | None,
    n_states: int | None,
    significance: float,
) -> IrreversibilityResult:
    """Test detailed balance on the transition counts."""
    counts = _transition_counts(values, n_states).astype(float)
    steps = counts.sum()
    upper = np.triu_indices_from(counts, k=1)
    forward, backward = counts[upper], counts.T[upper]
    visited = (forward + backward) > 0
    statistic = float(np.sum((forward - backward)[visited] ** 2 / (forward + backward)[visited]))
    p_value = float(stats.chi2.sf(statistic, max(int(visited.sum()), 1)))
    return IrreversibilityResult(
        values,
        duration,
        currents=(counts - counts.T) / steps,
        statistic=statistic,
        p_value=p_value,
        balanced=p_value > significance,
    )


def _fluctuation_theorem(
    values: npt.NDArray[np.int64] | Array,
    duration: float | None,
    bins: int,
) -> IrreversibilityResult:
    """Return the log ratios of producing `s` and `-s` and their slope."""
    array = values.ravel()
    reach = float(np.abs(array).max())
    counts, edges = np.histogram(array, bins=2 * (bins // 2), range=(-reach, reach))
    centres = (edges[1:] + edges[:-1]) / 2
    mirrored = counts[::-1]
    usable = (centres > 0) & (counts > 0) & (mirrored > 0)
    if usable.sum() < MIN_PAIRS:
        error_msg = f'Fewer than {MIN_PAIRS} bins hold both s and -s. Give more samples or fewer bins.'
        raise AnalysisError(error_msg)
    log_ratios = np.log(counts[usable] / mirrored[usable])
    kept = centres[usable]
    return IrreversibilityResult(
        values,
        duration,
        centres=kept,
        log_ratios=log_ratios,
        slope=float(np.sum(kept * log_ratios) / np.sum(kept**2)),
    )


def _get_product(sample: Array, entropy: float) -> float:
    """Return the squared relative fluctuation of sampled currents times the entropy."""
    return float(sample.var(ddof=1) / sample.mean() ** 2 * entropy)


def _uncertainty_relation(
    values: npt.NDArray[np.int64] | Array,
    duration: float | None,
    entropy: float,
    resamples: int,
    random_state: int | None,
) -> IrreversibilityResult:
    """Return the precision of the current against the entropy production."""
    currents = np.asarray(values, dtype=np.float64).ravel()
    if currents.size < MIN_PATHS or entropy < 0:
        error_msg = (
            f'The uncertainty relation needs at least {MIN_PATHS} paths and a nonnegative entropy production, but got '
            f'{currents.size} paths and {entropy}.'
        )
        raise AnalysisError(error_msg)
    mean, variance = float(currents.mean()), float(currents.var(ddof=1))
    if mean == 0:
        error_msg = 'The mean current is zero, so its relative fluctuation is undefined. Measure a current that flows.'
        raise AnalysisError(error_msg)
    rng = np.random.default_rng(random_state)
    boots = [_get_product(currents[rng.integers(currents.size, size=currents.size)], entropy) for _ in range(resamples)]
    product, error = _get_product(currents, entropy), float(np.std(boots, ddof=1))
    return IrreversibilityResult(
        values,
        duration,
        mean=mean,
        variance=variance,
        product=product,
        error=error,
        holds=product + 2 * error >= TUR_BOUND,
    )


def _entropy_production(rates: npt.ArrayLike) -> float:
    """Return the entropy production of a rate matrix.

    It is `Σ_{i<j} (p_i k_ij - p_j k_ji) ln(p_i k_ij / (p_j k_ji))` with `p` the stationary law of the rates.
    """
    matrix = np.asarray(rates, dtype=float)
    matrix = matrix - np.diag(np.diag(matrix))
    if np.any((matrix > 0) != (matrix.T > 0)):
        error_msg = 'A jump has no reverse jump, so the entropy production is infinite. Give every jump its reverse.'
        raise AnalysisError(error_msg)
    flux = _stationary_distribution(matrix, rates=True)[:, None] * matrix
    positive = flux > 0
    return float(0.5 * np.sum((flux - flux.T)[positive] * np.log(flux[positive] / flux.T[positive])))


def _count_entropy_production(states: npt.ArrayLike, n_states: int | None, dt: float) -> float:
    """Return `Σ_{i<j} (N_ij - N_ji) ln(N_ij / N_ji)` per unit time.

    The sum is over the pairs visited both ways.
    """
    counts = _transition_counts(states, n_states).astype(float)
    upper = np.triu(np.ones_like(counts, dtype=bool), k=1)
    both = upper & (counts > 0) & (counts.T > 0)
    forward, backward = counts[both], counts.T[both]
    return float(np.sum((forward - backward) * np.log(forward / backward)) / (counts.sum() * dt))


def _response(perturbed: npt.ArrayLike, unperturbed: npt.ArrayLike, field: float) -> Array:
    """Return the integrated response `chi(t) = (<x(t)>_h - <x(t)>_0) / h`.

    It compares ensembles with and without a field.
    """
    first, second = np.asarray(perturbed, dtype=float), np.asarray(unperturbed, dtype=float)
    return (first.mean(axis=0) - second.mean(axis=0)) / field


def _fluctuation_dissipation_ratio(correlation: npt.ArrayLike, susceptibility: Array, intensity: float) -> float:
    """Return the slope of `chi / b` against `C(t_w, t_w) - C(t, t_w)`.

    Here `b` is the intensity.
    """
    values = np.asarray(correlation, dtype=float)
    decay = values[0] - values
    scaled = susceptibility / intensity
    return float(np.sum(decay * scaled) / np.sum(decay**2))
