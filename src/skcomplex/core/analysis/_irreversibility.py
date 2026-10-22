"""Measure irreversibility: entropy production of jump and Langevin dynamics and the fluctuation theorem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..base import AnalysisError
from ._kinds import DataKind, analysis
from ._transitions import stationary_distribution, transition_counts

type Array = npt.NDArray[np.float64]
MIN_PAIRS = 3
MIN_PATHS = 2
TUR_BOUND = 2.0


@analysis(DataKind.PATH)
def entropy_production(rates: npt.ArrayLike) -> float:
    """Return the stationary entropy production rate of a jump process from its rates.

    It is `Σ_{i<j} (p_i k_ij - p_j k_ji) ln(p_i k_ij / (p_j k_ji))` with `p` the stationary law, 0 under detailed
    balance.

    Args:
        rates:
            The rate of every jump, one row per origin, whose diagonal is ignored.

    Returns:
        rate:
            The entropy production per unit time in units of `k_B`.

    Raises:
        AnalysisError:
            A jump has no reverse jump, so the production is infinite.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import entropy_production
        >>> ring = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
        >>> bool(abs(entropy_production(ring) - np.log(2)) < 1e-12)
        True
    """
    matrix = np.asarray(rates, dtype=float)
    matrix = matrix - np.diag(np.diag(matrix))
    if np.any((matrix > 0) != (matrix.T > 0)):
        error_msg = 'A jump has no reverse jump, so the entropy production is infinite.'
        raise AnalysisError(error_msg)
    flux = stationary_distribution(matrix, rates=True)[:, None] * matrix
    positive = flux > 0
    return float(0.5 * np.sum((flux - flux.T)[positive] * np.log(flux[positive] / flux.T[positive])))


@dataclass(frozen=True)
class EntropyProduction:
    """Entropy produced in the medium along every path, and its mean rate.

    Args:
        entropies:
            The entropy produced along every path.

        rate:
            The mean entropy per path divided by the duration of the paths.
    """

    entropies: Array
    rate: float


@analysis(DataKind.PATH)
def jump_entropy_production(paths: npt.ArrayLike, rates: npt.ArrayLike, duration: float) -> EntropyProduction:
    """Return the entropy every jump path produces in the medium, `Σ ln(k_ij / k_ji)` over its jumps.

    Args:
        paths:
            The integer states after every jump of one path, or one row per path of equal numbers of jumps.

        rates:
            The rate of every jump, one row per origin.

        duration:
            The duration of every path.

    Returns:
        production:
            The entropy of every path and the mean rate.

    Examples:
        >>> from skcomplex.core.analysis import jump_entropy_production
        >>> jump_entropy_production([0, 1, 0, 1], [[0, 2], [1, 0]], 3.0).entropies.round(3).tolist()
        [0.693]
    """
    states = np.asarray(paths, dtype=np.int64)
    states = states.reshape(1, -1) if states.ndim == 1 else states
    matrix = np.asarray(rates, dtype=float)
    origins, destinations = states[:, :-1], states[:, 1:]
    moved = origins != destinations
    steps = np.where(
        moved,
        np.log(matrix[origins, destinations] / np.where(moved, matrix[destinations, origins], 1)),
        0,
    )
    entropies = steps.sum(axis=1)
    return EntropyProduction(entropies, float(entropies.mean() / duration))


@analysis(DataKind.ENSEMBLE)
def langevin_entropy_production(
    positions: npt.ArrayLike,
    force: Callable[[Array], Array] | float,
    temperature: float,
    dt: float,
) -> EntropyProduction:
    """Return the heat every overdamped Langevin path releases over the temperature, `∫ F(x) ∘ dx / T`.

    Args:
        positions:
            The unwrapped positions of one path, or one row per path, on frames `dt` apart.

        force:
            The force as a function of position, or a constant force.

        temperature:
            The temperature of the bath.

        dt:
            The time between frames.

    Returns:
        production:
            The entropy of every path and the mean rate.

    Examples:
        >>> from skcomplex.core.analysis import langevin_entropy_production
        >>> langevin_entropy_production([[0.0, 0.5, 1.0]], 1.0, 1.0, 0.5).rate
        1.0
    """
    array = np.asarray(positions, dtype=float)
    array = array.reshape(1, -1) if array.ndim == 1 else array
    midpoints = (array[:, 1:] + array[:, :-1]) / 2
    forces = force(midpoints) if callable(force) else np.full_like(midpoints, force)
    entropies = np.sum(forces * np.diff(array, axis=1), axis=1) / temperature
    return EntropyProduction(entropies, float(entropies.mean() / (dt * (array.shape[1] - 1))))


@analysis(DataKind.PATH)
def count_entropy_production(states: npt.ArrayLike, n_states: int | None = None, dt: float = 1.0) -> float:
    """Return the entropy production rate of a discrete path estimated from its transition counts alone.

    It is `Σ_{i<j} (N_ij - N_ji) ln(N_ij / N_ji)` over the pairs visited both ways, per unit time.

    Args:
        states:
            The integer states of one path, or one row per path, recorded every `dt`.

        n_states:
            The number of states, or `None` for the largest state plus one.

        dt:
            The time between records.

    Returns:
        rate:
            The estimated entropy production per unit time.

    Examples:
        >>> from skcomplex.core.analysis import count_entropy_production
        >>> count_entropy_production([0, 1, 0, 1, 0])
        0.0
    """
    counts = transition_counts(states, n_states).astype(float)
    upper = np.triu(np.ones_like(counts, dtype=bool), k=1)
    both = upper & (counts > 0) & (counts.T > 0)
    forward, backward = counts[both], counts.T[both]
    return float(np.sum((forward - backward) * np.log(forward / backward)) / (counts.sum() * dt))


@dataclass(frozen=True)
class FluctuationTheorem:
    """Check of the fluctuation theorem `ln P(s) / P(-s) = s` on samples of the entropy produced.

    Args:
        entropies:
            The centres of the bins of entropy whose mirror bin is also populated.

        log_ratios:
            `ln P(s) / P(-s)` at every centre.

        slope:
            The slope of the log ratio against the entropy, 1 when the theorem holds.
    """

    entropies: Array
    log_ratios: Array
    slope: float


@analysis(DataKind.ENSEMBLE)
def fluctuation_theorem(entropies: npt.ArrayLike, bins: int = 40) -> FluctuationTheorem:
    """Return the log ratio of the probabilities of producing `s` and `-s` and its slope against `s`.

    Args:
        entropies:
            The entropy produced along every path.

        bins:
            The number of bins symmetric about 0.

    Returns:
        theorem:
            The bin centres, the log ratios, and their slope.

    Raises:
        AnalysisError:
            Fewer than 3 bins have both signs populated.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import fluctuation_theorem
        >>> samples = np.random.default_rng(0).normal(1.0, np.sqrt(2.0), 200_000)
        >>> round(fluctuation_theorem(samples).slope, 1)
        1.0
    """
    array = np.asarray(entropies, dtype=float)
    reach = float(np.abs(array).max())
    counts, edges = np.histogram(array, bins=2 * (bins // 2), range=(-reach, reach))
    centres = (edges[1:] + edges[:-1]) / 2
    mirrored = counts[::-1]
    usable = (centres > 0) & (counts > 0) & (mirrored > 0)
    if usable.sum() < MIN_PAIRS:
        error_msg = f'Fewer than {MIN_PAIRS} bins hold both s and -s. Give more samples or fewer bins.'
        raise AnalysisError(error_msg)
    ratios = np.log(counts[usable] / mirrored[usable])
    slope = float(np.sum(centres[usable] * ratios) / np.sum(centres[usable] ** 2))
    return FluctuationTheorem(centres[usable], ratios, slope)


@dataclass(frozen=True)
class UncertaintyRelation:
    """Check of the thermodynamic uncertainty relation `Var(J) / ⟨J⟩² · Σ ≥ 2` on samples of a current.

    Args:
        mean:
            The mean of the current over the paths.

        variance:
            The variance of the current over the paths.

        entropy_production:
            The mean entropy produced along a path, in units of the Boltzmann constant.

        product:
            The squared relative fluctuation of the current times the entropy production.

        error:
            The standard error of the product, from bootstrap resamples of the paths.

        holds:
            Whether the product is at least 2 within two standard errors.
    """

    mean: float
    variance: float
    entropy_production: float
    product: float
    error: float
    holds: bool


@analysis(DataKind.ENSEMBLE)
def uncertainty_relation(
    currents: npt.ArrayLike,
    entropy_production: float,
    *,
    resamples: int = 200,
    random_state: int | None = None,
) -> UncertaintyRelation:
    """Return the precision of a current against the entropy production, the thermodynamic uncertainty relation.

    For a steady-state current `J` integrated over a duration, the squared relative fluctuation `Var(J) / ⟨J⟩²` is at
    least `2 / Σ`, where `Σ` is the mean entropy produced over the same duration (Barato and Seifert 2015). The product
    of the two is therefore at least 2, with equality only near equilibrium.

    Args:
        currents:
            The integrated current of every path, all over the same duration.

        entropy_production:
            The mean entropy produced along a path over that duration.

        resamples:
            The number of bootstrap resamples of the standard error.

        random_state:
            The seed of the bootstrap.

    Returns:
        relation:
            The mean and variance of the current, the product, its error, and whether the relation holds.

    Raises:
        AnalysisError:
            Fewer than 2 paths, a mean current of zero, or a negative entropy production.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import uncertainty_relation
        >>> currents = np.random.default_rng(0).normal(10.0, np.sqrt(30.0), 5000)
        >>> relation = uncertainty_relation(currents, entropy_production=10.0, random_state=0)
        >>> round(relation.product, 1), relation.holds
        (3.0, True)
    """
    values = np.asarray(currents, dtype=float).ravel()
    if values.size < MIN_PATHS or entropy_production < 0:
        error_msg = (
            f'The uncertainty relation needs at least {MIN_PATHS} paths and a nonnegative entropy production, but got '
            f'{values.size} paths and {entropy_production}.'
        )
        raise AnalysisError(error_msg)
    mean, variance = float(values.mean()), float(values.var(ddof=1))
    if mean == 0:
        error_msg = 'The mean current is zero, so its relative fluctuation is undefined. Measure a current that flows.'
        raise AnalysisError(error_msg)

    def product_of(sample: Array) -> float:
        return float(sample.var(ddof=1) / sample.mean() ** 2 * entropy_production)

    rng = np.random.default_rng(random_state)
    boots = [product_of(values[rng.integers(values.size, size=values.size)]) for _ in range(resamples)]
    product, error = product_of(values), float(np.std(boots, ddof=1))
    holds = product + 2 * error >= TUR_BOUND
    return UncertaintyRelation(mean, variance, float(entropy_production), product, error, holds)
