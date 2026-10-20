"""Analyse transition and rate matrices: estimates from paths, stationary laws, spectral gaps, and detailed balance."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import linalg, stats

from ..base import AnalysisError
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]
TOLERANCE = 1e-10


def _paths(states: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Return integer states as one row per path."""
    array = np.asarray(states, dtype=np.int64)
    return array.reshape(1, -1) if array.ndim == 1 else array


@analysis(DataKind.PATH)
def transition_counts(states: npt.ArrayLike, n_states: int | None = None, lag: int = 1) -> npt.NDArray[np.int64]:
    """Return the number of transitions from every state to every state after `lag` steps along discrete paths.

    Args:
        states:
            The integer states of one path, or one row per path.

        n_states:
            The number of states, or `None` for the largest state plus one.

        lag:
            The number of steps of a transition.

    Returns:
        counts:
            The counts, one row per origin and one column per destination.

    Examples:
        >>> from skcomplex.core.analysis import transition_counts
        >>> transition_counts([0, 1, 1, 0]).tolist()
        [[0, 1], [1, 1]]
    """
    paths = _paths(states)
    size = int(paths.max()) + 1 if n_states is None else n_states
    origins, destinations = paths[:, :-lag].ravel(), paths[:, lag:].ravel()
    return np.bincount(origins * size + destinations, minlength=size * size).reshape(size, size)


@analysis(DataKind.PATH)
def transition_matrix(states: npt.ArrayLike, n_states: int | None = None, lag: int = 1) -> Array:
    """Return the maximum-likelihood transition matrix of discrete paths, the counts normalized by row.

    Args:
        states:
            The integer states of one path, or one row per path.

        n_states:
            The number of states, or `None` for the largest state plus one.

        lag:
            The number of steps of a transition.

    Returns:
        matrix:
            The probability of every destination from every origin, the identity row for unvisited states.

    Examples:
        >>> from skcomplex.core.analysis import transition_matrix
        >>> transition_matrix([0, 1, 0, 1, 1]).tolist()
        [[0.0, 1.0], [0.5, 0.5]]
    """
    counts = transition_counts(states, n_states, lag).astype(float)
    totals = counts.sum(axis=1)
    unvisited = totals == 0
    counts[unvisited, unvisited] = 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def _generator(matrix: npt.ArrayLike, *, rates: bool) -> Array:
    """Return a rate matrix with diagonal minus the escape rates, or a transition matrix minus the identity."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:  # noqa: PLR2004
        error_msg = f'A transition or rate matrix is square, not of shape {array.shape}.'
        raise AnalysisError(error_msg)
    if not rates:
        return array - np.eye(array.shape[0])
    off = array - np.diag(np.diag(array))
    return off - np.diag(off.sum(axis=1))


@analysis(DataKind.PATH)
def stationary_distribution(matrix: npt.ArrayLike, *, rates: bool = False) -> Array:
    """Return the stationary distribution `pi` of a transition matrix, `pi P = pi`, or of a rate matrix, `pi Q = 0`.

    Args:
        matrix:
            The transition matrix with rows summing to 1, or the rate matrix whose off-diagonal entries are rates.

        rates:
            Whether the matrix holds rates of a continuous-time chain, whose diagonal is ignored.

    Returns:
        distribution:
            The stationary probability of every state.

    Raises:
        AnalysisError:
            The matrix is not square.

    Examples:
        >>> from skcomplex.core.analysis import stationary_distribution
        >>> stationary_distribution([[0.5, 0.5], [0.25, 0.75]]).round(3).tolist()
        [0.333, 0.667]
    """
    generator = _generator(matrix, rates=rates)
    null = linalg.null_space(generator.T, rcond=TOLERANCE)
    vector = np.abs(null[:, 0]) if null.size else np.abs(linalg.eig(generator.T)[1][:, 0].real)
    return vector / vector.sum()


def _eigenvalues(matrix: npt.ArrayLike, *, rates: bool) -> npt.NDArray[np.complex128]:
    """Return the eigenvalues of the generator, the stationary 0 first, then by falling real part."""
    values = linalg.eigvals(_generator(matrix, rates=rates))
    return values[np.argsort(-values.real)]


@analysis(DataKind.PATH)
def spectral_gap(matrix: npt.ArrayLike, *, rates: bool = False) -> float:
    """Return the spectral gap, `1 - |lambda_2|` of a transition matrix or `-Re lambda_2` of a rate matrix.

    Args:
        matrix:
            The transition matrix, or the rate matrix.

        rates:
            Whether the matrix holds rates of a continuous-time chain.

    Returns:
        gap:
            The gap, the rate at which the slowest mode relaxes.

    Raises:
        AnalysisError:
            The matrix is not square.

    Examples:
        >>> from skcomplex.core.analysis import spectral_gap
        >>> round(spectral_gap([[0.5, 0.5], [0.5, 0.5]]), 12)
        1.0
    """
    values = _eigenvalues(matrix, rates=rates)
    if rates:
        return float(-values[1].real)
    return float(1 - np.abs(values[1:] + 1).max())


@analysis(DataKind.PATH)
def relaxation_times(matrix: npt.ArrayLike, *, rates: bool = False, dt: float = 1.0) -> Array:
    """Return the relaxation times of every mode but the stationary one, slowest first.

    Args:
        matrix:
            The transition matrix, or the rate matrix.

        rates:
            Whether the matrix holds rates of a continuous-time chain.

        dt:
            The time of one step of a transition matrix.

    Returns:
        times:
            `-dt / ln |lambda|` for a transition matrix, or `-1 / Re lambda` for a rate matrix.

    Raises:
        AnalysisError:
            The matrix is not square.

    Examples:
        >>> from skcomplex.core.analysis import relaxation_times
        >>> relaxation_times([[-1.0, 1.0], [1.0, -1.0]], rates=True).tolist()
        [0.5]
    """
    values = _eigenvalues(matrix, rates=rates)[1:]
    if rates:
        return -1 / values.real
    moduli = np.clip(np.abs(values + 1), 1e-300, 1 - 1e-15)
    return np.sort(-dt / np.log(moduli))[::-1]


@dataclass(frozen=True)
class DetailedBalance:
    """Test of detailed balance on the transition counts of discrete paths.

    Args:
        currents:
            The net number of transitions from every state to every other per step, antisymmetric.

        statistic:
            The statistic `Σ (N_ij - N_ji)² / (N_ij + N_ji)` over pairs of states.

        p_value:
            The chi-squared p-value of the statistic under detailed balance.

        balanced:
            Whether the p-value exceeds the significance level.
    """

    currents: Array
    statistic: float
    p_value: float
    balanced: bool


@analysis(DataKind.PATH)
def detailed_balance(
    states: npt.ArrayLike,
    n_states: int | None = None,
    *,
    significance: float = 0.05,
) -> DetailedBalance:
    """Test detailed balance on discrete paths, which holds when every transition is as frequent as its reverse.

    Args:
        states:
            The integer states of one path, or one row per path.

        n_states:
            The number of states, or `None` for the largest state plus one.

        significance:
            The level below which the p-value rejects detailed balance.

    Returns:
        balance:
            The net currents, the statistic, its p-value, and the verdict.

    Examples:
        >>> from skcomplex.core.analysis import detailed_balance
        >>> detailed_balance([0, 1, 2] * 1000).balanced
        False
    """
    counts = transition_counts(states, n_states).astype(float)
    steps = counts.sum()
    upper = np.triu_indices_from(counts, k=1)
    forward, backward = counts[upper], counts.T[upper]
    visited = (forward + backward) > 0
    statistic = float(np.sum((forward - backward)[visited] ** 2 / (forward + backward)[visited]))
    p_value = float(stats.chi2.sf(statistic, max(int(visited.sum()), 1)))
    return DetailedBalance((counts - counts.T) / steps, statistic, p_value, p_value > significance)
