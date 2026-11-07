"""Analyse Markov chains: transition matrices of paths, stationary laws, spectral gaps, and Markov state models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydtmc import MarkovChain
from scipy import linalg
from scipy.sparse import csgraph

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array
from ._base import _check_choice, _check_settings, _read_path
from .types import Source

MATRIX_DIMENSIONS = 2
MIN_CONNECTED_STATES = 2
MAX_ITERATIONS = 1_000_000

TOLERANCE = 1e-10
FIT_TOLERANCE = 1e-8
KINDS = ('transition-matrix', 'stationary', 'spectral-gap', 'state-system', 'timescales')
SETTINGS = {
    'transition-matrix': {'n_states', 'lag'},
    'stationary': {'matrix', 'n_states', 'lag'},
    'spectral-gap': {'matrix', 'n_states', 'lag', 'dt'},
    'state-system': {'lag', 'n_metastable', 'reversible', 'dt'},
    'timescales': {'lags', 'n_timescales', 'reversible', 'dt'},
}
"""Settings every kind reads, besides `kind` and `variables`."""

MATRICES = ('transition', 'rate')


@dataclass(frozen=True)
class MarkovResult:
    """Markov chain of the paths of a source, or of a matrix, with the quantities of one kind.

    Args:
        path:
            The integer states of the paths, one row per path, read without a matrix.

        matrix:
            The matrix read with `matrix`.

        counts:
            The transition counts, one row per origin, of `'transition-matrix'`.

        transition_matrix:
            The probability of every destination after one lag from every origin, of `'transition-matrix'` and
            `'state-system'`.

        stationary:
            The stationary law, of `'stationary'` and `'state-system'`.

        spectral_gap:
            The spectral gap, of `'spectral-gap'`.

        relaxation_times:
            `-dt / ln |lambda|`, or `-1 / Re lambda` for rates, of every mode slowest first, of `'spectral-gap'`.

        states:
            The states of the largest connected set, which index the rows of the matrix, of `'state-system'`.

        timescales:
            The implied timescales `-lag dt / ln |lambda_i|` slowest first, of `'state-system'`, and one row per lag
            of `'timescales'`.

        metastable_sets:
            The states of every metastable set, empty without a number of sets, of `'state-system'`.
    """

    path: npt.NDArray[np.int64] | None = None
    matrix: Array | None = None
    counts: npt.NDArray[np.int64] | None = None
    transition_matrix: Array | None = None
    stationary: Array | None = None
    spectral_gap: float | None = None
    relaxation_times: Array | None = None
    states: npt.NDArray[np.int64] | None = None
    timescales: Array | None = None
    metastable_sets: list[npt.NDArray[np.int64]] | None = None


def markov(
    source: Source,
    *,
    kind: str = 'transition-matrix',
    variables: Sequence[str] | None = None,
    matrix: str | None = None,
    n_states: int | None = None,
    lag: int = 1,
    lags: Sequence[int] = (1,),
    n_timescales: int = 1,
    n_metastable: int | None = None,
    reversible: bool = True,
    dt: float = 1.0,
) -> MarkovResult:
    """Return the Markov chain of the paths of a source, or of a matrix, with the quantities of one kind.

    The kinds read the integer states of one path, or one row per path, or a matrix with `matrix`:

    - `'transition-matrix'`: the transitions after `lag` steps counted along the paths and normalized by row.
    - `'stationary'`: the stationary law `pi` of the matrix, `pi P = pi` or `pi Q = 0` for rates.
    - `'spectral-gap'`: the gap `1 - |lambda_2|`, or `-Re lambda_2` for rates, and the relaxation time of every mode.
    - `'state-system'`: the maximum-likelihood Markov state model of the largest connected set, with its metastable
      sets found by PCCA+.
    - `'timescales'`: the slowest implied timescales of Markov state models at every lag of `lags`, which do not
      depend on the lag for a Markovian discretization.

    Args:
        source:
            The trajectory of a jump process or of a map with discrete states, an array of integer states with one
            row per path, or a transition or rate matrix with `matrix`.

        kind:
            The quantity to compute, one of the kinds above.

        variables:
            The discrete state of a trajectory to read, or `None` for the first one.

        matrix:
            `None` to read paths, `'transition'` to read the source as a transition matrix, or `'rate'` to read it as
            the rate matrix of a continuous-time chain, whose diagonal is ignored.

        n_states:
            The number of states of the paths, or `None` for the largest state plus one.

        lag:
            The lag time in steps.

        lags:
            The lag times in steps of the implied timescales.

        n_timescales:
            The number of implied timescales.

        n_metastable:
            The number of metastable sets PCCA+ groups the states into, or `None` for none.

        reversible:
            Whether the state systems obey detailed balance, which PCCA+ needs.

        dt:
            The time of one step.

    Returns:
        result:
            The paths or the matrix read, and the quantities of the kind.

    Raises:
        AnalysisError:
            A choice is unknown, a setting the kind does not read is changed, or a setting is out of range.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import markov
        >>> markov([0, 1, 0, 1, 1]).transition_matrix.tolist()
        [[0.0, 1.0], [0.5, 0.5]]
        >>> markov([[0.5, 0.5], [0.25, 0.75]], kind='stationary', matrix='transition').stationary.round(3).tolist()
        [0.333, 0.667]
        >>> path = np.random.default_rng(0).integers(0, 2, 10_000)
        >>> markov(path, kind='state-system').transition_matrix.round(1).tolist()
        [[0.5, 0.5], [0.5, 0.5]]
    """
    _check_choice('markov', 'kind', kind, KINDS)
    if matrix is not None:
        _check_choice('markov', 'matrix', matrix, MATRICES)
    _check_settings(
        markov,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        matrix=matrix,
        n_states=n_states,
        lag=lag,
        lags=lags,
        n_timescales=n_timescales,
        n_metastable=n_metastable,
        reversible=reversible,
        dt=dt,
    )
    _check(lag, lags, n_timescales, n_metastable, dt, reversible=reversible)
    if matrix is not None:
        return _of_matrix(np.asarray(source, dtype=float), kind, dt, rates=matrix == 'rate')
    if isinstance(source, Trajectory):
        name = variables[0] if variables else None
        path = _paths(_read_path(source, name).states)
    else:
        path = _paths(source)
    if kind == 'state-system':
        return _state_system(path, lag, n_metastable, dt, reversible=reversible)
    if kind == 'transition-matrix':
        return MarkovResult(
            path=path,
            counts=_transition_counts(path, n_states, lag),
            transition_matrix=_transition_matrix(path, n_states, lag),
        )
    if kind == 'stationary':
        return MarkovResult(path=path, stationary=_stationary_distribution(_transition_matrix(path, n_states, lag)))
    if kind == 'spectral-gap':
        estimated = _transition_matrix(path, n_states, lag)
        return MarkovResult(
            path=path,
            spectral_gap=_spectral_gap(estimated),
            relaxation_times=_relaxation_times(estimated, dt=dt),
        )
    timescales = np.asarray(
        [_implied_timescales(_fit(path, item, reversible=reversible)[1], item)[:n_timescales] * dt for item in lags],
        dtype=float,
    )
    return MarkovResult(path=path, timescales=timescales)


def _check(
    lag: int,
    lags: Sequence[int],
    n_timescales: int,
    n_metastable: int | None,
    dt: float,
    *,
    reversible: bool,
) -> None:
    """Raise an error for lags, timescales, or a step out of range, or for metastable sets of an irreversible fit."""
    if lag < 1 or any(item < 1 for item in lags) or n_timescales < 1:
        error_msg = f'markov has lag={lag!r}, lags={lags!r}, and n_timescales={n_timescales!r}. Use 1 or more.'
        raise AnalysisError(error_msg)
    if dt <= 0:
        error_msg = f'markov has dt={dt!r}. Use a positive time step.'
        raise AnalysisError(error_msg)
    if n_metastable is not None and not reversible:
        error_msg = 'markov finds metastable sets of reversible systems only. Use reversible=True.'
        raise AnalysisError(error_msg)


def _of_matrix(matrix: Array, kind: str, dt: float, *, rates: bool) -> MarkovResult:
    """Return the stationary law, or the spectral gap and the relaxation times, of a transition or rate matrix."""
    if kind == 'stationary':
        return MarkovResult(matrix=matrix, stationary=_stationary_distribution(matrix, rates=rates))
    return MarkovResult(
        matrix=matrix,
        spectral_gap=_spectral_gap(matrix, rates=rates),
        relaxation_times=_relaxation_times(matrix, rates=rates, dt=dt),
    )


def _state_system(
    path: npt.NDArray[np.int64],
    lag: int,
    n_metastable: int | None,
    dt: float,
    *,
    reversible: bool,
) -> MarkovResult:
    """Return the Markov state model of the paths and its metastable sets."""
    states, transition_matrix = _fit(path, lag, reversible=reversible)
    metastable_sets = []
    if n_metastable is not None:
        metastable_sets = _metastable_sets(transition_matrix, states, n_metastable)
    return MarkovResult(
        path=path,
        transition_matrix=transition_matrix,
        stationary=_stationary_distribution(transition_matrix),
        states=states,
        timescales=_implied_timescales(transition_matrix, lag) * dt,
        metastable_sets=metastable_sets,
    )


def _paths(states: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Return integer states as one row per path."""
    array = np.asarray(states, dtype=np.int64)
    return array.reshape(1, -1) if array.ndim == 1 else array


def _generator(matrix: npt.ArrayLike, *, rates: bool) -> Array:
    """Return a rate matrix with diagonal minus the escape rates, or a transition matrix minus the identity."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != MATRIX_DIMENSIONS or array.shape[0] != array.shape[1]:
        error_msg = f'A transition or rate matrix is square, not of shape {array.shape}.'
        raise AnalysisError(error_msg)
    if not rates:
        return array - np.eye(array.shape[0])
    off = array - np.diag(np.diag(array))
    return off - np.diag(off.sum(axis=1))


def _eigenvalues(matrix: npt.ArrayLike, *, rates: bool) -> npt.NDArray[np.complex128]:
    """Return the eigenvalues of the generator, the stationary 0 first, then by falling real part."""
    values = linalg.eigvals(_generator(matrix, rates=rates))
    return values[np.argsort(-values.real)]


def _transition_counts(states: npt.ArrayLike, n_states: int | None = None, lag: int = 1) -> npt.NDArray[np.int64]:
    """Return the number of transitions from every state to every state after `lag` steps along discrete paths."""
    paths = _paths(states)
    size = int(paths.max()) + 1 if n_states is None else n_states
    origins, destinations = paths[:, :-lag].ravel(), paths[:, lag:].ravel()
    return np.bincount(origins * size + destinations, minlength=size * size).reshape(size, size)


def _transition_matrix(states: npt.ArrayLike, n_states: int | None = None, lag: int = 1) -> Array:
    """Return the counts normalized by row, the identity row for unvisited states."""
    counts = _transition_counts(states, n_states, lag).astype(float)
    totals = counts.sum(axis=1)
    unvisited = totals == 0
    counts[unvisited, unvisited] = 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def _stationary_distribution(matrix: npt.ArrayLike, *, rates: bool = False) -> Array:
    """Return the stationary law `pi` of a transition matrix, `pi P = pi`, or of a rate matrix, `pi Q = 0`."""
    generator = _generator(matrix, rates=rates)
    null = linalg.null_space(generator.T, rcond=TOLERANCE)
    vector = np.abs(null[:, 0]) if null.size else np.abs(linalg.eig(generator.T)[1][:, 0].real)
    return vector / vector.sum()


def _spectral_gap(matrix: npt.ArrayLike, *, rates: bool = False) -> float:
    """Return `1 - |lambda_2|` of a transition matrix or `-Re lambda_2` of a rate matrix."""
    values = _eigenvalues(matrix, rates=rates)
    if rates:
        return float(-values[1].real)
    return float(1 - np.abs(values[1:] + 1).max())


def _relaxation_times(matrix: npt.ArrayLike, *, rates: bool = False, dt: float = 1.0) -> Array:
    """Return the relaxation times of every mode but the stationary one, slowest first."""
    values = _eigenvalues(matrix, rates=rates)[1:]
    if rates:
        return -1 / values.real
    moduli = np.clip(np.abs(values + 1), 1e-300, 1 - 1e-15)
    return np.sort(-dt / np.log(moduli))[::-1]


def _reversible_matrix(counts: Array) -> Array:
    """Return the reversible maximum-likelihood transition matrix of counts, by a fixed-point iteration on flows."""
    symmetric = counts + counts.T
    totals = counts.sum(axis=1)
    flows = symmetric / symmetric.sum()
    for _ in range(MAX_ITERATIONS):
        ratios = totals / flows.sum(axis=1)
        updated = symmetric / (ratios[:, np.newaxis] + ratios)
        updated /= updated.sum()
        change = np.abs(updated.sum(axis=1) - flows.sum(axis=1)).max()
        flows = updated
        if change < FIT_TOLERANCE:
            return flows / flows.sum(axis=1, keepdims=True)
    error_msg = f'The reversible fit did not converge in {MAX_ITERATIONS} iterations. Use more data.'
    raise AnalysisError(error_msg)


def _fit(states: npt.ArrayLike, lag: int, *, reversible: bool) -> tuple[npt.NDArray[np.int64], Array]:
    """Return the largest connected set of states and its maximum-likelihood transition matrix at a lag."""
    counts = _transition_counts(states, lag=lag).astype(float)
    labels = csgraph.connected_components(counts, connection='strong')[1]
    connected = np.flatnonzero(labels == np.bincount(labels).argmax())
    if connected.size < MIN_CONNECTED_STATES:
        error_msg = f'A Markov state model needs {MIN_CONNECTED_STATES} connected states, and the paths have fewer.'
        raise AnalysisError(error_msg)
    kept = counts[np.ix_(connected, connected)]
    if reversible:
        return connected, _reversible_matrix(kept)
    return connected, kept / kept.sum(axis=1, keepdims=True)


def _implied_timescales(matrix: Array, lag: int) -> Array:
    """Return the implied timescales in steps of every mode but the stationary one, slowest first."""
    return np.asarray(MarkovChain(matrix).implied_timescales, dtype=float)[1:] * lag


def _metastable_sets(matrix: Array, states: npt.NDArray[np.int64], n_sets: int) -> list[npt.NDArray[np.int64]]:
    """Return the states of every metastable set of a reversible transition matrix."""
    if not 1 <= n_sets <= states.size:
        error_msg = f'A Markov state model of {states.size} states has 1 to {states.size} sets, not {n_sets}.'
        raise AnalysisError(error_msg)
    if n_sets == 1:
        return [states]
    if n_sets == states.size:
        return list(states.reshape(-1, 1))
    chain = MarkovChain(matrix, [str(state) for state in states.tolist()])
    return [np.asarray(found.split(','), dtype=np.int64) for found in chain.metastable_aggregation(n_sets).states]
