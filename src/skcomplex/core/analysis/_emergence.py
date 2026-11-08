"""Measure emergence across scales.

The measures are effective information, causal emergence, integrated criteria, and closure.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import math
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..engine.runtime import Trajectory
from ..errors import AnalysisError, UnobservedStatesWarning
from ..specs import AnalysisTags
from ..types import AnyArray, Array, DataKind, IntArray
from ._base import _check_choice, _check_settings, _read_series
from ._information import _knn_chain, _log_determinant, _row_codes
from ._markov import _stationary_distribution
from .types import Partition, Source

MATRIX_DIMENSIONS = 2

KINDS = ('effective-information', 'causal', 'criteria', 'closure')
MATRIX_KINDS = ('effective-information', 'causal')
WEIGHTINGS = ('uniform', 'stationary')
SEARCHES = ('greedy', 'exhaustive')
ESTIMATORS = ('plugin', 'gaussian', 'knn')
SETTINGS = {
    'effective-information': set(),
    'causal': {'partition', 'weighting', 'search'},
    'criteria': {'partition', 'lag', 'estimator', 'k', 'base'},
    'closure': {'partition', 'history', 'base'},
}
"""Settings every kind reads, besides `kind` and `variables`."""

MAX_EXHAUSTIVE_STATES = 10
TOLERANCE = 1e-12

TAGS = AnalysisTags(DataKind.SERIES)
KIND_TAGS = {
    'effective-information': AnalysisTags(DataKind.PATH),
    'causal': AnalysisTags(DataKind.PATH),
}


@dataclass(frozen=True)
class EmergenceResult:
    """Emergence of a macro scale over a micro scale, of one kind.

    Args:
        matrix:
            The micro transition matrix, for `'effective-information'` and `'causal'`.

        values:
            The micro series, one column per micro variable, for `'criteria'` and `'closure'`.

        labels:
            The label of every micro series.

        information:
            The effective information in bits, for `'effective-information'`.

        determinism:
            `log2 n` minus the mean entropy of the rows, high when every state has one successor.

        degeneracy:
            `log2 n` minus the entropy of the mean row, high when many states share a successor.

        states:
            The states with observed transitions, the rows the effective information uses.

        partition:
            The macro label of every micro state, for `'causal'`.

        macro_matrix:
            The macro transition matrix of the partition, every row summing to one.

        micro_information:
            The effective information of the micro matrix.

        macro_information:
            The effective information of the macro matrix.

        gain:
            The macro minus the micro effective information in bits.

        psi:
            `I(V_t ; V_t') - Σ_j I(X_t^j ; V_t')`, positive when the macro feature predicts its own future beyond what
            every micro variable does, a sufficient condition for causal emergence.

        delta:
            `max_j [I(V_t ; X_t'^j) - Σ_i I(X_t^i ; X_t'^j)]`, positive when the macro feature predicts a micro
            variable beyond the micro variables, a sufficient condition for downward causation.

        gamma:
            `max_j I(V_t ; X_t'^j)`, zero for a causally decoupled feature when `psi` is positive.

        macro_uncertainty:
            `H(V_t | V_past)`, the entropy of the macro future given the macro past.

        micro_uncertainty:
            `H(V_t | X_past)`, the entropy of the macro future given the micro past.

        gap:
            The information the micro past adds about the macro future, zero for a closed coarse-graining.
    """

    matrix: Array | None = None
    values: AnyArray | None = None
    labels: list[str] | None = None
    information: float | None = None
    determinism: float | None = None
    degeneracy: float | None = None
    states: npt.NDArray[np.int64] | None = None
    partition: IntArray | None = None
    macro_matrix: Array | None = None
    micro_information: float | None = None
    macro_information: float | None = None
    gain: float | None = None
    psi: float | None = None
    delta: float | None = None
    gamma: float | None = None
    macro_uncertainty: float | None = None
    micro_uncertainty: float | None = None
    gap: float | None = None


def emergence(
    source: Source,
    *,
    kind: str = 'causal',
    variables: Sequence[str] | None = None,
    partition: Partition | None = None,
    weighting: str = 'uniform',
    search: str = 'greedy',
    lag: int = 1,
    history: int = 1,
    estimator: str = 'plugin',
    k: int = 3,
    base: float = 2.0,
) -> EmergenceResult:
    """Return the emergence of a macro scale over a micro scale, of one kind.

    The kinds read a transition matrix or the series of micro variables:

    - `'effective-information'`: the effective information of a transition matrix, `H(mean row) - mean H(row)` over
      the states with observed transitions, split into determinism and degeneracy. A matrix of counts is normalised
      by row first.
    - `'causal'`: the causal emergence of a transition matrix, the gain in effective information of the macro matrix
      of a given or searched partition over the micro matrix. The macro matrix sums the micro transitions from every
      block to every block, weighting the micro states of a block uniformly, as Hoel, Albantakis, and Tononi (2013)
      do, or by the stationary distribution, as Klein and Hoel (2020) do. The greedy search of Klein and Hoel (2020)
      merges the pair of blocks with the largest macro effective information while that grows, so it can miss the
      best partition. The exhaustive search tries every partition of at most 10 states.
    - `'criteria'`: the criteria Ψ, Δ, and Γ of Rosas et al. (2020) for the macro feature of the micro series given
      by `partition`, estimated `lag` steps ahead with the minimum mutual information redundancy.
    - `'closure'`: how well the macro past and the micro past of the macro series given by `partition` predict the
      macro future. The coarse-graining is closed when the gap is zero up to the bias of the plug-in estimator.

    Args:
        source:
            A transition matrix or transition counts for `'effective-information'` and `'causal'`, and the trajectory or
            an array of micro series for `'criteria'` and `'closure'`.

        kind:
            The emergence to measure, one of the kinds above.

        variables:
            The micro variables of a trajectory to read, or `None` for every scalar series.

        partition:
            The macro label of every integer micro state, numbered 0, 1, and so on, or a function from an array of
            micro states, or from the micro series with one column per variable, to their macro labels. `None` to
            search a partition of the matrix.

        weighting:
            `'uniform'` or `'stationary'`, the weight of the micro states within a block.

        search:
            `'greedy'` or `'exhaustive'`, the search when no partition is given.

        lag:
            The number of steps between the present and the future of the criteria.

        history:
            The number of past steps the closure conditions on.

        estimator:
            `'plugin'` for the frequencies of symbols, `'gaussian'` for the closed form of a fitted Gaussian, or
            `'knn'` for nearest-neighbour conditional mutual informations.

        k:
            The number of neighbours of the `'knn'` estimator.

        base:
            The base of the logarithm, 2 for bits.

    Returns:
        result:
            The matrix or the micro series read, and the measures of the kind.

    Raises:
        AnalysisError:
            A choice is unknown, a setting the kind does not read is changed, a setting is out of range, the macro
            feature is missing, the matrix is not square or has no observed transition, the partition is not valid, or
            the exhaustive search gets more than 10 states.

    Warns:
        UnobservedStatesWarning:
            Some rows of the micro matrix hold no transition, and the warning names their states.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import emergence
        >>> emergence(np.eye(4), kind='effective-information').information
        2.0
        >>> matrix = np.zeros((4, 4))
        >>> matrix[:3, :3], matrix[3, 3] = 1 / 3, 1.0
        >>> found = emergence(matrix, kind='causal')
        >>> found.partition.tolist(), round(found.gain, 3)
        ([0, 0, 0, 1], 0.189)
        >>> micro = np.random.default_rng(0).integers(0, 4, 5000)
        >>> round(emergence(micro, kind='closure', partition=[0, 0, 1, 1]).gap, 2)
        0.0
    """
    for name, value, choices in (
        ('kind', kind, KINDS),
        ('weighting', weighting, WEIGHTINGS),
        ('search', search, SEARCHES),
        ('estimator', estimator, ESTIMATORS),
    ):
        _check_choice('emergence', name, value, choices)
    _check_settings(
        emergence,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        partition=partition,
        weighting=weighting,
        search=search,
        lag=lag,
        history=history,
        estimator=estimator,
        k=k,
        base=base,
    )
    if kind in {'criteria', 'closure'} and partition is None:
        error_msg = (
            f'emergence of kind {kind!r} needs the macro feature. Pass the macro label of every micro state, or a '
            'function of the micro series, as `partition`.'
        )
        raise AnalysisError(error_msg)
    if lag < 1 or history < 1 or k < 1 or base <= 0:
        error_msg = (
            f'emergence has lag={lag!r}, history={history!r}, k={k!r}, and base={base!r}. Use a lag, a history, and '
            'neighbours of 1 or more and a positive base.'
        )
        raise AnalysisError(error_msg)
    if kind not in MATRIX_KINDS:
        series = _read_series(source, variables)
        macro = _macro_series(series.values, _get_partition(partition, kind))
        if kind == 'criteria':
            return _criteria(series.values, series.labels, macro, lag=lag, estimator=estimator, k=k, base=base)
        return _closure(series.values, series.labels, macro, history, base)
    if isinstance(source, Trajectory):
        error_msg = (
            f'emergence of kind {kind!r} reads a transition matrix. Pass the matrix or the transition counts as an '
            'array.'
        )
        raise AnalysisError(error_msg)
    matrix = _square(source)
    if kind == 'effective-information':
        information, determinism, degeneracy, states = _effective_information(matrix)
        return EmergenceResult(
            matrix=matrix,
            information=information,
            determinism=determinism,
            degeneracy=degeneracy,
            states=states,
        )
    return _causal(matrix, partition, weighting, search)


def _causal(micro: Array, partition: Partition | None, weighting: str, search: str) -> EmergenceResult:
    """Find or take the partition, and compare micro and macro effective information."""
    weights = _weights(micro, weighting)
    if partition is not None:
        labels = _blocks_of(partition, micro.shape[0])
    else:
        labels = _greedy(micro, weights) if search == 'greedy' else _exhaustive(micro, weights)
    macro_matrix = _coarse_matrix(micro, labels, weights)
    micro_information = _effective_information(micro)[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnobservedStatesWarning)
        macro_information = _effective_information(macro_matrix)[0]
    return EmergenceResult(
        matrix=micro,
        partition=labels,
        macro_matrix=macro_matrix,
        micro_information=micro_information,
        macro_information=macro_information,
        gain=macro_information - micro_information,
    )


def _get_partition(partition: Partition | None, kind: str) -> Partition:
    """Return the partition of the micro series, raising an error when none is given."""
    if partition is None:
        error_msg = f'emergence of kind {kind!r} needs the partition of the macro feature.'
        raise AnalysisError(error_msg)
    return partition


def _criteria(
    x: AnyArray,
    labels: list[str],
    v: npt.NDArray[np.generic],
    *,
    lag: int,
    estimator: str,
    k: int,
    base: float,
) -> EmergenceResult:
    """Compute the criteria Ψ, Δ, and Γ of the macro feature."""
    present_x, future_x, present_v, future_v = x[:-lag], x[lag:], v[:-lag], v[lag:]
    information = partial(_estimated_information, estimator=estimator, k=k)
    variables = range(x.shape[1])
    psi = information(present_v, future_v) - sum(information(present_x[:, [j]], future_v) for j in variables)
    gains = [information(present_v, future_x[:, [j]]) for j in variables]
    delta = max(gains[j] - sum(information(present_x[:, [i]], future_x[:, [j]]) for i in variables) for j in variables)
    scale = math.log(base)
    return EmergenceResult(
        values=x,
        labels=labels,
        psi=psi / scale,
        delta=delta / scale,
        gamma=max(gains) / scale,
    )


def _closure(
    x: AnyArray,
    labels: list[str],
    v: npt.NDArray[np.generic],
    history: int,
    base: float,
) -> EmergenceResult:
    """Compute the entropies of the macro future given the macro and the micro past."""
    future = v[history:]
    scale = math.log(base)
    macro_uncertainty = _conditional_entropy(future, _lagged(v, history)) / scale
    micro_uncertainty = _conditional_entropy(future, _lagged(x, history)) / scale
    return EmergenceResult(
        values=x,
        labels=labels,
        macro_uncertainty=macro_uncertainty,
        micro_uncertainty=micro_uncertainty,
        gap=macro_uncertainty - micro_uncertainty,
    )


def _read_columns(data: npt.ArrayLike) -> AnyArray:
    """Return data as one row per sample and one column per variable.

    A series is a single column.

    Args:
        data:
            A series, or one row per sample with the variables along the remaining axes.

    Returns:
        columns:
            The data as a two-dimensional array.
    """
    array = np.asarray(data)
    return array.reshape(-1, 1) if array.ndim == 1 else array.reshape(array.shape[0], -1)


def _labels_of(partition: Partition, states: npt.NDArray[np.int64]) -> IntArray:
    """Return the macro label of every micro state.

    It comes from a label array or a function of an array of states.
    """
    if callable(partition):
        return np.asarray(partition(states), dtype=np.int64).reshape(states.shape)
    labels = np.asarray(partition, dtype=np.int64)
    if labels.ndim != 1:
        error_msg = f'A partition holds one label per state, not an array of shape {labels.shape}.'
        raise AnalysisError(error_msg)
    if states.size and int(states.max()) >= labels.size:
        error_msg = f'The partition has {labels.size} labels but the states reach {int(states.max())}.'
        raise AnalysisError(error_msg)
    return labels[states]


def _blocks_of(partition: Partition, n_states: int) -> IntArray:
    """Return the label of every state of a matrix.

    The labels are checked to number the blocks from 0.
    """
    if not callable(partition) and np.size(partition) != n_states:
        error_msg = f'A partition of {n_states} states holds one label per state, not {np.size(partition)}.'
        raise AnalysisError(error_msg)
    labels = _labels_of(partition, np.arange(n_states))
    if labels.min() < 0:
        error_msg = f'The labels of a partition are 0, 1, and so on, but got {int(labels.min())}.'
        raise AnalysisError(error_msg)
    n_blocks = int(labels.max()) + 1
    empty = np.setdiff1d(np.arange(n_blocks), labels)
    if empty.size:
        error_msg = f'The partition uses no state for label {int(empty[0])}. Number the blocks 0, 1, and so on.'
        raise AnalysisError(error_msg)
    return labels


def _square(matrix: npt.ArrayLike) -> Array:
    """Return a square matrix of nonnegative floats."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != MATRIX_DIMENSIONS or array.shape[0] != array.shape[1]:
        error_msg = f'A transition matrix is square, not of shape {array.shape}.'
        raise AnalysisError(error_msg)
    return array


def _weights(matrix: Array, weighting: str) -> Array:
    """Return the weight of every micro state within its block, uniform or stationary."""
    if weighting == 'uniform':
        return np.ones(matrix.shape[0])
    totals = matrix.sum(axis=1, keepdims=True)
    rows = np.divide(matrix, totals, out=np.eye(matrix.shape[0]), where=totals > 0)
    return _stationary_distribution(rows)


def _coarse_matrix(matrix: Array, labels: IntArray, weights: Array) -> Array:
    """Return the macro transition matrix of a partition.

    The weighted micro rows are summed within every block.
    """
    n_blocks = int(labels.max()) + 1
    membership = np.eye(n_blocks)[labels]
    flows = membership.T @ (weights[:, None] * matrix) @ membership
    totals = flows.sum(axis=1, keepdims=True)
    return np.divide(flows, totals, out=np.zeros_like(flows), where=totals > 0)


def _canonical(labels: npt.ArrayLike) -> IntArray:
    """Return labels renumbered 0, 1, and so on in the order of their first state."""
    _, first, inverse = np.unique(np.asarray(labels), return_index=True, return_inverse=True)
    return np.argsort(np.argsort(first))[inverse].astype(np.int64)


def _effective_information(matrix: Array) -> tuple[float, float, float, npt.NDArray[np.int64]]:
    """Return the effective information, determinism, degeneracy, and observed states.

    They are those of a matrix.
    """
    totals = matrix.sum(axis=1)
    observed = np.flatnonzero(totals > 0)
    if observed.size == 0:
        error_msg = 'The transition matrix has no observed transition, so it has no effective information.'
        raise AnalysisError(error_msg)
    if observed.size < matrix.shape[0]:
        dropped = np.setdiff1d(np.arange(matrix.shape[0]), observed).tolist()
        warnings.warn(
            f'The states {dropped} have no observed transitions and are left out of the effective information.',
            UnobservedStatesWarning,
            stacklevel=2,
        )
    rows = matrix[observed] / totals[observed, None]
    noise = float(np.mean(stats.entropy(rows, base=2, axis=1)))
    spread = float(stats.entropy(rows.mean(axis=0), base=2))
    size = float(np.log2(observed.size))
    return spread - noise, size - noise, size - spread, observed


def _macro_information(matrix: Array, labels: IntArray, weights: Array) -> float:
    """Return the effective information of the macro matrix of a partition.

    Unobserved blocks raise no warning.
    """
    macro = _coarse_matrix(matrix, labels, weights)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnobservedStatesWarning)
        return _effective_information(macro)[0]


def _greedy(matrix: Array, weights: Array) -> IntArray:
    """Return the partition reached by greedy merges of blocks.

    The pair of blocks with the largest gain is merged until no merge gains.
    """
    labels = np.arange(matrix.shape[0], dtype=np.int64)
    best = _macro_information(matrix, labels, weights)
    while int(labels.max()) > 0:
        merges = [
            _canonical(np.where(labels == second, first, labels))
            for first, second in itertools.combinations(range(int(labels.max()) + 1), 2)
        ]
        scores = [_macro_information(matrix, merged, weights) for merged in merges]
        index = int(np.argmax(scores))
        if scores[index] <= best + TOLERANCE:
            break
        labels, best = merges[index], scores[index]
    return labels


def _set_partitions(n_states: int) -> Iterator[IntArray]:
    """Yield every partition of the states as restricted growth labels."""
    labels = np.zeros(n_states, dtype=np.int64)

    def extend(position: int, blocks: int) -> Iterator[IntArray]:
        if position == n_states:
            yield labels.copy()
            return
        for label in range(blocks + 1):
            labels[position] = label
            yield from extend(position + 1, max(blocks, label + 1))

    if n_states:
        yield from extend(1, 1)


def _exhaustive(matrix: Array, weights: Array) -> IntArray:
    """Return the partition with the largest macro effective information.

    Of equal ones, the finest is returned.
    """
    if matrix.shape[0] > MAX_EXHAUSTIVE_STATES:
        error_msg = (
            f'The exhaustive search covers at most {MAX_EXHAUSTIVE_STATES} states, not {matrix.shape[0]}. '
            "Use search='greedy'."
        )
        raise AnalysisError(error_msg)
    best, best_score = np.arange(matrix.shape[0], dtype=np.int64), -np.inf
    for labels in _set_partitions(matrix.shape[0]):
        score = _macro_information(matrix, labels, weights)
        blocks = int(labels.max())
        if score > best_score + TOLERANCE or (score > best_score - TOLERANCE and blocks > int(best.max())):
            best, best_score = labels, max(score, best_score)
    return best


def _macro_series(micro: npt.NDArray[np.generic], partition: Partition) -> npt.NDArray[np.generic]:
    """Return the macro series of a micro series under a partition.

    The partition is given as labels or as a function.
    """
    if callable(partition):
        return _read_columns(partition(micro))
    return _read_columns(np.asarray(partition)[np.asarray(micro).ravel().astype(np.int64)])


def _lagged(columns: npt.NDArray[np.generic], history: int) -> npt.NDArray[np.generic]:
    """Return the `history` rows before every time from `history` on, side by side."""
    size = columns.shape[0]
    return np.column_stack([columns[history - step : size - step] for step in range(1, history + 1)])


def _conditional_entropy(target: npt.NDArray[np.generic], given: npt.NDArray[np.generic]) -> float:
    """Return the plug-in entropy of the target given other columns in nats."""
    return _plugin_entropy(np.column_stack([target, given])) - _plugin_entropy(
        given,
    )


def _plugin_entropy(columns: AnyArray) -> float:
    """Return the plug-in entropy of the rows of discrete columns in nats."""
    return float(stats.entropy(np.bincount(_row_codes(columns))))


def _estimated_information(first: AnyArray, second: AnyArray, estimator: str, k: int) -> float:
    """Return the mutual information of two groups of columns in nats.

    It uses plug-in, Gaussian, or kNN estimates.
    """
    if estimator == 'plugin':
        joint = np.column_stack([_row_codes(first), _row_codes(second)])
        return _plugin_entropy(first) + _plugin_entropy(second) - _plugin_entropy(joint)
    x, y = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if estimator == 'gaussian':
        joint_value = _log_determinant(np.column_stack([x, y]))
        return (_log_determinant(x) + _log_determinant(y) - joint_value) / 2
    return sum(_knn_chain(x[:, index], y, k, x[:, :index]) for index in range(x.shape[1]))
