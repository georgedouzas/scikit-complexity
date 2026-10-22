"""Measure emergence across scales: partitions of states, effective information, and causal emergence."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..base import AnalysisError, UnobservedStatesWarning
from ._kinds import DataKind, analysis
from ._transitions import stationary_distribution

type Array = npt.NDArray[np.float64]
type Labels = npt.NDArray[np.int64]
type Partition = npt.ArrayLike | Callable[[int], int]

WEIGHTINGS = ('uniform', 'stationary')
SEARCHES = ('greedy', 'exhaustive')
MAX_EXHAUSTIVE_STATES = 10
TOLERANCE = 1e-12


def _labels_of(partition: Partition, states: npt.NDArray[np.int64]) -> Labels:
    """Return the macro label of every micro state, from a label array or a labelling function."""
    if callable(partition):
        return np.asarray([int(partition(int(state))) for state in states.ravel()], dtype=np.int64).reshape(
            states.shape,
        )
    labels = np.asarray(partition, dtype=np.int64)
    if labels.ndim != 1:
        error_msg = f'A partition holds one label per state, not an array of shape {labels.shape}.'
        raise AnalysisError(error_msg)
    if states.size and int(states.max()) >= labels.size:
        error_msg = f'The partition has {labels.size} labels but the states reach {int(states.max())}.'
        raise AnalysisError(error_msg)
    return labels[states]


def _blocks_of(partition: Partition, n_states: int) -> Labels:
    """Return the label of every state of a matrix, after checking that the labels number the blocks from 0."""
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
    if array.ndim != 2 or array.shape[0] != array.shape[1]:  # noqa: PLR2004
        error_msg = f'A transition matrix is square, not of shape {array.shape}.'
        raise AnalysisError(error_msg)
    return array


def _weights(matrix: Array, weighting: str) -> Array:
    """Return the weight of every micro state within its block, uniform or stationary."""
    if weighting not in WEIGHTINGS:
        error_msg = f'Unknown weighting {weighting!r}. Use one of {", ".join(WEIGHTINGS)}.'
        raise AnalysisError(error_msg)
    if weighting == 'uniform':
        return np.ones(matrix.shape[0])
    totals = matrix.sum(axis=1, keepdims=True)
    rows = np.divide(matrix, totals, out=np.eye(matrix.shape[0]), where=totals > 0)
    return stationary_distribution(rows)


def _coarse_matrix(matrix: Array, labels: Labels, weights: Array) -> Array:
    """Return the macro transition matrix of a partition, the weighted micro rows summed within every block."""
    n_blocks = int(labels.max()) + 1
    membership = np.eye(n_blocks)[labels]
    flows = membership.T @ (weights[:, None] * matrix) @ membership
    totals = flows.sum(axis=1, keepdims=True)
    return np.divide(flows, totals, out=np.zeros_like(flows), where=totals > 0)


@analysis(DataKind.PATH)
def partition_coarse_grain(
    source: npt.ArrayLike,
    partition: Partition,
    *,
    matrix: bool = False,
    weighting: str = 'uniform',
) -> npt.NDArray[np.int64] | Array:
    """Return the macro states of a discrete trajectory, or the macro transition matrix of a micro one.

    The macro matrix sums the micro transitions from every block to every block, weighting the micro states of a block
    uniformly, as Hoel, Albantakis, and Tononi (2013) do, or by the stationary distribution, as Klein and Hoel (2020)
    do. Every macro row sums to one.

    Args:
        source:
            The integer micro states of one path or one row per path, or a transition matrix when `matrix` is true.

        partition:
            The macro label of every micro state, numbered 0, 1, and so on, or a function from a micro state to its
            label.

        matrix:
            Whether the source is a transition matrix rather than a trajectory.

        weighting:
            `'uniform'` or `'stationary'`, the weight of the micro states within a block of a matrix.

    Returns:
        coarse:
            The macro label of every frame of the trajectory, or the macro transition matrix.

    Raises:
        AnalysisError:
            The matrix is not square, the weighting is unknown, or the partition has the wrong length or an empty
            block.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import partition_coarse_grain
        >>> partition_coarse_grain([0, 1, 2, 3, 2], [0, 0, 1, 1]).tolist()
        [0, 0, 1, 1, 1]
        >>> partition_coarse_grain(np.full((4, 4), 0.25), [0, 0, 0, 1], matrix=True).tolist()
        [[0.75, 0.25], [0.75, 0.25]]
    """
    if not matrix:
        return _labels_of(partition, np.asarray(source, dtype=np.int64))
    micro = _square(source)
    return _coarse_matrix(micro, _blocks_of(partition, micro.shape[0]), _weights(micro, weighting))


@dataclass(frozen=True)
class EffectiveInformation:
    """Effective information of a transition matrix and its two parts, in bits.

    Args:
        information:
            The mutual information between a uniform intervention on the current state and the next state.

        determinism:
            `log2 n` minus the mean entropy of the rows, high when every state has one successor.

        degeneracy:
            `log2 n` minus the entropy of the mean row, high when many states share a successor.

        states:
            The states with observed transitions, the rows the measure uses.
    """

    information: float
    determinism: float
    degeneracy: float
    states: npt.NDArray[np.int64]


@analysis(DataKind.PATH)
def effective_information(matrix: npt.ArrayLike) -> EffectiveInformation:
    """Return the effective information of a transition matrix, its determinism minus its degeneracy.

    The effective information is `H(mean row) - mean H(row)` over the `n` states with observed transitions. A matrix
    of counts is normalised by row first.

    Args:
        matrix:
            The transition matrix, or the transition counts, one row per origin and one column per destination.

    Returns:
        information:
            The effective information, the determinism, the degeneracy, and the states used.

    Raises:
        AnalysisError:
            The matrix is not square or has no observed transition.

    Warns:
        UnobservedStatesWarning:
            Some rows hold no transition, and the warning names their states.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import effective_information
        >>> effective_information(np.eye(4)).information
        2.0
        >>> round(effective_information([[0.5, 0.5], [0.5, 0.5]]).information, 12)
        0.0
    """
    array = _square(matrix)
    totals = array.sum(axis=1)
    observed = np.flatnonzero(totals > 0)
    if observed.size == 0:
        error_msg = 'The transition matrix has no observed transition, so it has no effective information.'
        raise AnalysisError(error_msg)
    if observed.size < array.shape[0]:
        dropped = np.setdiff1d(np.arange(array.shape[0]), observed).tolist()
        warnings.warn(
            f'The states {dropped} have no observed transitions and are left out of the effective information.',
            UnobservedStatesWarning,
            stacklevel=2,
        )
    rows = array[observed] / totals[observed, None]
    noise = float(np.mean(stats.entropy(rows, base=2, axis=1)))
    spread = float(stats.entropy(rows.mean(axis=0), base=2))
    size = float(np.log2(observed.size))
    return EffectiveInformation(spread - noise, size - noise, size - spread, observed)


@dataclass(frozen=True)
class CausalEmergence:
    """Causal emergence of a partition of the states of a transition matrix.

    Args:
        partition:
            The macro label of every micro state.

        gain:
            The effective information of the macro matrix minus that of the micro matrix, in bits.

        micro:
            The effective information of the micro matrix.

        macro:
            The effective information of the macro matrix.
    """

    partition: Labels
    gain: float
    micro: EffectiveInformation
    macro: EffectiveInformation


def _canonical(labels: npt.ArrayLike) -> Labels:
    """Return labels renumbered 0, 1, and so on in the order of their first state."""
    _, first, inverse = np.unique(np.asarray(labels), return_index=True, return_inverse=True)
    return np.argsort(np.argsort(first))[inverse].astype(np.int64)


def _macro_information(matrix: Array, labels: Labels, weights: Array) -> float:
    """Return the effective information of the macro matrix of a partition, without warnings on unobserved blocks."""
    macro = _coarse_matrix(matrix, labels, weights)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnobservedStatesWarning)
        return effective_information(macro).information


def _greedy(matrix: Array, weights: Array) -> Labels:
    """Return the partition reached by merging the pair of blocks with the largest gain until no merge gains."""
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


def _set_partitions(n_states: int) -> Iterator[Labels]:
    """Yield every partition of the states as restricted growth labels."""
    labels = np.zeros(n_states, dtype=np.int64)

    def extend(position: int, blocks: int) -> Iterator[Labels]:
        if position == n_states:
            yield labels.copy()
            return
        for label in range(blocks + 1):
            labels[position] = label
            yield from extend(position + 1, max(blocks, label + 1))

    if n_states:
        yield from extend(1, 1)


def _exhaustive(matrix: Array, weights: Array) -> Labels:
    """Return the partition with the largest macro effective information, the finest of equal ones."""
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


@analysis(DataKind.PATH)
def causal_emergence(
    matrix: npt.ArrayLike,
    partition: Partition | None = None,
    *,
    weighting: str = 'uniform',
    search: str = 'greedy',
) -> CausalEmergence:
    """Return the causal emergence of a given or searched partition, its gain in effective information.

    The greedy search of Klein and Hoel (2020) starts from the micro states and merges the pair of blocks with the
    largest macro effective information while that grows, so it can miss the best partition. The exhaustive search
    tries every partition of at most 10 states.

    Args:
        matrix:
            The micro transition matrix, or the transition counts.

        partition:
            The macro label of every micro state or a function from a micro state to its label, or `None` to search.

        weighting:
            `'uniform'` or `'stationary'`, the weight of the micro states within a block.

        search:
            `'greedy'` or `'exhaustive'`, the search when no partition is given.

    Returns:
        emergence:
            The partition, the gain, and the micro and macro effective information.

    Raises:
        AnalysisError:
            The matrix is not square, the weighting or the search is unknown, the partition is not valid, or the
            exhaustive search gets more than 10 states.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import causal_emergence
        >>> matrix = np.zeros((4, 4))
        >>> matrix[:3, :3], matrix[3, 3] = 1 / 3, 1.0
        >>> found = causal_emergence(matrix)
        >>> found.partition.tolist(), round(found.gain, 3)
        ([0, 0, 0, 1], 0.189)
    """
    micro = _square(matrix)
    weights = _weights(micro, weighting)
    if partition is not None:
        labels = _blocks_of(partition, micro.shape[0])
    elif search not in SEARCHES:
        error_msg = f'Unknown search {search!r}. Use one of {", ".join(SEARCHES)}.'
        raise AnalysisError(error_msg)
    else:
        labels = _greedy(micro, weights) if search == 'greedy' else _exhaustive(micro, weights)
    micro_information = effective_information(micro)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnobservedStatesWarning)
        macro_information = effective_information(_coarse_matrix(micro, labels, weights))
    return CausalEmergence(
        labels,
        macro_information.information - micro_information.information,
        micro_information,
        macro_information,
    )
