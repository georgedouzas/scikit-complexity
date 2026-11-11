"""Reconstruct the ε-machine of a discrete series by causal-state splitting."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..enums import DataKind
from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, IntArray, IntTuple
from ._base import _read_series
from .types import Source

TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class EpsilonMachineResult:
    """A result of `epsilon_machine`, the causal states of a discrete series.

    Args:
        values:
            The series of symbols.

        labels:
            The label of the series.

        alphabet:
            The distinct symbols, whose positions are the codes of the histories and of the columns.

        histories:
            The histories of every causal state, as codes of their symbols, oldest first.

        transitions:
            The state reached from every state on every symbol, -1 for symbols a state never emits.

        probabilities:
            The probability of every symbol from every state.

        stationary:
            The fraction of time spent in every state.

        statistical_complexity:
            The entropy of the stationary distribution of the causal states in bits.

        entropy_rate:
            The mean entropy of the next symbol given the causal state in bits.
    """

    values: Array
    labels: list[str]
    alphabet: Array
    histories: tuple[tuple[IntTuple, ...], ...]
    transitions: IntArray
    probabilities: Array
    stationary: Array
    statistical_complexity: float
    entropy_rate: float


def epsilon_machine(
    source: Source,
    *,
    variables: Sequence[str] | None = None,
    max_length: int = 3,
    significance: float = 0.001,
) -> EpsilonMachineResult:
    """Return the epsilon-machine of a discrete series.

    It is reconstructed by causal-state splitting (CSSR). Histories up to `max_length` symbols join the state whose law
    of the next symbol they share by a chi-square test, and states are then split until their transitions are
    deterministic.

    It appears in many fields. It finds the hidden states of a cellular automaton, of the spikes of a neuron, and of a
    written text.

    Args:
        source:
            The trajectory, or an array that holds one series of symbols.

        variables:
            The variable of a trajectory to read, a list of one name, or `None` for the only scalar series.

        max_length:
            The longest history, kept well below the logarithm of the series length in the alphabet size.

        significance:
            The significance level of the test that splits a history from a state.

    Returns:
        result:
            The series, the causal states with their histories, transitions, and laws, and their entropies.

    Raises:
        AnalysisError:
            A setting is out of range, the source does not hold one series, or the longest history is not shorter than
            the series.

    Examples:
        >>> from skcomplex.core.analysis import epsilon_machine
        >>> machine = epsilon_machine([0, 0, 1] * 300, max_length=3)
        >>> len(machine.histories), round(machine.statistical_complexity, 3), round(machine.entropy_rate, 3)
        (3, 1.585, 0.0)
    """
    if max_length < 1 or not 0 < significance < 1:
        error_msg = (
            f'epsilon_machine has max_length={max_length!r} and significance={significance!r}. Use a longest history '
            'of 1 or more and a significance between 0 and 1.'
        )
        raise AnalysisError(error_msg)
    series = _read_series(source, variables)
    if series.values.shape[1] != 1:
        error_msg = (
            f'epsilon_machine reads one series of symbols, but the source has {series.values.shape[1]}. Select one '
            'with `variables`.'
        )
        raise AnalysisError(error_msg)
    values = series.values[:, 0]
    alphabet, codes = np.unique(values, return_inverse=True)
    if not max_length < codes.size:
        error_msg = f'The longest history must be shorter than the series of length {codes.size}. Lower max_length.'
        raise AnalysisError(error_msg)
    codes = codes.astype(np.int64)
    size = alphabet.size
    counts = [_history_counts(codes, length, size) for length in range(max_length + 1)]
    states = _determinize(_homogenize(counts, size, significance), counts[max_length])
    index = {history: number for number, state in enumerate(states) for history in state}
    pooled = np.asarray([_pooled(state, counts) for state in states], dtype=float)
    totals = pooled.sum(axis=1)
    probabilities = pooled / totals[:, None]
    stationary = totals / totals.sum()
    return EpsilonMachineResult(
        values,
        series.labels,
        alphabet,
        tuple(tuple(state) for state in states),
        np.asarray([_transitions(state, counts[max_length], index) for state in states], dtype=np.int64),
        probabilities,
        stationary,
        float(stats.entropy(stationary, base=2)),
        float(stationary @ stats.entropy(probabilities, base=2, axis=1)),
    )


def _history_counts(codes: IntArray, length: int, size: int) -> dict[IntTuple, IntArray]:
    """Return the counts of the next symbol after every observed history of a length.

    Histories are read oldest symbol first.
    """
    windows = np.lib.stride_tricks.sliding_window_view(codes, length + 1)
    weights = size ** np.arange(length, -1, -1, dtype=np.int64)
    table = np.bincount(windows @ weights, minlength=size ** (length + 1)).reshape(-1, size)
    observed = np.flatnonzero(table.sum(axis=1))
    return {
        tuple(int(index // size ** (length - 1 - place) % size) for place in range(length)): table[index]
        for index in observed
    }


def _is_same_law(first: IntArray, second: IntArray, significance: float) -> bool:
    """Return whether two counts of next symbols come from one law.

    The test is a chi-square test of homogeneity.
    """
    table = np.vstack([first, second])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] <= 1:
        return True
    return bool(stats.chi2_contingency(table, correction=False).pvalue > significance)


def _pooled(state: list[IntTuple], counts: list[dict[IntTuple, IntArray]]) -> IntArray:
    """Return the counts of the next symbol pooled over the histories of a state."""
    return np.sum([counts[len(history)][history] for history in state], axis=0)


def _assign(
    child: IntTuple,
    states: list[list[IntTuple]],
    parent: list[IntTuple],
    counts: list[dict[IntTuple, IntArray]],
    significance: float,
) -> None:
    """Add a history to a state.

    The state is its parent's, the first other state with its law, or a new one.
    """
    child_counts = counts[len(child)][child]
    for state in [parent, *(state for state in states if state is not parent)]:
        if _is_same_law(
            child_counts,
            _pooled(state, counts),
            significance,
        ):
            state.append(child)
            return
    states.append([child])


def _homogenize(counts: list[dict[IntTuple, IntArray]], size: int, significance: float) -> list[list[IntTuple]]:
    """Return the states of histories with the same law of the next symbol.

    They are grown one past symbol at a time.
    """
    states: list[list[IntTuple]] = [[()]]
    for length in range(len(counts) - 1):
        for state in list(states):
            children = [(symbol, *history) for history in state if len(history) == length for symbol in range(size)]
            for child in [child for child in children if child in counts[length + 1]]:
                _assign(child, states, state, counts, significance)
    longest = len(counts) - 1
    kept = [[history for history in state if len(history) == longest] for state in states]
    return [state for state in kept if state]


def _successors(history: IntTuple, counts: IntArray, index: dict[IntTuple, int]) -> tuple[int, ...]:
    """Return the state reached from a history on every symbol.

    It is -1 for symbols the history never emits.
    """
    return tuple(index.get((*history[1:], symbol), -1) if count else -1 for symbol, count in enumerate(counts.tolist()))


def _determinize(states: list[list[IntTuple]], counts: dict[IntTuple, IntArray]) -> list[list[IntTuple]]:
    """Return the states split until every symbol leads from a state to a single state."""
    while True:
        index = {history: number for number, state in enumerate(states) for history in state}
        split: list[list[IntTuple]] = []
        for state in states:
            groups: dict[IntTuple, list[IntTuple]] = {}
            for history in state:
                groups.setdefault(_successors(history, counts[history], index), []).append(
                    history,
                )
            split.extend(groups.values())
        if len(split) == len(states):
            return split
        states = split


def _transitions(state: list[IntTuple], counts: dict[IntTuple, IntArray], index: dict[IntTuple, int]) -> list[int]:
    """Return the state most histories of a state reach on every symbol.

    It is -1 for symbols none of them emits.
    """
    signatures = [_successors(history, counts[history], index) for history in state]
    reached = [
        Counter(signature[symbol] for signature in signatures if signature[symbol] >= 0)
        for symbol in range(len(signatures[0]))
    ]
    return [found.most_common(1)[0][0] if found else -1 for found in reached]
