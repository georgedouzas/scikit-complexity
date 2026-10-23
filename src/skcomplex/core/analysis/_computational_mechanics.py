"""Reconstruct the ε-machine of a discrete series by causal-state splitting."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..base import AnalysisError
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]
type Counts = npt.NDArray[np.int64]
type History = tuple[int, ...]


def _history_counts(codes: Counts, length: int, size: int) -> dict[History, Counts]:
    """Return the counts of the next symbol after every observed history of a length, oldest symbol first."""
    windows = np.lib.stride_tricks.sliding_window_view(codes, length + 1)
    weights = size ** np.arange(length, -1, -1, dtype=np.int64)
    table = np.bincount(windows @ weights, minlength=size ** (length + 1)).reshape(-1, size)
    observed = np.flatnonzero(table.sum(axis=1))
    return {
        tuple(int(index // size ** (length - 1 - place) % size) for place in range(length)): table[index]
        for index in observed
    }


def _is_same_law(first: Counts, second: Counts, significance: float) -> bool:
    """Return whether two counts of next symbols come from one law by a chi-square test of homogeneity."""
    table = np.vstack([first, second])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] <= 1:
        return True
    return bool(stats.chi2_contingency(table, correction=False).pvalue > significance)


def _pooled(state: list[History], counts: list[dict[History, Counts]]) -> Counts:
    """Return the counts of the next symbol pooled over the histories of a state."""
    return np.sum([counts[len(history)][history] for history in state], axis=0)


def _assign(
    child: History,
    states: list[list[History]],
    parent: list[History],
    counts: list[dict[History, Counts]],
    significance: float,
) -> None:
    """Add a history to its parent's state, to the first other state with its law, or to a new state."""
    child_counts = counts[len(child)][child]
    for state in [parent, *(state for state in states if state is not parent)]:
        if _is_same_law(child_counts, _pooled(state, counts), significance):
            state.append(child)
            return
    states.append([child])


def _homogenize(counts: list[dict[History, Counts]], size: int, significance: float) -> list[list[History]]:
    """Return the states of histories with the same law of the next symbol, grown one past symbol at a time."""
    states: list[list[History]] = [[()]]
    for length in range(len(counts) - 1):
        for state in list(states):
            children = [(symbol, *history) for history in state if len(history) == length for symbol in range(size)]
            for child in [child for child in children if child in counts[length + 1]]:
                _assign(child, states, state, counts, significance)
    longest = len(counts) - 1
    kept = [[history for history in state if len(history) == longest] for state in states]
    return [state for state in kept if state]


def _successors(history: History, counts: Counts, index: dict[History, int]) -> tuple[int, ...]:
    """Return the state reached from a history on every symbol, -1 for symbols it never emits."""
    return tuple(index.get((*history[1:], symbol), -1) if count else -1 for symbol, count in enumerate(counts.tolist()))


def _determinize(states: list[list[History]], counts: dict[History, Counts]) -> list[list[History]]:
    """Return the states split until every symbol leads from a state to a single state."""
    while True:
        index = {history: number for number, state in enumerate(states) for history in state}
        split: list[list[History]] = []
        for state in states:
            groups: dict[tuple[int, ...], list[History]] = {}
            for history in state:
                groups.setdefault(_successors(history, counts[history], index), []).append(history)
            split.extend(groups.values())
        if len(split) == len(states):
            return split
        states = split


def _transitions(state: list[History], counts: dict[History, Counts], index: dict[History, int]) -> list[int]:
    """Return the state most histories of a state reach on every symbol, -1 for symbols none of them emits."""
    signatures = [_successors(history, counts[history], index) for history in state]
    reached = [
        Counter(signature[symbol] for signature in signatures if signature[symbol] >= 0)
        for symbol in range(len(signatures[0]))
    ]
    return [found.most_common(1)[0][0] if found else -1 for found in reached]


@dataclass(frozen=True)
class EpsilonMachine:
    """Epsilon-machine of a discrete series: its causal states and the labelled transitions between them.

    Args:
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
    """

    alphabet: npt.NDArray[np.generic]
    histories: tuple[tuple[History, ...], ...]
    transitions: Counts
    probabilities: Array
    stationary: Array

    @property
    def statistical_complexity(self) -> float:
        """Entropy of the stationary distribution of the causal states in bits."""
        p = self.stationary[self.stationary > 0]
        return float(np.sum(p * np.log2(1 / p)))

    @property
    def entropy_rate(self) -> float:
        """Mean entropy of the next symbol given the causal state in bits."""
        surprises = np.log2(1 / np.where(self.probabilities > 0, self.probabilities, 1.0))
        return float(np.sum(self.stationary[:, None] * self.probabilities * surprises))


@analysis(DataKind.SERIES)
def epsilon_machine(symbols: npt.ArrayLike, max_length: int = 3, *, significance: float = 0.001) -> EpsilonMachine:
    """Return the ε-machine of a discrete series reconstructed by causal-state splitting (CSSR).

    Histories up to `max_length` symbols join the state whose law of the next symbol they share by a chi-square
    test, and states are then split until their transitions are deterministic.

    Args:
        symbols:
            The series of symbols.

        max_length:
            The longest history, kept well below the logarithm of the series length in the alphabet size.

        significance:
            The significance level of the test that splits a history from a state.

    Returns:
        machine:
            The causal states, their transitions, and their stationary distribution.

    Raises:
        AnalysisError:
            The longest history is shorter than one symbol, or not shorter than the series.

    Examples:
        >>> from skcomplex.core.analysis import epsilon_machine
        >>> machine = epsilon_machine([0, 0, 1] * 300, 3)
        >>> len(machine.histories), round(machine.statistical_complexity, 3), round(machine.entropy_rate, 3)
        (3, 1.585, 0.0)
    """
    alphabet, codes = np.unique(np.asarray(symbols).ravel(), return_inverse=True)
    if not 1 <= max_length < codes.size:
        error_msg = f'The longest history must be between 1 and the series length {codes.size} less one.'
        raise AnalysisError(error_msg)
    codes = codes.astype(np.int64)
    size = alphabet.size
    counts = [_history_counts(codes, length, size) for length in range(max_length + 1)]
    states = _determinize(_homogenize(counts, size, significance), counts[max_length])
    index = {history: number for number, state in enumerate(states) for history in state}
    pooled = np.asarray([_pooled(state, counts) for state in states], dtype=float)
    transitions = np.asarray([_transitions(state, counts[max_length], index) for state in states], dtype=np.int64)
    totals = pooled.sum(axis=1)
    return EpsilonMachine(
        alphabet=alphabet,
        histories=tuple(tuple(state) for state in states),
        transitions=transitions,
        probabilities=pooled / totals[:, None],
        stationary=totals / totals.sum(),
    )
