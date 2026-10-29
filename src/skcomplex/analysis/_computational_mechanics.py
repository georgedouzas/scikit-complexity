"""Reconstruct the ε-machine of a discrete series by causal-state splitting."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from collections.abc import Sequence
from typing import Self

import numpy as np
from scipy import stats

from ..errors import AnalysisError
from ..types import DataKind, IntArray, IntTuple
from ._base import BaseAnalysis, Source


class EpsilonMachineAnalysis(BaseAnalysis):
    """Epsilon-machine of a discrete series reconstructed by causal-state splitting (CSSR).

    Histories up to `max_length` symbols join the state whose law of the next symbol they share by a chi-square
    test, and states are then split until their transitions are deterministic.

    Args:
        variables:
            The variable of a trajectory to read, a list of one name, or `None` for the only scalar series.

        max_length:
            The longest history, kept well below the logarithm of the series length in the alphabet size.

        significance:
            The significance level of the test that splits a history from a state.

    Attributes:
        values_:
            The series of symbols.

        labels_:
            The label of the series.

        alphabet_:
            The distinct symbols, whose positions are the codes of the histories and of the columns.

        histories_:
            The histories of every causal state, as codes of their symbols, oldest first.

        transitions_:
            The state reached from every state on every symbol, -1 for symbols a state never emits.

        probabilities_:
            The probability of every symbol from every state.

        stationary_:
            The fraction of time spent in every state.

        statistical_complexity_:
            The entropy of the stationary distribution of the causal states in bits.

        entropy_rate_:
            The mean entropy of the next symbol given the causal state in bits.

    Examples:
        >>> from skcomplex.analysis import EpsilonMachineAnalysis
        >>> machine = EpsilonMachineAnalysis(max_length=3).check().prepare([0, 0, 1] * 300).run()
        >>> len(machine.histories_), round(machine.statistical_complexity_, 3), round(machine.entropy_rate_, 3)
        (3, 1.585, 0.0)
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        max_length: int = 3,
        significance: float = 0.001,
    ) -> None:
        self.variables = variables
        self.max_length = max_length
        self.significance = significance

    def _check(self) -> Self:
        """Validate the longest history and the significance level."""
        if self.max_length < 1 or not 0 < self.significance < 1:
            error_msg = (
                f'{type(self).__name__} has max_length={self.max_length!r} and significance={self.significance!r}. '
                'Use a longest history of 1 or more and a significance between 0 and 1.'
            )
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the one series of symbols of the source."""
        series = self._read_series(source, self.variables)
        if series.values.shape[1] != 1:
            error_msg = (
                f'{type(self).__name__} reads one series of symbols, but the source has {series.values.shape[1]}. '
                'Select one with `variables`.'
            )
            raise AnalysisError(error_msg)
        self.values_ = series.values[:, 0]
        self.labels_ = series.labels
        return self

    def _run(self, inputs: object = None) -> Self:
        """Reconstruct the causal states, their transitions, and their stationary distribution."""
        alphabet, codes = np.unique(self.values_, return_inverse=True)
        if not self.max_length < codes.size:
            error_msg = f'The longest history must be shorter than the series of length {codes.size}. Lower max_length.'
            raise AnalysisError(error_msg)
        codes = codes.astype(np.int64)
        size = alphabet.size
        counts = [self._history_counts(codes, length, size) for length in range(self.max_length + 1)]
        states = self._determinize(self._homogenize(counts, size, self.significance), counts[self.max_length])
        index = {history: number for number, state in enumerate(states) for history in state}
        pooled = np.asarray([self._pooled(state, counts) for state in states], dtype=float)
        totals = pooled.sum(axis=1)
        self.alphabet_ = alphabet
        self.histories_ = tuple(tuple(state) for state in states)
        self.transitions_ = np.asarray(
            [self._transitions(state, counts[self.max_length], index) for state in states],
            dtype=np.int64,
        )
        self.probabilities_ = pooled / totals[:, None]
        self.stationary_ = totals / totals.sum()
        self.statistical_complexity_ = float(stats.entropy(self.stationary_, base=2))
        self.entropy_rate_ = float(self.stationary_ @ stats.entropy(self.probabilities_, base=2, axis=1))
        return self

    @staticmethod
    def _history_counts(codes: IntArray, length: int, size: int) -> dict[IntTuple, IntArray]:
        """Return the counts of the next symbol after every observed history of a length, oldest symbol first."""
        windows = np.lib.stride_tricks.sliding_window_view(codes, length + 1)
        weights = size ** np.arange(length, -1, -1, dtype=np.int64)
        table = np.bincount(windows @ weights, minlength=size ** (length + 1)).reshape(-1, size)
        observed = np.flatnonzero(table.sum(axis=1))
        return {
            tuple(int(index // size ** (length - 1 - place) % size) for place in range(length)): table[index]
            for index in observed
        }

    @staticmethod
    def _is_same_law(first: IntArray, second: IntArray, significance: float) -> bool:
        """Return whether two counts of next symbols come from one law by a chi-square test of homogeneity."""
        table = np.vstack([first, second])
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] <= 1:
            return True
        return bool(stats.chi2_contingency(table, correction=False).pvalue > significance)

    @staticmethod
    def _pooled(state: list[IntTuple], counts: list[dict[IntTuple, IntArray]]) -> IntArray:
        """Return the counts of the next symbol pooled over the histories of a state."""
        return np.sum([counts[len(history)][history] for history in state], axis=0)

    @staticmethod
    def _assign(
        child: IntTuple,
        states: list[list[IntTuple]],
        parent: list[IntTuple],
        counts: list[dict[IntTuple, IntArray]],
        significance: float,
    ) -> None:
        """Add a history to its parent's state, to the first other state with its law, or to a new state."""
        child_counts = counts[len(child)][child]
        for state in [parent, *(state for state in states if state is not parent)]:
            if EpsilonMachineAnalysis._is_same_law(
                child_counts,
                EpsilonMachineAnalysis._pooled(state, counts),
                significance,
            ):
                state.append(child)
                return
        states.append([child])

    @staticmethod
    def _homogenize(counts: list[dict[IntTuple, IntArray]], size: int, significance: float) -> list[list[IntTuple]]:
        """Return the states of histories with the same law of the next symbol, grown one past symbol at a time."""
        states: list[list[IntTuple]] = [[()]]
        for length in range(len(counts) - 1):
            for state in list(states):
                children = [(symbol, *history) for history in state if len(history) == length for symbol in range(size)]
                for child in [child for child in children if child in counts[length + 1]]:
                    EpsilonMachineAnalysis._assign(child, states, state, counts, significance)
        longest = len(counts) - 1
        kept = [[history for history in state if len(history) == longest] for state in states]
        return [state for state in kept if state]

    @staticmethod
    def _successors(history: IntTuple, counts: IntArray, index: dict[IntTuple, int]) -> tuple[int, ...]:
        """Return the state reached from a history on every symbol, -1 for symbols it never emits."""
        return tuple(
            index.get((*history[1:], symbol), -1) if count else -1 for symbol, count in enumerate(counts.tolist())
        )

    @staticmethod
    def _determinize(states: list[list[IntTuple]], counts: dict[IntTuple, IntArray]) -> list[list[IntTuple]]:
        """Return the states split until every symbol leads from a state to a single state."""
        while True:
            index = {history: number for number, state in enumerate(states) for history in state}
            split: list[list[IntTuple]] = []
            for state in states:
                groups: dict[IntTuple, list[IntTuple]] = {}
                for history in state:
                    groups.setdefault(EpsilonMachineAnalysis._successors(history, counts[history], index), []).append(
                        history,
                    )
                split.extend(groups.values())
            if len(split) == len(states):
                return split
            states = split

    @staticmethod
    def _transitions(state: list[IntTuple], counts: dict[IntTuple, IntArray], index: dict[IntTuple, int]) -> list[int]:
        """Return the state most histories of a state reach on every symbol, -1 for symbols none of them emits."""
        signatures = [EpsilonMachineAnalysis._successors(history, counts[history], index) for history in state]
        reached = [
            Counter(signature[symbol] for signature in signatures if signature[symbol] >= 0)
            for symbol in range(len(signatures[0]))
        ]
        return [found.most_common(1)[0][0] if found else -1 for found in reached]
