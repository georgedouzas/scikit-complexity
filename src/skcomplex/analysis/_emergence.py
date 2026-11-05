"""Measure emergence across scales: effective information, causal emergence, integrated criteria, and closure."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import math
import warnings
from collections.abc import Iterator, Sequence
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..engine.runtime import Trajectory
from ..errors import AnalysisError, UnobservedStatesWarning
from ..types import AnyArray, Array, DataKind, IntArray
from ._base import BaseAnalysis
from ._information import InformationAnalysis
from ._markov import MarkovAnalysis
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


class EmergenceAnalysis(BaseAnalysis):
    """Emergence of a macro scale over a micro scale, of one kind.

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

    Attributes:
        matrix_:
            The micro transition matrix, for `'effective-information'` and `'causal'`.

        values_:
            The micro series, one column per micro variable, for `'criteria'` and `'closure'`.

        labels_:
            The label of every micro series.

        information_:
            The effective information in bits, for `'effective-information'`.

        determinism_:
            `log2 n` minus the mean entropy of the rows, high when every state has one successor.

        degeneracy_:
            `log2 n` minus the entropy of the mean row, high when many states share a successor.

        states_:
            The states with observed transitions, the rows the effective information uses.

        partition_:
            The macro label of every micro state, for `'causal'`.

        macro_matrix_:
            The macro transition matrix of the partition, every row summing to one.

        micro_information_:
            The effective information of the micro matrix.

        macro_information_:
            The effective information of the macro matrix.

        gain_:
            The macro minus the micro effective information in bits.

        psi_:
            `I(V_t ; V_t') - Σ_j I(X_t^j ; V_t')`, positive when the macro feature predicts its own future beyond what
            every micro variable does, a sufficient condition for causal emergence.

        delta_:
            `max_j [I(V_t ; X_t'^j) - Σ_i I(X_t^i ; X_t'^j)]`, positive when the macro feature predicts a micro
            variable beyond the micro variables, a sufficient condition for downward causation.

        gamma_:
            `max_j I(V_t ; X_t'^j)`, zero for a causally decoupled feature when `psi_` is positive.

        macro_uncertainty_:
            `H(V_t | V_past)`, the entropy of the macro future given the macro past.

        micro_uncertainty_:
            `H(V_t | X_past)`, the entropy of the macro future given the micro past.

        gap_:
            The information the micro past adds about the macro future, zero for a closed coarse-graining.

    Raises:
        AnalysisError:
            The matrix is not square or has no observed transition, the partition is not valid, or the exhaustive
            search gets more than 10 states.

    Warns:
        UnobservedStatesWarning:
            Some rows of the micro matrix hold no transition, and the warning names their states.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import EmergenceAnalysis
        >>> EmergenceAnalysis('effective-information').prepare(np.eye(4)).run().information_
        2.0
        >>> matrix = np.zeros((4, 4))
        >>> matrix[:3, :3], matrix[3, 3] = 1 / 3, 1.0
        >>> found = EmergenceAnalysis('causal').prepare(matrix).run()
        >>> found.partition_.tolist(), round(found.gain_, 3)
        ([0, 0, 0, 1], 0.189)
        >>> micro = np.random.default_rng(0).integers(0, 4, 5000)
        >>> round(EmergenceAnalysis('closure', partition=[0, 0, 1, 1]).prepare(micro).run().gap_, 2)
        0.0
    """

    def __init__(
        self,
        kind: str = 'causal',
        variables: Sequence[str] | None = None,
        *,
        partition: Partition | None = None,
        weighting: str = 'uniform',
        search: str = 'greedy',
        lag: int = 1,
        history: int = 1,
        estimator: str = 'plugin',
        k: int = 3,
        base: float = 2.0,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.partition = partition
        self.weighting = weighting
        self.search = search
        self.lag = lag
        self.history = history
        self.estimator = estimator
        self.k = k
        self.base = base

    def _check(self) -> None:
        """Validate the kind, the choices, the numeric settings, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_choice('weighting', self.weighting, WEIGHTINGS)
        self._check_choice('search', self.search, SEARCHES)
        self._check_choice('estimator', self.estimator, ESTIMATORS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.kind in {'criteria', 'closure'} and self.partition is None:
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} needs the macro feature. Pass the macro label of every '
                'micro state, or a function of the micro series, as `partition`.'
            )
            raise AnalysisError(error_msg)
        if self.lag < 1 or self.history < 1 or self.k < 1 or self.base <= 0:
            error_msg = (
                f'{type(self).__name__} has lag={self.lag!r}, history={self.history!r}, k={self.k!r}, and '
                f'base={self.base!r}. Use a lag, a history, and neighbours of 1 or more and a positive base.'
            )
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the path kind for a transition matrix, and the series kind for micro series."""
        return DataKind.PATH if self.kind in MATRIX_KINDS else DataKind.SERIES

    def _prepare(self, source: Source) -> None:
        """Read the transition matrix, or the micro series as columns."""
        if self.kind not in MATRIX_KINDS:
            series = self._read_series(source, self.variables)
            self.values_ = series.values
            self.labels_ = series.labels
            return
        if isinstance(source, Trajectory):
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} reads a transition matrix. Pass the matrix or the '
                'transition counts as an array.'
            )
            raise AnalysisError(error_msg)
        self.matrix_ = self._square(source)

    def _run_causal(self) -> None:
        """Find or take the partition, and compare the micro and macro effective information."""
        micro = self.matrix_
        weights = self._weights(micro, self.weighting)
        if self.partition is not None:
            labels = self._blocks_of(self.partition, micro.shape[0])
        else:
            labels = self._greedy(micro, weights) if self.search == 'greedy' else self._exhaustive(micro, weights)
        self.partition_ = labels
        self.macro_matrix_ = self._coarse_matrix(micro, labels, weights)
        self.micro_information_ = self._effective_information(micro)[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UnobservedStatesWarning)
            self.macro_information_ = self._effective_information(self.macro_matrix_)[0]
        self.gain_ = self.macro_information_ - self.micro_information_

    def _get_partition(self) -> Partition:
        """Return the partition of the micro series, raising an error when none is given."""
        if self.partition is None:
            error_msg = f'{type(self).__name__} of kind {self.kind!r} needs the partition of the macro feature.'
            raise AnalysisError(error_msg)
        return self.partition

    def _run_criteria(self) -> None:
        """Compute the criteria Ψ, Δ, and Γ of the macro feature."""
        x, v = self.values_, self._macro_series(self.values_, self._get_partition())
        lag = self.lag
        present_x, future_x, present_v, future_v = x[:-lag], x[lag:], v[:-lag], v[lag:]
        information = partial(self._estimated_information, estimator=self.estimator, k=self.k)
        variables = range(x.shape[1])
        psi = information(present_v, future_v) - sum(information(present_x[:, [j]], future_v) for j in variables)
        gains = [information(present_v, future_x[:, [j]]) for j in variables]
        delta = max(
            gains[j] - sum(information(present_x[:, [i]], future_x[:, [j]]) for i in variables) for j in variables
        )
        scale = math.log(self.base)
        self.psi_ = psi / scale
        self.delta_ = delta / scale
        self.gamma_ = max(gains) / scale

    def _run_closure(self) -> None:
        """Compute the entropies of the macro future given the macro past and the micro past."""
        x = self.values_
        v = self._macro_series(x, self._get_partition())
        future = v[self.history :]
        scale = math.log(self.base)
        self.macro_uncertainty_ = self._conditional_entropy(future, self._lagged(v, self.history)) / scale
        self.micro_uncertainty_ = self._conditional_entropy(future, self._lagged(x, self.history)) / scale
        self.gap_ = self.macro_uncertainty_ - self.micro_uncertainty_

    def _run(self, inputs: object = None) -> None:
        """Compute the emergence of the kind."""
        if self.kind == 'effective-information':
            information, determinism, degeneracy, states = self._effective_information(self.matrix_)
            self.information_ = information
            self.determinism_ = determinism
            self.degeneracy_ = degeneracy
            self.states_ = states
        elif self.kind == 'causal':
            self._run_causal()
        elif self.kind == 'criteria':
            self._run_criteria()
        else:
            self._run_closure()

    @staticmethod
    def _read_columns(data: npt.ArrayLike) -> AnyArray:
        """Return data as one row per sample and one column per variable, a series as a single column.

        Args:
            data:
                A series, or one row per sample with the variables along the remaining axes.

        Returns:
            columns:
                The data as a two-dimensional array.
        """
        array = np.asarray(data)
        return array.reshape(-1, 1) if array.ndim == 1 else array.reshape(array.shape[0], -1)

    @staticmethod
    def _labels_of(partition: Partition, states: npt.NDArray[np.int64]) -> IntArray:
        """Return the macro label of every micro state, from a label array or a function of an array of states."""
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

    @staticmethod
    def _blocks_of(partition: Partition, n_states: int) -> IntArray:
        """Return the label of every state of a matrix, after checking that the labels number the blocks from 0."""
        if not callable(partition) and np.size(partition) != n_states:
            error_msg = f'A partition of {n_states} states holds one label per state, not {np.size(partition)}.'
            raise AnalysisError(error_msg)
        labels = EmergenceAnalysis._labels_of(partition, np.arange(n_states))
        if labels.min() < 0:
            error_msg = f'The labels of a partition are 0, 1, and so on, but got {int(labels.min())}.'
            raise AnalysisError(error_msg)
        n_blocks = int(labels.max()) + 1
        empty = np.setdiff1d(np.arange(n_blocks), labels)
        if empty.size:
            error_msg = f'The partition uses no state for label {int(empty[0])}. Number the blocks 0, 1, and so on.'
            raise AnalysisError(error_msg)
        return labels

    @staticmethod
    def _square(matrix: npt.ArrayLike) -> Array:
        """Return a square matrix of nonnegative floats."""
        array = np.asarray(matrix, dtype=float)
        if array.ndim != MATRIX_DIMENSIONS or array.shape[0] != array.shape[1]:
            error_msg = f'A transition matrix is square, not of shape {array.shape}.'
            raise AnalysisError(error_msg)
        return array

    @staticmethod
    def _weights(matrix: Array, weighting: str) -> Array:
        """Return the weight of every micro state within its block, uniform or stationary."""
        if weighting == 'uniform':
            return np.ones(matrix.shape[0])
        totals = matrix.sum(axis=1, keepdims=True)
        rows = np.divide(matrix, totals, out=np.eye(matrix.shape[0]), where=totals > 0)
        return MarkovAnalysis._stationary_distribution(rows)

    @staticmethod
    def _coarse_matrix(matrix: Array, labels: IntArray, weights: Array) -> Array:
        """Return the macro transition matrix of a partition, the weighted micro rows summed within every block."""
        n_blocks = int(labels.max()) + 1
        membership = np.eye(n_blocks)[labels]
        flows = membership.T @ (weights[:, None] * matrix) @ membership
        totals = flows.sum(axis=1, keepdims=True)
        return np.divide(flows, totals, out=np.zeros_like(flows), where=totals > 0)

    @staticmethod
    def _canonical(labels: npt.ArrayLike) -> IntArray:
        """Return labels renumbered 0, 1, and so on in the order of their first state."""
        _, first, inverse = np.unique(np.asarray(labels), return_index=True, return_inverse=True)
        return np.argsort(np.argsort(first))[inverse].astype(np.int64)

    @staticmethod
    def _effective_information(matrix: Array) -> tuple[float, float, float, npt.NDArray[np.int64]]:
        """Return the effective information, the determinism, the degeneracy, and the observed states of a matrix."""
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

    @staticmethod
    def _macro_information(matrix: Array, labels: IntArray, weights: Array) -> float:
        """Return the effective information of the macro matrix of a partition, without warning on unobserved blocks."""
        macro = EmergenceAnalysis._coarse_matrix(matrix, labels, weights)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UnobservedStatesWarning)
            return EmergenceAnalysis._effective_information(macro)[0]

    @staticmethod
    def _greedy(matrix: Array, weights: Array) -> IntArray:
        """Return the partition reached by merging the pair of blocks with the largest gain until no merge gains."""
        labels = np.arange(matrix.shape[0], dtype=np.int64)
        best = EmergenceAnalysis._macro_information(matrix, labels, weights)
        while int(labels.max()) > 0:
            merges = [
                EmergenceAnalysis._canonical(np.where(labels == second, first, labels))
                for first, second in itertools.combinations(range(int(labels.max()) + 1), 2)
            ]
            scores = [EmergenceAnalysis._macro_information(matrix, merged, weights) for merged in merges]
            index = int(np.argmax(scores))
            if scores[index] <= best + TOLERANCE:
                break
            labels, best = merges[index], scores[index]
        return labels

    @staticmethod
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

    @staticmethod
    def _exhaustive(matrix: Array, weights: Array) -> IntArray:
        """Return the partition with the largest macro effective information, the finest of equal ones."""
        if matrix.shape[0] > MAX_EXHAUSTIVE_STATES:
            error_msg = (
                f'The exhaustive search covers at most {MAX_EXHAUSTIVE_STATES} states, not {matrix.shape[0]}. '
                "Use search='greedy'."
            )
            raise AnalysisError(error_msg)
        best, best_score = np.arange(matrix.shape[0], dtype=np.int64), -np.inf
        for labels in EmergenceAnalysis._set_partitions(matrix.shape[0]):
            score = EmergenceAnalysis._macro_information(matrix, labels, weights)
            blocks = int(labels.max())
            if score > best_score + TOLERANCE or (score > best_score - TOLERANCE and blocks > int(best.max())):
                best, best_score = labels, max(score, best_score)
        return best

    @staticmethod
    def _macro_series(micro: npt.NDArray[np.generic], partition: Partition) -> npt.NDArray[np.generic]:
        """Return the macro series of a micro series under a partition given as labels or as a function."""
        if callable(partition):
            return EmergenceAnalysis._read_columns(partition(micro))
        return EmergenceAnalysis._read_columns(np.asarray(partition)[np.asarray(micro).ravel().astype(np.int64)])

    @staticmethod
    def _lagged(columns: npt.NDArray[np.generic], history: int) -> npt.NDArray[np.generic]:
        """Return the `history` rows before every time from `history` on, side by side."""
        size = columns.shape[0]
        return np.column_stack([columns[history - step : size - step] for step in range(1, history + 1)])

    @staticmethod
    def _conditional_entropy(target: npt.NDArray[np.generic], given: npt.NDArray[np.generic]) -> float:
        """Return the plug-in entropy of the target given other columns in nats."""
        return EmergenceAnalysis._plugin_entropy(np.column_stack([target, given])) - EmergenceAnalysis._plugin_entropy(
            given,
        )

    @staticmethod
    def _plugin_entropy(columns: AnyArray) -> float:
        """Return the plug-in entropy of the rows of discrete columns in nats."""
        return float(stats.entropy(np.bincount(InformationAnalysis._row_codes(columns))))

    @staticmethod
    def _estimated_information(first: AnyArray, second: AnyArray, estimator: str, k: int) -> float:
        """Return the mutual information of two groups of columns in nats by plug-in, Gaussian, or kNN estimates."""
        if estimator == 'plugin':
            joint = np.column_stack([InformationAnalysis._row_codes(first), InformationAnalysis._row_codes(second)])
            return (
                EmergenceAnalysis._plugin_entropy(first)
                + EmergenceAnalysis._plugin_entropy(second)
                - EmergenceAnalysis._plugin_entropy(joint)
            )
        x, y = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        if estimator == 'gaussian':
            joint_value = InformationAnalysis._log_determinant(np.column_stack([x, y]))
            return (InformationAnalysis._log_determinant(x) + InformationAnalysis._log_determinant(y) - joint_value) / 2
        return sum(InformationAnalysis._knn_chain(x[:, index], y, k, x[:, :index]) for index in range(x.shape[1]))
