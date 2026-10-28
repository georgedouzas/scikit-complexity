"""Measure emergence across scales: effective information, causal emergence, integrated criteria, and closure."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
import itertools
import math
import warnings
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..core.errors import AnalysisError, UnobservedStatesWarning
from ..engine.runtime import Trajectory
from ._base import BaseAnalysis, Source
from ._information import _estimated_information, _plugin_entropy
from ._kinds import DataKind
from ._transitions import stationary_distribution
from ._utils import sample_columns

type Array = npt.NDArray[np.float64]
type Labels = npt.NDArray[np.int64]
type Partition = npt.ArrayLike | Callable[[npt.NDArray[np.generic]], npt.ArrayLike]

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


def _is_changed(value: object, default: object) -> bool:
    """Return whether a setting differs from its default, comparing arrays and functions by identity to `None`."""
    if default is None:
        return value is not None
    return bool(value != default)


def _labels_of(partition: Partition, states: npt.NDArray[np.int64]) -> Labels:
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


def _canonical(labels: npt.ArrayLike) -> Labels:
    """Return labels renumbered 0, 1, and so on in the order of their first state."""
    _, first, inverse = np.unique(np.asarray(labels), return_index=True, return_inverse=True)
    return np.argsort(np.argsort(first))[inverse].astype(np.int64)


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


def _macro_information(matrix: Array, labels: Labels, weights: Array) -> float:
    """Return the effective information of the macro matrix of a partition, without warnings on unobserved blocks."""
    macro = _coarse_matrix(matrix, labels, weights)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnobservedStatesWarning)
        return _effective_information(macro)[0]


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


def _macro_series(micro: npt.NDArray[np.generic], partition: Partition) -> npt.NDArray[np.generic]:
    """Return the macro series of a micro series under a partition given as labels or as a function."""
    if callable(partition):
        return sample_columns(partition(micro))
    return sample_columns(np.asarray(partition)[np.asarray(micro).ravel().astype(np.int64)])


def _lagged(columns: npt.NDArray[np.generic], history: int) -> npt.NDArray[np.generic]:
    """Return the `history` rows before every time from `history` on, side by side."""
    size = columns.shape[0]
    return np.column_stack([columns[history - step : size - step] for step in range(1, history + 1)])


def _conditional_entropy(target: npt.NDArray[np.generic], given: npt.NDArray[np.generic]) -> float:
    """Return the plug-in entropy of the target given other columns in nats."""
    return _plugin_entropy(np.column_stack([target, given])) - _plugin_entropy(given)


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
        >>> EmergenceAnalysis('effective-information').check().prepare(np.eye(4)).run().information_
        2.0
        >>> matrix = np.zeros((4, 4))
        >>> matrix[:3, :3], matrix[3, 3] = 1 / 3, 1.0
        >>> found = EmergenceAnalysis('causal').check().prepare(matrix).run()
        >>> found.partition_.tolist(), round(found.gain_, 3)
        ([0, 0, 0, 1], 0.189)
        >>> micro = np.random.default_rng(0).integers(0, 4, 5000)
        >>> round(EmergenceAnalysis('closure', partition=[0, 0, 1, 1]).check().prepare(micro).run().gap_, 2)
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

    def _check_choice(self, name: str, value: str, choices: tuple[str, ...]) -> None:
        """Raise an error for a setting that is not one of its choices."""
        if value not in choices:
            error_msg = f'{type(self).__name__} has {name}={value!r}. Use one of {", ".join(choices)}.'
            raise AnalysisError(error_msg)

    def _check_ignored(self) -> None:
        """Raise an error for a setting changed from its default that the kind does not read."""
        defaults = {
            name: parameter.default for name, parameter in inspect.signature(type(self).__init__).parameters.items()
        }
        read = SETTINGS[self.kind] | {'kind', 'variables'}
        ignored = [
            name for name, value in self.get_params().items() if name not in read and _is_changed(value, defaults[name])
        ]
        if ignored:
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} does not read {", ".join(ignored)}. It reads '
                f'{", ".join(sorted(SETTINGS[self.kind])) or "no other setting"}.'
            )
            raise AnalysisError(error_msg)

    def _check(self) -> Self:
        """Validate the kind, the choices, the numeric settings, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_choice('weighting', self.weighting, WEIGHTINGS)
        self._check_choice('search', self.search, SEARCHES)
        self._check_choice('estimator', self.estimator, ESTIMATORS)
        self._check_ignored()
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
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the path kind for a transition matrix, and the series kind for micro series."""
        return DataKind.PATH if self.kind in MATRIX_KINDS else DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the transition matrix, or the micro series as columns."""
        if self.kind not in MATRIX_KINDS:
            series = self._read_series(source, self.variables)
            self.values_ = series.values
            self.labels_ = series.labels
            return self
        if isinstance(source, Trajectory):
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} reads a transition matrix. Pass the matrix or the '
                'transition counts as an array.'
            )
            raise AnalysisError(error_msg)
        self.matrix_ = _square(source)
        return self

    def _run_causal(self) -> None:
        """Find or take the partition, and compare the micro and macro effective information."""
        micro = self.matrix_
        weights = _weights(micro, self.weighting)
        if self.partition is not None:
            labels = _blocks_of(self.partition, micro.shape[0])
        else:
            labels = _greedy(micro, weights) if self.search == 'greedy' else _exhaustive(micro, weights)
        self.partition_ = labels
        self.macro_matrix_ = _coarse_matrix(micro, labels, weights)
        self.micro_information_ = _effective_information(micro)[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UnobservedStatesWarning)
            self.macro_information_ = _effective_information(self.macro_matrix_)[0]
        self.gain_ = self.macro_information_ - self.micro_information_

    def _run_criteria(self) -> None:
        """Compute the criteria Ψ, Δ, and Γ of the macro feature."""
        x, v = self.values_, _macro_series(self.values_, self.partition)
        lag = self.lag
        present_x, future_x, present_v, future_v = x[:-lag], x[lag:], v[:-lag], v[lag:]
        information = partial(_estimated_information, estimator=self.estimator, k=self.k)
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
        v = _macro_series(x, self.partition)
        future = v[self.history :]
        scale = math.log(self.base)
        self.macro_uncertainty_ = _conditional_entropy(future, _lagged(v, self.history)) / scale
        self.micro_uncertainty_ = _conditional_entropy(future, _lagged(x, self.history)) / scale
        self.gap_ = self.macro_uncertainty_ - self.micro_uncertainty_

    def _run(self) -> Self:
        """Compute the emergence of the kind."""
        if self.kind == 'effective-information':
            information, determinism, degeneracy, states = _effective_information(self.matrix_)
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
        return self
