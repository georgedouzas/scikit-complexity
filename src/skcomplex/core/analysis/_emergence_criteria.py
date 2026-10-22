"""Test whether a macro feature is emergent by the integrated information criteria and by closure."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt

from ._kinds import DataKind, analysis
from ._multivariate import MULTIVARIATE_ESTIMATORS, _check_estimator, _mutual_information, _plugin_entropy
from ._utils import sample_columns

type Partition = npt.ArrayLike | Callable[[npt.NDArray[Any]], npt.ArrayLike]


def _macro_series(micro: npt.NDArray[Any], partition: Partition) -> npt.NDArray[Any]:
    """Return the macro series of a micro series under a partition given as labels or as a function."""
    if callable(partition):
        return sample_columns(partition(micro))
    return sample_columns(np.asarray(partition)[np.asarray(micro).ravel()])


def _lagged(columns: npt.NDArray[Any], history: int) -> npt.NDArray[Any]:
    """Return the `history` rows before every time from `history` on, side by side."""
    size = columns.shape[0]
    return np.column_stack([columns[history - step : size - step] for step in range(1, history + 1)])


def _conditional_entropy(target: npt.NDArray[Any], given: npt.NDArray[Any]) -> float:
    """Return the plug-in entropy of the target given other columns in nats."""
    return _plugin_entropy(np.column_stack([target, given])) - _plugin_entropy(given)


@dataclass(frozen=True)
class EmergenceCriteria:
    """Integrated information criteria of a macro feature with the minimum mutual information redundancy.

    Args:
        psi:
            `I(V_t ; V_t') - Σ_j I(X_t^j ; V_t')`, positive when the macro feature predicts its own future beyond what
            every micro variable does, a sufficient condition for causal emergence.

        delta:
            `max_j [I(V_t ; X_t'^j) - Σ_i I(X_t^i ; X_t'^j)]`, positive when the macro feature predicts a micro
            variable beyond the micro variables, a sufficient condition for downward causation.

        gamma:
            `max_j I(V_t ; X_t'^j)`, zero for a causally decoupled feature when `psi` is positive.
    """

    psi: float
    delta: float
    gamma: float


@dataclass(frozen=True)
class Closure:
    """How well the macro and the micro past predict the macro future of a coarse-graining.

    Args:
        macro_uncertainty:
            `H(V_t | V_past)`, the entropy of the macro future given the macro past.

        micro_uncertainty:
            `H(V_t | X_past)`, the entropy of the macro future given the micro past.
    """

    macro_uncertainty: float
    micro_uncertainty: float

    @property
    def gap(self) -> float:
        """Information the micro past adds about the macro future, zero for a closed coarse-graining."""
        return self.macro_uncertainty - self.micro_uncertainty


@analysis(DataKind.SERIES)
def emergence_criteria(
    micro: npt.ArrayLike,
    macro: npt.ArrayLike,
    *,
    lag: int = 1,
    estimator: str = 'plugin',
    base: float = 2.0,
    k: int = 3,
) -> EmergenceCriteria:
    """Return the criteria Ψ, Δ, and Γ of Rosas et al. (2020) for a macro feature of a micro system.

    Args:
        micro:
            One row per time and one column per micro variable.

        macro:
            The macro feature at every time, a series or one column per macro variable.

        lag:
            The number of steps between the present and the future.

        estimator:
            `'plugin'` for the frequencies of symbols, `'gaussian'` for the closed form of a fitted Gaussian, or
            `'knn'` for nearest-neighbour conditional mutual informations.

        base:
            The base of the logarithm, 2 for bits.

        k:
            The number of neighbours of the `'knn'` estimator.

    Returns:
        criteria:
            The criteria Ψ, Δ, and Γ.

    Raises:
        AnalysisError:
            The estimator is unknown.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import emergence_criteria
        >>> bits = np.random.default_rng(0).integers(0, 2, (5000, 3))
        >>> bool(abs(emergence_criteria(bits, bits.sum(axis=1) % 2).psi) < 0.01)
        True
    """
    _check_estimator(estimator, MULTIVARIATE_ESTIMATORS)
    x, v = sample_columns(micro), sample_columns(macro)
    present_x, future_x, present_v, future_v = x[:-lag], x[lag:], v[:-lag], v[lag:]

    information = partial(_mutual_information, estimator=estimator, k=k)
    variables = range(x.shape[1])
    psi = information(present_v, future_v) - sum(information(present_x[:, [j]], future_v) for j in variables)
    gains = [information(present_v, future_x[:, [j]]) for j in variables]
    delta = max(gains[j] - sum(information(present_x[:, [i]], future_x[:, [j]]) for i in variables) for j in variables)
    scale = math.log(base)
    return EmergenceCriteria(psi=psi / scale, delta=delta / scale, gamma=max(gains) / scale)


@analysis(DataKind.SERIES)
def closure(micro: npt.ArrayLike, partition: Partition, *, history: int = 1, base: float = 2.0) -> Closure:
    """Return how well the macro past and the micro past of a coarse-graining predict the macro future.

    The coarse-graining is closed when the micro past adds nothing, so the gap between the two conditional entropies
    is zero up to the bias of the plug-in estimator.

    Args:
        micro:
            The micro states, a series of symbols or one row per time and one column per micro variable.

        partition:
            The macro label of every micro state, indexed by the integer micro state, or a function from the micro
            series to the macro series.

        history:
            The number of past steps conditioned on.

        base:
            The base of the logarithm, 2 for bits.

    Returns:
        closure:
            The conditional entropies of the macro future and their gap.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import closure
        >>> micro = np.random.default_rng(0).integers(0, 4, 5000)
        >>> round(closure(micro, [0, 0, 1, 1]).gap, 2)
        0.0
    """
    x = sample_columns(micro)
    v = _macro_series(np.asarray(micro), partition)
    future = v[history:]
    scale = math.log(base)
    return Closure(
        macro_uncertainty=_conditional_entropy(future, _lagged(v, history)) / scale,
        micro_uncertainty=_conditional_entropy(future, _lagged(x, history)) / scale,
    )
