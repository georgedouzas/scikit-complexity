"""Measure redundancy and synergy among many variables: total correlations, O-information, and its decomposition."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import ennemi
import numpy as np
import numpy.typing as npt

from ..base import AnalysisError
from ._kinds import DataKind, analysis
from ._lazy import module
from ._utils import real_array, sample_columns

type Array = npt.NDArray[np.float64]
MULTIVARIATE_ESTIMATORS = ('plugin', 'gaussian', 'knn')
REDUNDANCIES = {
    'mmi': 'PID_MMI',
    'williams_beer': 'PID_WB',
    'broja': 'PID_BROJA',
    'ccs': 'PID_CCS',
    'pm': 'PID_PM',
    'dep': 'PID_dep',
}


def _check_estimator(estimator: str, estimators: tuple[str, ...]) -> None:
    """Raise when the estimator is not one of the given ones."""
    if estimator not in estimators:
        error_msg = f'Unknown estimator {estimator!r}. Use one of {", ".join(estimators)}.'
        raise AnalysisError(error_msg)


def _row_codes(columns: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
    """Return one integer code per distinct row."""
    _, codes = np.unique(columns, axis=0, return_inverse=True)
    return np.asarray(codes, dtype=np.int64).ravel()


def _plugin_entropy(columns: npt.NDArray[Any]) -> float:
    """Return the plug-in entropy of the rows of discrete columns in nats."""
    counts = np.bincount(_row_codes(columns)).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _log_determinant(columns: Array) -> float:
    """Return the logarithm of the determinant of the sample covariance of the columns."""
    if columns.shape[1] == 0:
        return 0.0
    _, value = np.linalg.slogdet(np.atleast_2d(np.cov(columns, rowvar=False, bias=True)))
    return float(value)


def _knn_chain(target: Array, others: Array, k: int, given: Array | None = None) -> float:
    """Return `I(target ; others | given)` in nats as a chain of nearest-neighbour conditional mutual informations."""
    condition = np.empty((target.size, 0)) if given is None else given
    total = 0.0
    for column in range(others.shape[1]):
        cond = np.column_stack([condition, others[:, :column]])
        estimate = ennemi.estimate_mi(target, others[:, column], cond=cond if cond.shape[1] else None, k=k)
        total += float(np.asarray(estimate).item())
    return total


def _mutual_information(first: npt.NDArray[Any], second: npt.NDArray[Any], estimator: str, k: int) -> float:
    """Return the mutual information of two groups of columns in nats."""
    if estimator == 'plugin':
        joint = np.column_stack([_row_codes(first), _row_codes(second)])
        return _plugin_entropy(first) + _plugin_entropy(second) - _plugin_entropy(joint)
    x, y = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if estimator == 'gaussian':
        joint_value = _log_determinant(np.column_stack([x, y]))
        return (_log_determinant(x) + _log_determinant(y) - joint_value) / 2
    return sum(_knn_chain(x[:, index], y, k, x[:, :index]) for index in range(x.shape[1]))


def _distribution(columns: npt.NDArray[Any]) -> object:
    """Return the `dit` distribution of the empirical frequencies of the rows."""
    counts = Counter(tuple(row) for row in columns.tolist())
    total = sum(counts.values())
    return module('dit').Distribution(list(counts), [count / total for count in counts.values()])


def _total(columns: npt.NDArray[Any], estimator: str, k: int) -> float:
    """Return the total correlation of the columns in nats."""
    if estimator == 'plugin':
        return float(module('dit').multivariate.total_correlation(_distribution(columns))) * math.log(2)
    values = real_array(columns, 'data')
    if estimator == 'gaussian':
        return -_log_determinant(values / values.std(axis=0)) / 2
    return sum(_knn_chain(values[:, index], values[:, :index], k) for index in range(1, values.shape[1]))


def _dual(columns: npt.NDArray[Any], estimator: str, k: int) -> float:
    """Return the dual total correlation of the columns in nats."""
    if estimator == 'plugin':
        return float(module('dit').multivariate.dual_total_correlation(_distribution(columns))) * math.log(2)
    values = real_array(columns, 'data')
    if estimator == 'gaussian':
        precision = np.linalg.inv(np.atleast_2d(np.cov(values, rowvar=False, bias=True)))
        return (_log_determinant(values) + float(np.sum(np.log(np.diag(precision))))) / 2
    parts = sum(_knn_chain(values[:, index], np.delete(values, index, axis=1), k) for index in range(values.shape[1]))
    return parts - _total(values, estimator, k)


@analysis(DataKind.SERIES)
def total_correlation(data: npt.ArrayLike, *, estimator: str = 'plugin', base: float = 2.0, k: int = 3) -> float:
    """Return the total correlation `Σ H(X_i) - H(X)` of many variables, the information shared at any order.

    Args:
        data:
            One row per sample and one column per variable, symbols for the plug-in estimator and real values for
            the others.

        estimator:
            `'plugin'` for the frequencies of symbols, `'gaussian'` for the closed form of a fitted Gaussian, or
            `'knn'` for chains of nearest-neighbour conditional mutual informations.

        base:
            The base of the logarithm, 2 for bits.

        k:
            The number of neighbours of the `'knn'` estimator.

    Returns:
        correlation:
            The total correlation, 0 for independent variables.

    Raises:
        AnalysisError:
            The estimator is unknown.

    Examples:
        >>> from skcomplex.core.analysis import total_correlation
        >>> round(total_correlation([[0, 0, 0], [1, 1, 1]]), 6)
        2.0
    """
    _check_estimator(estimator, MULTIVARIATE_ESTIMATORS)
    return _total(sample_columns(data), estimator, k) / math.log(base)


@analysis(DataKind.SERIES)
def dual_total_correlation(
    data: npt.ArrayLike,
    *,
    estimator: str = 'plugin',
    base: float = 2.0,
    k: int = 3,
) -> float:
    """Return the dual total correlation `H(X) - Σ H(X_i | X_-i)`, the information that two or more variables share.

    Args:
        data:
            One row per sample and one column per variable, symbols for the plug-in estimator and real values for
            the others.

        estimator:
            `'plugin'` for the frequencies of symbols, `'gaussian'` for the closed form of a fitted Gaussian, or
            `'knn'` for chains of nearest-neighbour conditional mutual informations.

        base:
            The base of the logarithm, 2 for bits.

        k:
            The number of neighbours of the `'knn'` estimator.

    Returns:
        correlation:
            The dual total correlation, 0 for independent variables.

    Raises:
        AnalysisError:
            The estimator is unknown.

    Examples:
        >>> from skcomplex.core.analysis import dual_total_correlation
        >>> round(dual_total_correlation([[0, 0, 0], [1, 1, 1]]), 6)
        1.0
    """
    _check_estimator(estimator, MULTIVARIATE_ESTIMATORS)
    return _dual(sample_columns(data), estimator, k) / math.log(base)


@analysis(DataKind.SERIES)
def o_information(data: npt.ArrayLike, *, estimator: str = 'plugin', base: float = 2.0, k: int = 3) -> float:
    """Return the O-information, total minus dual total correlation, positive for redundancy and negative for synergy.

    Args:
        data:
            One row per sample and one column per variable, symbols for the plug-in estimator and real values for
            the others.

        estimator:
            `'plugin'` for the frequencies of symbols, `'gaussian'` for the closed form of a fitted Gaussian, or
            `'knn'` for chains of nearest-neighbour conditional mutual informations.

        base:
            The base of the logarithm, 2 for bits.

        k:
            The number of neighbours of the `'knn'` estimator.

    Returns:
        information:
            The O-information.

    Raises:
        AnalysisError:
            The estimator is unknown.

    Examples:
        >>> from skcomplex.core.analysis import o_information
        >>> xor = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
        >>> round(o_information(xor), 6), round(o_information([[0, 0, 0], [1, 1, 1]]), 6)
        (-1.0, 1.0)
    """
    _check_estimator(estimator, MULTIVARIATE_ESTIMATORS)
    columns = sample_columns(data)
    return (_total(columns, estimator, k) - _dual(columns, estimator, k)) / math.log(base)


@analysis(DataKind.SERIES)
def o_information_gradients(
    data: npt.ArrayLike,
    *,
    estimator: str = 'plugin',
    base: float = 2.0,
    k: int = 3,
) -> Array:
    """Return the O-information gradient of every variable, the O-information it adds to the others.

    Args:
        data:
            One row per sample and one column per variable, symbols for the plug-in estimator and real values for
            the others.

        estimator:
            `'plugin'` for the frequencies of symbols, `'gaussian'` for the closed form of a fitted Gaussian, or
            `'knn'` for chains of nearest-neighbour conditional mutual informations.

        base:
            The base of the logarithm, 2 for bits.

        k:
            The number of neighbours of the `'knn'` estimator.

    Returns:
        gradients:
            `Ω(X) - Ω(X_-i)` for every variable `i`, negative where the variable takes part in synergy.

    Raises:
        AnalysisError:
            The estimator is unknown.

    Examples:
        >>> from skcomplex.core.analysis import o_information_gradients
        >>> xor = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
        >>> o_information_gradients(xor).round(6).tolist()
        [-1.0, -1.0, -1.0]
    """
    _check_estimator(estimator, MULTIVARIATE_ESTIMATORS)
    columns = sample_columns(data)
    whole = o_information(columns, estimator=estimator, base=base, k=k)
    parts = [
        o_information(np.delete(columns, index, axis=1), estimator=estimator, base=base, k=k)
        for index in range(columns.shape[1])
    ]
    return whole - np.asarray(parts, dtype=float)


@dataclass(frozen=True)
class PartialInformation:
    """Partial information decomposition of what two sources tell about a target.

    Args:
        redundancy:
            The information both sources give.

        unique_first:
            The information only the first source gives.

        unique_second:
            The information only the second source gives.

        synergy:
            The information only the two sources together give.
    """

    redundancy: float
    unique_first: float
    unique_second: float
    synergy: float

    @property
    def total(self) -> float:
        """Mutual information of the two sources together with the target, the sum of the four parts."""
        return self.redundancy + self.unique_first + self.unique_second + self.synergy


@analysis(DataKind.SERIES)
def partial_information(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    redundancy: str = 'mmi',
    base: float = 2.0,
) -> PartialInformation:
    """Return the partial information decomposition of what two sources of symbols tell about a target.

    Args:
        first:
            The symbols of the first source.

        second:
            The symbols of the second source, aligned with the first.

        target:
            The symbols of the target, aligned with the sources.

        redundancy:
            The redundancy measure of `dit`: `'mmi'` for the minimum mutual information, `'williams_beer'` for
            `I_min`, `'broja'`, `'ccs'`, `'pm'`, or `'dep'`.

        base:
            The base of the logarithm, 2 for bits.

    Returns:
        decomposition:
            The redundant, unique, and synergistic information.

    Raises:
        AnalysisError:
            The redundancy measure is unknown.

    Examples:
        >>> from skcomplex.core.analysis import partial_information
        >>> parts = partial_information([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0])
        >>> round(parts.synergy, 6), round(parts.redundancy, 6)
        (1.0, 0.0)
    """
    if redundancy not in REDUNDANCIES:
        error_msg = f'Unknown redundancy measure {redundancy!r}. Use one of {", ".join(REDUNDANCIES)}.'
        raise AnalysisError(error_msg)
    columns = np.column_stack([np.asarray(values).ravel() for values in (first, second, target)])
    measure = getattr(module('dit').pid, REDUNDANCIES[redundancy])
    lattice = measure(_distribution(columns), [[0], [1]], [2])
    scale = math.log(2) / math.log(base)
    return PartialInformation(
        redundancy=float(lattice.get_pi(((0,), (1,)))) * scale,
        unique_first=float(lattice.get_pi(((0,),))) * scale,
        unique_second=float(lattice.get_pi(((1,),))) * scale,
        synergy=float(lattice.get_pi(((0, 1),))) * scale,
    )
