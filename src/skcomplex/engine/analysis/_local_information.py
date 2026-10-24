"""Measure information sample by sample: local entropy, local mutual information, and local transfer entropy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from ._kinds import DataKind, analysis
from ._multivariate import _check_estimator, _row_codes
from ._utils import real_array, sample_columns

type Array = npt.NDArray[np.float64]
LOCAL_ESTIMATORS = ('plugin', 'gaussian')


def _log_probabilities(columns: npt.NDArray[Any], estimator: str) -> Array:
    """Return the logarithm of the probability, or of the fitted Gaussian density, of every row in nats."""
    if estimator == 'plugin':
        codes = _row_codes(columns)
        counts = np.bincount(codes).astype(float)
        return np.log(counts[codes] / codes.size)
    values = real_array(columns, 'values')
    covariance = np.atleast_2d(np.cov(values, rowvar=False, bias=True))
    density = stats.multivariate_normal(values.mean(axis=0), covariance)
    return np.asarray(density.logpdf(values), dtype=float).reshape(-1)


@analysis(DataKind.SERIES)
def local_entropy(values: npt.ArrayLike, *, estimator: str = 'plugin', base: float = 2.0) -> Array:
    """Return the local entropy `-log p(x)` of every sample, whose mean is the entropy.

    Args:
        values:
            A series, or one row per sample and one column per variable, symbols for the plug-in estimator.

        estimator:
            `'plugin'` for the frequencies of symbols, or `'gaussian'` for the density of a fitted Gaussian.

        base:
            The base of the logarithm, 2 for bits.

    Returns:
        entropies:
            One local entropy per sample.

    Raises:
        AnalysisError:
            The estimator is unknown.

    Examples:
        >>> from skcomplex.core.analysis import local_entropy
        >>> local_entropy([0, 0, 0, 1]).round(3).tolist()
        [0.415, 0.415, 0.415, 2.0]
    """
    _check_estimator(estimator, LOCAL_ESTIMATORS)
    return -_log_probabilities(sample_columns(values), estimator) / math.log(base)


@analysis(DataKind.SERIES)
def local_mutual_information(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    *,
    estimator: str = 'plugin',
    base: float = 2.0,
) -> Array:
    """Return the local mutual information `log p(x, y) / (p(x) p(y))` of every sample, whose mean is the global one.

    Args:
        first:
            The first series, or one row per sample and one column per variable.

        second:
            The second series, aligned with the first.

        estimator:
            `'plugin'` for the frequencies of symbols, or `'gaussian'` for the densities of a fitted Gaussian.

        base:
            The base of the logarithm, 2 for bits.

    Returns:
        information:
            One local mutual information per sample, negative where the variables mislead about each other.

    Raises:
        AnalysisError:
            The estimator is unknown.

    Examples:
        >>> from skcomplex.core.analysis import local_mutual_information
        >>> local_mutual_information([0, 0, 1, 1], [0, 0, 1, 0]).round(3).tolist()
        [0.415, 0.415, 1.0, -0.585]
    """
    _check_estimator(estimator, LOCAL_ESTIMATORS)
    x, y = sample_columns(first), sample_columns(second)
    joint = _log_probabilities(np.column_stack([x, y]), estimator)
    return (joint - _log_probabilities(x, estimator) - _log_probabilities(y, estimator)) / math.log(base)


@analysis(DataKind.SERIES)
def local_transfer_entropy(
    source: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    history: int = 1,
    lag: int = 1,
    estimator: str = 'plugin',
    base: float = 2.0,
) -> Array:
    """Return the local transfer entropy from a source to a target at every time, whose mean is the global one.

    It is `log p(y_t | y_past, x_{t-lag}) / p(y_t | y_past)` at every time from `max(history, lag)` on.

    Args:
        source:
            The source series.

        target:
            The target series, aligned with the source.

        history:
            The number of past target values conditioned on.

        lag:
            The delay of the source.

        estimator:
            `'plugin'` for the frequencies of symbols, or `'gaussian'` for the densities of a fitted Gaussian.

        base:
            The base of the logarithm, 2 for bits.

    Returns:
        transfer:
            One local transfer entropy per time from `max(history, lag)` on.

    Raises:
        AnalysisError:
            The estimator is unknown.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import local_transfer_entropy
        >>> x = np.random.default_rng(0).integers(0, 2, 10000)
        >>> round(float(local_transfer_entropy(x, np.roll(x, 1)).mean()), 1)
        1.0
    """
    _check_estimator(estimator, LOCAL_ESTIMATORS)
    x, y = np.asarray(source).ravel(), np.asarray(target).ravel()
    start = max(history, lag)
    future = y[start:, None]
    past = np.column_stack([y[start - step : y.size - step] for step in range(1, history + 1)])
    driver = x[start - lag : x.size - lag, None]
    terms = (
        _log_probabilities(np.column_stack([future, past, driver]), estimator)
        - _log_probabilities(np.column_stack([past, driver]), estimator)
        - _log_probabilities(np.column_stack([future, past]), estimator)
        + _log_probabilities(past, estimator)
    )
    return terms / math.log(base)
