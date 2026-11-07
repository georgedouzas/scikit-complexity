"""Measure information sample by sample: local entropy, local mutual information, and local transfer entropy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..errors import AnalysisError
from ..types import AnyArray, Array
from ._base import _check_choice, _check_settings, _read_real, _read_series
from ._information import _row_codes
from .types import Source

KINDS = ('entropy', 'mutual', 'transfer')
SETTINGS = {
    'entropy': {'estimator', 'base'},
    'mutual': {'estimator', 'base'},
    'transfer': {'history', 'lag', 'estimator', 'base'},
}
"""Settings every kind reads, besides `kind` and `variables`."""

ESTIMATORS = ('plugin', 'gaussian')


@dataclass(frozen=True)
class LocalInformationResult:
    """Local information of every series or of every pair of series of a source.

    Args:
        values:
            The series, one per column.

        labels:
            The label of every series.

        information:
            One row of local entropies per series for `'entropy'`, and for `'mutual'` and `'transfer'` one row of
            local values from the series of every first index to the series of every second index, `nan` from a series
            to itself.
    """

    values: Array
    labels: list[str]
    information: Array


def local_information(
    source: Source,
    *,
    kind: str = 'entropy',
    variables: Sequence[str] | None = None,
    history: int = 1,
    lag: int = 1,
    estimator: str = 'plugin',
    base: float = 2.0,
) -> LocalInformationResult:
    """Return the local information of every series or of every pair of series of a source, of one kind.

    The kinds measure every series or every pair of series:

    - `'entropy'`: the local entropy `-log p(x)` of every sample of every series.
    - `'mutual'`: the local mutual information `log p(x, y) / (p(x) p(y))` of every sample of every pair of series,
      negative where the series mislead about each other.
    - `'transfer'`: the local transfer entropy `log p(y_t | y_past, x_{t-lag}) / p(y_t | y_past)` from every series
      to every series at every time from `max(history, lag)` on.

    The probabilities are the frequencies of symbols, or the densities of a fitted Gaussian.

    Args:
        source:
            The trajectory, or an array with one row per time and one column per series.

        kind:
            The local information to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        history:
            The number of past target values of the transfer entropy.

        lag:
            The delay of the source of the transfer entropy.

        estimator:
            `'plugin'` for the frequencies of symbols, or `'gaussian'` for the densities of a fitted Gaussian.

        base:
            The base of the logarithm, 2 for bits.

    Returns:
        result:
            The series and their local information.

    Raises:
        AnalysisError:
            A choice is unknown, a setting the kind does not read is changed, or a setting is out of range.

    Examples:
        >>> from skcomplex.core.analysis import local_information
        >>> local = local_information([[0, 0], [0, 0], [1, 1], [1, 0]], kind='mutual')
        >>> local.information[0, 1].round(3).tolist()
        [0.415, 0.415, 1.0, -0.585]
    """
    _check_choice('local_information', 'kind', kind, KINDS)
    _check_choice('local_information', 'estimator', estimator, ESTIMATORS)
    _check_settings(
        local_information,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        history=history,
        lag=lag,
        estimator=estimator,
        base=base,
    )
    if history < 1 or lag < 1 or base <= 0:
        error_msg = (
            f'local_information has history={history!r}, lag={lag!r}, and base={base!r}. Use a history and a lag of 1 '
            'or more and a positive base.'
        )
        raise AnalysisError(error_msg)
    series = _read_series(source, variables)
    columns = series.values.T
    local: list[Array] | list[list[Array]]
    if kind == 'entropy':
        local = [-_log_probabilities(column[:, None], estimator) for column in columns]
    else:
        local = [
            [
                _compute_pair(kind, first, second, history=history, lag=lag, estimator=estimator, same=row == column)
                for column, second in enumerate(columns)
            ]
            for row, first in enumerate(columns)
        ]
    return LocalInformationResult(series.values, series.labels, np.asarray(local, dtype=float) / math.log(base))


def _compute_pair(
    kind: str,
    first: AnyArray,
    second: AnyArray,
    *,
    history: int,
    lag: int,
    estimator: str,
    same: bool,
) -> Array:
    """Return the local mutual information or the local transfer entropy from one series to another in nats."""
    if same:
        size = first.size - max(history, lag) if kind == 'transfer' else first.size
        return np.full(size, np.nan)
    if kind == 'transfer':
        return _local_transfer_entropy(first, second, history, lag, estimator)
    x, y = first[:, None], second[:, None]
    joint = _log_probabilities(np.column_stack([x, y]), estimator)
    return joint - _log_probabilities(x, estimator) - _log_probabilities(y, estimator)


def _log_probabilities(columns: AnyArray, estimator: str) -> Array:
    """Return the logarithm of the probability, or of the fitted Gaussian density, of every row in nats."""
    if estimator == 'plugin':
        codes = _row_codes(columns)
        counts = np.bincount(codes).astype(float)
        return np.log(counts[codes] / codes.size)
    values = _read_real(columns, 'values')
    covariance = np.atleast_2d(np.cov(values, rowvar=False, bias=True))
    density = stats.multivariate_normal(values.mean(axis=0), covariance)
    return np.asarray(density.logpdf(values), dtype=float).reshape(-1)


def _local_transfer_entropy(x: AnyArray, y: AnyArray, history: int, lag: int, estimator: str) -> Array:
    """Return `log p(y_t | y_past, x_{t-lag}) / p(y_t | y_past)` in nats at each time from `max(history, lag)`."""
    start = max(history, lag)
    future = y[start:, None]
    past = np.column_stack([y[start - step : y.size - step] for step in range(1, history + 1)])
    driver = x[start - lag : x.size - lag, None]
    return (
        _log_probabilities(np.column_stack([future, past, driver]), estimator)
        - _log_probabilities(np.column_stack([past, driver]), estimator)
        - _log_probabilities(np.column_stack([future, past]), estimator)
        + _log_probabilities(past, estimator)
    )
