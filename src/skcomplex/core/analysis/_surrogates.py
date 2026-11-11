"""Test a measure of dependence against surrogates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..enums import DataKind
from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array
from ._base import _read_series
from .types import Source

ALTERNATIVES = ('greater', 'less', 'two-sided')
TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class SurrogateTestResult:
    """A result of `surrogate_test`, a measure tested against surrogates.

    The surrogates shuffle every variable independently.

    Args:
        values:
            The data, one row per sample and one column per variable.

        value:
            The measure on the data.

        surrogates:
            The measure on every surrogate.

        p_value:
            The p-value of the data against the surrogates.
    """

    values: Array
    value: float
    surrogates: Array
    p_value: float


def surrogate_test(
    source: Source,
    *,
    analysis: Callable[[Array], object],
    attribute: str,
    variables: Sequence[str] | None = None,
    n: int = 99,
    alternative: str = 'two-sided',
    random_state: int | None = None,
) -> SurrogateTestResult:
    """Return the test of an analysis of many variables against surrogates.

    The surrogates shuffle every variable independently. The analysis runs on the data and on `n` surrogates, and the
    field of its result gives the value of each. The p-value is the fraction of surrogates, with the data counted among
    them, at least as extreme as the data.

    It appears in many fields. It tests whether a coupling between brain regions, between markets, or between climate
    indices is more than chance.

    Args:
        source:
            The trajectory, or an array with one row per sample and one column per variable.

        analysis:
            The analysis of the variables, a function of an array such as one `bind_analysis` returns, applied to the
            data and to every surrogate.

        attribute:
            The scalar field of the result of the analysis that measures the dependence, such as `'information'`.

        variables:
            The variables of a trajectory, or `None` for every scalar state or observable.

        n:
            The number of surrogates.

        alternative:
            `'greater'`, `'less'`, or `'two-sided'`, the direction in which the data differ from independence.

        random_state:
            The seed of the shuffles.

    Returns:
        result:
            The data, the measure on the data and on every surrogate, and the p-value.

    Raises:
        AnalysisError:
            The alternative is unknown, the number of surrogates is below 1, or the analysis is not callable.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import bind_analysis, information, surrogate_test
        >>> bit = np.random.default_rng(0).integers(0, 2, 500)
        >>> measure = bind_analysis(information, kind='o-information')
        >>> data = np.column_stack([bit, bit, bit])
        >>> test = surrogate_test(data, analysis=measure, attribute='information', n=19, random_state=0)
        >>> round(test.value, 1), test.p_value
        (1.0, 0.05)
    """
    if alternative not in ALTERNATIVES:
        error_msg = f'Unknown alternative {alternative!r}. Use one of {", ".join(ALTERNATIVES)}.'
        raise AnalysisError(error_msg)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        error_msg = f'n = {n!r} must be a positive number of surrogates.'
        raise AnalysisError(error_msg)
    if not callable(analysis):
        error_msg = f'The measure of a surrogate test is an analysis function, not {analysis!r}.'
        raise AnalysisError(error_msg)
    values = np.asarray(_read_series(source, variables).values, dtype=float)
    rng = np.random.default_rng(random_state)
    value = _measure(analysis, attribute, values)
    surrogates = np.asarray(
        [
            _measure(analysis, attribute, np.column_stack([rng.permutation(column) for column in values.T]))
            for _ in range(n)
        ],
        dtype=float,
    )
    if alternative == 'greater':
        extreme = int(np.sum(surrogates >= value))
    elif alternative == 'less':
        extreme = int(np.sum(surrogates <= value))
    else:
        center = float(np.mean(surrogates))
        extreme = int(np.sum(np.abs(surrogates - center) >= abs(value - center)))
    return SurrogateTestResult(values, value, surrogates, (1 + extreme) / (1 + surrogates.size))


def _measure(analysis: Callable[[Array], object], attribute: str, values: Array) -> float:
    """Return the field of the result of the analysis applied to the values."""
    return float(getattr(analysis(values), attribute))
