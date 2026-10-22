"""Test a measure of dependence against surrogates whose variables are shuffled independently."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..base import AnalysisError
from ._kinds import DataKind, analysis
from ._utils import sample_columns

type Array = npt.NDArray[np.float64]
ALTERNATIVES = ('greater', 'less', 'two-sided')


@dataclass(frozen=True)
class SurrogateTest:
    """Value of a measure and its values on surrogates with independent variables.

    Args:
        value:
            The measure on the data.

        surrogates:
            The measure on every surrogate.

        alternative:
            `'greater'`, `'less'`, or `'two-sided'`, the direction in which the data differ from independence.
    """

    value: float
    surrogates: Array
    alternative: str

    @property
    def p_value(self) -> float:
        """Fraction of surrogates, with the data counted among them, at least as extreme as the data."""
        if self.alternative == 'greater':
            extreme = int(np.sum(self.surrogates >= self.value))
        elif self.alternative == 'less':
            extreme = int(np.sum(self.surrogates <= self.value))
        else:
            center = float(np.mean(self.surrogates))
            extreme = int(np.sum(np.abs(self.surrogates - center) >= abs(self.value - center)))
        return (1 + extreme) / (1 + self.surrogates.size)


@analysis(DataKind.SERIES)
def surrogate_test(
    measure: Callable[[npt.NDArray[Any]], float],
    data: npt.ArrayLike,
    *,
    n: int = 99,
    alternative: str = 'two-sided',
    random_state: int | None = None,
) -> SurrogateTest:
    """Return a measure of many variables with the measure on surrogates that shuffle every variable independently.

    Args:
        measure:
            The measure, a function of one row per sample and one column per variable.

        data:
            One row per sample and one column per variable.

        n:
            The number of surrogates.

        alternative:
            `'greater'`, `'less'`, or `'two-sided'`, the direction in which the data differ from independence.

        random_state:
            The seed of the shuffles.

    Returns:
        test:
            The value on the data, the values on the surrogates, and the p-value.

    Raises:
        AnalysisError:
            The alternative is unknown.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import o_information, surrogate_test
        >>> bit = np.random.default_rng(0).integers(0, 2, 500)
        >>> test = surrogate_test(o_information, np.column_stack([bit, bit, bit]), n=19, random_state=0)
        >>> round(test.value, 1), test.p_value
        (1.0, 0.05)
    """
    if alternative not in ALTERNATIVES:
        error_msg = f'Unknown alternative {alternative!r}. Use one of {", ".join(ALTERNATIVES)}.'
        raise AnalysisError(error_msg)
    columns = sample_columns(data)
    rng = np.random.default_rng(random_state)
    values = [measure(np.column_stack([rng.permutation(column) for column in columns.T])) for _ in range(n)]
    return SurrogateTest(float(measure(columns)), np.asarray(values, dtype=float), alternative)
