"""Test a measure of dependence against surrogates whose variables are shuffled independently."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from sklearn.base import clone

from ..core.errors import AnalysisError
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

type Array = npt.NDArray[np.float64]
ALTERNATIVES = ('greater', 'less', 'two-sided')


class SurrogateTestAnalysis(BaseAnalysis):
    """Test of an analysis of many variables against surrogates that shuffle every variable independently.

    The analysis runs on the data and on `n` surrogates, and the attribute it sets gives the value of each. The p-value
    is the fraction of surrogates, with the data counted among them, at least as extreme as the data.

    Args:
        analysis:
            The unprepared analysis of the variables, cloned for the data and for every surrogate.

        attribute:
            The scalar attribute of the analysis that measures the dependence, such as `'information_'`.

        variables:
            The variables of a trajectory, or `None` for every scalar state or observable.

        n:
            The number of surrogates.

        alternative:
            `'greater'`, `'less'`, or `'two-sided'`, the direction in which the data differ from independence.

        random_state:
            The seed of the shuffles.

    Attributes:
        values_:
            The data, one row per sample and one column per variable.

        value_:
            The measure on the data.

        surrogates_:
            The measure on every surrogate.

        p_value_:
            The p-value of the data against the surrogates.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import InformationAnalysis, SurrogateTestAnalysis
        >>> bit = np.random.default_rng(0).integers(0, 2, 500)
        >>> test = SurrogateTestAnalysis(InformationAnalysis('o-information'), 'information_', n=19, random_state=0)
        >>> test = test.check().prepare(np.column_stack([bit, bit, bit])).run()
        >>> round(test.value_, 1), test.p_value_
        (1.0, 0.05)
    """

    def __init__(
        self,
        analysis: BaseAnalysis,
        attribute: str,
        variables: Sequence[str] | None = None,
        *,
        n: int = 99,
        alternative: str = 'two-sided',
        random_state: int | None = None,
    ) -> None:
        self.analysis = analysis
        self.attribute = attribute
        self.variables = variables
        self.n = n
        self.alternative = alternative
        self.random_state = random_state

    def _check(self) -> Self:
        """Raise an error for an unknown alternative, a number of surrogates below 1, or an analysis that is not one."""
        if self.alternative not in ALTERNATIVES:
            error_msg = f'Unknown alternative {self.alternative!r}. Use one of {", ".join(ALTERNATIVES)}.'
            raise AnalysisError(error_msg)
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            error_msg = f'n = {self.n!r} must be a positive number of surrogates.'
            raise AnalysisError(error_msg)
        if not isinstance(self.analysis, BaseAnalysis):
            error_msg = f'The measure of a surrogate test is an analysis, not {self.analysis!r}.'
            raise AnalysisError(error_msg)
        self.analysis.check()
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the series of the variables as columns."""
        self.values_ = np.asarray(self._read_series(source, self.variables).values, dtype=float)
        return self

    def _measure(self, values: Array) -> float:
        """Return the attribute of a clone of the analysis run on the values."""
        return float(getattr(clone(self.analysis).check().prepare(values).run(), self.attribute))

    def _run(self) -> Self:
        """Measure the data and every surrogate, and set the p-value."""
        rng = np.random.default_rng(self.random_state)
        self.value_ = self._measure(self.values_)
        self.surrogates_ = np.asarray(
            [
                self._measure(np.column_stack([rng.permutation(column) for column in self.values_.T]))
                for _ in range(self.n)
            ],
            dtype=float,
        )
        if self.alternative == 'greater':
            extreme = int(np.sum(self.surrogates_ >= self.value_))
        elif self.alternative == 'less':
            extreme = int(np.sum(self.surrogates_ <= self.value_))
        else:
            center = float(np.mean(self.surrogates_))
            extreme = int(np.sum(np.abs(self.surrogates_ - center) >= abs(self.value_ - center)))
        self.p_value_ = (1 + extreme) / (1 + self.surrogates_.size)
        return self
