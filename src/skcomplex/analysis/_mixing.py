"""Measure how categorical states mix or cluster on grids and networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

type Array = npt.NDArray[np.float64]


class MixingAnalysis(BaseAnalysis):
    """Duncan's dissimilarity index of two groups of agents across units, the share of one that would have to move.

    It reads one row per agent with its group and its unit, such as a district or a block of cells, and gives
    `1/2 Σ_u |a_u / A - b_u / B|`, 0 for even mixing and 1 for complete separation.

    Attributes:
        values_:
            The group and the unit of every agent.

        dissimilarity_:
            The dissimilarity index.

    Examples:
        >>> from skcomplex.analysis import MixingAnalysis
        >>> MixingAnalysis().check().prepare([[0, 0], [0, 0], [1, 1], [1, 1]]).run().dissimilarity_
        1.0
    """

    def _get_data_kind(self) -> DataKind:
        """Return the agents kind."""
        return DataKind.AGENTS

    def _prepare(self, source: Source) -> Self:
        """Read the group and the unit of every agent."""
        self.values_ = np.asarray(source)
        if self.values_.ndim != 2 or self.values_.shape[1] != 2:  # noqa: PLR2004
            error_msg = (
                f'{type(self).__name__} reads one row per agent with its group and its unit, not an array of shape '
                f'{self.values_.shape}.'
            )
            raise AnalysisError(error_msg)
        return self

    def _run(self) -> Self:
        """Compute Duncan's dissimilarity index of the two groups."""
        groups = self.values_[:, 0]
        labels = np.unique(groups)
        if labels.size != 2:  # noqa: PLR2004
            error_msg = f'The dissimilarity index compares two groups, not {labels.size}. Give agents of two groups.'
            raise AnalysisError(error_msg)
        _, places = np.unique(self.values_[:, 1], return_inverse=True)
        first = np.bincount(places[groups == labels[0]], minlength=places.max() + 1)
        second = np.bincount(places[groups == labels[1]], minlength=places.max() + 1)
        self.dissimilarity_ = float(0.5 * np.sum(np.abs(first / first.sum() - second / second.sum())))
        return self
