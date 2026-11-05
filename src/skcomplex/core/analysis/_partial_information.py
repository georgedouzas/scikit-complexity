"""Decompose what two sources tell about a target into redundant, unique, and synergistic information."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import numpy as np

from ..errors import AnalysisError
from ..types import DataKind
from ._base import BaseAnalysis
from ._information import InformationAnalysis
from .types import Source

REDUNDANCIES = {
    'mmi': 'PID_MMI',
    'williams_beer': 'PID_WB',
    'broja': 'PID_BROJA',
    'ccs': 'PID_CCS',
    'pm': 'PID_PM',
    'dep': 'PID_dep',
}
SERIES = 3


class PartialInformationAnalysis(BaseAnalysis):
    """Partial information decomposition of what two series of symbols tell about a third, the target.

    The source holds three series: the first source, the second source, and the target. The mutual information of the
    two sources together with the target splits into the information both give, the information only one of them
    gives, and the information only the two together give, with a redundancy measure of `dit`.

    Args:
        variables:
            The variables of a trajectory to read, the two sources and the target, or `None` for every scalar series.

        redundancy:
            The redundancy measure of `dit`: `'mmi'` for the minimum mutual information, `'williams_beer'` for
            `I_min`, `'broja'`, `'ccs'`, `'pm'`, or `'dep'`.

        base:
            The base of the logarithm, 2 for bits.

    Attributes:
        values_:
            The symbols of the two sources and the target, one per column.

        labels_:
            The label of every series.

        redundancy_:
            The information both sources give.

        unique_first_:
            The information only the first source gives.

        unique_second_:
            The information only the second source gives.

        synergy_:
            The information only the two sources together give.

        total_:
            The mutual information of the two sources together with the target, the sum of the four parts.

    Examples:
        >>> from skcomplex.core.analysis import PartialInformationAnalysis
        >>> xor = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
        >>> parts = PartialInformationAnalysis().prepare(xor).run()
        >>> round(parts.synergy_, 6), round(parts.redundancy_, 6)
        (1.0, 0.0)
    """

    def __init__(self, variables: Sequence[str] | None = None, *, redundancy: str = 'mmi', base: float = 2.0) -> None:
        self.variables = variables
        self.redundancy = redundancy
        self.base = base

    def _check(self) -> None:
        """Validate the redundancy measure and the base."""
        if self.redundancy not in REDUNDANCIES:
            error_msg = (
                f'{type(self).__name__} has redundancy={self.redundancy!r}. Use one of {", ".join(REDUNDANCIES)}.'
            )
            raise AnalysisError(error_msg)
        if self.base <= 0:
            error_msg = f'{type(self).__name__} has base={self.base!r}. Use a positive base.'
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: Source) -> None:
        """Read the two sources and the target as three columns."""
        series = self._read_series(source, self.variables)
        if series.values.shape[1] != SERIES:
            error_msg = (
                f'{type(self).__name__} reads two sources and a target, but the source has {series.values.shape[1]} '
                'series. Select three with `variables` or pass an array of three columns.'
            )
            raise AnalysisError(error_msg)
        self.values_ = series.values
        self.labels_ = series.labels

    def _run(self, inputs: object = None) -> None:
        """Decompose the information of the sources about the target."""
        columns = np.column_stack([column.ravel() for column in self.values_.T])
        measure = getattr(self._import('dit').pid, REDUNDANCIES[self.redundancy])
        lattice = measure(InformationAnalysis._dit_distribution(columns), [[0], [1]], [2])
        scale = math.log(2) / math.log(self.base)
        self.redundancy_ = float(lattice.get_pi(((0,), (1,)))) * scale
        self.unique_first_ = float(lattice.get_pi(((0,),))) * scale
        self.unique_second_ = float(lattice.get_pi(((1,),))) * scale
        self.synergy_ = float(lattice.get_pi(((0, 1),))) * scale
        self.total_ = self.redundancy_ + self.unique_first_ + self.unique_second_ + self.synergy_
