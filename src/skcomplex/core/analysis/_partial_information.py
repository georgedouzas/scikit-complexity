"""Decompose what two sources tell about a target."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import _read_series
from ._information import _dit_distribution
from ._utils import load_module
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
TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class PartialInformationResult:
    """A result of `partial_information`, the parts of the information of two sources.

    Args:
        values:
            The symbols of the two sources and the target, one per column.

        labels:
            The label of every series.

        redundancy:
            The information both sources give.

        unique_first:
            The information only the first source gives.

        unique_second:
            The information only the second source gives.

        synergy:
            The information only the two sources together give.

        total:
            The mutual information of the two sources together with the target, the sum of the four parts.
    """

    values: Array
    labels: list[str]
    redundancy: float
    unique_first: float
    unique_second: float
    synergy: float
    total: float


def partial_information(
    source: Source,
    *,
    variables: Sequence[str] | None = None,
    redundancy: str = 'mmi',
    base: float = 2.0,
) -> PartialInformationResult:
    """Return the partial information decomposition of two sources about a target.

    The source holds three series: the first source, the second source, and the target. The mutual information of the
    two sources together with the target splits into the information both give, the information only one of them
    gives, and the information only the two together give, with a redundancy measure of `dit`.

    It appears in many fields. It splits what two genes tell about a trait, or two neurons about a stimulus, into
    unique, shared, and synergistic parts.

    Args:
        source:
            The trajectory, or an array of three columns: the two sources and the target.

        variables:
            The variables of a trajectory to read, the two sources and the target, or `None` for every scalar series.

        redundancy:
            The redundancy measure of `dit`: `'mmi'` for the minimum mutual information, `'williams_beer'` for
            `I_min`, `'broja'`, `'ccs'`, `'pm'`, or `'dep'`.

        base:
            The base of the logarithm, 2 for bits.

    Returns:
        result:
            The symbols and the four parts of the information with their sum.

    Raises:
        AnalysisError:
            The redundancy measure is unknown, the base is not positive, or the source does not hold three series.

    Examples:
        >>> from skcomplex.core.analysis import partial_information
        >>> xor = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
        >>> parts = partial_information(xor)
        >>> round(parts.synergy, 6), round(parts.redundancy, 6)
        (1.0, 0.0)
    """
    if redundancy not in REDUNDANCIES:
        error_msg = f'partial_information has redundancy={redundancy!r}. Use one of {", ".join(REDUNDANCIES)}.'
        raise AnalysisError(error_msg)
    if base <= 0:
        error_msg = f'partial_information has base={base!r}. Use a positive base.'
        raise AnalysisError(error_msg)
    series = _read_series(source, variables)
    if series.values.shape[1] != SERIES:
        error_msg = (
            f'partial_information reads two sources and a target, but the source has {series.values.shape[1]} series. '
            'Select three with `variables` or pass an array of three columns.'
        )
        raise AnalysisError(error_msg)
    columns = np.column_stack([column.ravel() for column in series.values.T])
    measure = getattr(load_module('dit').pid, REDUNDANCIES[redundancy])
    lattice = measure(_dit_distribution(columns), [[0], [1]], [2])
    scale = math.log(2) / math.log(base)
    shared = float(lattice.get_pi(((0,), (1,)))) * scale
    unique_first = float(lattice.get_pi(((0,),))) * scale
    unique_second = float(lattice.get_pi(((1,),))) * scale
    synergy = float(lattice.get_pi(((0, 1),))) * scale
    return PartialInformationResult(
        series.values,
        series.labels,
        shared,
        unique_first,
        unique_second,
        synergy,
        shared + unique_first + unique_second + synergy,
    )
