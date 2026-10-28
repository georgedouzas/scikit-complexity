"""Define the table values spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ..types import Indices
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class TableValuesSpec(BaseSpec):
    """Values of selected entries of a table of numbers or parameter expressions at a parameter vector.

    Args:
        table:
            The table, a float array or an object array of Sage expressions.

        positions:
            The indices of the selected entries, one array per dimension.

        parameters:
            The flat index in the parameter vector of every parameter symbol the selected entries read.

        time:
            The time symbol.
    """

    table: npt.NDArray[Any]
    positions: tuple[Indices, ...]
    parameters: Mapping[Expression, int]
    time: Expression
