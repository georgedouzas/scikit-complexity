"""Define the table values spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ..types import AnyArray, Indices
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class TableValuesSpec(BaseSpec):
    """Values of selected entries of a table at a parameter vector.

    The table holds numbers or parameter expressions.

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

    table: AnyArray
    positions: tuple[Indices, ...]
    parameters: Mapping[Expression, int]
    time: Expression
