"""Define the table values spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec
from ...types import AnyArray, Indices


@dataclass(frozen=True)
class _TableValuesSpec(BaseSpec):
    """A spec of the values of chosen entries of a table at a parameter vector.

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
