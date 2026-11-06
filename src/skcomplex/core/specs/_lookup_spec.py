"""Define the lookup spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class LookupSpec(BaseSpec):
    """Lookup array of a table or totalistic rule.

    Args:
        kind:
            `'table'`, `'totalistic'`, or `'outer'`.

        base:
            The number of state values.

        values:
            The next value by pattern code, by sum, or by cell value and sum in row-major order.

        columns:
            The number of sums of an outer totalistic rule.
    """

    kind: str
    base: int
    values: Array
    columns: int = 1
