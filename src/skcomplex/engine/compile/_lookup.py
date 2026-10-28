"""Lookup array of a table or totalistic rule."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

type Vector = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Lookup:
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
    values: Vector
    columns: int = 1
