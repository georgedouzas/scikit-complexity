"""How a symbol reads a field at a neighbouring site of every member."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...core.types import Indices
from ._base import BaseValue

type Offset = tuple[int, ...]


@dataclass(frozen=True, eq=False)
class NeighborRead(BaseValue):
    """How a symbol reads a field at a neighbouring site of every member.

    Args:
        table:
            The member index of the neighbour of every member, with the number of members for the fixed value.

        ghost:
            The value read outside fixed edges, or `None` when no neighbour is outside.

        sizes:
            The lattice shape, or `None` on a network.

        steps:
            The offset of the neighbour on a lattice, or `None` on a network.

        mode:
            0 for periodic, 1 for zero-flux, and 2 for fixed edges.
    """

    table: Indices
    ghost: float | None
    sizes: tuple[int, ...] | None
    steps: Offset | None
    mode: int

    def read(self, column: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the neighbour's value of every member from the column of values.

        Args:
            column:
                One value per member.

        Returns:
            values:
                The value at the neighbour of every member.
        """
        if self.ghost is None:
            return column[self.table]
        return np.append(column, self.ghost)[self.table]
