"""Define the neighbor read spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..types import Array, Indices, IntTuple
from ._base_spec import BaseSpec
from ._entry_spec import EntrySpec


@dataclass(frozen=True, eq=False)
class NeighborReadSpec(BaseSpec):
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

        entry:
            The parameter entry whose flat values at the table hold the member of every row, for members that change
            during a run, such as those of a growing hypergraph, or `None` when the table holds the members.

        order:
            The member at every position of an order drawn again at every step, whose positions the table holds, or
            `None` for a table of members.
    """

    table: Indices
    ghost: float | None
    sizes: tuple[int, ...] | None
    steps: IntTuple | None
    mode: int
    entry: EntrySpec | None = None
    order: Indices | None = None

    def members(self, p: Array) -> Indices:
        """Return the member every row reads.

        It reads it from the table or the parameter entry at the table, through the order.

        Args:
            p:
                The parameter vector.

        Returns:
            members:
                The member index of every row.
        """
        rows = self.table if self.entry is None else self.entry.view(p).ravel()[self.table].astype(np.intp)
        return rows if self.order is None else self.order[rows]

    def read(self, column: npt.NDArray[np.float64], p: Array) -> npt.NDArray[np.float64]:
        """Return the neighbour's value of every member from the column of values.

        Args:
            column:
                One value per member.

            p:
                The parameter vector, which holds the members of a table that changes during a run.

        Returns:
            values:
                The value at the neighbour of every member.
        """
        if self.ghost is None:
            return column[self.members(p)]
        return np.append(column, self.ghost)[self.table]
