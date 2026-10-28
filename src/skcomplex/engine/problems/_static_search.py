"""Neighbours that do not change during a run, the edges of a network or the offsets of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...core.types import Indices
from ._base import BaseSearch
from ._edge_row_search import Found
from ._state import State


class StaticSearch(BaseSearch):
    """Neighbours that do not change during a run, the edges of a network or the offsets of a lattice.

    Args:
        indptr:
            The start of the neighbours of every member, and the number of pairs last.

        indices:
            The neighbours of every member in turn.

        edges:
            The edge of a network layer that joins every member to every neighbour, or `None` on a lattice.
    """

    def __init__(self, indptr: Indices, indices: Indices, edges: Indices | None = None) -> None:
        self.indptr = indptr
        self.indices = indices
        self.edges = edges

    def _run(self, inputs: State) -> Found:
        """Return the neighbours of every member."""
        del inputs
        return self.indptr, self.indices, self.edges
