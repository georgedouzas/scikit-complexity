"""Neighbours that do not change during a run, the edges of a network or the offsets of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...specs import RunStateSpec
from ...types import Indices
from ..types import EdgesFound
from ._base import BaseSearch


class StaticSearch(BaseSearch):
    """Neighbours that do not change during a run, the edges of a network or the offsets of a lattice.

    Args:
        indptr:
            The start of the neighbours of every member, and the number of pairs last.

        indices:
            The neighbours of every member in turn.

        edges:
            The edge of a network layer that joins every member to every neighbour, or `None` on a lattice.

    Attributes:
        pairs_:
            The row starts, the neighbours, and the edges of the last run.
    """

    pairs_: EdgesFound

    def __init__(self, indptr: Indices, indices: Indices, edges: Indices | None = None) -> None:
        self.indptr = indptr
        self.indices = indices
        self.edges = edges

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the neighbours of every member."""
        state = self._state_of(inputs)
        del state
        self.pairs_ = self.indptr, self.indices, self.edges
