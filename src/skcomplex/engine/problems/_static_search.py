"""Neighbours that do not change during a run, the edges of a network or the offsets of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...core.types import Indices, Vector
from ._edge_rows import Found


@dataclass(frozen=True)
class StaticSearch:
    """Neighbours that do not change during a run, the edges of a network or the offsets of a lattice.

    Args:
        indptr:
            The start of the neighbours of every member, and the number of pairs last.

        indices:
            The neighbours of every member in turn.

        edges:
            The edge of a network layer that joins every member to every neighbour, or `None` on a lattice.
    """

    indptr: Indices
    indices: Indices
    edges: Indices | None = None

    def find(self, x: Vector, q: Vector) -> Found:
        """Return the neighbours of every member.

        Args:
            x:
                The state vector, which the neighbours do not depend on.

            q:
                The extended parameter vector, which the neighbours do not depend on.

        Returns:
            rows:
                The row starts, the neighbours, and the edge of every neighbour on a network.
        """
        del x, q
        return self.indptr, self.indices, self.edges
