"""Neighbours of the nodes of a network layer whose edges change, read from the rows of its variable edge group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np

from ...core.types import Indices, Vector
from ._entry import Entry

type Found = tuple[Indices, Indices, Indices | None]


ACTIVE_THRESHOLD = 0.5


@dataclass(frozen=True)
class EdgeRows:
    """Neighbours of the nodes of a network layer whose edges change, read from the rows of its variable edge group.

    Args:
        source:
            The parameter entry of the source node of every edge row.

        target:
            The parameter entry of the target node of every edge row.

        active:
            The parameter entry of the activity of every edge row.

        size:
            The number of node rows, the capacity of the groups on the nodes.

        directed:
            Whether the edges are directed, so a node reads its successors.

        center:
            Whether a node is its own neighbour.
    """

    source: Entry
    target: Entry
    active: Entry
    size: int
    directed: bool = False
    center: bool = False

    def find(self, x: Vector, q: Vector) -> Found:
        """Return the neighbours of every node along the active edges.

        Args:
            x:
                The state vector, which the neighbours do not depend on.

            q:
                The extended parameter vector, which holds the edge rows.

        Returns:
            rows:
                The row starts, the neighbours, and the edge row of every neighbour, -1 for the node itself.
        """
        del x
        keep = self.active.view(q)[:, 0] > ACTIVE_THRESHOLD
        edges = np.flatnonzero(keep).astype(np.intp)
        sources = self.source.view(q)[keep, 0].astype(np.intp)
        targets = self.target.view(q)[keep, 0].astype(np.intp)
        readers = sources if self.directed else np.concatenate([sources, targets])
        others = targets if self.directed else np.concatenate([targets, sources])
        ids = edges if self.directed else np.concatenate([edges, edges])
        if self.center:
            own = np.arange(self.size, dtype=np.intp)
            readers, others = np.concatenate([readers, own]), np.concatenate([others, own])
            ids = np.concatenate([ids, np.full(self.size, -1, dtype=np.intp)])
        order = np.argsort(readers * self.size + others, kind='stable')
        counts = np.bincount(readers, minlength=self.size)
        indptr = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(counts)]).astype(np.intp)
        return indptr, others[order].astype(np.intp), ids[order].astype(np.intp)
