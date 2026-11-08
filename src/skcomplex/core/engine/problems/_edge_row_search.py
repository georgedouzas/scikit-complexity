"""Neighbours of the nodes of a network layer whose edges change.

They are read from the rows of its variable edge group.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ...specs import EntrySpec, RunStateSpec
from ...specs._capacity_rows_spec import ACTIVE_THRESHOLD
from ..types import EdgesFound
from ._base import BaseSearch


class EdgeRowSearch(BaseSearch):
    """Neighbours of the nodes of a network layer whose edges change.

    They are read from the rows of its variable edge group.

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

    Attributes:
        pairs_:
            The row starts, the neighbours, and the edge rows of every node of the last run.
    """

    pairs_: EdgesFound

    def __init__(
        self,
        source: EntrySpec,
        target: EntrySpec,
        active: EntrySpec,
        size: int,
        directed: bool = False,
        *,
        center: bool = False,
    ) -> None:
        self.source = source
        self.target = target
        self.active = active
        self.size = size
        self.directed = directed
        self.center = center

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the neighbours of every node along the active edges."""
        state = self._state_of(inputs)
        q = state.p
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
        self.pairs_ = indptr, others[order].astype(np.intp), ids[order].astype(np.intp)
