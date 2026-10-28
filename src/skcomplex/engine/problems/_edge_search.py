"""Edges of a network layer that an interaction over neighbours runs its laws on, one evaluation index per edge."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...specs import EntrySpec, RunStateSpec
from ...types import Indices
from ._base import BaseSearch


class EdgeSearch(BaseSearch):
    """Edges of a network layer that an interaction over neighbours runs its laws on, one evaluation index per edge.

    The edges are fixed node index arrays, or on a network whose topology changes, hidden parameters of its variable
    edge group that topology requests rewrite, read as node indices at every evaluation.

    Args:
        sources:
            The node index of the source of every edge of a fixed layer.

        targets:
            The node index of the target of every edge of a fixed layer.

        source:
            The parameter entry of the sources of a changing layer, or `None` for fixed edges.

        target:
            The parameter entry of the targets of a changing layer, or `None` for fixed edges.
    """

    def __init__(
        self,
        sources: Indices,
        targets: Indices,
        source: EntrySpec | None = None,
        target: EntrySpec | None = None,
    ) -> None:
        self.sources = sources
        self.targets = targets
        self.source = source
        self.target = target

    @property
    def size(self) -> int:
        """Number of edges, the capacity of a changing layer."""
        return int(self.sources.size if self.source is None else self.source.rows)

    def _run(self, inputs: RunStateSpec) -> tuple[Indices, Indices]:
        """Return the source and target node of every edge."""
        p = inputs.p
        if self.source is None or self.target is None:
            return self.sources, self.targets
        return self.source.view(p)[:, 0].astype(np.intp), self.target.view(p)[:, 0].astype(np.intp)
