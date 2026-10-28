"""Edges of a network layer that an interaction over neighbours runs its laws on, one evaluation index per edge."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

import numpy as np

from ...types import Array, Indices
from ._entry import Entry


@dataclass
class EdgeSearch:
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

    sources: Indices
    targets: Indices
    source: Entry | None = None
    target: Entry | None = None
    second: None = field(default=None, init=False, repr=False)

    @property
    def size(self) -> int:
        """Number of edges, the capacity of a changing layer."""
        return int(self.sources.size if self.source is None else self.source.rows)

    def find(self, x: Array, p: Array) -> tuple[Indices, Indices]:
        """Return the source and target node of every edge.

        Args:
            x:
                The state vector, which the edges do not depend on.

            p:
                The parameter vector, which holds the edges of a changing layer.

        Returns:
            pairs:
                The source and target node indices.
        """
        del x
        if self.source is None or self.target is None:
            return self.sources, self.targets
        return self.source.view(p)[:, 0].astype(np.intp), self.target.view(p)[:, 0].astype(np.intp)
