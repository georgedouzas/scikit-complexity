"""Edges of a network layer that an interaction over neighbours runs its laws on."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...components import EntrySpec
from ...types import Indices
from ._base import BaseSearch
from ._run_state_spec import RunStateSpec


class EdgeSearch(BaseSearch):
    """A search of the edges of a network that the laws of an interaction run along.

    An interaction with `along` runs its laws once for every edge, and this search gives the source and target node of
    every edge, such as the two people of a contact that may pass on an infection. The edges are fixed arrays, or on a
    network that rewires, values of its edge group that are read again at every evaluation. It runs the contacts of an
    epidemic, the couplings of oscillators, and the trades along a supply network.

    Read more in the user guide, under Extend the library.

    Args:
        sources:
            The node index of the source of every edge of a fixed layer.

        targets:
            The node index of the target of every edge of a fixed layer.

        source:
            The parameter entry of the sources of a changing layer, or `None` for fixed edges.

        target:
            The parameter entry of the targets of a changing layer, or `None` for fixed edges.

    Attributes:
        pairs_:
            The source and target node of every edge of the last run.

    See Also:
        StaticSearch: The neighbours of every member of a fixed network.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.problems import RunStateSpec
        >>> from skcomplex.core.engine.problems._edge_search import EdgeSearch
        >>> state = RunStateSpec(0.0, np.zeros(0), np.zeros(0))
        >>> [ends.tolist() for ends in EdgeSearch(np.array([0, 1]), np.array([1, 2])).prepare().run(state).pairs_]
        [[0, 1], [1, 2]]
    """

    pairs_: tuple[Indices, Indices]

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

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the source and target node of every edge."""
        state = self._get_state_of(inputs)
        p = state.p
        if self.source is None or self.target is None:
            self.pairs_ = self.sources, self.targets
        else:
            self.pairs_ = self.source.get_view(p)[:, 0].astype(np.intp), self.target.get_view(p)[:, 0].astype(np.intp)
