"""Neighbours that do not change during a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...types import Indices
from ..types import EdgesFound
from ._base import BaseSearch
from ._run_state_spec import RunStateSpec


class StaticSearch(BaseSearch):
    """A search of neighbours that stay the same during a run.

    On a fixed network or a lattice the neighbours of every member never change, such as the friends of a person or the
    eight cells around a cell, so the search holds them once as the start of every member's list and the neighbours in
    turn, and gives them at every evaluation. It finds the contacts of a person, the cells around a cell, and the trade
    partners of a country.

    Read more in the user guide, under Spaces.

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

    See Also:
        EdgeRowSearch: Neighbours on a network whose edges change.
        MetricSearch: Neighbours in a continuous space.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.problems import RunStateSpec
        >>> from skcomplex.core.engine.problems._static_search import StaticSearch
        >>> search = StaticSearch(np.array([0, 1, 2]), np.array([1, 0])).prepare()
        >>> search.run(RunStateSpec(0.0, np.zeros(0), np.zeros(0))).pairs_[1].tolist()
        [1, 0]
    """

    pairs_: EdgesFound

    def __init__(self, indptr: Indices, indices: Indices, edges: Indices | None = None) -> None:
        self.indptr = indptr
        self.indices = indices
        self.edges = edges

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the neighbours of every member."""
        state = self._get_state_of(inputs)
        del state
        self.pairs_ = self.indptr, self.indices, self.edges
