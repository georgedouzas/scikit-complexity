"""Neighbour search that returns the pairs of members closer than a cutoff."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
from scipy.spatial import cKDTree

from ...specs import EntrySpec, RunStateSpec
from ...types import Array, Indices
from ._base import BaseSearch


class PairSearch(BaseSearch):
    """A search that returns the pairs of members closer than a cutoff.

    Args:
        first:
            The entry of the positions of the first members.

        second:
            The entry of the positions of the second members.

        cutoff:
            The parameter entry of the cutoff distance.

        period:
            The side of a periodic box, or `None` for an open space.

    Attributes:
        pairs_:
            The indices of the first and second members of every pair of the last run.
    """

    pairs_: tuple[Indices, Indices]

    def __init__(self, first: EntrySpec, second: EntrySpec, cutoff: EntrySpec, period: float | None) -> None:
        self.first = first
        self.second = second
        self.cutoff = cutoff
        self.period = period

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the indices of the first and second members of every pair within the cutoff."""
        self.pairs_ = self._get_pairs(self._get_state_of(inputs))

    def _get_pairs(self, state: RunStateSpec) -> tuple[Indices, Indices]:
        """Return the indices of the members of every pair within the cutoff.

        They are the first and second members of every pair.
        """
        x, p = state.x, state.p
        cutoff = float(self.cutoff.get_view(p)[0, 0])
        first, second = self._get_positions(self.first.get_view(x)), self._get_positions(self.second.get_view(x))
        boxsize = self.period
        first_tree = cKDTree(first, boxsize=boxsize)
        if self.first == self.second:
            pairs = first_tree.query_pairs(cutoff, output_type='ndarray')
            return pairs[:, 0].astype(np.intp), pairs[:, 1].astype(np.intp)
        matrix = first_tree.sparse_distance_matrix(cKDTree(second, boxsize=boxsize), cutoff, output_type='ndarray')
        return matrix['i'].astype(np.intp), matrix['j'].astype(np.intp)

    def _get_positions(self, positions: Array) -> Array:
        """Return positions wrapped into the periodic box when there is one."""
        return np.mod(positions, self.period) if self.period is not None else positions

    def get_second_images(self, x: Array, first: Indices, second: Indices) -> Array:
        """Return the positions of the second members as seen from the first.

        They are taken at minimum image distances.

        Args:
            x:
                The state vector.

            first:
                The first member of every pair.

            second:
                The second member of every pair.

        Returns:
            positions:
                One row of second member positions per pair.
        """
        first_positions, second_positions = self.first.get_view(x)[first], self.second.get_view(x)[second]
        if self.period is None:
            return second_positions
        difference = first_positions - second_positions
        return first_positions - (difference - self.period * np.rint(difference / self.period))
