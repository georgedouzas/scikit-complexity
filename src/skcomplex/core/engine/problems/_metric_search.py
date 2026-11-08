"""Neighbours in a continuous space found from the positions with a k-d tree.

They are found at every evaluation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ...components.spaces import (
    ConeNeighborhood,
    NearestNeighborhood,
    RadiusNeighborhood,
)
from ...specs import DirectionSpec, EntrySpec, RunStateSpec
from ...types import Array, Indices
from ..types import EdgesFound, Pairs
from ._base import BaseSearch


class MetricSearch(BaseSearch):
    """Neighbours in a continuous space found from the positions with a k-d tree.

    They are found at every evaluation.

    Args:
        neighbourhood:
            The radius, nearest, or cone neighbourhood.

        positions:
            The entry of the position state of the members.

        periods:
            The sides of a periodic box, or `None` for distances without minimum images.

        direction:
            The state with the direction of a cone, or `None`.

    Attributes:
        pairs_:
            The row starts and the neighbours of every member of the last run, with no edges.
    """

    pairs_: EdgesFound

    def __init__(
        self,
        neighbourhood: RadiusNeighborhood | NearestNeighborhood | ConeNeighborhood,
        positions: EntrySpec,
        periods: tuple[float, ...] | None = None,
        direction: DirectionSpec | None = None,
    ) -> None:
        self.neighbourhood = neighbourhood
        self.positions = positions
        self.periods = periods
        self.direction = direction

    def wrapped(self, x: Array) -> Array:
        """Return the positions of the members, wrapped into a periodic box.

        Args:
            x:
                The state vector.

        Returns:
            positions:
                One row of position components per member.
        """
        positions = self.positions.view(x)
        if self.periods is None:
            return positions
        periods = np.asarray(self.periods)
        wrapped = np.mod(positions, periods)
        return np.where(wrapped >= periods, wrapped - periods, wrapped)

    def differences(self, x: Array, readers: Indices, others: Indices) -> Array:
        """Return the displacement from every reader to its neighbour.

        In a periodic box, it is the minimum image.

        Args:
            x:
                The state vector.

            readers:
                The reader of every pair.

            others:
                The neighbour of every pair.

        Returns:
            differences:
                One row of components per pair, `x_reader - x_other` wrapped.
        """
        positions = self.positions.view(x)
        difference = positions[readers] - positions[others]
        if self.periods is None:
            return difference
        periods = np.asarray(self.periods)
        return difference - periods * np.rint(difference / periods)

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the neighbours of every member at the current positions."""
        state = self._state_of(inputs)
        x, q = state.x, state.p
        del q
        positions = self.wrapped(x)
        size = positions.shape[0]
        tree = cKDTree(positions, boxsize=np.asarray(self.periods) if self.periods is not None else None)
        neighbourhood = self.neighbourhood
        if isinstance(neighbourhood, NearestNeighborhood):
            readers, others = self._nearest(tree, positions, neighbourhood.k)
        else:
            pairs = tree.query_pairs(neighbourhood.radius, output_type='ndarray').astype(np.intp)
            readers = np.concatenate([pairs[:, 0], pairs[:, 1]])
            others = np.concatenate([pairs[:, 1], pairs[:, 0]])
            if isinstance(neighbourhood, ConeNeighborhood):
                keep = self._inside(x, readers, others, neighbourhood)
                readers, others = readers[keep], others[keep]
        if neighbourhood.center:
            readers, others = self._with_center(readers, others, size)
        self.pairs_ = (*self._rows(readers, others, size), None)

    @staticmethod
    def _nearest(tree: cKDTree, positions: Array, k: int) -> tuple[Indices, Indices]:
        """Return the pairs of every member with its `k` nearest other members."""
        size = positions.shape[0]
        k = min(k, size - 1)
        if k < 1:
            return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
        _, found = tree.query(positions, k=k + 1)
        found = np.asarray(found, dtype=np.intp).reshape(size, k + 1)
        own = found == np.arange(size)[:, None]
        own[~own.any(axis=1), -1] = True
        return np.repeat(np.arange(size, dtype=np.intp), k), found[~own].astype(np.intp)

    def _inside(self, x: Array, readers: Indices, others: Indices, cone: ConeNeighborhood) -> npt.NDArray[np.bool_]:
        """Return whether every pair has its neighbour inside the reader's cone."""
        assert self.direction is not None
        separation = -self.differences(x, readers, others)
        direction = self.direction.vectors(x)[readers]
        dot = np.sum(direction * separation, axis=1)
        norms = np.sqrt(np.sum(direction * direction, axis=1)) * np.sqrt(np.sum(separation * separation, axis=1))
        return dot >= np.cos(cone.angle) * norms

    @staticmethod
    def _rows(readers: Indices, others: Indices, size: int) -> Pairs:
        """Return the row starts and the neighbours of a list of pairs.

        They are sorted by reader and then by neighbour.
        """
        order = np.lexsort((others, readers))
        counts = np.bincount(readers, minlength=size)
        indptr = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(counts)]).astype(np.intp)
        return indptr, others[order].astype(np.intp)

    @staticmethod
    def _with_center(readers: Indices, others: Indices, size: int) -> tuple[Indices, Indices]:
        """Return pairs with every member paired with itself."""
        own = np.arange(size, dtype=np.intp)
        return np.concatenate([readers, own]), np.concatenate([others, own])
