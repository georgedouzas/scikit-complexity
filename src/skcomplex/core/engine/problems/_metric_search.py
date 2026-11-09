"""Neighbours in a continuous space found from the positions with a k-d tree or a Delaunay triangulation.

They are found at every evaluation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import math

import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay, QhullError, cKDTree

from ...components.spaces import (
    DelaunayNeighborhood,
    NearestNeighborhood,
    RadiusNeighborhood,
)
from ...specs import DirectionSpec, EntrySpec, RunStateSpec
from ...types import Array, Indices
from ..types import EdgesFound, Pairs
from ._base import BaseSearch


class MetricSearch(BaseSearch):
    """Neighbours in a continuous space found from the positions with a k-d tree or a Delaunay triangulation.

    They are found at every evaluation.

    Args:
        neighbourhood:
            The radius, nearest, cone, or Delaunay neighbourhood.

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
        neighbourhood: RadiusNeighborhood | NearestNeighborhood | DelaunayNeighborhood,
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
        if isinstance(neighbourhood, DelaunayNeighborhood):
            readers, others = self._delaunay(positions, self.periods)
        elif isinstance(neighbourhood, NearestNeighborhood):
            readers, others = self._nearest(tree, positions, neighbourhood.k)
        else:
            pairs = tree.query_pairs(neighbourhood.radius, output_type='ndarray').astype(np.intp)
            readers = np.concatenate([pairs[:, 0], pairs[:, 1]])
            others = np.concatenate([pairs[:, 1], pairs[:, 0]])
            if isinstance(neighbourhood, RadiusNeighborhood) and float(neighbourhood.angle) < math.pi:
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

    @staticmethod
    def _delaunay(positions: Array, periods: tuple[float, ...] | None) -> tuple[Indices, Indices]:
        """Return the pairs of members joined by an edge of the Delaunay triangulation of the positions.

        In a periodic box of the periods the positions are triangulated with their periodic images, and every edge
        joins the members of its two ends. Too few members, or members in a degenerate position, give every member all
        the others.
        """
        size, dimension = positions.shape
        if periods is None:
            points, owners = positions, np.arange(size, dtype=np.intp)
        else:
            grid = np.meshgrid(*([[-1.0, 0.0, 1.0]] * dimension), indexing='ij')
            shifts = np.stack(grid, axis=-1).reshape(-1, dimension) * np.asarray(periods)
            points = (positions[None, :, :] + shifts[:, None, :]).reshape(-1, dimension)
            owners = np.tile(np.arange(size, dtype=np.intp), len(shifts))
        try:
            simplices = Delaunay(points).simplices
        except (QhullError, ValueError):
            readers, others = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
            keep = readers != others
            return readers[keep].astype(np.intp), others[keep].astype(np.intp)
        first, second = np.triu_indices(simplices.shape[1], k=1)
        starts = np.concatenate([simplices[:, first].ravel(), simplices[:, second].ravel()])
        ends = np.concatenate([simplices[:, second].ravel(), simplices[:, first].ravel()])
        readers, others = owners[starts], owners[ends]
        keep = readers != others
        pairs = np.unique(np.stack([readers[keep], others[keep]], axis=1), axis=0)
        return pairs[:, 0].astype(np.intp), pairs[:, 1].astype(np.intp)

    def _inside(self, x: Array, readers: Indices, others: Indices, cone: RadiusNeighborhood) -> npt.NDArray[np.bool_]:
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
