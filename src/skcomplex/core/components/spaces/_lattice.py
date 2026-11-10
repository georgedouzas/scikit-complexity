"""Define lattice spaces.

They are boxes of Z^d or of the triangular lattice whose sites hold lattice group members.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ...errors import ValidationError
from ._ball import BallNeighborhood
from ._base import BaseNeighborhood, BaseSpace
from ._lattice_neighborhood import LatticeNeighborhood

BOUNDARIES = ('periodic', 'fixed', 'reflecting')
GEOMETRIES = ('square', 'triangular')
SQUARE_METRICS = ('chebyshev', 'manhattan')
TRIANGULAR_ROW_HEIGHT = float(np.sqrt(3.0) / 2)
TRIANGULAR_DIMENSIONS = 2


class LatticeSpace(BaseSpace):
    """Box of `Z^d`, or of the triangular lattice in 2D, holding lattice group members.

    `Z^d` is the integer lattice, and the sites hold the members. Sites are numbered in row-major order, so site `(i,
    j)` of a lattice of shape `(n, m)` is member `i * m + j`. A triangular site `(i, j)` keeps its integer coordinates
    on a skewed grid and sits at `(j + i / 2, i sqrt(3) / 2)` in the plane, so it has six neighbours at unit distance.

    Args:
        shape:
            The number of sites along every dimension.

        geometry:
            `'square'` for the integer lattice or `'triangular'` for the triangular lattice in two dimensions.

        boundary:
            `'periodic'` to join opposite edges, `'fixed'` to read the value `fill` outside the box, or `'reflecting'`
            to read the mirror image of the box outside it.

        fill:
            The value read outside a lattice with fixed edges.

        spacing:
            The distance between neighbouring sites, or one distance per dimension.

        metric:
            The metric of the ball of radius 1 that agents query. It is `'chebyshev'` for the 8 surrounding cells of a
            two-dimensional square lattice or `'manhattan'` for the 4 adjacent ones, and `'hexagonal'` for the six
            neighbours on a triangular lattice. `None` means `'chebyshev'` on a square lattice and `'hexagonal'` on a
            triangular one.

        capacity:
            The largest number of agents in a cell, or `None` for no limit.

        label:
            The label of the space in its system.

    Attributes:
        spacings_:
            The distance between neighbouring sites along every dimension.

        metric_:
            The metric of the ball agents query.

    Examples:
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> space = LatticeSpace((10, 20)).prepare()
        >>> space.size_, space.n_dim_, space.shape_
        (200, 2, (10, 20))
        >>> LatticeSpace((2, 2), geometry='triangular').prepare().positions_.round(3).tolist()
        [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [1.5, 0.866]]
    """

    space_tags: ClassVar[tuple[str, ...]] = ('lattice',)

    def __init__(
        self,
        shape: Sequence[int] | int,
        *,
        geometry: str = 'square',
        boundary: str = 'periodic',
        fill: float = 0.0,
        spacing: float | Sequence[float] = 1.0,
        metric: str | None = None,
        capacity: int | None = None,
        label: str = 'space',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.geometry = geometry
        self.boundary = boundary
        self.fill = fill
        self.spacing = spacing
        self.metric = metric
        self.capacity = capacity

    def _prepare(self, context: None = None) -> None:
        """Set the shape, the size, the spacings, and the position of every site.

        It first raises an error when the shape, the geometry, or the boundary is not valid.
        """
        super()._prepare(context)
        shape = tuple(np.ravel(self.shape).tolist())
        if not shape or any(int(size) != size or size < 1 for size in shape):
            error_msg = (
                f'A lattice has a positive number of sites along every dimension, but its shape is {self.shape}.'
            )
            raise ValidationError(error_msg)
        if self.boundary not in BOUNDARIES:
            error_msg = f'Unknown lattice boundary {self.boundary!r}. Use one of {", ".join(BOUNDARIES)}.'
            raise ValidationError(error_msg)
        if self.geometry not in GEOMETRIES:
            error_msg = f'Unknown lattice geometry {self.geometry!r}. Use one of {", ".join(GEOMETRIES)}.'
            raise ValidationError(error_msg)
        if self.geometry == 'triangular' and len(shape) != TRIANGULAR_DIMENSIONS:
            error_msg = f'A triangular lattice has two dimensions, but its shape is {self.shape}.'
            raise ValidationError(error_msg)
        self._check_metric()
        self.shape_ = tuple(int(size) for size in np.ravel(self.shape).tolist())
        self.n_dim_ = len(self.shape_)
        self.size_ = int(np.prod(self.shape_))
        spacings = np.ravel(np.asarray(self.spacing, dtype=float))
        if spacings.size == 1:
            spacings = np.repeat(spacings, self.n_dim_)
        self.spacings_ = tuple(float(item) for item in spacings)
        coordinates = np.indices(self.shape_).reshape(self.n_dim_, -1).T.astype(float)
        if self.geometry == 'triangular':
            rows, columns = coordinates.T
            coordinates = np.column_stack([columns + rows / 2, rows * TRIANGULAR_ROW_HEIGHT])
        self.positions_ = coordinates * np.asarray(self.spacings_)
        self.periodic_ = self.boundary == 'periodic'
        self.default_visual_ = 'image' if self.geometry == 'square' else 'sites'
        if self.metric is not None:
            self.metric_ = self.metric
        else:
            self.metric_ = 'chebyshev' if self.geometry == 'square' else 'hexagonal'

    def _check_metric(self) -> None:
        """Raise an error for a metric the geometry of the lattice does not take.

        A square lattice takes the Chebyshev or the Manhattan metric, and a triangular one the hexagonal metric.
        """
        metrics = SQUARE_METRICS if self.geometry == 'square' else ('hexagonal',)
        if self.metric is not None and self.metric not in metrics:
            error_msg = (
                f'A {self.geometry} lattice has the metric {self.metric!r}. Use one of {", ".join(metrics)}, or None.'
            )
            raise ValidationError(error_msg)

    def _get_axis_boundary(self, axis: int) -> str:
        """Return the edges of an axis, the one boundary of the lattice."""
        return self.boundary

    def _copy_with_edges(self, boundary: str, fill: float) -> LatticeSpace:
        """Return the sites of the lattice with other edges.

        The reads of a state with its own boundary use them.
        """
        return LatticeSpace(self.shape_, boundary=boundary, fill=fill).prepare()

    def _get_default_neighborhood(self) -> LatticeNeighborhood:
        """Return the neighbourhood agents query, the unit ball under the lattice metric."""
        return BallNeighborhood(metric=self.metric_, n_dim=self.n_dim_)

    def _get_index_table(self, neighborhood: BaseNeighborhood) -> npt.NDArray[np.intp]:
        """Return the member index of every neighbour of every site.

        Sites outside fixed edges get `size_`.
        """
        if not isinstance(neighborhood, LatticeNeighborhood):
            error_msg = f'A lattice reads a lattice neighbourhood of offsets, but it was given {neighborhood!r}.'
            raise ValidationError(error_msg)
        offsets = neighborhood.prepare().offsets_
        if offsets and len(offsets[0]) != self.n_dim_:
            error_msg = (
                f'The neighbourhood has offsets of dimension {len(offsets[0])}, but the lattice has {self.n_dim_} '
                'dimensions.'
            )
            raise ValidationError(error_msg)
        coordinates = np.indices(self.shape_).reshape(self.n_dim_, -1)
        table = np.empty((self.size_, len(offsets)), dtype=np.intp)
        for column, offset in enumerate(offsets):
            shifted = coordinates + np.asarray(offset)[:, None]
            outside = np.zeros(self.size_, dtype=bool)
            for axis, size in enumerate(self.shape_):
                edges = self._get_axis_boundary(axis)
                if edges == 'periodic':
                    shifted[axis] %= size
                elif edges == 'reflecting':
                    shifted[axis] %= 2 * size
                    shifted[axis] = np.where(shifted[axis] < size, shifted[axis], 2 * size - 1 - shifted[axis])
                else:
                    outside |= (shifted[axis] < 0) | (shifted[axis] >= size)
                    shifted[axis] = np.clip(shifted[axis], 0, size - 1)
            flat = np.ravel_multi_index(tuple(shifted), self.shape_)
            table[:, column] = np.where(outside, self.size_, flat)
        return table

    def _get_ball(self, radius: int) -> LatticeNeighborhood:
        """Return the box of offsets within a radius along every dimension.

        It is the Chebyshev ball with the site itself in its lexicographic place, on both geometries.
        """
        return BallNeighborhood(radius, n_dim=self.n_dim_, center='inline')

    def _get_neighbor_pairs(self, layer: str | None = None) -> list[tuple[int, int]]:
        """Return the pairs of distinct sites one step apart along a dimension.

        Each pair comes once.
        """
        sites = np.arange(self.size_).reshape(self.shape_)
        found: dict[frozenset[int], tuple[int, int]] = {}
        for axis, length in enumerate(self.shape_):
            shifted = np.roll(sites, -1, axis=axis)
            pairs = np.stack([sites.ravel(), shifted.ravel()], axis=1)
            if self._get_axis_boundary(axis) != 'periodic':
                edge = np.indices(self.shape_)[axis].ravel() == length - 1
                pairs = pairs[~edge]
            for first, second in pairs.tolist():
                if first != second:
                    found.setdefault(frozenset((first, second)), (first, second))
        return list(found.values())
