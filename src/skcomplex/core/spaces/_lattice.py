"""Define lattice spaces, boxes of Z^d or of the triangular lattice whose sites hold lattice group members."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt

from ...errors import ValidationError
from ._base import BaseNeighborhood, BaseSpace
from ._box import BoxNeighborhood
from ._lattice_neighborhood import LatticeNeighborhood
from ._moore import MooreNeighborhood
from ._triangular import TriangularNeighborhood
from ._von_neumann import VonNeumannNeighborhood

BOUNDARIES = ('periodic', 'fixed', 'reflecting')
GEOMETRIES = ('square', 'triangular')
TRIANGULAR_ROW_HEIGHT = float(np.sqrt(3.0) / 2)


class LatticeSpace(BaseSpace):
    """Box of the integer lattice `Z^d`, or of the triangular lattice in 2D, whose sites hold lattice group members.

    Sites are numbered in row-major order, so site `(i, j)` of a lattice of shape `(n, m)` is member `i * m + j`. A
    triangular site `(i, j)` keeps its integer coordinates on a skewed grid and sits at `(j + i / 2, i sqrt(3) / 2)` in
    the plane, so it has six neighbours at unit distance.

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

        neighborhood:
            The neighbourhood agents on a two-dimensional square lattice query, `'moore'` for the 8 surrounding cells
            or `'von_neumann'` for the 4 adjacent ones.

        capacity:
            The largest number of agents in a cell, or `None` for no limit.

        label:
            The label of the space in its system.

    Attributes:
        spacings_:
            The distance between neighbouring sites along every dimension.

    Examples:
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> space = LatticeSpace((10, 20)).check().prepare()
        >>> space.size_, space.n_dim_, space.shape_
        (200, 2, (10, 20))
        >>> LatticeSpace((2, 2), geometry='triangular').check().prepare().positions_.round(3).tolist()
        [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [1.5, 0.866]]
    """

    def __init__(
        self,
        shape: Sequence[int] | int,
        *,
        geometry: str = 'square',
        boundary: str = 'periodic',
        fill: float = 0.0,
        spacing: float | Sequence[float] = 1.0,
        neighborhood: str = 'moore',
        capacity: int | None = None,
        label: str = 'space',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.geometry = geometry
        self.boundary = boundary
        self.fill = fill
        self.spacing = spacing
        self.neighborhood = neighborhood
        self.capacity = capacity

    def _check(self) -> Self:
        """Raise an error when the shape, the geometry, or the boundary is not valid."""
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
        if self.geometry == 'triangular' and len(shape) != 2:  # noqa: PLR2004
            error_msg = f'A triangular lattice has two dimensions, but its shape is {self.shape}.'
            raise ValidationError(error_msg)
        return self

    def _prepare(self, context: object = None) -> Self:
        """Set the shape, the size, the spacings, and the position of every site."""
        self.shape_ = tuple(int(size) for size in np.ravel(self.shape).tolist())
        self.n_dim_ = len(self.shape_)
        self.size_ = int(np.prod(self.shape_))
        if np.ndim(self.spacing) == 0:
            self.spacings_ = (float(self.spacing),) * self.n_dim_  # type: ignore[arg-type]
        else:
            self.spacings_ = tuple(float(item) for item in self.spacing)  # type: ignore[union-attr]
        coordinates = np.indices(self.shape_).reshape(self.n_dim_, -1).T.astype(float)
        if self.geometry == 'triangular':
            rows, columns = coordinates.T
            coordinates = np.column_stack([columns + rows / 2, rows * TRIANGULAR_ROW_HEIGHT])
        self.positions_ = coordinates * np.asarray(self.spacings_)
        self.periodic_ = self.boundary == 'periodic'
        self.default_visual_ = 'image' if self.geometry == 'square' else 'sites'
        return self

    def _default_neighborhood(self) -> LatticeNeighborhood:
        """Return the neighbourhood agents query, the triangular, von Neumann, or Moore neighbourhood of radius 1."""
        if self.geometry == 'triangular':
            return TriangularNeighborhood()
        if self.neighborhood == 'von_neumann':
            return VonNeumannNeighborhood(n_dim=self.n_dim_)
        return MooreNeighborhood(n_dim=self.n_dim_)

    def _index_table(self, neighborhood: BaseNeighborhood) -> npt.NDArray[np.intp]:
        """Return the member index of every neighbour of every site, with `size_` for sites outside fixed edges."""
        if not isinstance(neighborhood, LatticeNeighborhood):
            error_msg = f'A lattice reads a lattice neighbourhood of offsets, but it was given {neighborhood!r}.'
            raise ValidationError(error_msg)
        offsets = neighborhood.check().prepare().offsets_
        if offsets and len(offsets[0]) != self.n_dim_:
            error_msg = (
                f'The neighbourhood has offsets of dimension {len(offsets[0])}, but the lattice has {self.n_dim_} '
                'dimensions.'
            )
            raise ValidationError(error_msg)
        coordinates = np.indices(self.shape_).reshape(self.n_dim_, -1)
        sizes = np.asarray(self.shape_)[:, None]
        table = np.empty((self.size_, len(offsets)), dtype=np.intp)
        for column, offset in enumerate(offsets):
            shifted = coordinates + np.asarray(offset)[:, None]
            outside = np.zeros(self.size_, dtype=bool)
            if self.boundary == 'periodic':
                shifted %= sizes
            elif self.boundary == 'reflecting':
                shifted %= 2 * sizes
                shifted = np.where(shifted < sizes, shifted, 2 * sizes - 1 - shifted)
            else:
                outside = ((shifted < 0) | (shifted >= sizes)).any(axis=0)
                shifted = np.clip(shifted, 0, sizes - 1)
            flat = np.ravel_multi_index(tuple(shifted), self.shape_)
            table[:, column] = np.where(outside, self.size_, flat)
        return table

    def _ball(self, radius: int) -> LatticeNeighborhood:
        """Return the box of offsets within a radius along every dimension, the site itself included."""
        return BoxNeighborhood(radius, n_dim=self.n_dim_)

    def _neighbor_pairs(self, layer: str | None = None) -> list[tuple[int, int]]:
        """Return the pairs of distinct sites one step apart along a dimension, each pair once."""
        sites = np.arange(self.size_).reshape(self.shape_)
        found: dict[frozenset[int], tuple[int, int]] = {}
        for axis, length in enumerate(self.shape_):
            shifted = np.roll(sites, -1, axis=axis)
            pairs = np.stack([sites.ravel(), shifted.ravel()], axis=1)
            if self.boundary != 'periodic':
                edge = np.indices(self.shape_)[axis].ravel() == length - 1
                pairs = pairs[~edge]
            for first, second in pairs.tolist():
                if first != second:
                    found.setdefault(frozenset((first, second)), (first, second))
        return list(found.values())
