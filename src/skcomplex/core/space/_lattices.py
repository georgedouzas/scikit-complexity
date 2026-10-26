"""Define lattice spaces, boxes of Z^d with their boundaries, and the neighbourhoods of their sites."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt

from ..base import Requires, SpaceKind, ValidationError
from ._bases import BaseNeighborhood, BaseSpace

BOUNDARIES = ('periodic', 'fixed', 'reflecting')
GEOMETRIES = ('square', 'triangular')
TRIANGULAR_OFFSETS = ((0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1))
TRIANGULAR_ROW_HEIGHT = float(np.sqrt(3.0) / 2)

type Offset = tuple[int, ...]


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

    Attributes:
        spacings_:
            The distance between neighbouring sites along every dimension.

    Examples:
        >>> from skcomplex.core.space import LatticeSpace
        >>> space = LatticeSpace((10, 20)).prepare()
        >>> space.size_, space.n_dim_, space.shape_
        (200, 2, (10, 20))
        >>> space.run(0).tolist()
        [199, 180, 181, 19, 1, 39, 20, 21]
        >>> LatticeSpace((2, 2), geometry='triangular').prepare().positions_.round(3).tolist()
        [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [1.5, 0.866]]
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.LATTICE
    boundaries: ClassVar[tuple[str, ...]] = BOUNDARIES

    size_: int
    spacings_: tuple[float, ...]

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
    ) -> None:
        self.shape = shape
        self.geometry = geometry
        self.boundary = boundary
        self.fill = fill
        self.spacing = spacing
        self.neighborhood = neighborhood
        self.capacity = capacity

    def _check(self) -> None:
        """Raise an error when the shape, the geometry, or the boundary is not valid.

        Raises:
            ValidationError:
                The shape has a size that is not a positive integer, the geometry or the boundary is unknown, or a
                triangular lattice does not have two dimensions.
        """
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

    def _prepare(self) -> None:
        """Set the shape, the size, the spacings, and the position of every site."""
        shape = tuple(np.ravel(self.shape).tolist())
        self.shape_ = tuple(int(size) for size in shape)
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

    def _default_neighborhood(self) -> 'Neighborhood':
        """Return the neighbourhood agents query, the von Neumann or the Moore neighbourhood of radius 1."""
        if self.geometry == 'triangular':
            return Neighborhood.triangular()
        if self.neighborhood == 'von_neumann':
            return Neighborhood.von_neumann(n_dim=self.n_dim_)
        return Neighborhood.moore(n_dim=self.n_dim_)

    def _index_table(self, neighborhood: 'Neighborhood') -> npt.NDArray[np.intp]:
        """Return the member index of every neighbour of every site, with `size_` for sites outside fixed edges."""
        if not isinstance(neighborhood, Neighborhood):
            error_msg = f'A lattice reads a neighbourhood of offsets, but it was given {neighborhood!r}.'
            raise ValidationError(error_msg)
        neighborhood = neighborhood.check()
        if neighborhood.offsets and len(neighborhood.offsets[0]) != self.n_dim_:
            error_msg = (
                f'The neighbourhood has offsets of dimension {len(neighborhood.offsets[0])}, but the lattice has '
                f'{self.n_dim_} dimensions.'
            )
            raise ValidationError(error_msg)
        coordinates = np.indices(self.shape_).reshape(self.n_dim_, -1)
        sizes = np.asarray(self.shape_)[:, None]
        table = np.empty((self.size_, len(neighborhood.offsets)), dtype=np.intp)
        for column, offset in enumerate(neighborhood.offsets):
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

    def _ball(self, radius: int) -> 'Neighborhood':
        """Return the box of offsets within a radius along every dimension, the site itself included."""
        return Neighborhood.radius(radius, n_dim=self.n_dim_)

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


@dataclass(frozen=True)
class Neighborhood(BaseNeighborhood):
    """Ordered offsets of the sites a local rule reads, with the site itself as the zero offset when included.

    Args:
        offsets:
            The offsets, one tuple of integers per neighbour, in the order rules receive their values.

    Examples:
        >>> from skcomplex.core.space import Neighborhood
        >>> Neighborhood.von_neumann().offsets
        ((-1, 0), (0, -1), (0, 1), (1, 0))
        >>> len(Neighborhood.moore(center=True).offsets)
        9
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.LATTICE})

    offsets: tuple[Offset, ...]

    def _check(self) -> Self:
        """Return the neighbourhood with its arguments checked and normalized."""
        changes: dict[str, Any] = {}
        offsets = tuple(tuple(int(step) for step in np.atleast_1d(offset)) for offset in self.offsets)
        if len({len(offset) for offset in offsets}) > 1:
            error_msg = 'Every offset of a neighbourhood has the same number of dimensions.'
            raise ValueError(error_msg)
        changes['offsets'] = offsets
        return replace(self, **changes) if changes else self

    @classmethod
    def _of(cls, radius: int, n_dim: int, keep: Callable[[Offset], bool], center: bool) -> 'Neighborhood':
        """Return the offsets within a radius that a condition keeps, in lexicographic order."""
        offsets = [
            offset
            for offset in itertools.product(range(-radius, radius + 1), repeat=n_dim)
            if any(offset) and keep(offset)
        ]
        return cls(((0,) * n_dim, *offsets) if center else tuple(offsets))

    @classmethod
    def moore(cls, radius: int = 1, *, n_dim: int = 2, center: bool = False) -> 'Neighborhood':
        """Return the sites within a radius in every dimension, the 8 surrounding cells in 2D for radius 1.

        Args:
            radius:
                The largest offset along any dimension.

            n_dim:
                The number of dimensions.

            center:
                Whether the site itself comes first.

        Returns:
            neighborhood:
                The neighbourhood.
        """
        return cls._of(radius, n_dim, lambda _: True, center)

    @classmethod
    def von_neumann(cls, radius: int = 1, *, n_dim: int = 2, center: bool = False) -> 'Neighborhood':
        """Return the sites within a Manhattan distance, the 4 adjacent cells in 2D for radius 1.

        Args:
            radius:
                The largest Manhattan distance.

            n_dim:
                The number of dimensions.

            center:
                Whether the site itself comes first.

        Returns:
            neighborhood:
                The neighbourhood.
        """
        return cls._of(radius, n_dim, lambda offset: sum(map(abs, offset)) <= radius, center)

    @classmethod
    def triangular(cls, *, center: bool = False) -> 'Neighborhood':
        """Return the six neighbours of a triangular lattice, counterclockwise from the direction of the second axis.

        Direction `k` points at `60 k` degrees in the plane, so direction `k + 3` modulo 6 is opposite to direction `k`.

        Args:
            center:
                Whether the site itself comes first.

        Returns:
            neighborhood:
                The neighbourhood.

        Examples:
            >>> from skcomplex.core.space import Neighborhood
            >>> Neighborhood.triangular().offsets
            ((0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1))
        """
        return cls(((0, 0), *TRIANGULAR_OFFSETS) if center else TRIANGULAR_OFFSETS)

    @classmethod
    def radius(cls, radius: int = 1, *, n_dim: int = 1) -> 'Neighborhood':
        """Return the sites within a radius, the site itself included in place, as one-dimensional automata read them.

        For radius 1 in one dimension the order is left, centre, right, the order of Wolfram's rule numbers.

        Args:
            radius:
                The largest offset along any dimension.

            n_dim:
                The number of dimensions.

        Returns:
            neighborhood:
                The neighbourhood.

        Examples:
            >>> from skcomplex.core.space import Neighborhood
            >>> Neighborhood.radius().offsets
            ((-1,), (0,), (1,))
        """
        return cls(tuple(itertools.product(range(-radius, radius + 1), repeat=n_dim)))
