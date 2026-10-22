"""Define lattice spaces, boxes of Z^d with their boundaries, and the neighbourhoods of their sites."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..base import (
    SpaceKind,
)
from ._spaces import Space

BOUNDARIES = ('periodic', 'fixed', 'reflecting')
GEOMETRIES = ('square', 'triangular')
TRIANGULAR_OFFSETS = ((0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1))
TRIANGULAR_ROW_HEIGHT = float(np.sqrt(3.0) / 2)

type Cell = tuple[int, ...]
type Offset = tuple[int, ...]


class LatticeSpace(Space):
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

    Examples:
        >>> from skcomplex.core.space import LatticeSpace
        >>> space = LatticeSpace((10, 20))
        >>> space.size, space.n_dim
        (200, 2)
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.LATTICE

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

    @property
    def width(self) -> int:
        """Return the number of sites along the first dimension."""
        return self.dimensions[0]

    @property
    def height(self) -> int:
        """Return the number of sites along the second dimension."""
        return self.dimensions[1]

    @property
    def torus(self) -> bool:
        """Check whether opposite edges are joined."""
        return self.boundary == 'periodic'

    @property
    def dimensions(self) -> tuple[int, ...]:
        """Number of sites along every dimension."""
        return (int(self.shape),) if np.ndim(self.shape) == 0 else tuple(int(size) for size in self.shape)  # type: ignore[arg-type, union-attr]

    @property
    def spacings(self) -> tuple[float, ...]:
        """Distance between neighbouring sites along every dimension."""
        if np.ndim(self.spacing) == 0:
            return (float(self.spacing),) * self.n_dim  # type: ignore[arg-type]
        return tuple(float(item) for item in self.spacing)  # type: ignore[union-attr]

    @property
    def n_dim(self) -> int:  # type: ignore[override]
        """Number of dimensions."""
        return len(self.dimensions)

    @property
    def size(self) -> int:
        """Number of sites."""
        return int(np.prod(self.dimensions))

    @property
    def positions(self) -> npt.NDArray[np.float64]:
        """Position of every site in space, one row per site in member order, scaled by the spacing.

        Examples:
            >>> from skcomplex.core.space import LatticeSpace
            >>> LatticeSpace((2, 2), geometry='triangular').positions.round(3).tolist()
            [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [1.5, 0.866]]
        """
        coordinates = np.indices(self.dimensions).reshape(self.n_dim, -1).T.astype(float)
        if self.geometry == 'triangular':
            rows, columns = coordinates.T
            coordinates = np.column_stack([columns + rows / 2, rows * TRIANGULAR_ROW_HEIGHT])
        return coordinates * np.asarray(self.spacings)

    def check(self) -> None:
        """Raise an error when the shape, the geometry, or the boundary is not valid.

        Raises:
            ValueError:
                The shape has a size that is not a positive integer, the geometry or the boundary is unknown, or a
                triangular lattice does not have two dimensions.
        """
        if not self.dimensions or any(size < 1 for size in self.dimensions):
            error_msg = (
                f'A lattice has a positive number of sites along every dimension, but its shape is {self.shape}.'
            )
            raise ValueError(error_msg)
        if self.boundary not in BOUNDARIES:
            error_msg = f'Unknown lattice boundary {self.boundary!r}. Use one of {", ".join(BOUNDARIES)}.'
            raise ValueError(error_msg)
        if self.geometry not in GEOMETRIES:
            error_msg = f'Unknown lattice geometry {self.geometry!r}. Use one of {", ".join(GEOMETRIES)}.'
            raise ValueError(error_msg)
        if self.geometry == 'triangular' and self.n_dim != 2:  # noqa: PLR2004
            error_msg = f'A triangular lattice has two dimensions, but its shape is {self.shape}.'
            raise ValueError(error_msg)

    def neighbor(self, cell: Cell, offset: Offset) -> Cell | None:
        """Return the site at an offset from a site, or `None` outside a lattice with fixed edges.

        Args:
            cell:
                The coordinates of the site.

            offset:
                The offset along every dimension.

        Returns:
            neighbor:
                The coordinates of the neighbouring site, wrapped or mirrored at the edges.
        """
        coordinates = []
        for position, step, size in zip(cell, offset, self.dimensions, strict=True):
            value = position + step
            if self.boundary == 'periodic':
                value %= size
            elif self.boundary == 'reflecting':
                period = 2 * size
                value %= period
                value = value if value < size else period - 1 - value
            elif not 0 <= value < size:
                return None
            coordinates.append(int(value))
        return tuple(coordinates)

    def neighbors(self, cell: Cell, neighborhood: 'Neighborhood') -> list[Cell | None]:
        """Return the sites of a neighbourhood of a site, in the neighbourhood's order.

        Args:
            cell:
                The coordinates of the site.

            neighborhood:
                The neighbourhood.

        Returns:
            neighbors:
                The coordinates of every neighbouring site, or `None` outside a lattice with fixed edges.

        Examples:
            >>> from skcomplex.core.space import LatticeSpace, Neighborhood
            >>> LatticeSpace((10, 10)).neighbors((0, 0), Neighborhood.moore())
            [(9, 9), (9, 0), (9, 1), (0, 9), (0, 1), (1, 9), (1, 0), (1, 1)]
        """
        return [self.neighbor(cell, offset) for offset in neighborhood.offsets]

    def index_table(self, neighborhood: 'Neighborhood') -> npt.NDArray[np.intp]:
        """Return the member index of every neighbour of every site, with `size` for sites outside fixed edges.

        Args:
            neighborhood:
                The neighbourhood.

        Returns:
            table:
                One row per site and one column per offset.
        """
        self.check()
        if not isinstance(neighborhood, Neighborhood):
            error_msg = f'A lattice reads a neighbourhood of offsets, but it was given {neighborhood!r}.'
            raise ValueError(error_msg)  # noqa: TRY004
        if neighborhood.offsets and len(neighborhood.offsets[0]) != self.n_dim:
            error_msg = (
                f'The neighbourhood has offsets of dimension {len(neighborhood.offsets[0])}, but the lattice has '
                f'{self.n_dim} dimensions.'
            )
            raise ValueError(error_msg)
        coordinates = np.indices(self.dimensions).reshape(self.n_dim, -1)
        sizes = np.asarray(self.dimensions)[:, None]
        table = np.empty((self.size, len(neighborhood.offsets)), dtype=np.intp)
        for column, offset in enumerate(neighborhood.offsets):
            shifted = coordinates + np.asarray(offset)[:, None]
            outside = np.zeros(self.size, dtype=bool)
            if self.boundary == 'periodic':
                shifted %= sizes
            elif self.boundary == 'reflecting':
                shifted %= 2 * sizes
                shifted = np.where(shifted < sizes, shifted, 2 * sizes - 1 - shifted)
            else:
                outside = ((shifted < 0) | (shifted >= sizes)).any(axis=0)
                shifted = np.clip(shifted, 0, sizes - 1)
            flat = np.ravel_multi_index(tuple(shifted), self.dimensions)
            table[:, column] = np.where(outside, self.size, flat)
        return table


@dataclass(frozen=True)
class Neighborhood:
    """Ordered offsets of the sites a local rule reads, with the site itself as the zero offset when included.

    Args:
        offsets:
            The offsets, one tuple of integers per neighbour, in the order rules receive their values.

    Examples:
        >>> from skcomplex.core.space import Neighborhood
        >>> Neighborhood.von_neumann().offsets
        ((-1, 0), (0, -1), (0, 1), (1, 0))
        >>> len(Neighborhood.moore(center=True))
        9
    """

    offsets: tuple[Offset, ...]

    def __post_init__(self) -> None:
        offsets = tuple(tuple(int(step) for step in np.atleast_1d(offset)) for offset in self.offsets)
        if len({len(offset) for offset in offsets}) > 1:
            error_msg = 'Every offset of a neighbourhood has the same number of dimensions.'
            raise ValueError(error_msg)
        object.__setattr__(self, 'offsets', offsets)

    def __len__(self) -> int:
        """Return the number of offsets."""
        return len(self.offsets)

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
