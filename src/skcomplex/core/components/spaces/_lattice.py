"""Define the space of a grid of sites, square or triangular."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ...enums import BoundaryKind
from ...errors import ValidationError
from ._base import BaseNeighborhood, BaseSpace
from ._continuous import AXES
from ._finite_volume_spec import FiniteVolumeSpec
from ._lattice_neighborhood import LatticeNeighborhood
from ._radius import RadiusNeighborhood

BOUNDARIES = (BoundaryKind.PERIODIC, BoundaryKind.FIXED, BoundaryKind.REFLECTING)
GEOMETRIES = ('square', 'triangular')
SQUARE_METRICS = ('chebyshev', 'manhattan')
TRIANGULAR_ROW_HEIGHT = float(np.sqrt(3.0) / 2)
TRIANGULAR_DIMENSIONS = 2
DISCRETIZATIONS = ('finite-difference', 'spectral', 'finite-volume')


class LatticeSpace(BaseSpace):
    """A space of a box of `Z^d`, or of the triangular lattice in two dimensions.

    It places the members of a group on a grid of sites with integer coordinates, one member per site, such as the 200
    cells of `LatticeSpace((10, 20))`. The sites are numbered row by row, so site `(i, j)` of a grid of shape `(n, m)`
    is member `i * m + j`. A triangular site `(i, j)` sits at `(j + i / 2, i sqrt(3) / 2)`, so it has six neighbours
    at distance one. A field on the grid turns operators such as `laplacian` into sums over neighbouring sites. The
    spins of a magnet sit on a square grid, the trees of a forest fire on a map of cells, and the temperature of a rod
    on a row of points.

    Read more in the user guide, under Spaces.

    Args:
        shape:
            The number of sites along every dimension.

        geometry:
            `'square'` for the integer lattice or `'triangular'` for the triangular lattice in two dimensions.

        boundary:
            The edges of the lattice, or the edge of each state of a field on it by name.

            - `'periodic'`: opposite edges join, as on a ring or a torus.
            - `'fixed'`: the value `fill` is read outside the box.
            - `'reflecting'`: the mirror image of the box is read outside it.
            - A mapping such as `{'T': ('fixed', 0.0), 'c': 'zero-flux'}`: each named state of a field takes
              `'periodic'`, `'zero-flux'` for an insulated wall, or `('fixed', value)` for a wall held at a value,
              and the other states and the neighbours of members join periodically.

        discretization:
            How the operators of a field on the lattice, such as `laplacian`, become numbers.

            - `'finite-difference'`: stencils of neighbouring sites.
            - `'spectral'`: products of Fourier modes on a periodic lattice, with the two-thirds rule, which drops the
              highest modes so that their products create no false ones.
            - `'finite-volume'`: averages over cells, so a rate `-divergence(flux)` of a state is a conservation law
              solved with the scheme of `scheme`.

        scheme:
            The limiter, numerical flux, and wave speed of a finite-volume field, or `None` for the minmod limiter and
            the Rusanov flux.

        symmetry:
            The parity of each state of a field by name along every axis of a periodic lattice, `'odd'`, `'even'`, or
            `None`, which the rates keep exactly, such as `{'T': ('even', 'odd')}`. A lattice twice as long as a
            domain along an axis, with states odd or even about its middle, stands for walls there, such as the
            free-slip walls of convection. A vector or tensor state takes one such sequence for its first component,
            or one sequence per component.

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
            The name of the space in its system, such as `'grid'`.

    Attributes:
        axes_:
            The names of the coordinates of a cell, such as `('x', 'y')` in two dimensions, which an agent's values of
            the same names place it at.

        spacings_:
            The distance between neighbouring sites along every dimension.

        metric_:
            The metric of the ball agents query.

    See Also:
        NetworkSpace: Places members on the nodes of any graph.
        ContinuousSpace: Lets members move to any real position.
        PhaseSpace: Adds axes such as velocity to a lattice of positions.

    Examples:
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> space = LatticeSpace((10, 20)).prepare()
        >>> space.size_, space.n_dim_, space.shape_
        (200, 2, (10, 20))
    """

    space_tags: ClassVar[tuple[str, ...]] = ('lattice',)

    def __init__(
        self,
        shape: Sequence[int] | int,
        *,
        geometry: str = 'square',
        boundary: str | Mapping[str, object] = 'periodic',
        discretization: str = 'finite-difference',
        scheme: FiniteVolumeSpec | None = None,
        symmetry: Mapping[str, Sequence[object]] | None = None,
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
        self.discretization = discretization
        self.scheme = scheme
        self.symmetry = symmetry
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
        if not isinstance(self.boundary, Mapping) and self.boundary not in BOUNDARIES:
            error_msg = (
                f'Unknown lattice boundary {self.boundary!r}. Use one of {", ".join(BOUNDARIES)}, or the edge of each '
                "state, such as {'T': ('fixed', 0.0)}."
            )
            raise ValidationError(error_msg)
        self._check_field()
        if self.geometry not in GEOMETRIES:
            error_msg = f'Unknown lattice geometry {self.geometry!r}. Use one of {", ".join(GEOMETRIES)}.'
            raise ValidationError(error_msg)
        if self.geometry == 'triangular' and len(shape) != TRIANGULAR_DIMENSIONS:
            error_msg = f'A triangular lattice has two dimensions, but its shape is {self.shape}.'
            raise ValidationError(error_msg)
        self._check_metric()
        self.shape_ = tuple(int(size) for size in np.ravel(self.shape).tolist())
        self.n_dim_ = len(self.shape_)
        many = self.n_dim_ > len(AXES)
        self.axes_ = tuple(f'x{axis}' for axis in range(self.n_dim_)) if many else AXES[: self.n_dim_]
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
        self.periodic_ = self._get_edges() == 'periodic'
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

    def _check_field(self) -> None:
        """Raise an error for an unknown discretization or a misplaced scheme.

        A scheme goes only with the finite-volume discretization. The field on the lattice checks the edge of each of
        its states.
        """
        if self.discretization not in DISCRETIZATIONS:
            error_msg = (
                f'Unknown discretization {self.discretization!r} of a lattice. Use one of {", ".join(DISCRETIZATIONS)}.'
            )
            raise ValidationError(error_msg)
        if self.scheme is None:
            return
        if not isinstance(self.scheme, FiniteVolumeSpec) or self.discretization != 'finite-volume':
            error_msg = (
                f'The scheme of a lattice is {self.scheme!r}. Give a `FiniteVolumeSpec` to a lattice with '
                "discretization='finite-volume'."
            )
            raise ValidationError(error_msg)
        self.scheme._check()

    def _get_edges(self) -> str:
        """Return the edges of the lattice.

        They join periodically when the boundary names the edge of each state.
        """
        return 'periodic' if isinstance(self.boundary, Mapping) else str(self.boundary)

    def _get_axis_boundary(self, axis: int) -> str:
        """Return the edges of an axis, the one boundary of the lattice."""
        return self._get_edges()

    def _copy_with_edges(self, boundary: str, fill: float) -> LatticeSpace:
        """Return the sites of the lattice with other edges.

        The reads of a state with its own boundary use them.
        """
        return LatticeSpace(self.shape_, boundary=boundary, fill=fill).prepare()

    def _has_sites(self) -> bool:
        """Check whether the space has sites joined by neighbours, which its cells are."""
        return True

    def _get_default_neighborhood(self) -> RadiusNeighborhood:
        """Return the neighbourhood agents query, the unit ball under the lattice metric."""
        return RadiusNeighborhood(1, metric=self.metric_)

    def _get_index_table(self, neighborhood: BaseNeighborhood) -> npt.NDArray[np.intp]:
        """Return the member index of every neighbour of every site.

        Sites outside fixed edges get `size_`.
        """
        offsets = neighborhood._get_offsets(self.n_dim_)
        if offsets is None:
            error_msg = f'A lattice reads a lattice neighbourhood of offsets, but it was given {neighborhood!r}.'
            raise ValidationError(error_msg)
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
        return LatticeNeighborhood(tuple(itertools.product(range(-radius, radius + 1), repeat=self.n_dim_)))

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
