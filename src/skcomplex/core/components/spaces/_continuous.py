"""Define the space of real positions, a box with walls or with no edges."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import shapely
from shapely import Polygon, STRtree

from ...errors import SpaceError, ValidationError
from ._base import BaseNeighborhood, BaseSpace

BOUNDARIES = ('open', 'periodic', 'reflecting')
AXES = ('x', 'y', 'z')
REGION_DIMENSIONS = 2


class ContinuousSpace(BaseSpace):
    """A space of real positions, a box with a boundary or the whole of `R^d`.

    Members move to any real position, and a value named after an axis is the position along it, such as `x` in
    `Element('walker', values={'x': 0.0}, space=ContinuousSpace(n_dim=1))`. The axes are `x`, `y`, and `z`, or `x0`,
    `x1`, and so on beyond three dimensions. A box runs from 0 to `extent` along every dimension, and without an extent
    the space has no edges. The space has no sites, since the positions are values of the members. A plane can hold
    named regions of a map, which agents start in, move to, and ask about. Atoms move in the box of a gas, birds fly in
    a flock, pedestrians walk through a station, and households live in the districts of a city.

    Read more in the user guide, under Spaces.

    Args:
        extent:
            The side of the box along every dimension, or empty for an unbounded space.

        n_dim:
            The number of dimensions of an unbounded space, or `None` for the number of sides of the box.

        boundary:
            What happens at the walls of the box. Positions stay as they are with `'open'`. Opposite sides join with
            `'periodic'`, and distances use the nearest copy across a wall. Positions mirror back into the box with
            `'reflecting'`, and the velocity across the wall reverses.

        regions:
            The polygon of every region, or the polygons of every layer of regions, by name, or `None` for no regions.
            A region is one polygon, such as `{'tract_1': polygon}`, and a layer is a list, such as
            `{'parks': [first, second]}`. Agents whose value `region` names a region start at a random point inside it.

        label:
            The name of the space in its system, such as `'grid'`.

    Attributes:
        extent_:
            The side of the box along every dimension, or `()` for an unbounded space.

        axes_:
            The names of the axes, such as `('x', 'y')` in two dimensions.

        regions_:
            The Shapely `STRtree` of the polygons of every region or layer of regions by name, one polygon for a region.

    Raises:
        SpaceError:
            A region is not a valid polygon in two dimensions, or the space has regions and other than two dimensions,
            raised by `prepare`.

    See Also:
        LatticeSpace: Places members on fixed sites of a grid.
        RadiusNeighborhood: Picks the members within a distance in the box.
        NearestNeighborhood: Picks the closest members in the box.

    Examples:
        >>> from skcomplex.core.components.spaces import ContinuousSpace
        >>> space = ContinuousSpace(extent=(10.0, 5.0), boundary='periodic').prepare()
        >>> space.extent_, space.axes_
        ((10.0, 5.0), ('x', 'y'))
    """

    space_tags: ClassVar[tuple[str, ...]] = ('continuous',)

    def __init__(
        self,
        extent: Sequence[float] = (),
        *,
        n_dim: int | None = None,
        boundary: str = 'open',
        regions: Mapping[str, Polygon | Sequence[Polygon]] | None = None,
        label: str = 'space',
    ) -> None:
        super().__init__(label)
        self.extent = extent
        self.n_dim = n_dim
        self.boundary = boundary
        self.regions = regions

    def _check_region(self, layer: str, index: int, region: object) -> None:
        """Raise an error for a region that is not a valid polygon in two dimensions."""
        name = f'The region {index} of the layer {layer!r}'
        if not isinstance(region, Polygon) or region.has_z:
            error_msg = f'{name} is {region!r}. Give a Shapely polygon in two dimensions.'
            raise SpaceError(error_msg)
        if region.is_empty or not region.is_valid:
            error_msg = f'{name} is not a valid polygon: {shapely.is_valid_reason(region)}. Repair it.'
            raise SpaceError(error_msg)

    def _get_regions(self) -> dict[str, STRtree]:
        """Return the index of every region or layer of regions.

        It checks the regions and the dimension of the space first. A named region is a layer of one polygon.
        """
        layers = self.regions or {}
        if layers and self.n_dim_ != REGION_DIMENSIONS:
            error_msg = (
                f'The layers {", ".join(layers)} hold regions of a map, and the space has {self.n_dim_} dimensions. '
                f'Give the space {REGION_DIMENSIONS} dimensions.'
            )
            raise SpaceError(error_msg)
        polygons = {name: [region] if isinstance(region, Polygon) else list(region) for name, region in layers.items()}
        for layer, regions in polygons.items():
            for index, region in enumerate(regions):
                self._check_region(layer, index, region)
        return {layer: STRtree(regions) for layer, regions in polygons.items()}

    def _prepare(self, context: None = None) -> None:
        """Set the extent, the dimension, the boundary, and the region index of the box.

        It first raises an error for an unknown boundary, no dimension, a bad side, or bad walls. The side must be
        positive, and there are no walls in `R^d`.

        The box has no sites.
        """
        super()._prepare(context)
        extent = tuple(self.extent)
        n_dim = self.n_dim if self.n_dim is not None else len(extent)
        if self.boundary not in BOUNDARIES:
            error_msg = f'Unknown boundary {self.boundary!r}. Use one of {", ".join(BOUNDARIES)}.'
            raise ValidationError(error_msg)
        if not n_dim:
            error_msg = 'A continuous space needs the sides of its box or a number of dimensions.'
            raise ValidationError(error_msg)
        if extent and (len(extent) != n_dim or min(extent) <= 0):
            error_msg = f'The box {extent} needs one positive side for each of its {n_dim} dimensions.'
            raise ValidationError(error_msg)
        if not extent and self.boundary != 'open':
            error_msg = f'An unbounded space has no walls for its boundary {self.boundary!r}. Give the sides of a box.'
            raise ValidationError(error_msg)
        self.extent_ = tuple(float(side) for side in self.extent)
        self.n_dim_ = int(self.n_dim) if self.n_dim is not None else len(self.extent_)
        many = self.n_dim_ > len(AXES)
        self.axes_ = tuple(f'x{axis}' for axis in range(self.n_dim_)) if many else AXES[: self.n_dim_]
        self.size_ = None
        self.shape_ = ()
        self.positions_ = np.zeros((0, self.n_dim_))
        self.periodic_ = self.boundary == 'periodic'
        self.default_visual_ = None
        self.regions_ = self._get_regions()

    def _get_neighbors(self, site: int, neighborhood: BaseNeighborhood | None = None) -> npt.NDArray[np.intp]:
        """Return no neighbours, since the space has no sites."""
        return np.zeros(0, dtype=np.intp)
