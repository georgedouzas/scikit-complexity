"""Define continuous spaces, boxes of real positions or the whole of R^d."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import shapely
from shapely import Polygon, STRtree

from ...errors import SpaceError, ValidationError
from ._base import BaseNeighborhood, BaseSpace

BOUNDARIES = ('open', 'periodic', 'reflecting')
REGION_DIMENSIONS = 2


class ContinuousSpace(BaseSpace):
    """Continuous space of real positions, a box with a boundary or the whole of `R^d`.

    A box has an extent along every dimension, from 0 to the extent. Its boundary acts on every position state of the
    members on it: `'open'` leaves positions alone, `'periodic'` joins opposite sides and measures pair distances by the
    minimum image, and `'reflecting'` mirrors positions back into the box and reverses the velocity across the wall.
    Without an extent the space is unbounded and has `n_dim` dimensions. The positions are states of the members, so
    the space has no sites, and a site has no neighbours in it. A space of two dimensions can hold layers of regions,
    the polygons of a map in the coordinates of the space, which agents are placed in, moved to, and ask about.

    Args:
        extent:
            The side of the box along every dimension, or empty for an unbounded space.

        n_dim:
            The number of dimensions of an unbounded space, or `None` for the number of sides of the box.

        boundary:
            `'open'`, `'periodic'`, or `'reflecting'`.

        regions:
            The polygons of every layer of regions by the name of the layer, or `None` for no layers.

        label:
            The label of the space in its system.

    Attributes:
        extent_:
            The side of the box along every dimension, or `()` for an unbounded space.

        regions_:
            The Shapely `STRtree` of the polygons of every layer of regions by the name of the layer.

    Raises:
        SpaceError:
            A region is not a valid polygon in two dimensions, or the space has regions and other than two dimensions,
            raised by `prepare`.

    Examples:
        >>> from skcomplex.core.spaces import ContinuousSpace
        >>> ContinuousSpace(extent=(10.0, 5.0), boundary='periodic').check().prepare().extent_
        (10.0, 5.0)
        >>> ContinuousSpace(n_dim=3).check().prepare().n_dim_
        3
        >>> from shapely import Point, box
        >>> layers = {'parks': [box(0.0, 0.0, 1.0, 1.0), box(2.0, 2.0, 3.0, 3.0)]}
        >>> space = ContinuousSpace(extent=(5.0, 5.0), regions=layers).check().prepare()
        >>> space.regions_['parks'].query(Point(2.5, 2.5), predicate='intersects').tolist()
        [1]
    """

    def __init__(
        self,
        extent: Sequence[float] = (),
        *,
        n_dim: int | None = None,
        boundary: str = 'open',
        regions: Mapping[str, Sequence[Polygon]] | None = None,
        label: str = 'space',
    ) -> None:
        super().__init__(label)
        self.extent = extent
        self.n_dim = n_dim
        self.boundary = boundary
        self.regions = regions

    def _check(self) -> None:
        """Raise an error for an unknown boundary, no dimension, a side that is not positive, or walls in `R^d`."""
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
        """Return the index of every layer of regions, after checking the regions and the dimension of the space."""
        layers = self.regions or {}
        if layers and self.n_dim_ != REGION_DIMENSIONS:
            error_msg = (
                f'The layers {", ".join(layers)} hold regions of a map, and the space has {self.n_dim_} dimensions. '
                f'Give the space {REGION_DIMENSIONS} dimensions.'
            )
            raise SpaceError(error_msg)
        for layer, regions in layers.items():
            for index, region in enumerate(regions):
                self._check_region(layer, index, region)
        return {layer: STRtree(list(regions)) for layer, regions in layers.items()}

    def _prepare(self, context: None = None) -> None:
        """Set the extent of the box, its dimension, its boundary, and the index of its regions, with no sites."""
        self.extent_ = tuple(float(side) for side in self.extent)
        self.n_dim_ = int(self.n_dim) if self.n_dim is not None else len(self.extent_)
        self.size_ = None
        self.shape_ = ()
        self.positions_ = np.zeros((0, self.n_dim_))
        self.periodic_ = self.boundary == 'periodic'
        self.default_visual_ = None
        self.regions_ = self._get_regions()

    def _get_neighbors(self, site: int, neighborhood: BaseNeighborhood | None = None) -> npt.NDArray[np.intp]:
        """Return no neighbours, since the space has no sites."""
        return np.zeros(0, dtype=np.intp)
