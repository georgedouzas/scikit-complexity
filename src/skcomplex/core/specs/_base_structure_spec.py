"""Define the base structure spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import shapely

from ..errors import ValidationError
from ..types import AnyArray, Array
from ._base_spec import BaseSpec

SPACE_DIMENSIONS = 3


@dataclass(frozen=True)
class BaseStructureSpec(BaseSpec):
    """Base of the structures of a rewriting element at one generation.

    A structure is a word, a patch of tiles, or a graph. The readers `parts`, `segments`, `polygons`, `geometry`, and
    `raster` describe the structure. Subclasses give the parts, the segments, and the polygons, and the measures follow
    from them with Shapely. Every subclass has the field `generation`, and its arrays are read-only, built with
    `_frozen`, so later generations cannot change a recorded structure.
    """

    @property
    def parts(self) -> int:
        """Number of parts, symbols, tiles, or nodes."""
        raise NotImplementedError

    @property
    def has_geometry(self) -> bool:
        """Whether the structure has a geometry, a patch or a word with a word geometry."""
        return False

    def get_segments(self) -> AnyArray:
        """Return the line segments of the geometry.

        Returns:
            segments:
                The array of shape `(k, 2, d)`, empty without segments.
        """
        return np.zeros((0, 2, 2))

    def get_polygons(self) -> AnyArray:
        """Return the polygons of the geometry.

        There is one coordinate array of shape `(t, k, 2)` per vertex count `k`.

        Returns:
            polygons:
                An object array of coordinate arrays, empty without polygons.
        """
        return np.zeros(0, dtype=object)

    def get_geometry(self) -> AnyArray:
        """Return the segments and polygons as an array of Shapely geometries.

        Returns:
            geometry:
                The line strings of the segments followed by the polygons.
        """
        return self._geometry

    @functools.cached_property
    def _geometry(self) -> AnyArray:
        """Line strings of the segments followed by the polygons, built once."""
        segments = self.get_segments()
        lines = shapely.linestrings(segments[..., :2]) if len(segments) else np.zeros(0, dtype=object)
        shapes = [shapely.polygons(coordinates) for coordinates in self.get_polygons()]
        return np.concatenate([lines, *shapes]) if shapes else lines

    def get_length(self) -> float:
        """Return the total length of the segments and the perimeters of the polygons.

        Returns:
            length:
                The total length.
        """
        segments = np.asarray(self.get_segments(), dtype=float)
        if segments.shape[-1] == SPACE_DIMENSIONS:
            return float(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=-1).sum())
        return float(shapely.length(self.get_geometry()).sum())

    def get_area(self) -> float:
        """Return the total area of the polygons.

        Returns:
            area:
                The sum of the areas of the polygons, 0 without polygons.
        """
        return float(shapely.area(self.get_geometry()).sum())

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Return the bounding box of the geometry in the plane.

        Returns:
            bounds:
                `(xmin, ymin, xmax, ymax)`, or zeros without geometry.
        """
        geometry = self.get_geometry()
        if not len(geometry):
            return (0.0, 0.0, 0.0, 0.0)
        box = shapely.bounds(geometry)
        return (
            float(box[:, 0].min()),
            float(box[:, 1].min()),
            float(box[:, 2].max()),
            float(box[:, 3].max()),
        )

    def get_raster(self, resolution: int, bounds: Sequence[float] | None = None) -> npt.NDArray[np.bool_]:
        """Return a square boolean grid of the cells the geometry covers.

        The first row of the grid is at the top. Segments mark the cells of points spaced at most half a cell apart
        along them, and polygons mark the cells whose centres they contain, so the grid suits
        `scaling(kind='box-counting')`.

        Args:
            resolution:
                The number of cells per side.

            bounds:
                The box `(xmin, ymin, xmax, ymax)` to rasterise, or `None` for the bounding box of the geometry. The
                grid covers a square with the longer side of the box.

        Returns:
            grid:
                The boolean grid of shape `(resolution, resolution)`.

        Raises:
            ValidationError:
                The resolution is not a positive integer.
        """
        if not isinstance(resolution, int | np.integer) or resolution < 1:
            error_msg = f'resolution = {resolution!r} must be a positive integer.'
            raise ValidationError(error_msg)
        xmin, ymin, xmax, ymax = self.get_bounds() if bounds is None else bounds
        side = max(xmax - xmin, ymax - ymin) or 1.0
        cell = side / resolution
        grid = np.zeros((resolution, resolution), dtype=bool)
        geometry = self.get_geometry()
        if not len(geometry):
            return grid
        lines = geometry[shapely.get_type_id(geometry) == shapely.GeometryType.LINESTRING]
        if len(lines):
            points = shapely.get_coordinates(shapely.segmentize(lines, cell / 2))
            self._mark(grid, points, xmin, ymax, cell)
        shapes = geometry[shapely.get_type_id(geometry) == shapely.GeometryType.POLYGON]
        if len(shapes):
            centres = (np.arange(resolution) + 0.5) * cell
            x, y = np.meshgrid(xmin + centres, ymax - centres)
            tree = shapely.STRtree(shapes)
            found = tree.query(shapely.points(x.ravel(), y.ravel()), predicate='intersects')[0]
            grid.ravel()[np.unique(found)] = True
        return grid

    @staticmethod
    def _mark(grid: npt.NDArray[np.bool_], points: Array, xmin: float, ymax: float, cell: float) -> None:
        """Mark the cells of a grid that hold points."""
        resolution = grid.shape[0]
        columns = np.clip(((points[:, 0] - xmin) / cell).astype(int), 0, resolution - 1)
        rows = np.clip(((ymax - points[:, 1]) / cell).astype(int), 0, resolution - 1)
        grid[rows, columns] = True

    @staticmethod
    def _get_frozen[S: np.generic](array: npt.ArrayLike, dtype: type[S]) -> npt.NDArray[S]:
        """Return a read-only copy of an array.

        Later generations cannot change a recorded structure through it.
        """
        result: npt.NDArray[S] = np.array(array, dtype=dtype)
        result.setflags(write=False)
        return result
