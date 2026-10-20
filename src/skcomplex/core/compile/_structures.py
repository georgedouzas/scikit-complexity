"""Define the structures of rewriting elements at one generation, with their geometry and geometric measures."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
import shapely

from ..base import ValidationError
from ..model import Module, Prototile

type Array = npt.NDArray[Any]


def _frozen(array: npt.ArrayLike, dtype: type) -> Array:
    """Return a read-only copy of an array, so later generations cannot change a recorded structure."""
    result: Array = np.array(array, dtype=dtype)
    result.setflags(write=False)
    return result


class Structure:
    """Base class of the structure of a rewriting element at one generation.

    Subclasses give their line segments and polygons, from which the measures follow with Shapely.

    Args:
        generation:
            The generation.
    """

    def __init__(self, generation: int) -> None:
        self.generation = generation
        self._geometry: Array | None = None

    @property
    def parts(self) -> int:
        """Number of parts, symbols, tiles, or nodes."""
        raise NotImplementedError

    @property
    def has_geometry(self) -> bool:
        """Whether the structure has a geometry, a patch or a word with a turtle."""
        return False

    def segments(self) -> Array:
        """Return the line segments of the geometry.

        Returns:
            segments:
                The array of shape `(k, 2, d)`, empty without segments.
        """
        return np.zeros((0, 2, 2))

    def polygons(self) -> Array:
        """Return the polygons of the geometry, one coordinate array of shape `(t, k, 2)` per vertex count `k`.

        Returns:
            polygons:
                An object array of coordinate arrays, empty without polygons.
        """
        return np.zeros(0, dtype=object)

    def geometry(self) -> Array:
        """Return the segments and polygons as an array of Shapely geometries.

        Returns:
            geometry:
                The line strings of the segments followed by the polygons.
        """
        if self._geometry is None:
            segments = self.segments()
            lines = shapely.linestrings(segments[..., :2]) if len(segments) else np.zeros(0, dtype=object)
            shapes = [shapely.polygons(coordinates) for coordinates in self.polygons()]
            self._geometry = np.concatenate([lines, *shapes]) if shapes else lines
        return self._geometry

    def length(self) -> float:
        """Return the total length of the segments and the perimeters of the polygons.

        Returns:
            length:
                The total length.
        """
        segments = self.segments()
        if segments.shape[-1] == 3:  # noqa: PLR2004
            return float(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=-1).sum())
        return float(shapely.length(self.geometry()).sum())

    def area(self) -> float:
        """Return the total area of the polygons.

        Returns:
            area:
                The sum of the areas of the polygons, 0 without polygons.
        """
        return float(shapely.area(self.geometry()).sum())

    def bounds(self) -> tuple[float, float, float, float]:
        """Return the bounding box of the geometry in the plane.

        Returns:
            bounds:
                `(xmin, ymin, xmax, ymax)`, or zeros without geometry.
        """
        geometry = self.geometry()
        if not len(geometry):
            return (0.0, 0.0, 0.0, 0.0)
        box = shapely.bounds(geometry)
        return (
            float(box[:, 0].min()),
            float(box[:, 1].min()),
            float(box[:, 2].max()),
            float(box[:, 3].max()),
        )

    def raster(self, resolution: int, bounds: Sequence[float] | None = None) -> npt.NDArray[np.bool_]:
        """Return a square boolean grid of the cells the geometry covers, with the first row at the top.

        Segments mark the cells of points spaced at most half a cell apart along them, and polygons mark the cells
        whose centres they contain, so the grid suits `box_counting_dimension`.

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
        xmin, ymin, xmax, ymax = self.bounds() if bounds is None else bounds
        side = max(xmax - xmin, ymax - ymin) or 1.0
        cell = side / resolution
        grid = np.zeros((resolution, resolution), dtype=bool)
        geometry = self.geometry()
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


class WordStructure(Structure):
    """Word of a string rewriting system at one generation, with the turtle geometry where a turtle is declared.

    Args:
        alphabet:
            The symbol names, indexed by code.

        codes:
            The symbol code of every module.

        parameters:
            The parameters of every module, padded with NaN.

        generation:
            The generation.

        segments:
            The turtle segments of shape `(k, 2, d)`, or `None` without a turtle.

        depths:
            The branch depth of every segment, or `None` without a turtle.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import WordStructure
        >>> str(WordStructure(('A', 'B'), np.array([0, 1, 0]), np.zeros((3, 0)), 2))
        'ABA'
    """

    def __init__(
        self,
        alphabet: Sequence[str],
        codes: npt.ArrayLike,
        parameters: npt.ArrayLike,
        generation: int,
        *,
        segments: npt.ArrayLike | None = None,
        depths: npt.ArrayLike | None = None,
    ) -> None:
        super().__init__(generation)
        self.alphabet = tuple(alphabet)
        self.codes = _frozen(codes, np.int64)
        self.parameters = _frozen(parameters, np.float64)
        self._segments = None if segments is None else _frozen(segments, np.float64)
        self.depths = None if depths is None else _frozen(depths, np.int64)

    def __len__(self) -> int:
        """Return the number of modules."""
        return len(self.codes)

    @property
    def parts(self) -> int:
        """Number of modules."""
        return len(self.codes)

    @property
    def has_geometry(self) -> bool:
        """Whether a turtle has drawn the word."""
        return self._segments is not None

    def modules(self) -> list[Module]:
        """Return the modules of the word with their parameters.

        Returns:
            modules:
                The modules in order.
        """
        modules = []
        for code, row in zip(self.codes.tolist(), self.parameters, strict=True):
            values = [float(value) for value in row if not np.isnan(value)]
            modules.append(Module(self.alphabet[code], *values))
        return modules

    def __str__(self) -> str:
        """Return the word, with modules separated by spaces when one has parameters or a long name."""
        modules = self.modules()
        texts = [
            module.name if not module.parameters else f'{module.name}({", ".join(f"{v:g}" for v in module.parameters)})'
            for module in modules
        ]
        spaced = any(module.parameters or len(module.name) > 1 for module in modules)
        return (' ' if spaced else '').join(texts)

    def __repr__(self) -> str:
        """Return the generation and the size."""
        return f'WordStructure(generation={self.generation}, size={len(self)})'

    def count(self, name: str) -> int:
        """Return the number of modules of a symbol.

        Args:
            name:
                The symbol.

        Returns:
            count:
                The number of modules with that symbol.
        """
        if name not in self.alphabet:
            return 0
        return int(np.count_nonzero(self.codes == self.alphabet.index(name)))

    def segments(self) -> Array:
        """Return the turtle segments.

        Returns:
            segments:
                The array of shape `(k, 2, d)`, empty without a turtle.
        """
        return np.zeros((0, 2, 2)) if self._segments is None else self._segments


class PatchStructure(Structure):
    """Patch of tiles at one generation, each a prototile under a homogeneous affine matrix.

    Args:
        prototiles:
            The prototiles, indexed by code.

        codes:
            The prototile code of every tile.

        matrices:
            The placement of every tile, of shape `(t, 3, 3)`.

        generation:
            The generation.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import PatchStructure
        >>> from skcomplex.core.model import Prototile
        >>> patch = PatchStructure([Prototile('interval', [(0, 0), (1, 0)])], [0], np.eye(3)[None], 0)
        >>> patch.length()
        1.0
    """

    def __init__(
        self,
        prototiles: Sequence[Prototile],
        codes: npt.ArrayLike,
        matrices: npt.ArrayLike,
        generation: int,
    ) -> None:
        super().__init__(generation)
        self.prototiles = tuple(prototiles)
        self.codes = _frozen(codes, np.int64)
        self.matrices = _frozen(matrices, np.float64).reshape(-1, 3, 3)

    def __len__(self) -> int:
        """Return the number of tiles."""
        return len(self.codes)

    def __repr__(self) -> str:
        """Return the generation and the size."""
        return f'PatchStructure(generation={self.generation}, size={len(self)})'

    @property
    def parts(self) -> int:
        """Number of tiles."""
        return len(self.codes)

    @property
    def has_geometry(self) -> bool:
        """Whether the patch has a geometry, which it always has."""
        return True

    def count(self, name: str) -> int:
        """Return the number of tiles of a prototile.

        Args:
            name:
                The name of the prototile.

        Returns:
            count:
                The number of tiles.
        """
        codes = [index for index, prototile in enumerate(self.prototiles) if prototile.name == name]
        return int(np.count_nonzero(np.isin(self.codes, codes)))

    def vertices(self, code: int) -> Array:
        """Return the vertices of the tiles of one prototile.

        Args:
            code:
                The code of the prototile.

        Returns:
            vertices:
                The array of shape `(t, k, 2)`.
        """
        points = self.prototiles[code].points()
        homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        matrices = self.matrices[self.codes == code]
        return np.einsum('tij,kj->tki', matrices, homogeneous)[..., :2]

    def segments(self) -> Array:
        """Return the tiles of segment prototiles as segments.

        Returns:
            segments:
                The array of shape `(k, 2, 2)`.
        """
        parts = [self.vertices(code) for code, item in enumerate(self.prototiles) if item.is_segment]
        return np.concatenate(parts) if parts else np.zeros((0, 2, 2))

    def polygons(self) -> Array:
        """Return the tiles of polygon prototiles, one coordinate array per prototile.

        Returns:
            polygons:
                An object array of arrays of shape `(t, k, 2)`.
        """
        parts = [self.vertices(code) for code, item in enumerate(self.prototiles) if not item.is_segment]
        result = np.empty(len(parts), dtype=object)
        result[:] = parts
        return result


class GraphStructure(Structure):
    """Host graph of a graph rewriting system at one generation.

    Args:
        graph:
            The graph, which the structure copies and freezes.

        generation:
            The generation.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.compile import GraphStructure
        >>> GraphStructure(nx.cycle_graph(4), 0).parts
        4
    """

    def __init__(self, graph: nx.Graph, generation: int) -> None:
        super().__init__(generation)
        self.graph = nx.freeze(graph.copy())

    def __len__(self) -> int:
        """Return the number of nodes."""
        return int(self.graph.number_of_nodes())

    def __repr__(self) -> str:
        """Return the generation and the size."""
        return f'GraphStructure(generation={self.generation}, size={len(self)})'

    @property
    def parts(self) -> int:
        """Number of nodes."""
        return len(self)

    def count(self, name: str) -> int:
        """Return the number of nodes with a label.

        Args:
            name:
                The label.

        Returns:
            count:
                The number of nodes whose `label` attribute is the label.
        """
        return sum(1 for _, label in self.graph.nodes(data='label') if label == name)
