"""Define the space of an unstructured mesh of triangles or tetrahedra."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt
import skfem

from ...errors import ValidationError
from ._base import BaseNeighborhood, BaseSpace

TRIANGLE_DIMENSIONS = 2
MESH_DIMENSIONS = (2, 3)
ARRAY_RANK = 2


class MeshSpace(BaseSpace):
    """A space of an unstructured mesh of triangles or tetrahedra.

    The nodes of the mesh hold the members of finite element fields. Build the nodes and cells with scikit-fem, such as
    `skfem.MeshTri.init_lshaped().refined(3)`, and pass its `p` and `t` arrays.

    Unlike `LatticeSpace`, which is a regular grid, it fits domains of any shape. Choose it when a partial differential
    equation on an irregular domain is solved by finite elements.

    It appears in many fields. Heat flows through a machine part, groundwater seeps through an aquifer, and a membrane
    deforms under a load.

    Args:
        points:
            The coordinates of the nodes, one column per node and one row per dimension.

        cells:
            The node indices of every triangle or tetrahedron, one column per cell.

        label:
            The name of the space in its system, such as `'grid'`.

    Attributes:
        mesh_:
            The scikit-fem mesh of the nodes and cells, a `MeshTri` in 2D or a `MeshTet` in 3D.

        boundary_nodes_:
            The indices of the nodes on the boundary of the mesh.

    Examples:
        >>> import numpy as np
        >>> import skfem
        >>> from skcomplex.core.components.spaces import MeshSpace
        >>> grid = skfem.MeshTri.init_tensor(np.linspace(0, 1, 5), np.linspace(0, 1, 4))
        >>> mesh = MeshSpace(grid.p, grid.t).prepare()
        >>> mesh.size_, mesh.n_dim_, mesh.positions_.shape
        (20, 2, (20, 2))
    """

    space_tags: ClassVar[tuple[str, ...]] = ('continuous',)

    def __init__(self, points: npt.ArrayLike, cells: npt.ArrayLike, *, label: str = 'space') -> None:
        super().__init__(label)
        self.points = points
        self.cells = cells

    def _check_points(self, points: npt.NDArray[np.float64]) -> None:
        """Raise an error for points that are not two or three rows of coordinates.

        Args:
            points:
                The coordinates of the nodes.

        Raises:
            ValidationError:
                The points are not a two-dimensional array of two or three rows.
        """
        if points.ndim != ARRAY_RANK or points.shape[0] not in MESH_DIMENSIONS:
            error_msg = (
                f'The points of the mesh space {self.label!r} have the shape {points.shape}. Give two or three rows, '
                'one per coordinate, with one column per node.'
            )
            raise ValidationError(error_msg)

    def _check_cells(self, cells: npt.NDArray[np.generic], points: npt.NDArray[np.float64]) -> None:
        """Raise an error for cells that are not columns of node indices of the points.

        Args:
            cells:
                The node indices of every cell.

            points:
                The coordinates of the nodes.

        Raises:
            ValidationError:
                The cells are not a nonempty two-dimensional array of integers, have the wrong number of rows, or name a
                node outside the points.
        """
        rows = points.shape[0] + 1
        problem = None
        if cells.ndim != ARRAY_RANK:
            problem = f'have the shape {cells.shape}, but they must be a two-dimensional array'
        elif cells.dtype.kind not in 'iu':
            problem = f'hold values of the type {cells.dtype}, but they must hold integers'
        elif cells.shape[0] != rows:
            problem = f'have {cells.shape[0]} rows, but a mesh in {points.shape[0]}D needs {rows} rows'
        elif cells.shape[1] == 0:
            problem = 'hold no cell, but they must be nonempty'
        elif cells.min() < 0 or cells.max() >= points.shape[1]:
            index = cells.min() if cells.min() < 0 else cells.max()
            problem = f'hold the index {index}, but the mesh has {points.shape[1]} points'
        if problem is not None:
            error_msg = (
                f'The cells of the mesh space {self.label!r} {problem}. Give one column of node indices per cell.'
            )
            raise ValidationError(error_msg)

    def _prepare(self, context: None = None) -> None:
        """Set the nodes, the scikit-fem mesh, and the boundary nodes."""
        super()._prepare(context)
        points = np.asarray(self.points, dtype=float)
        cells = np.asarray(self.cells)
        self._check_points(points)
        self._check_cells(cells, points)
        kind = skfem.MeshTri if points.shape[0] == TRIANGLE_DIMENSIONS else skfem.MeshTet
        self.mesh_ = kind(points, cells.astype(np.int64))
        self.boundary_nodes_ = np.asarray(self.mesh_.boundary_nodes(), dtype=np.intp)
        self.n_dim_ = int(points.shape[0])
        self.size_ = int(points.shape[1])
        self.shape_ = (self.size_,)
        self.positions_ = points.T
        self.periodic_ = False
        self.default_visual_ = None

    def _get_neighbors(self, site: int, neighborhood: BaseNeighborhood | None = None) -> npt.NDArray[np.intp]:
        """Return the nodes that share a cell with a node, in increasing order."""
        cells = np.asarray(self.cells, dtype=np.intp)
        around = np.unique(cells[:, (cells == site).any(axis=0)])
        return around[around != site]
