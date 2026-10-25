"""Define unstructured triangular and tetrahedral meshes that fields discretised by finite elements live on."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ...core.base import BaseSpace, SpaceKind


def _skfem() -> Any:  # noqa: ANN401
    """Return the scikit-fem module, imported only when meshes are built."""
    import skfem  # noqa: PLC0415  # finite elements are the only part of the framework that needs scikit-fem

    return skfem


class MeshSpace(BaseSpace):
    """Unstructured mesh of triangles or tetrahedra, whose nodes hold the members of finite element fields.

    Args:
        points:
            The coordinates of the nodes, one column per node and one row per dimension.

        cells:
            The node indices of every triangle or tetrahedron, one column per cell.

    Attributes:
        mesh_:
            The scikit-fem mesh of the nodes and cells, a `MeshTri` in 2D or a `MeshTet` in 3D.

        boundary_nodes_:
            The indices of the nodes on the boundary of the mesh.

    Examples:
        >>> from skcomplex.formalisms.fields import MeshSpace
        >>> mesh = MeshSpace.rectangle(4, 3).prepare()
        >>> mesh.size_, mesh.n_dim_, mesh.positions_.shape
        (20, 2, (20, 2))
        >>> mesh.run(0).tolist()
        [1, 4, 5]
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.MESH

    size_: int
    mesh_: Any
    boundary_nodes_: npt.NDArray[np.intp]

    def __init__(self, points: npt.ArrayLike, cells: npt.ArrayLike) -> None:
        self.points = points
        self.cells = cells

    def _prepare(self) -> None:
        """Set the nodes, the scikit-fem mesh, and the boundary nodes."""
        points = np.asarray(self.points, dtype=float)
        skfem = _skfem()
        kind = skfem.MeshTri if points.shape[0] == 2 else skfem.MeshTet  # noqa: PLR2004
        self.mesh_ = kind(points, np.asarray(self.cells, dtype=np.int64))
        self.boundary_nodes_ = np.asarray(self.mesh_.boundary_nodes(), dtype=np.intp)
        self.n_dim_ = int(points.shape[0])
        self.size_ = int(points.shape[1])
        self.shape_ = (self.size_,)
        self.positions_ = points.T
        self.periodic_ = False
        self.default_visual_ = None

    def _run(self, site: int, neighborhood: Any = None) -> npt.NDArray[np.intp]:  # noqa: ANN401
        """Return the nodes that share a cell with a node, in increasing order."""
        cells = np.asarray(self.cells, dtype=np.intp)
        around = np.unique(cells[:, (cells == site).any(axis=0)])
        return around[around != site]

    @classmethod
    def _of(cls, mesh: Any) -> 'MeshSpace':  # noqa: ANN401
        """Return the mesh of a scikit-fem mesh."""
        return cls(np.asarray(mesh.p, dtype=float), np.asarray(mesh.t, dtype=np.int64))

    @classmethod
    def rectangle(cls, nx: int, ny: int, *, width: float = 1.0, height: float = 1.0) -> 'MeshSpace':
        """Return a rectangle split into triangles on a grid of cells.

        Args:
            nx:
                The number of cells along the width.

            ny:
                The number of cells along the height.

            width:
                The width.

            height:
                The height.

        Returns:
            mesh:
                The mesh.
        """
        skfem = _skfem()
        return cls._of(skfem.MeshTri.init_tensor(np.linspace(0, width, nx + 1), np.linspace(0, height, ny + 1)))

    @classmethod
    def l_shape(cls, refine: int = 3) -> 'MeshSpace':
        """Return the L-shaped domain `[-1, 1]^2` without its quarter `[0, 1] x [-1, 0]`, refined uniformly.

        Args:
            refine:
                The number of uniform refinements, each halving the mesh size.

        Returns:
            mesh:
                The mesh.
        """
        return cls._of(_skfem().MeshTri.init_lshaped().refined(refine))

    @classmethod
    def disk(cls, refine: int = 3) -> 'MeshSpace':
        """Return the unit disk, refined uniformly.

        Args:
            refine:
                The number of uniform refinements.

        Returns:
            mesh:
                The mesh.
        """
        return cls._of(_skfem().MeshTri.init_circle(refine))

    @classmethod
    def load(cls, path: str | Path) -> 'MeshSpace':
        """Return a mesh read from a file in any format that meshio reads, such as Gmsh.

        Args:
            path:
                The path of the file.

        Returns:
            mesh:
                The mesh.
        """
        return cls._of(_skfem().Mesh.load(str(path)))
