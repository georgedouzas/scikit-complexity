"""Define unstructured triangular and tetrahedral meshes that fields discretised by finite elements live on."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ..base import SpaceKind
from ._spaces import Space


def _skfem() -> Any:  # noqa: ANN401
    """Return the scikit-fem module, imported only when meshes are built."""
    import skfem  # noqa: PLC0415  # finite elements are the only part of the framework that needs scikit-fem

    return skfem


class Mesh(Space):
    """Unstructured mesh of triangles or tetrahedra, whose nodes hold the members of finite element fields.

    Args:
        points:
            The coordinates of the nodes, one column per node and one row per dimension.

        cells:
            The node indices of every triangle or tetrahedron, one column per cell.

    Examples:
        >>> from skcomplex.core.space import Mesh
        >>> mesh = Mesh.rectangle(4, 3)
        >>> mesh.size, mesh.n_dim
        (20, 2)
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.MESH

    def __init__(self, points: npt.ArrayLike, cells: npt.ArrayLike) -> None:
        self.points = points
        self.cells = cells

    @property
    def nodes(self) -> npt.NDArray[np.float64]:
        """Coordinates of the nodes, one row per node."""
        return np.asarray(self.points, dtype=float).T

    @property
    def n_dim(self) -> int:  # type: ignore[override]
        """Number of dimensions."""
        return int(np.shape(self.points)[0])

    @property
    def size(self) -> int:
        """Number of nodes."""
        return int(np.shape(self.points)[1])

    def skfem_mesh(self) -> Any:  # noqa: ANN401
        """Return the scikit-fem mesh of the nodes and cells.

        Returns:
            mesh:
                A `MeshTri` in 2D or a `MeshTet` in 3D.
        """
        skfem = _skfem()
        kind = skfem.MeshTri if self.n_dim == 2 else skfem.MeshTet  # noqa: PLR2004
        return kind(np.asarray(self.points, dtype=float), np.asarray(self.cells, dtype=np.int64))

    def boundary_nodes(self) -> npt.NDArray[np.intp]:
        """Return the indices of the nodes on the boundary of the mesh.

        Returns:
            nodes:
                The boundary node indices.
        """
        return np.asarray(self.skfem_mesh().boundary_nodes(), dtype=np.intp)

    @classmethod
    def _of(cls, mesh: Any) -> 'Mesh':  # noqa: ANN401
        """Return the mesh of a scikit-fem mesh."""
        return cls(np.asarray(mesh.p, dtype=float), np.asarray(mesh.t, dtype=np.int64))

    @classmethod
    def rectangle(cls, nx: int, ny: int, *, width: float = 1.0, height: float = 1.0) -> 'Mesh':
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
    def l_shape(cls, refine: int = 3) -> 'Mesh':
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
    def disk(cls, refine: int = 3) -> 'Mesh':
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
    def load(cls, path: str | Path) -> 'Mesh':
        """Return a mesh read from a file in any format that meshio reads, such as Gmsh.

        Args:
            path:
                The path of the file.

        Returns:
            mesh:
                The mesh.
        """
        return cls._of(_skfem().Mesh.load(str(path)))
