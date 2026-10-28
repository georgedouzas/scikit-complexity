"""Define the patch structure spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..types import AnyArray
from ._affine_spec import AffineSpec
from ._base_structure_spec import BaseStructureSpec
from ._prototile_spec import PrototileSpec


@dataclass(frozen=True)
class PatchStructureSpec(BaseStructureSpec):
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
        >>> from skcomplex.specs import PatchStructureSpec
        >>> from skcomplex.specs import PrototileSpec
        >>> patch = PatchStructureSpec((PrototileSpec('interval', [(0, 0), (1, 0)]),), np.array([0]), np.eye(3)[None], 0)
        >>> patch.length()
        1.0
    """

    prototiles: tuple[PrototileSpec, ...]
    codes: npt.NDArray[np.int64]
    matrices: npt.NDArray[np.float64]
    generation: int

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

    def vertices(self, code: int) -> AnyArray:
        """Return the vertices of the tiles of one prototile.

        Args:
            code:
                The code of the prototile.

        Returns:
            vertices:
                The array of shape `(t, k, 2)`.
        """
        points = np.asarray(self.prototiles[code].vertices, dtype=float)
        homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        matrices = self.matrices[self.codes == code]
        return np.einsum('tij,kj->tki', matrices, homogeneous)[..., :2]

    def segments(self) -> AnyArray:
        """Return the tiles of segment prototiles as segments.

        Returns:
            segments:
                The array of shape `(k, 2, 2)`.
        """
        parts = [self.vertices(code) for code, item in enumerate(self.prototiles) if self._is_segment(item)]
        return np.concatenate(parts) if parts else np.zeros((0, 2, 2))

    def polygons(self) -> AnyArray:
        """Return the tiles of polygon prototiles, one coordinate array per prototile.

        Returns:
            polygons:
                An object array of arrays of shape `(t, k, 2)`.
        """
        parts = [self.vertices(code) for code, item in enumerate(self.prototiles) if not self._is_segment(item)]
        result = np.empty(len(parts), dtype=object)
        result[:] = parts
        return result

    @staticmethod
    def _affine_matrix(affine: AffineSpec, values: Mapping[object, float]) -> npt.NDArray[np.float64]:
        """Return the homogeneous matrix of an affine map at values of the parameters its expressions read.

        The map reflects across the x axis, rotates, scales, and translates, in this order.

        Args:
            affine:
                The affine map.

            values:
                A mapping from parameter symbol to value.

        Returns:
            matrix:
                The 3 by 3 matrix.

        Examples:
            >>> from skcomplex.specs import AffineSpec
            >>> from skcomplex.specs import PatchStructureSpec
            >>> PatchStructureSpec._affine_matrix(AffineSpec(scale=0.5, translation=(1, 0)), {}).tolist()
            [[0.5, 0.0, 1.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
        """
        if affine.matrix is not None:
            return np.asarray(affine.matrix, dtype=float)
        scale, rotation, *translation = (
            float(item.subs(values)) if hasattr(item, 'subs') else float(item)
            for item in (affine.scale, affine.rotation, *affine.translation)
        )
        angle = np.deg2rad(rotation)
        turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        reflection = np.diag([1.0, -1.0]) if affine.reflect else np.eye(2)
        matrix = np.eye(3)
        matrix[:2, :2] = scale * turn @ reflection
        matrix[:2, 2] = translation
        return matrix

    @staticmethod
    def _is_segment(prototile: PrototileSpec) -> bool:
        """Check whether a prototile is a segment, with two vertices, rather than a polygon.

        Args:
            prototile:
                The prototile.

        Returns:
            is_segment:
                Whether the prototile has two vertices.
        """
        return len(prototile.vertices) == 2  # noqa: PLR2004
