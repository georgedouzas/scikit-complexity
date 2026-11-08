"""Define the patch structure spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..types import AnyArray
from ._base_structure_spec import BaseStructureSpec
from ._prototile_spec import PrototileSpec


@dataclass(frozen=True)
class PatchStructureSpec(BaseStructureSpec):
    """Patch of tiles at one generation.

    Each tile is a prototile under a homogeneous affine matrix.

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
        >>> from skcomplex.core.specs import PatchStructureSpec
        >>> from skcomplex.core.specs import PrototileSpec
        >>> interval = PrototileSpec('interval', [(0, 0), (1, 0)])
        >>> patch = PatchStructureSpec((interval,), np.array([0]), np.eye(3)[None], 0)
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
        """Whether the patch has a geometry.

        It always has one.
        """
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
        parts = [self.vertices(code) for code, item in enumerate(self.prototiles) if item.is_segment]
        return np.concatenate(parts) if parts else np.zeros((0, 2, 2))

    def polygons(self) -> AnyArray:
        """Return the tiles of polygon prototiles, one coordinate array per prototile.

        Returns:
            polygons:
                An object array of arrays of shape `(t, k, 2)`.
        """
        parts = [self.vertices(code) for code, item in enumerate(self.prototiles) if not item.is_segment]
        result = np.empty(len(parts), dtype=object)
        result[:] = parts
        return result
