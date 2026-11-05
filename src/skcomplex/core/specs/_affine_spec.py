"""Define the affine spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ..errors import RewritingError
from ._base_spec import BaseSpec

PLANE_DIMENSION = 2


@dataclass(frozen=True)
class AffineSpec(BaseSpec):
    """Spec of an affine map of the plane: a reflection across the x axis, a rotation, a scale, and a translation.

    The scale, the rotation, and the translation can be expressions of the parameters of the element whose laws hold
    the substitution, which are evaluated at every generation.

    Args:
        scale:
            The scale factor.

        rotation:
            The rotation in degrees.

        translation:
            The translation.

        reflect:
            Whether to reflect across the x axis first.

        matrix:
            A homogeneous 3 by 3 matrix that replaces the other arguments.

    Examples:
        >>> from skcomplex.core.specs import AffineSpec
        >>> AffineSpec(scale=0.5, translation=(1, 0)).scale
        0.5
    """

    scale: float | Expression = 1.0
    rotation: float | Expression = 0.0
    translation: Sequence[float | Expression] = (0.0, 0.0)
    reflect: bool = False
    matrix: npt.ArrayLike | None = None

    def _check(self) -> None:
        """Raise an error for a translation that is not a point of the plane, or a matrix that is not 3 by 3.

        Raises:
            RewritingError:
                The translation or the matrix has the wrong shape.
        """
        if len(self.translation) != PLANE_DIMENSION:
            error_msg = f'The translation of an affine map is a point of the plane, not {self.translation!r}.'
            raise RewritingError(error_msg)
        if self.matrix is not None and np.shape(self.matrix) != (3, 3):
            error_msg = (
                f'The matrix of an affine map is homogeneous 3 by 3, but it has the shape {np.shape(self.matrix)}.'
            )
            raise RewritingError(error_msg)

    def matrix_at(self, values: Mapping[object, float]) -> npt.NDArray[np.float64]:
        """Return the homogeneous matrix of the map at values of the parameters its expressions read.

        The map reflects across the x axis, rotates, scales, and translates, in this order.

        Args:
            values:
                A mapping from parameter symbol to value.

        Returns:
            matrix:
                The 3 by 3 matrix.

        Examples:
            >>> from skcomplex.core.specs import AffineSpec
            >>> AffineSpec(scale=0.5, translation=(1, 0)).matrix_at({}).tolist()
            [[0.5, 0.0, 1.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
        """
        if self.matrix is not None:
            return np.asarray(self.matrix, dtype=float)
        scale, rotation, *translation = (
            float(item.subs(values)) if hasattr(item, 'subs') else float(item)
            for item in (self.scale, self.rotation, *self.translation)
        )
        angle = np.deg2rad(rotation)
        turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        reflection = np.diag([1.0, -1.0]) if self.reflect else np.eye(2)
        matrix = np.eye(3)
        matrix[:2, :2] = scale * turn @ reflection
        matrix[:2, 2] = translation
        return matrix
