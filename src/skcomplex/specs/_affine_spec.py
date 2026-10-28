"""Define the affine spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from ..errors import RewritingError
from ._base_spec import BaseSpec


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
        >>> from skcomplex.specs import AffineSpec
        >>> AffineSpec(scale=0.5, translation=(1, 0)).check().scale
        0.5
    """

    scale: object = 1.0
    rotation: object = 0.0
    translation: Sequence[object] = (0.0, 0.0)
    reflect: bool = False
    matrix: Sequence[Sequence[float]] | None = None

    def _check(self) -> Self:
        """Raise an error for a translation that is not a point of the plane, or a matrix that is not 3 by 3.

        Returns:
            spec:
                The spec itself.

        Raises:
            RewritingError:
                The translation or the matrix has the wrong shape.
        """
        if len(self.translation) != 2:  # noqa: PLR2004
            error_msg = f'The translation of an affine map is a point of the plane, not {self.translation!r}.'
            raise RewritingError(error_msg)
        if self.matrix is not None and np.shape(self.matrix) != (3, 3):
            error_msg = (
                f'The matrix of an affine map is homogeneous 3 by 3, but it has the shape {np.shape(self.matrix)}.'
            )
            raise RewritingError(error_msg)
        return self
