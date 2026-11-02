"""Define the prototile spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ....specs import BaseSpec
from ._rewriting_error import RewritingError

PLANE_DIMENSION = 2
SEGMENT_VERTICES = 2


@dataclass(frozen=True)
class PrototileSpec(BaseSpec):
    """Spec of a shape that tiles of a patch are placed copies of, a segment or a polygon in its own frame.

    Args:
        name:
            The name of the prototile, which names its count state.

        vertices:
            The vertices in the plane, two for a segment and at least three for a polygon.

    Examples:
        >>> from skcomplex.models.mathematics.fractals import PrototileSpec
        >>> len(PrototileSpec('interval', [(0, 0), (1, 0)]).check().vertices)
        2
    """

    name: str
    vertices: Sequence[Sequence[float]]

    def _check(self) -> None:
        """Raise an error for vertices that are not at least two points of the plane.

        Raises:
            RewritingError:
                The vertices are not points of the plane, or fewer than two.
        """
        points = np.asarray(self.vertices, dtype=float)
        if points.shape[1:] != (PLANE_DIMENSION,) or len(points) < SEGMENT_VERTICES:
            error_msg = f'Prototile {self.name!r} needs at least two vertices in the plane, but got {self.vertices!r}.'
            raise RewritingError(error_msg)

    @property
    def is_segment(self) -> bool:
        """Whether the prototile is a segment, with two vertices, rather than a polygon."""
        return len(self.vertices) == SEGMENT_VERTICES
