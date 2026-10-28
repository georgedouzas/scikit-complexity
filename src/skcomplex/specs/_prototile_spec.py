"""Define the prototile spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from ..errors import RewritingError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class PrototileSpec(BaseSpec):
    """Spec of a shape that tiles of a patch are placed copies of, a segment or a polygon in its own frame.

    Args:
        name:
            The name of the prototile, which names its count state.

        vertices:
            The vertices in the plane, two for a segment and at least three for a polygon.

    Examples:
        >>> from skcomplex.specs import PrototileSpec
        >>> len(PrototileSpec('interval', [(0, 0), (1, 0)]).check().vertices)
        2
    """

    name: str
    vertices: Sequence[Sequence[float]]

    def _check(self) -> Self:
        """Raise an error for vertices that are not at least two points of the plane.

        Returns:
            spec:
                The spec itself.

        Raises:
            RewritingError:
                The vertices are not points of the plane, or fewer than two.
        """
        points = np.asarray(self.vertices, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:  # noqa: PLR2004
            error_msg = f'Prototile {self.name!r} needs at least two vertices in the plane, but got {self.vertices!r}.'
            raise RewritingError(error_msg)
        return self

    @property
    def is_segment(self) -> bool:
        """Whether the prototile is a segment, with two vertices, rather than a polygon."""
        return len(self.vertices) == 2  # noqa: PLR2004
