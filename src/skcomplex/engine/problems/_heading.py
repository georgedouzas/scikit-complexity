"""State that holds the heading of the members, a vector or a heading angle in two dimensions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np

from ...core.types import Vector
from ._base import BaseValue
from ._entry import Entry


@dataclass(frozen=True)
class Heading(BaseValue):
    """State that holds the heading of the members, a vector or a heading angle in two dimensions.

    Args:
        entry:
            The entry of the state.

        angle:
            Whether the state is a scalar angle, whose heading is `(cos, sin)`.
    """

    entry: Entry
    angle: bool

    def vectors(self, x: Vector) -> Vector:
        """Return the heading of every member as one row of components.

        Args:
            x:
                The state vector.

        Returns:
            headings:
                One row per member.
        """
        values = self.entry.view(x)
        if self.angle:
            return np.column_stack([np.cos(values[:, 0]), np.sin(values[:, 0])])
        return values
