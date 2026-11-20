"""Define the direction spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np

from ...specs._base_spec import BaseSpec
from ...specs._entry_spec import EntrySpec
from ...types import Array


@dataclass(frozen=True)
class _DirectionSpec(BaseSpec):
    """A spec of the value that holds the direction of the members.

    The direction is a vector, or an angle in two dimensions. It is a state, such as a heading that turns, or a
    parameter, such as the fixed heading of a sensor.

    Args:
        entry:
            The entry of the value.

        angle:
            Whether the value is a scalar angle, whose direction is `(cos, sin)`.

        is_state:
            Whether the value is a state, read from the state vector, or a parameter, read from the parameters.
    """

    entry: EntrySpec
    angle: bool
    is_state: bool = True

    def get_vectors(self, x: Array, p: Array) -> Array:
        """Return the direction of every member as one row of components.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            directions:
                One row per member.
        """
        values = self.entry.get_view(x if self.is_state else p)
        if self.angle:
            return np.column_stack([np.cos(values[:, 0]), np.sin(values[:, 0])])
        return values
