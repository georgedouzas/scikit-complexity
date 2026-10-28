"""Define the heading spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np

from ..types import Array
from ._base_spec import BaseSpec
from ._entry_spec import EntrySpec


@dataclass(frozen=True)
class HeadingSpec(BaseSpec):
    """State that holds the heading of the members, a vector or a heading angle in two dimensions.

    Args:
        entry:
            The entry of the state.

        angle:
            Whether the state is a scalar angle, whose heading is `(cos, sin)`.
    """

    entry: EntrySpec
    angle: bool

    def vectors(self, x: Array) -> Array:
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
