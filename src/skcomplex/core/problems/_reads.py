"""Locate the values a symbol of a reduction stands for in the state or the extended parameter vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np

from ..base import Entry, Indices


@dataclass(frozen=True)
class Read:
    """Values a symbol of a reduction stands for, one per reduced member or one shared value.

    Args:
        is_state:
            Whether the values come from the state vector, as opposed to the extended parameter vector.

        index:
            The flat index of a shared value, or the flat index of the value of every reduced member.

        entry:
            The layout entry the values belong to, or `None` for an element set.

        column:
            The component of the entry's member that the symbol stands for.

        stride:
            The distance between the values of two consecutive members in the vector.
    """

    is_state: bool
    index: int | Indices
    entry: Entry | None = None
    column: int = 0
    stride: int = 1

    @property
    def per_member(self) -> bool:
        """Whether the symbol has one value per reduced member."""
        return np.ndim(self.index) > 0
