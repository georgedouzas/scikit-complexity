"""Define the reference spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...specs._base_spec import BaseSpec
from ...specs._entry_spec import EntrySpec
from ...types import Array, Indices, ReadKind
from ._edge_search import EdgeSearch
from ._pair_search import PairSearch


@dataclass(frozen=True)
class ReferenceSpec(BaseSpec):
    """Component of a state or parameter that a symbol stands for.

    Args:
        entry:
            The layout entry of the variable.

        is_state:
            Whether the variable is a state, as opposed to a parameter.

        column:
            The component of the variable of one member, or for a single read of one group member, the component
            counted from the group's first component.

        source:
            How the component is read for each evaluation index.

        neighbor:
            How a neighbour read finds the neighbour of every member, for the neighbour source.
    """

    entry: EntrySpec
    is_state: bool
    column: int
    source: ReadKind = ReadKind.SINGLE
    neighbor: Any = field(default=None, compare=False, repr=False)

    @functools.cached_property
    def index(self) -> int:
        """Flat index of the component of a single read, or of the first member."""
        return self.entry.start + self.column

    @functools.cached_property
    def members(self) -> slice:
        """Strided slice of the component of every member in the flat vector."""
        return slice(self.entry.start + self.column, self.entry.start + self.entry.size, self.entry.columns)

    def column_of(self, x: Array, p: Array) -> Array:
        """Return the component of every member as a view into the state or parameter vector.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            column:
                A view with one value per member, or with the one value of a single read.
        """
        vector = x if self.is_state else p
        if self.source is ReadKind.SINGLE:
            return vector[self.index : self.index + 1]
        return vector[self.members]

    @property
    def flat_indices(self) -> Indices:
        """Indices of the component of every member in the flat vector."""
        return self.entry.start + np.arange(self.entry.rows) * self.entry.columns + self.column


type Search = PairSearch | EdgeSearch
