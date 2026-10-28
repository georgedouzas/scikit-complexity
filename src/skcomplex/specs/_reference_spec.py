"""Define the reference spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import ValidationError
from ..types import Array, Indices, ReadKind
from ._base_spec import BaseSpec
from ._entry_spec import EntrySpec

if TYPE_CHECKING:
    from ..engine.problems._edge_search import EdgeSearch
    from ..engine.problems._pair_search import PairSearch


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

    @staticmethod
    def _add_into(output: Array, target: ReferenceSpec, values: Array, pairs: tuple[Indices, Indices] | None) -> None:
        """Add the values of one output to the target component of a flat vector.

        Args:
            output:
                The flat vector to add to.

            target:
                The target component.

            values:
                One value per evaluation index.

            pairs:
                The pairs of a pair kernel, or `None` in member mode.
        """
        if target.source is ReadKind.SINGLE:
            output[target.index] += values.sum()
            return
        column = output[target.members]
        if target.source is ReadKind.MEMBER:
            column += values
        else:
            assert pairs is not None
            np.add.at(column, pairs[0] if target.source is ReadKind.FIRST else pairs[1], values)

    @staticmethod
    def _assign_into(output: Array, target: ReferenceSpec, values: Array) -> None:
        """Set the target component of a flat vector to the values of one output.

        Args:
            output:
                The flat vector to write to.

            target:
                The target component, read per member or as a single element.

            values:
                One value per evaluation index.
        """
        if target.source is not ReadKind.MEMBER:
            output[target.index] = values[0]
            return
        output[target.members] = values

    def _member_indices(self, rows: int, pairs: Search | None = None) -> Indices:
        """Return the flat index of a target component for every evaluation index of a member or edge kernel.

        Args:
            rows:
                The number of evaluation indices of the kernel.

            pairs:
                The edges of an edge kernel, whose ends targets of the source and target roles index, or `None`.

        Returns:
            indices:
                One flat index per evaluation index.

        Raises:
            ValidationError:
                A target of an end of an edge lies on a network whose edges change.
        """
        from ..engine.problems._edge_search import EdgeSearch
        if self.source is ReadKind.MEMBER:
            return self.flat_indices
        if self.source in {ReadKind.FIRST, ReadKind.SECOND} and isinstance(pairs, EdgeSearch):
            if pairs.source is not None:
                error_msg = (
                    f'A law targets {self.entry.name} at an end of the edges of a network whose edges change, whose '
                    'ends a fixed channel cannot follow. Target the edge member, or the node through its own laws.'
                )
                raise ValidationError(error_msg)
            ends = pairs.sources if self.source is ReadKind.FIRST else pairs.targets
            return (self.entry.start + ends * self.entry.columns + self.column).astype(np.intp)
        if self.source is ReadKind.NEIGHBOR:
            table = np.asarray(self.neighbor.table, dtype=np.intp)
            inside = table < self.entry.rows
            flat = self.entry.start + table * self.entry.columns + self.column
            return np.where(inside, flat, -1).astype(np.intp)
        return np.full(rows, self.entry.start + self.column, dtype=np.intp)


type Search = PairSearch | EdgeSearch
