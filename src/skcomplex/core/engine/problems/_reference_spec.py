"""Define the reference spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ...base import BaseSpec
from ...components import EntrySpec
from ...errors import ValidationError
from ...types import Array, Indices, IntTuple
from ..enums import ReadKind
from ._edge_search import EdgeSearch
from ._field_operator_kernel import FieldOperatorKernel
from ._pair_search import PairSearch


@dataclass(frozen=True, eq=False)
class _NeighborRead:
    """A record of how a symbol reads a field at a neighbouring site of a member.

    Args:
        table:
            The member index of the neighbour of every member, with the number of members for the fixed value.

        ghost:
            The value read outside fixed edges, or `None` when no neighbour is outside.

        sizes:
            The lattice shape, or `None` on a network.

        steps:
            The offset of the neighbour on a lattice, or `None` on a network.

        mode:
            0 for periodic, 1 for zero-flux, and 2 for fixed edges.

        entry:
            The parameter entry whose flat values at the table hold the member of every row, for members that change
            during a run, such as those of a growing hypergraph, or `None` when the table holds the members.

        order:
            The member at every position of an order drawn again at every step, whose positions the table holds, or
            `None` for a table of members.
    """

    table: Indices
    ghost: float | None
    sizes: tuple[int, ...] | None
    steps: IntTuple | None
    mode: int
    entry: EntrySpec | None = None
    order: Indices | None = None

    def get_members(self, p: Array) -> Indices:
        """Return the member every row reads.

        It reads it from the table or the parameter entry at the table, through the order.

        Args:
            p:
                The parameter vector.

        Returns:
            members:
                The member index of every row.
        """
        rows = self.table if self.entry is None else self.entry.get_view(p).ravel()[self.table].astype(np.intp)
        return rows if self.order is None else self.order[rows]

    def read(self, column: npt.NDArray[np.float64], p: Array) -> npt.NDArray[np.float64]:
        """Return the neighbour's value of every member from the column of values.

        Args:
            column:
                One value per member.

            p:
                The parameter vector, which holds the members of a table that changes during a run.

        Returns:
            values:
                The value at the neighbour of every member.
        """
        if self.ghost is None:
            return column[self.get_members(p)]
        return np.append(column, self.ghost)[self.table]


@dataclass(frozen=True)
class ReferenceSpec(BaseSpec):
    """A spec of the value that a symbol of a law stands for, and how it is read.

    Every symbol of a compiled law stands for one number of a state or parameter, such as the size of a population, and
    the spec says where it sits, whether it is a state, which number of the value it is, and how it is read for every
    member: its own value, a shared value, a neighbour, or a partner in a pair. It locates the size of a population, the
    position of a particle, and the price of a stock.

    Read more in the user guide, under Extend the library.

    Args:
        entry:
            The layout entry of the variable.

        is_state:
            Whether the variable is a state, as opposed to a parameter.

        column:
            The component of the variable of one member, or for a single read of one group member, the component counted
            from the group's first component.

        source:
            How the component is read for each evaluation index.

        neighbor:
            How a neighbour read finds the neighbour of every member, for the neighbour source, or the global operator
            of a derived read, for the derived source.

    See Also:
        EntrySpec: Where a value sits in a vector.
        ReadKind: The kinds of read.

    Examples:
        >>> from sage.all__sagemath_symbolics import RR
        >>> from skcomplex.core.components import EntrySpec
        >>> from skcomplex.core.engine.problems._reference_spec import ReferenceSpec
        >>> ReferenceSpec(EntrySpec('population.x', 0, (), RR), True, 0).source
        <ReadKind.SINGLE: 'single'>
    """

    entry: EntrySpec
    is_state: bool
    column: int
    source: ReadKind = ReadKind.SINGLE
    neighbor: _NeighborRead | FieldOperatorKernel | None = field(default=None, compare=False, repr=False)

    @functools.cached_property
    def index(self) -> int:
        """Flat index of the component of a single read, or of the first member."""
        return self.entry.start + self.column

    @functools.cached_property
    def members(self) -> slice:
        """Strided slice of the component of every member in the flat vector."""
        return slice(self.entry.start + self.column, self.entry.start + self.entry.size, self.entry.columns)

    def get_column_of(self, x: Array, p: Array) -> Array:
        """Return the component of every member.

        It is a view into the state or parameter vector.

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
    def shifted(self) -> _NeighborRead:
        """How the neighbour read finds the neighbour of every member.

        Raises:
            ValidationError:
                The reference is not a neighbour read.
        """
        if not isinstance(self.neighbor, _NeighborRead):
            error_msg = f'The reference to {self.entry.name} is not a neighbour read.'
            raise ValidationError(error_msg)
        return self.neighbor

    @property
    def derived(self) -> FieldOperatorKernel:
        """Global operator of the derived read.

        Raises:
            ValidationError:
                The reference is not a derived read.
        """
        if not isinstance(self.neighbor, FieldOperatorKernel):
            error_msg = f'The reference to {self.entry.name} is not a derived read.'
            raise ValidationError(error_msg)
        return self.neighbor

    @property
    def flat_indices(self) -> Indices:
        """Indices of the component of every member in the flat vector."""
        return self.entry.start + np.arange(self.entry.rows) * self.entry.columns + self.column


type PairsSearch = PairSearch | EdgeSearch
