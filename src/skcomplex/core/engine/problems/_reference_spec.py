"""Define the reference spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np

from ...base import BaseSpec
from ...components import EntrySpec
from ...errors import ValidationError
from ...types import Array, Indices
from ..enums import ReadKind
from ._edge_search import EdgeSearch
from ._field_operator_kernel import FieldOperatorKernel
from ._neighbor_read_spec import _NeighborReadSpec
from ._pair_search import PairSearch


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
    neighbor: _NeighborReadSpec | FieldOperatorKernel | None = field(default=None, compare=False, repr=False)

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
    def shifted(self) -> _NeighborReadSpec:
        """How the neighbour read finds the neighbour of every member.

        Raises:
            ValidationError:
                The reference is not a neighbour read.
        """
        if not isinstance(self.neighbor, _NeighborReadSpec):
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
