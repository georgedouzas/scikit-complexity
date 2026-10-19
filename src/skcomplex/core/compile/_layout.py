"""Lay out named variables in flat numeric vectors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..base import AnyDomain


@dataclass(frozen=True)
class Entry:
    """Position of one named variable in a flat vector.

    Args:
        name:
            The dotted name of the variable, such as `meadow/rabbits.n`.

        start:
            The index of the first component in the vector.

        shape:
            The shape of the variable, `()` for a scalar and `(n_dim,)` for a vector.

        domain:
            The set of values the variable takes.

        members:
            The size of the group that holds the variable, or `None` for a single element or an interaction.
    """

    name: str
    start: int
    shape: tuple[int, ...]
    domain: AnyDomain
    members: int | None = None
    _size: int = field(init=False, repr=False, compare=False)
    _indices: slice = field(init=False, repr=False, compare=False)
    _rows: int = field(init=False, repr=False, compare=False)
    _columns: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Compute the size, slice, rows, and columns of the variable once."""
        size = math.prod(self.shape)
        object.__setattr__(self, '_size', size)
        object.__setattr__(self, '_indices', slice(self.start, self.start + size))
        object.__setattr__(self, '_rows', self.members if self.members is not None else 1)
        object.__setattr__(self, '_columns', math.prod(self.member_shape))

    @property
    def size(self) -> int:
        """Number of components of the variable."""
        return self._size

    @property
    def indices(self) -> slice:
        """Slice of the variable's components in the vector."""
        return self._indices

    @property
    def member_shape(self) -> tuple[int, ...]:
        """Shape of the variable of one member, `()` for a scalar and `(n_dim,)` for a vector."""
        return self.shape[1:] if self.members is not None else self.shape

    @property
    def rows(self) -> int:
        """Number of members, 1 for a single element or an interaction."""
        return self._rows

    @property
    def columns(self) -> int:
        """Number of components of the variable of one member."""
        return self._columns

    @property
    def component_names(self) -> list[str]:
        """Names of the components, with an index suffix for vectors and a member index for groups."""
        if self.members is None:
            if not self.shape:
                return [self.name]
            return [f'{self.name}[{index}]' for index in range(self.size)]
        if not self.member_shape:
            return [f'{self.name}[{member}]' for member in range(self.rows)]
        return [f'{self.name}[{member}, {index}]' for member in range(self.rows) for index in range(self.columns)]

    @property
    def generic_names(self) -> list[str]:
        """Names of the components of one member, with the member index `i` for groups."""
        if self.members is None:
            return self.component_names
        if not self.member_shape:
            return [f'{self.name}[i]']
        return [f'{self.name}[i, {index}]' for index in range(self.columns)]

    def view(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the variable's components of a flat vector as a view with one row per member.

        Args:
            vector:
                A flat vector.

        Returns:
            view:
                A `(rows, columns)` view that writes through to the vector.
        """
        return vector[self._indices].reshape(self._rows, self._columns)


class Layout:
    """Ordered collection of named variables stored in one flat vector.

    Args:
        entries:
            The entries, in vector order.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import Entry, Layout
        >>> from skcomplex.core.base import Domain
        >>> layout = Layout([Entry('a.n', 0, (), Domain.REAL), Entry('b.x', 1, (2,), Domain.REAL)])
        >>> layout.read(np.array([1.0, 2.0, 3.0]), 'b.x')
        array([2., 3.])
    """

    def __init__(self, entries: Iterable[Entry]) -> None:
        self.entries = tuple(entries)
        self._by_name = {entry.name: entry for entry in self.entries}

    def __contains__(self, name: object) -> bool:
        """Check whether a variable name is part of the layout."""
        return name in self._by_name

    def __iter__(self) -> Iterator[Entry]:
        """Iterate over the entries in vector order."""
        return iter(self.entries)

    def __len__(self) -> int:
        """Return the number of variables."""
        return len(self.entries)

    @property
    def size(self) -> int:
        """Number of components of the flat vector."""
        return sum(entry.size for entry in self.entries)

    @property
    def names(self) -> list[str]:
        """Dotted names of the variables, in vector order."""
        return [entry.name for entry in self.entries]

    @property
    def component_names(self) -> list[str]:
        """Names of the components, in vector order."""
        return [name for entry in self.entries for name in entry.component_names]

    def entry(self, name: str) -> Entry:
        """Return the entry of a variable.

        Args:
            name:
                The dotted name of the variable.

        Returns:
            entry:
                The entry of the variable.

        Raises:
            KeyError:
                The name is not part of the layout.
        """
        if name not in self._by_name:
            error_msg = f'No variable named {name!r}. Available names are: {", ".join(self.names)}.'
            raise KeyError(error_msg)
        return self._by_name[name]

    def locate(self, name: str) -> tuple[Entry, slice]:
        """Return the entry of a variable or of one of its components, and the slice of the vector it covers.

        Args:
            name:
                The dotted name of a variable, or of one component such as `regulation.weights[1]`.

        Returns:
            location:
                The entry of the variable, a scalar entry for a component, and the slice of its values.

        Raises:
            KeyError:
                The name is neither a variable nor a component of the layout.

        Examples:
            >>> from skcomplex.core.compile import Entry, Layout
            >>> from skcomplex.core.base import Domain
            >>> layout = Layout([Entry('a.n', 0, (), Domain.REAL), Entry('b.w', 1, (3,), Domain.REAL)])
            >>> layout.locate('b.w[1]')
            (Entry(name='b.w[1]', start=2, shape=(), domain=<Domain.REAL: 'real'>, members=None), slice(2, 3, None))
        """
        if name in self._by_name:
            entry = self._by_name[name]
            return entry, entry.indices
        found = self._by_name.get(name.partition('[')[0])
        if found is not None and name in found.component_names:
            start = found.start + found.component_names.index(name)
            return Entry(name, start, (), found.domain), slice(start, start + 1)
        error_msg = f'No variable or component named {name!r}. Available names are: {", ".join(self.names)}.'
        raise KeyError(error_msg)

    def read(self, vector: npt.NDArray[np.float64], name: str) -> npt.NDArray[np.float64] | float:
        """Return the value of a variable from a flat vector or from rows of flat vectors.

        Args:
            vector:
                A flat vector, or a 2D array with one flat vector per row.

            name:
                The dotted name of the variable.

        Returns:
            value:
                A number for a scalar in a flat vector, or an array whose shape is the leading shape of `vector`
                followed by the shape of the variable.
        """
        entry = self.entry(name)
        values = vector[..., entry.indices].reshape(vector.shape[:-1] + entry.shape)
        return float(values) if values.ndim == 0 else values

    def values(self, vector: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64] | float]:
        """Return the value of every variable of a flat vector.

        Args:
            vector:
                A flat vector.

        Returns:
            values:
                A mapping from dotted name to value.
        """
        return {name: self.read(vector, name) for name in self._by_name}
