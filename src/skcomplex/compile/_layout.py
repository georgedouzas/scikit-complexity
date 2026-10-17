"""Lay out named variables in flat numeric vectors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..core import Domain


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
    """

    name: str
    start: int
    shape: tuple[int, ...]
    domain: Domain

    @property
    def size(self) -> int:
        """Number of components of the variable."""
        return int(np.prod(self.shape, dtype=int))

    @property
    def indices(self) -> slice:
        """Slice of the variable's components in the vector."""
        return slice(self.start, self.start + self.size)

    @property
    def component_names(self) -> list[str]:
        """Names of the components, with an index suffix for vectors."""
        if not self.shape:
            return [self.name]
        return [f'{self.name}[{index}]' for index in range(self.size)]


class Layout:
    """Ordered collection of named variables stored in one flat vector.

    Args:
        entries:
            The entries, in vector order.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.compile import Entry, Layout
        >>> from skcomplex.core import Domain
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

    def read(self, vector: npt.NDArray[np.float64], name: str) -> npt.NDArray[np.float64] | float:
        """Return the value of a variable from a flat vector or from rows of flat vectors.

        Args:
            vector:
                A flat vector, or a 2D array with one flat vector per row.

            name:
                The dotted name of the variable.

        Returns:
            value:
                A number for a scalar in a flat vector, or an array otherwise.
        """
        entry = self.entry(name)
        values = vector[..., entry.indices]
        if entry.shape:
            return values
        values = values[..., 0]
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
