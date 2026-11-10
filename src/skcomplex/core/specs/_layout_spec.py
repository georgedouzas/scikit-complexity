"""Define the layout spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import RR

from ..types import Array, ComplexArray
from ._base_spec import BaseSpec
from ._entry_spec import EntrySpec


@dataclass(frozen=True)
class LayoutSpec(BaseSpec):
    """A spec of the named variables stored in one flat vector, in order.

    Args:
        entries:
            The entries, in vector order.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.specs import EntrySpec, LayoutSpec
        >>> from sage.all__sagemath_symbolics import CC, RR
        >>> layout = LayoutSpec((EntrySpec('a.n', 0, (), RR), EntrySpec('b.x', 1, (2,), RR)))
        >>> layout.read(np.array([1.0, 2.0, 3.0]), 'b.x')
        array([2., 3.])
    """

    entries: tuple[EntrySpec, ...]

    @functools.cached_property
    def _by_name(self) -> dict[str, EntrySpec]:
        """Entries by dotted name."""
        return {entry.name: entry for entry in self.entries}

    def has(self, name: str) -> bool:
        """Check whether a variable name is part of the layout.

        Args:
            name:
                The dotted name of a variable.

        Returns:
            found:
                Whether the layout has a variable of that name.
        """
        return name in self._by_name

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

    def get_entry(self, name: str) -> EntrySpec:
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

    def locate(self, name: str) -> tuple[EntrySpec, slice]:
        """Return the entry of a variable or of one of its components.

        It also returns the slice of the vector that the variable covers.

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
            >>> from skcomplex.core.specs import EntrySpec, LayoutSpec
            >>> from sage.all__sagemath_symbolics import CC, RR
            >>> layout = LayoutSpec((EntrySpec('a.n', 0, (), RR), EntrySpec('b.w', 1, (3,), RR)))
            >>> entry, indices = layout.locate('b.w[1]')
            >>> entry.name, entry.shape, indices
            ('b.w[1]', (), slice(2, 3, None))
        """
        if name in self._by_name:
            entry = self._by_name[name]
            return entry, entry.indices
        found = self._by_name.get(name.partition('[')[0].removesuffix('.real').removesuffix('.imag'))
        if found is not None and name in found.component_names:
            start = found.start + found.component_names.index(name)
            return EntrySpec(name, start, (), RR if found.is_complex else found.domain), slice(
                start,
                start + 1,
            )
        error_msg = f'No variable or component named {name!r}. Available names are: {", ".join(self.names)}.'
        raise KeyError(error_msg)

    def read(self, vector: npt.NDArray[np.float64], name: str) -> Array | ComplexArray | float | complex:
        """Return the value of a variable from a flat vector or from rows of flat vectors.

        Args:
            vector:
                A flat vector, or a 2D array with one flat vector per row.

            name:
                The dotted name of the variable.

        Returns:
            value:
                A number for a scalar in a flat vector, or an array whose shape is the leading shape of `vector`
                followed by the shape of the variable, complex for a complex variable.
        """
        entry = self.get_entry(name)
        values = entry.decode(vector[..., entry.indices]).reshape(vector.shape[:-1] + entry.shape)
        if values.ndim:
            return values
        return complex(values) if entry.is_complex else float(values)

    def get_values(self, vector: npt.NDArray[np.float64]) -> dict[str, Array | ComplexArray | float | complex]:
        """Return the value of every variable of a flat vector.

        Args:
            vector:
                A flat vector.

        Returns:
            values:
                A mapping from dotted name to value.
        """
        return {name: self.read(vector, name) for name in self._by_name}
