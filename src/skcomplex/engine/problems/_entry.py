"""Position of one named variable in a flat vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._base import BaseValue
from ._domains import encode

PARTS = 2


_PART_NAMES = ('real', 'imag')


@dataclass(frozen=True)
class Entry(BaseValue):
    """Position of one named variable in a flat vector.

    Args:
        name:
            The dotted name of the variable, such as `meadow/rabbits.n`.

        start:
            The index of the first component in the vector.

        shape:
            The shape of the variable, `()` for a scalar and `(n_dim,)` for a vector.

        domain:
            The Sage set of the values the variable takes.

        members:
            The size of the group that holds the variable, or `None` for a single element or an interaction.

        is_complex:
            Whether the variable is complex, with the real and imaginary part of every component stored in a row.

        period:
            The period that values of the variable wrap around, or `None`.

    Examples:
        >>> from sage.all__sagemath_symbolics import CC, RR
        >>> from skcomplex.engine.problems import Entry
        >>> entry = Entry('a.z', 0, (2,), CC, is_complex=True)
        >>> entry.size, entry.component_names
        (4, ['a.z[0].real', 'a.z[0].imag', 'a.z[1].real', 'a.z[1].imag'])
    """

    name: str
    start: int
    shape: tuple[int, ...]
    domain: object
    members: int | None = None
    is_complex: bool = False
    period: float | None = None

    def _parts(self, names: list[str]) -> list[str]:
        """Return names with the suffixes of the real and imaginary parts for a complex variable."""
        if not self.is_complex:
            return names
        return [f'{name}.{part}' for name in names for part in _PART_NAMES]

    @functools.cached_property
    def size(self) -> int:
        """Number of components of the variable."""
        return math.prod(self.shape) * (PARTS if self.is_complex else 1)

    @functools.cached_property
    def indices(self) -> slice:
        """Slice of the variable's components in the vector."""
        return slice(self.start, self.start + self.size)

    @property
    def member_shape(self) -> tuple[int, ...]:
        """Shape of the variable of one member, `()` for a scalar and `(n_dim,)` for a vector."""
        return self.shape[1:] if self.members is not None else self.shape

    @functools.cached_property
    def rows(self) -> int:
        """Number of members, 1 for a single element or an interaction."""
        return self.members if self.members is not None else 1

    @functools.cached_property
    def columns(self) -> int:
        """Number of components of the variable of one member."""
        return math.prod(self.member_shape) * (PARTS if self.is_complex else 1)

    @property
    def component_names(self) -> list[str]:
        """Names of the components, with an index suffix for vectors, a member index for groups, and complex parts."""
        components = math.prod(self.member_shape)
        if self.members is None:
            if not self.shape:
                return self._parts([self.name])
            return self._parts([f'{self.name}[{index}]' for index in range(components)])
        if not self.member_shape:
            return self._parts([f'{self.name}[{member}]' for member in range(self.rows)])
        return self._parts(
            [f'{self.name}[{member}, {index}]' for member in range(self.rows) for index in range(components)],
        )

    @property
    def generic_names(self) -> list[str]:
        """Names of the components of one member, with the member index `i` for groups."""
        if self.members is None:
            return self.component_names
        if not self.member_shape:
            return self._parts([f'{self.name}[i]'])
        return self._parts([f'{self.name}[i, {index}]' for index in range(math.prod(self.member_shape))])

    def view(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the variable's components of a flat vector as a view with one row per member.

        Args:
            vector:
                A flat vector.

        Returns:
            view:
                A `(rows, columns)` view that writes through to the vector.
        """
        return vector[self.indices].reshape(self.rows, self.columns)

    def encode(self, value: object) -> npt.NDArray[np.float64]:
        """Return a value broadcast to the shape of the variable as flat floats, in pairs for a complex variable.

        Args:
            value:
                A value in the domain of the variable, of its shape or of a shape that broadcasts to it.

        Returns:
            components:
                The flat components, as stored in a vector.

        Raises:
            ValueError:
                The value does not broadcast to the shape of the variable.

        Examples:
            >>> from sage.all__sagemath_symbolics import CC, RR
            >>> from skcomplex.engine.problems import Entry
            >>> Entry('a.z', 0, (), CC, is_complex=True).encode(1 + 2j).tolist()
            [1.0, 2.0]
        """
        values = np.broadcast_to(encode(self.domain, value, self.period), self.shape)
        if self.is_complex:
            return np.ascontiguousarray(values, dtype=complex).ravel().view(float).copy()
        return values.ravel().copy()

    def decode(self, values: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        """Return the stored components of the variable, the last axis of an array, with complex parts joined.

        Args:
            values:
                The stored components on the last axis.

        Returns:
            values:
                The values, complex for a complex variable, with the last axis of the shape of the variable.
        """
        if not self.is_complex:
            return values
        return np.ascontiguousarray(values).view(complex)
