"""Position of one named variable in a flat vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import CC, NN, ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ...specs import BaseSpec
from ._domain_kind import DomainKind

PARTS = 2


_PART_NAMES = ('real', 'imag')


_BOOLEAN_LABELS = (frozenset({0, 1}), frozenset({False, True}))


@dataclass(frozen=True)
class Entry(BaseSpec):
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
        values = np.broadcast_to(Entry._encode(self.domain, value, self.period), self.shape)
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

    @property
    def kind(self) -> DomainKind:
        """Kind of value the variable takes, an angle for a periodic variable."""
        return Entry._domain_kind(self.domain, self.period)

    def contains(self, value: object) -> bool:
        """Check that every component of a value belongs to the domain of the variable.

        Args:
            value:
                A value or an array of values.

        Returns:
            is_contained:
                Whether every component belongs to the domain.
        """
        return Entry._contains(self.domain, value)

    def read(self, values: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Return stored numbers as values of the domain: labels for a categorical domain, booleans for a boolean one.

        Args:
            values:
                The stored numbers.

        Returns:
            values:
                The values.

        Examples:
            >>> from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
            >>> from skcomplex.engine.problems import Entry
            >>> Entry('a.s', 0, (), FiniteEnumeratedSet(['S', 'I'])).read([1.0, 0.0]).tolist()
            ['I', 'S']
        """
        return Entry._decode(self.domain, values)

    @staticmethod
    def _wrap_angle(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return angles in `(-pi, pi]`, with `-pi` moved to `pi` and angles already inside left exactly as they are.

        Args:
            values:
                An angle or an array of angles in radians.

        Returns:
            angles:
                The wrapped angles.
        """
        angles = np.asarray(values, dtype=float)
        inside = (angles > -np.pi) & (angles <= np.pi)
        return np.where(inside, angles, np.pi - np.mod(np.pi - angles, 2 * np.pi))

    @staticmethod
    def _is_number(item: object) -> bool:
        """Check whether an element of a finite set is a number, which a state stores as it is."""
        return isinstance(item, int | float) or getattr(item, 'is_numeric', lambda: False)()

    @staticmethod
    def _labels_of(domain: object) -> tuple[object, ...] | None:
        """Return the labels of a finite domain stored as indices, or `None` for a numeric or a boolean domain.

        A finite set of numbers, such as the spins `FiniteEnumeratedSet([-1, 1])`, stores its values as they are.

        Args:
            domain:
                The Sage set of the values.

        Returns:
            labels:
                The labels in order, or `None`.
        """
        if not isinstance(domain, FiniteEnumeratedSet) or frozenset(domain) in _BOOLEAN_LABELS:
            return None
        if all(Entry._is_number(item) for item in domain):
            return None
        return tuple(domain)

    @staticmethod
    def _domain_kind(domain: object, period: object = None) -> DomainKind:
        """Return the kind of value a Sage set holds, with an angle for a periodic state.

        Args:
            domain:
                The Sage set of the values.

            period:
                The period the values wrap around, or `None`.

        Returns:
            kind:
                The kind of value.
        """
        if period is not None:
            return DomainKind.ANGLE
        if domain is CC:
            return DomainKind.COMPLEX
        if isinstance(domain, FiniteEnumeratedSet):
            if Entry._labels_of(domain) is not None:
                return DomainKind.CATEGORICAL
            return DomainKind.BOOLEAN if frozenset(domain) in _BOOLEAN_LABELS else DomainKind.INTEGER
        if domain is ZZ or domain is NN:
            return DomainKind.INTEGER
        return DomainKind.REAL

    @staticmethod
    def _sage_assumption(domain: object) -> str:
        """Return the Sage domain of the symbols of a Sage set: real, positive, integer, or complex.

        Args:
            domain:
                The Sage set of the values.

        Returns:
            assumption:
                The Sage domain of the symbols.
        """
        kind = Entry._domain_kind(domain)
        if kind == DomainKind.COMPLEX:
            return 'complex'
        if kind in {DomainKind.INTEGER, DomainKind.BOOLEAN, DomainKind.CATEGORICAL}:
            return 'integer'
        if isinstance(domain, RealSet) and domain.inf() >= 0 and 0 not in domain:
            return 'positive'
        return 'real'

    @staticmethod
    def _encode(domain: object, value: object, period: object = None) -> npt.NDArray[np.generic]:
        """Return a value as the stored numbers of a state vector.

        Labels become their indices, booleans 0 and 1, angles their values in `(-pi, pi]`, and a complex domain keeps
        complex numbers. A real domain takes the real part of a complex number.

        Args:
            domain:
                The Sage set of the values.

            value:
                A value or an array of values.

            period:
                The period the values wrap around, or `None`.

        Returns:
            array:
                The stored numbers.
        """
        labels = Entry._labels_of(domain)
        if labels is not None:
            array = np.asarray(value, dtype=object)
            index = np.vectorize(lambda item: float(labels.index(item) if item in labels else item), otypes=[float])
            return index(array)
        array = np.asarray(value)
        if domain is CC:
            return np.asarray(array, dtype=complex)
        array = np.asarray(array.real if np.iscomplexobj(array) else array, dtype=float)
        return Entry._wrap_angle(array) if period is not None else array

    @staticmethod
    def _decode(domain: object, values: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Return stored numbers as values of a Sage set: labels for a categorical domain, booleans for a boolean one.

        Args:
            domain:
                The Sage set of the values.

            values:
                The stored numbers.

        Returns:
            values:
                The values.
        """
        labels = Entry._labels_of(domain)
        if labels is not None:
            indices = np.rint(np.asarray(values)).astype(int)
            return np.asarray(labels, dtype=object)[np.clip(indices, 0, len(labels) - 1)]
        if Entry._domain_kind(domain) == DomainKind.BOOLEAN:
            return np.asarray(values) > 0.5  # noqa: PLR2004
        return np.asarray(values)

    @staticmethod
    def _contains(domain: object, value: object) -> bool:
        """Check that every component of a value belongs to a Sage set.

        Args:
            domain:
                The Sage set of the values.

            value:
                A value or an array of values.

        Returns:
            is_contained:
                Whether every component belongs to the set.
        """
        labels = Entry._labels_of(domain)
        items = np.asarray(value, dtype=object).ravel()
        if labels is not None:
            return all(item in labels or (isinstance(item, int) and 0 <= item < len(labels)) for item in items)
        try:
            return all(complex(item) in domain if domain is CC else item in domain for item in items)
        except (TypeError, ValueError):
            return False
