"""Define the entry spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from collections.abc import Container
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import CC, NN, RR, ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ..types import Array, ComplexArray, DomainKind
from ._base_spec import BaseSpec
from ._power_series_spec import PowerSeriesSpec
from ._subshift_spec import SubshiftSpec
from ._tensor_spec import TensorSpec

PARTS = 2
PART_NAMES = ('real', 'imag')
BOOLEAN_LABELS = (frozenset({0, 1}), frozenset({False, True}))
BOOLEAN_THRESHOLD = 0.5


@dataclass(frozen=True)
class EntrySpec(BaseSpec):
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

        tensor:
            The independent components of a rank-2 tensor variable, or `None`.

    Examples:
        >>> from sage.all__sagemath_symbolics import CC, RR
        >>> from skcomplex.core.specs import EntrySpec
        >>> entry = EntrySpec('a.z', 0, (2,), CC, is_complex=True)
        >>> entry.size, entry.component_names
        (4, ['a.z[0].real', 'a.z[0].imag', 'a.z[1].real', 'a.z[1].imag'])
        >>> from skcomplex.core.specs import TensorSpec
        >>> tensor = EntrySpec('a.Q', 0, (2,), RR, tensor=TensorSpec('traceless', 2))
        >>> tensor.component_names, tensor.kind.value
        (['a.Q_xx', 'a.Q_xy'], 'tensor')
    """

    name: str
    start: int
    shape: tuple[int, ...]
    domain: object
    members: int | None = None
    is_complex: bool = False
    period: float | None = None
    tensor: TensorSpec | None = None

    def _get_parts(self, names: list[str]) -> list[str]:
        """Return names with the suffixes of the real and imaginary parts.

        The suffixes are added for a complex variable.
        """
        if not self.is_complex:
            return names
        return [f'{name}.{part}' for name in names for part in PART_NAMES]

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
        """Shape of the variable of one member.

        It is `()` for a scalar and `(n_dim,)` for a vector.
        """
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
        """Names of the components.

        They carry an index suffix for vectors, a member index for groups, and complex parts.
        """
        components = math.prod(self.member_shape)
        if self.tensor is not None:
            names = [f'{self.name}_{name}' for name in self.tensor.names]
            if self.members is None:
                return names
            return [f'{name}[{member}]' for member in range(self.rows) for name in names]
        if self.members is None:
            if not self.shape:
                return self._get_parts([self.name])
            return self._get_parts([f'{self.name}[{index}]' for index in range(components)])
        if not self.member_shape:
            return self._get_parts([f'{self.name}[{member}]' for member in range(self.rows)])
        return self._get_parts(
            [f'{self.name}[{member}, {index}]' for member in range(self.rows) for index in range(components)],
        )

    @property
    def generic_names(self) -> list[str]:
        """Names of the components of one member, with the member index `i` for groups."""
        if self.members is None:
            return self.component_names
        if self.tensor is not None:
            return [f'{self.name}_{name}[i]' for name in self.tensor.names]
        if not self.member_shape:
            return self._get_parts([f'{self.name}[i]'])
        return self._get_parts([f'{self.name}[i, {index}]' for index in range(math.prod(self.member_shape))])

    def get_view(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the variable's components of a flat vector as a view.

        The view has one row per member.

        Args:
            vector:
                A flat vector.

        Returns:
            view:
                A `(rows, columns)` view that writes through to the vector.
        """
        return vector[self.indices].reshape(self.rows, self.columns)

    def encode(self, value: object) -> npt.NDArray[np.float64]:
        """Return a value broadcast to the shape of the variable as flat floats.

        A complex variable gives the floats in pairs.

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
            >>> from skcomplex.core.specs import EntrySpec
            >>> EntrySpec('a.z', 0, (), CC, is_complex=True).encode(1 + 2j).tolist()
            [1.0, 2.0]
        """
        values = np.broadcast_to(EntrySpec._encode(self.domain, value, self.period), self.shape)
        if self.is_complex:
            return np.ascontiguousarray(values, dtype=complex).ravel().view(float).copy()
        return values.ravel().copy()

    def decode(self, values: Array) -> Array | ComplexArray:
        """Return the stored components of the variable, with complex parts joined.

        The components are the last axis of an array.

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
        """Kind of value the variable takes.

        It is an angle for a periodic variable and a tensor for a tensor component.
        """
        if self.tensor is not None:
            return DomainKind.TENSOR
        return EntrySpec._get_domain_kind(self.domain, self.period)

    def contains(self, value: object) -> bool:
        """Check that every component of a value belongs to the domain of the variable.

        Args:
            value:
                A value or an array of values.

        Returns:
            is_contained:
                Whether every component belongs to the domain.
        """
        return EntrySpec._contains(self.domain, value)

    def read(self, values: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Return stored numbers as values of the domain.

        They are labels for a categorical domain and booleans for a boolean one.

        Args:
            values:
                The stored numbers.

        Returns:
            values:
                The values.

        Examples:
            >>> from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
            >>> from skcomplex.core.specs import EntrySpec
            >>> EntrySpec('a.s', 0, (), FiniteEnumeratedSet(['S', 'I'])).read([1.0, 0.0]).tolist()
            ['I', 'S']
        """
        return EntrySpec._decode(self.domain, values)

    @staticmethod
    def _wrap_angle(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return angles in `(-pi, pi]`.

        An angle of `-pi` moves to `pi`, and angles already inside are left exactly as they are.

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
        """Check whether an element of a finite set is a number.

        A state stores a number as it is.
        """
        return isinstance(item, int | float) or getattr(item, 'is_numeric', lambda: False)()

    @staticmethod
    def _get_labels_of(domain: object) -> tuple[object, ...] | None:
        """Return the labels of a finite domain stored as indices.

        It returns `None` for a numeric or a boolean domain. A finite set of numbers, such as the spins
        `FiniteEnumeratedSet([-1, 1])`, stores its values as they are.

        Args:
            domain:
                The Sage set of the values.

        Returns:
            labels:
                The labels in order, or `None`.
        """
        if not isinstance(domain, FiniteEnumeratedSet) or frozenset(domain) in BOOLEAN_LABELS:
            return None
        if all(EntrySpec._is_number(item) for item in domain):
            return None
        return tuple(domain)

    @staticmethod
    def _get_cell_domain(domain: object) -> object:
        """Return the Sage set of one component.

        It is the alphabet of a subshift or a power series, or the domain itself.
        """
        return domain.alphabet if isinstance(domain, SubshiftSpec | PowerSeriesSpec) else domain

    @staticmethod
    def _get_domain_kind(domain: object, period: object = None) -> DomainKind:
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
        domain = EntrySpec._get_cell_domain(domain)
        if domain is CC:
            return DomainKind.COMPLEX
        if isinstance(domain, FiniteEnumeratedSet):
            if EntrySpec._get_labels_of(domain) is not None:
                return DomainKind.CATEGORICAL
            return DomainKind.BOOLEAN if frozenset(domain) in BOOLEAN_LABELS else DomainKind.INTEGER
        if domain is ZZ or domain is NN:
            return DomainKind.INTEGER
        return DomainKind.REAL

    @staticmethod
    def _get_sage_assumption(domain: object) -> str:
        """Return the Sage domain of the symbols of a Sage set.

        The domain is real, positive, integer, or complex.

        Args:
            domain:
                The Sage set of the values.

        Returns:
            assumption:
                The Sage domain of the symbols.
        """
        kind = EntrySpec._get_domain_kind(domain)
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
        labels = EntrySpec._get_labels_of(domain)
        if labels is not None:
            array = np.asarray(value, dtype=object)
            index = np.vectorize(lambda item: float(labels.index(item) if item in labels else item), otypes=[float])
            return index(array)
        array = np.asarray(value)
        if domain is CC:
            return np.asarray(array, dtype=complex)
        array = np.asarray(array.real if np.iscomplexobj(array) else array, dtype=float)
        return EntrySpec._wrap_angle(array) if period is not None else array

    @staticmethod
    def _decode(domain: object, values: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Return stored numbers as values of a Sage set.

        They are labels for a categorical domain and booleans for a boolean one.

        Args:
            domain:
                The Sage set of the values.

            values:
                The stored numbers.

        Returns:
            values:
                The values.
        """
        labels = EntrySpec._get_labels_of(domain)
        if labels is not None:
            indices = np.rint(np.asarray(values)).astype(int)
            return np.asarray(labels, dtype=object)[np.clip(indices, 0, len(labels) - 1)]
        if EntrySpec._get_domain_kind(domain) == DomainKind.BOOLEAN:
            return np.asarray(values) > BOOLEAN_THRESHOLD
        return np.asarray(values)

    @staticmethod
    def _contains_numbers(domain: object, value: object) -> bool | None:
        """Check at once that real numbers belong to the real line or to intervals.

        The check is the one Sage makes for each number, against the real line or a union of intervals. A large array,
        such as a tensor parameter of an experiment, is checked in one pass instead of one Sage call per component. Not
        a number belongs to neither, and infinities belong to the real line alone.

        Args:
            domain:
                The Sage set of the values.

            value:
                A value or an array of values.

        Returns:
            is_contained:
                Whether every component belongs to the set, or `None` when the values are not real numbers or the set
                is neither the real line nor a union of intervals.
        """
        array = np.asarray(value)
        if array.dtype.kind not in 'biuf' or not (domain is RR or isinstance(domain, RealSet)):
            return None
        numbers = array.astype(float).ravel()
        if domain is RR:
            return not bool(np.isnan(numbers).any())
        inside = np.zeros(numbers.shape, dtype=bool)
        for interval in domain:
            lower, upper = float(interval.lower()), float(interval.upper())
            above = (numbers > lower) | (interval.lower_closed() & (numbers == lower))
            below = (numbers < upper) | (interval.upper_closed() & (numbers == upper))
            inside |= above & below
        return bool(inside.all())

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
        domain = EntrySpec._get_cell_domain(domain)
        numbers = EntrySpec._contains_numbers(domain, value)
        if numbers is not None:
            return numbers
        labels = EntrySpec._get_labels_of(domain)
        items = np.asarray(value, dtype=object).ravel()
        if labels is not None:
            return all(item in labels or (isinstance(item, int) and 0 <= item < len(labels)) for item in items)
        if not isinstance(domain, Container):
            return False
        try:
            return all(complex(item) in domain if domain is CC else item in domain for item in items)
        except (TypeError, ValueError):
            return False
