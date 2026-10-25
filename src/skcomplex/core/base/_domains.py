"""Define the domains that variables take values from."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from ._kinds import DomainKind
from ._utils import wrap_angle


class BaseDomain:
    """Base of the domains, the sets of values a variable takes.

    A domain names its kind, which decides the laws that may target a variable of it, checks that a value belongs to it,
    and turns values to and from the stored numbers of a state vector. Its methods raise `NotImplementedError` rather
    than being abstract, so an enumeration of domains can derive from it. Its `value`, the name or parameters of the
    domain, is declared and not defined, so it does not hide the value of an enumeration member.
    """

    value: Any

    def _check(self) -> Self:
        """Return the domain with its values checked, the domain itself by default."""
        return self

    def check(self) -> Self:
        """Return the domain with its values checked.

        Domains are values, so checking returns a domain rather than changing this one. Declaring a state or a
        parameter checks its domain.

        Returns:
            domain:
                The checked domain.

        Raises:
            ValidationError:
                A value of the domain is invalid, such as an unknown kind of structure.
        """
        return self._check()

    @property
    def domain_kind(self) -> DomainKind:
        """Kind of value the domain holds."""
        raise NotImplementedError

    @property
    def sage_domain(self) -> str:
        """Sage domain of the symbols of the domain: real, positive, integer, or complex."""
        raise NotImplementedError

    def encode(self, value: object) -> npt.NDArray[Any]:
        """Return a value as the stored numbers of a state vector.

        Args:
            value:
                A value of the domain.

        Returns:
            array:
                The stored numbers.
        """
        raise NotImplementedError

    def decode(self, values: npt.NDArray[np.float64]) -> Any:  # noqa: ANN401
        """Return stored numbers as values of the domain.

        Args:
            values:
                The stored numbers.

        Returns:
            values:
                The values of the domain.
        """
        raise NotImplementedError

    def contains(self, value: object) -> bool:
        """Check that a value belongs to the domain.

        Args:
            value:
                A value.

        Returns:
            is_contained:
                Whether the value belongs to the domain.
        """
        raise NotImplementedError


class NumericDomain(BaseDomain, Enum):
    """Set of values a variable can take.

    Examples:
        >>> from skcomplex.core.base import NumericDomain
        >>> NumericDomain.POSITIVE.contains(2.0)
        True
        >>> NumericDomain.NONNEGATIVE_INTEGER.contains(1.5)
        False
        >>> NumericDomain.COMPLEX.contains(1 + 2j), NumericDomain.REAL.contains(1 + 2j), NumericDomain.REAL.contains(
        ...     3 + 0j
        ... )
        (True, False, True)
    """

    REAL = 'real'
    POSITIVE = 'positive'
    NONNEGATIVE = 'nonnegative'
    INTEGER = 'integer'
    NONNEGATIVE_INTEGER = 'nonnegative integer'
    BOOLEAN = 'boolean'
    ANGLE = 'angle'
    COMPLEX = 'complex'

    @staticmethod
    def categorical(*labels: Any) -> 'CategoricalDomain':  # noqa: ANN401
        """Return the domain of a finite alphabet, whose values are stored as the indices of its labels.

        Args:
            *labels:
                The distinct labels of the alphabet, in order.

        Returns:
            domain:
                The categorical domain.

        Examples:
            >>> from skcomplex.core.base import NumericDomain
            >>> health = NumericDomain.categorical('susceptible', 'infected', 'recovered')
            >>> health.encode(['infected', 'recovered']).tolist()
            [1.0, 2.0]
        """
        return CategoricalDomain(tuple(labels))

    @property
    def domain_kind(self) -> DomainKind:
        """Kind of value the domain holds."""
        kinds = {
            NumericDomain.INTEGER: DomainKind.INTEGER,
            NumericDomain.NONNEGATIVE_INTEGER: DomainKind.INTEGER,
            NumericDomain.BOOLEAN: DomainKind.BOOLEAN,
            NumericDomain.ANGLE: DomainKind.ANGLE,
            NumericDomain.COMPLEX: DomainKind.COMPLEX,
        }
        return kinds.get(self, DomainKind.REAL)

    @property
    def sage_domain(self) -> str:
        """Sage domain of the symbols of the domain, which has no nonnegative domains."""
        return {
            NumericDomain.REAL: 'real',
            NumericDomain.POSITIVE: 'positive',
            NumericDomain.NONNEGATIVE: 'real',
            NumericDomain.INTEGER: 'integer',
            NumericDomain.NONNEGATIVE_INTEGER: 'integer',
            NumericDomain.BOOLEAN: 'integer',
            NumericDomain.ANGLE: 'real',
            NumericDomain.COMPLEX: 'complex',
        }[self]

    def encode(self, value: object) -> npt.NDArray[Any]:
        """Return a value as floats, with booleans as 0 and 1 and angles in `(-pi, pi]`, or as complex numbers.

        Args:
            value:
                A number, a boolean, or an array of them. A real domain takes the real part of a complex number.

        Returns:
            array:
                The float values, or the complex values of the complex domain.

        Examples:
            >>> from skcomplex.core.base import NumericDomain
            >>> NumericDomain.COMPLEX.encode([1, 2j]).tolist()
            [(1+0j), 2j]
        """
        array = np.asarray(value)
        if self is NumericDomain.COMPLEX:
            return np.asarray(array, dtype=complex)
        array = np.asarray(array.real if np.iscomplexobj(array) else array, dtype=float)
        return wrap_angle(array) if self is NumericDomain.ANGLE else array

    def decode(self, values: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        """Return stored values in the domain's own type, booleans for the boolean domain and numbers otherwise.

        Args:
            values:
                The stored float values.

        Returns:
            values:
                The decoded values.
        """
        return np.asarray(values) > 0.5 if self is NumericDomain.BOOLEAN else np.asarray(values)  # noqa: PLR2004

    def contains(self, value: object) -> bool:
        """Check that every component of a value belongs to the domain.

        A real domain holds a complex number whose imaginary part is zero.

        Args:
            value:
                A number or an array of numbers.

        Returns:
            is_contained:
                Whether every component is finite and belongs to the domain.
        """
        try:
            numbers = np.asarray(value, dtype=complex)
        except (TypeError, ValueError):
            return False
        if not np.all(np.isfinite(numbers)):
            return False
        if self is NumericDomain.COMPLEX:
            return True
        if np.any(numbers.imag != 0):
            return False
        array = numbers.real
        integral = bool(np.all(array == np.round(array)))
        checks = {
            NumericDomain.BOOLEAN: bool(np.all((array == 0) | (array == 1))),
            NumericDomain.ANGLE: True,
            NumericDomain.REAL: True,
            NumericDomain.POSITIVE: bool(np.all(array > 0)),
            NumericDomain.NONNEGATIVE: bool(np.all(array >= 0)),
            NumericDomain.INTEGER: integral,
            NumericDomain.NONNEGATIVE_INTEGER: integral and bool(np.all(array >= 0)),
        }
        return checks[self]


@dataclass(frozen=True)
class CategoricalDomain(BaseDomain):
    """Domain of a finite alphabet, whose values are stored as the indices of its labels.

    Args:
        labels:
            The distinct labels of the alphabet, in order.

    Examples:
        >>> from skcomplex.core.base import NumericDomain
        >>> NumericDomain.categorical('low', 'high').contains(['high', 1])
        True
    """

    labels: tuple[Any, ...]

    @property
    def value(self) -> str:
        """Name of the domain, used in error messages."""
        return f'categorical {{{", ".join(map(str, self.labels))}}}'

    @property
    def domain_kind(self) -> DomainKind:
        """Kind of value the domain holds."""
        return DomainKind.CATEGORICAL

    @property
    def sage_domain(self) -> str:
        """Sage domain of the symbols of the domain."""
        return 'integer'

    def _index(self, item: Any) -> float:  # noqa: ANN401
        """Return the index of a label, or an index given as an integer."""
        if item in self.labels:
            return float(self.labels.index(item))
        numeric = isinstance(item, int | float | np.integer | np.floating)
        if numeric and float(item).is_integer() and 0 <= item < len(self.labels):
            return float(item)
        error_msg = f'{item!r} is neither a label nor an index of the alphabet {self.labels}.'
        raise ValueError(error_msg)

    def encode(self, value: object) -> npt.NDArray[np.float64]:
        """Return labels or indices as float indices.

        Args:
            value:
                A label, an index, or an array of them.

        Returns:
            array:
                The float indices.
        """
        array = np.asarray(value, dtype=object)
        return np.vectorize(self._index, otypes=[float])(array) if array.size else np.zeros(array.shape)

    def decode(self, values: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        """Return stored indices as labels.

        Args:
            values:
                The stored float indices.

        Returns:
            labels:
                The labels, as an object array of the same shape.
        """
        indices = np.rint(np.asarray(values)).astype(int)
        return np.asarray(self.labels, dtype=object)[np.clip(indices, 0, len(self.labels) - 1)]

    def contains(self, value: object) -> bool:
        """Check that every component of a value is a label or an index of the alphabet.

        Args:
            value:
                A label, an index, or an array of them.

        Returns:
            is_contained:
                Whether every component belongs to the alphabet.
        """
        try:
            self.encode(value)
        except (TypeError, ValueError):
            return False
        return True


@dataclass(frozen=True)
class TensorDomain(BaseDomain):
    """Domain of a state that is a tensor of a rank over the dimensions of the space, with a declared symmetry.

    Fields of tensors, such as the nematic order of active matter, are specified by `add-tensor-fields`.

    Args:
        rank:
            The number of indices.

        symmetry:
            `'symmetric'`, `'antisymmetric'`, `'traceless'` for symmetric and traceless, or `None` for none.

    Examples:
        >>> from skcomplex.core.base import TensorDomain
        >>> TensorDomain(2, 'traceless').value
        'traceless tensor of rank 2'
    """

    rank: int
    symmetry: str | None = None

    @property
    def value(self) -> str:
        """Name of the domain, used in error messages."""
        kind = f'{self.symmetry} tensor' if self.symmetry else 'tensor'
        return f'{kind} of rank {self.rank}'

    @property
    def domain_kind(self) -> DomainKind:
        """Kind of value the domain holds."""
        return DomainKind.TENSOR

    @property
    def sage_domain(self) -> str:
        """Sage domain of the symbols of the components, which are real."""
        return 'real'

    def encode(self, value: object) -> npt.NDArray[Any]:
        """Return the components of a tensor as floats.

        Args:
            value:
                The components.

        Returns:
            array:
                The float values.
        """
        return np.asarray(value, dtype=float)

    def decode(self, values: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        """Return stored values unchanged.

        Args:
            values:
                The stored values.

        Returns:
            values:
                The values.
        """
        return np.asarray(values)

    def contains(self, value: object) -> bool:
        """Check that every component of a tensor is a finite real number.

        Args:
            value:
                The components.

        Returns:
            is_contained:
                Whether every component is finite and real.
        """
        try:
            numbers = np.asarray(value, dtype=complex)
        except (TypeError, ValueError):
            return False
        return bool(np.all(np.isfinite(numbers)) and np.all(numbers.imag == 0))
