"""Define the domains that variables take values from."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt


class Domain(Enum):
    """Set of values a variable can take.

    Examples:
        >>> from skcomplex.core.base import Domain
        >>> Domain.POSITIVE.contains(2.0)
        True
        >>> Domain.NONNEGATIVE_INTEGER.contains(1.5)
        False
        >>> Domain.COMPLEX.contains(1 + 2j), Domain.REAL.contains(1 + 2j), Domain.REAL.contains(3 + 0j)
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
    def categorical(*labels: Any) -> 'Categorical':  # noqa: ANN401
        """Return the domain of a finite alphabet, whose values are stored as the indices of its labels.

        Args:
            *labels:
                The distinct labels of the alphabet, in order.

        Returns:
            domain:
                The categorical domain.

        Examples:
            >>> from skcomplex.core.base import Domain
            >>> health = Domain.categorical('susceptible', 'infected', 'recovered')
            >>> health.encode(['infected', 'recovered']).tolist()
            [1.0, 2.0]
        """
        return Categorical(tuple(labels))

    @property
    def sage_domain(self) -> str:
        """Sage domain of the symbols of the domain, which has no nonnegative domains."""
        return {
            Domain.REAL: 'real',
            Domain.POSITIVE: 'positive',
            Domain.NONNEGATIVE: 'real',
            Domain.INTEGER: 'integer',
            Domain.NONNEGATIVE_INTEGER: 'integer',
            Domain.BOOLEAN: 'integer',
            Domain.ANGLE: 'real',
            Domain.COMPLEX: 'complex',
        }[self]

    def encode(self, value: object) -> npt.NDArray[Any]:
        """Return a value as floats, with booleans as 0 and 1 and angles wrapped into one turn, or as complex numbers.

        Args:
            value:
                A number, a boolean, or an array of them. A real domain takes the real part of a complex number.

        Returns:
            array:
                The float values, or the complex values of the complex domain.

        Examples:
            >>> from skcomplex.core.base import Domain
            >>> Domain.COMPLEX.encode([1, 2j]).tolist()
            [(1+0j), 2j]
        """
        array = np.asarray(value)
        if self is Domain.COMPLEX:
            return np.asarray(array, dtype=complex)
        array = np.asarray(array.real if np.iscomplexobj(array) else array, dtype=float)
        return np.mod(array, 2 * np.pi) if self is Domain.ANGLE else array

    def decode(self, values: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        """Return stored values in the domain's own type, booleans for the boolean domain and numbers otherwise.

        Args:
            values:
                The stored float values.

        Returns:
            values:
                The decoded values.
        """
        return np.asarray(values) > 0.5 if self is Domain.BOOLEAN else np.asarray(values)  # noqa: PLR2004

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
        if self is Domain.COMPLEX:
            return True
        if np.any(numbers.imag != 0):
            return False
        array = numbers.real
        integral = bool(np.all(array == np.round(array)))
        checks = {
            Domain.BOOLEAN: bool(np.all((array == 0) | (array == 1))),
            Domain.ANGLE: True,
            Domain.REAL: True,
            Domain.POSITIVE: bool(np.all(array > 0)),
            Domain.NONNEGATIVE: bool(np.all(array >= 0)),
            Domain.INTEGER: integral,
            Domain.NONNEGATIVE_INTEGER: integral and bool(np.all(array >= 0)),
        }
        return checks[self]


@dataclass(frozen=True)
class Categorical:
    """Domain of a finite alphabet, whose values are stored as the indices of its labels.

    Args:
        labels:
            The distinct labels of the alphabet, in order.

    Examples:
        >>> from skcomplex.core.base import Domain
        >>> Domain.categorical('low', 'high').contains(['high', 1])
        True
    """

    labels: tuple[Any, ...]

    @property
    def value(self) -> str:
        """Name of the domain, used in error messages."""
        return f'categorical {{{", ".join(map(str, self.labels))}}}'

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


type AnyDomain = Domain | Categorical
