"""Define the domains that variables take values from."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt


def wrap_angle(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return angles wrapped into one turn centred on zero, the half-open interval `(-pi, pi]`.

    Args:
        values:
            An angle or an array of angles in radians.

    Returns:
        angles:
            The angles in `(-pi, pi]`, with `-pi` moved to `pi` and angles already inside left exactly as they are.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.base import wrap_angle
        >>> wrap_angle(np.array([3 * np.pi / 2, -np.pi, 0.5])).round(4).tolist()
        [-1.5708, 3.1416, 0.5]
    """
    angles = np.asarray(values, dtype=float)
    inside = (angles > -np.pi) & (angles <= np.pi)
    return np.where(inside, angles, np.pi - np.mod(np.pi - angles, 2 * np.pi))


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
        """Return a value as floats, with booleans as 0 and 1 and angles in `(-pi, pi]`, or as complex numbers.

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
        return wrap_angle(array) if self is Domain.ANGLE else array

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


@dataclass(frozen=True)
class HilbertSpace:
    """Domain of a state that is a vector of a Hilbert space, with the local operators laws build Hamiltonians from.

    An element whose state lies in a Hilbert space is a quantum subsystem. Its laws read every local operator by name,
    and a group of such elements is the tensor product of its members' spaces, in member order.

    Args:
        dimension:
            The dimension of the space.

        operators:
            A mapping from operator name to its matrix and whether it is Hermitian, such as `{'z': (matrix, True)}`.

        basis:
            A mapping from the label of a basis state to its index, such as `{'up': 0, 'down': 1}`.

        display:
            The name of the operator whose expectation visuals show for a group of such elements, or `None` for the
            probabilities of the basis states.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.base import HilbertSpace
        >>> space = HilbertSpace(2, {'z': (np.diag([1.0, -1.0]), True)}, basis={'up': 0, 'down': 1})
        >>> space.index('down'), space.index(0)
        (1, 0)
    """

    dimension: int
    operators: Mapping[str, tuple[Any, bool]] = field(default_factory=dict)
    basis: Mapping[str, int] = field(default_factory=dict)
    display: str | None = 'z'

    @property
    def value(self) -> str:
        """Name of the domain, used in error messages."""
        return f'Hilbert space of dimension {self.dimension}'

    def index(self, label: object) -> int:
        """Return the index of a basis state from its label or its index.

        Args:
            label:
                The label of a basis state, or its index.

        Returns:
            index:
                The index of the basis state.

        Raises:
            ValueError:
                The label names no basis state.
        """
        if isinstance(label, str) and label in self.basis:
            return self.basis[label]
        if isinstance(label, int | np.integer) and not isinstance(label, bool) and 0 <= label < self.dimension:
            return int(label)
        labels = ', '.join(map(repr, self.basis)) or 'none'
        error_msg = f'{label!r} is no basis state. Use a label ({labels}) or an index from 0 to {self.dimension - 1}.'
        raise ValueError(error_msg)

    @staticmethod
    def family(function: Callable[[Any], 'HilbertSpace']) -> 'HilbertFamily':
        """Return the Hilbert spaces of an element type that depend on the values of each element.

        Args:
            function:
                A function of an element that returns its Hilbert space, such as the space of a spin of quantum
                number `element.j`.

        Returns:
            family:
                The family, which a state declares as its domain.
        """
        return HilbertFamily(function)


@dataclass(frozen=True)
class HilbertFamily:
    """Hilbert spaces of the elements of a type, each given by a function of the element.

    Args:
        function:
            A function of an element that returns its Hilbert space.
    """

    function: Callable[[Any], HilbertSpace]

    @property
    def value(self) -> str:
        """Name of the domain, used in error messages."""
        return 'Hilbert space'

    def of(self, element: object) -> HilbertSpace:
        """Return the Hilbert space of one element.

        Args:
            element:
                The element.

        Returns:
            space:
                Its Hilbert space.
        """
        return self.function(element)


STRUCTURES = ('sequence', 'tiling', 'graph')


@dataclass(frozen=True)
class Structure:
    """Domain of a state that is a structure rewritten as a whole: a sequence of symbols, a tiling, or a graph.

    Args:
        kind:
            `'sequence'`, `'tiling'`, or `'graph'`.

    Raises:
        ValueError:
            The kind is unknown.

    Examples:
        >>> from skcomplex.core.base import Structure
        >>> Structure('sequence').value
        'sequence'
    """

    kind: str

    def __post_init__(self) -> None:
        """Check the kind of structure."""
        if self.kind not in STRUCTURES:
            error_msg = f'Unknown structure {self.kind!r}. Use one of {", ".join(STRUCTURES)}.'
            raise ValueError(error_msg)

    @property
    def value(self) -> str:
        """Name of the domain, used in error messages."""
        return self.kind


@dataclass(frozen=True)
class Tensor:
    """Domain of a state that is a tensor of a rank over the dimensions of the space, with a declared symmetry.

    Fields of tensors, such as the nematic order of active matter, are specified by `add-tensor-fields`.

    Args:
        rank:
            The number of indices.

        symmetry:
            `'symmetric'`, `'antisymmetric'`, `'traceless'` for symmetric and traceless, or `None` for none.

    Examples:
        >>> from skcomplex.core.base import Tensor
        >>> Tensor(2, 'traceless').value
        'traceless tensor of rank 2'
    """

    rank: int
    symmetry: str | None = None

    @property
    def value(self) -> str:
        """Name of the domain, used in error messages."""
        kind = f'{self.symmetry} tensor' if self.symmetry else 'tensor'
        return f'{kind} of rank {self.rank}'


type AnyDomain = Domain | Categorical | HilbertSpace | HilbertFamily | Structure | Tensor
