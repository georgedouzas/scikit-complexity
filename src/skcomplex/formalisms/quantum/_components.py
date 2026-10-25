"""Define views of quantum elements, whose states lie in Hilbert spaces, as the quantum compiler reads them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ...core.base import BaseDomain, DomainKind, ValidationError, VariableKind, declarations_of
from ...core.components import ElementComponent, GroupComponent
from ...core.laws import BaseLaw, Symbols
from ._laws import LocalOperator, Operator, sparse_matrix

NORM_TOLERANCE = 1e-8
DIGEST_SIZE = 8


def _members(values: Any) -> Any:  # noqa: ANN401
    """Return the values of the members of a group unreduced."""
    return np.asarray(values)


def hilbert_state(element_type: type) -> str | None:
    """Return the name of the state of an element type that lies in a Hilbert space, or `None` for a classical type.

    Args:
        element_type:
            The element type.

    Returns:
        name:
            The name of the state.
    """
    for name, declaration in declarations_of(element_type, VariableKind.STATE).items():
        if isinstance(declaration.domain, HilbertDomain | HilbertFamilyDomain):
            return name
    return None


def is_quantum(element_type: type) -> bool:
    """Check whether an element type is a quantum subsystem, an element with a state in a Hilbert space.

    Args:
        element_type:
            The element type.

    Returns:
        quantum:
            Whether a state of the type lies in a Hilbert space.
    """
    return (
        isinstance(element_type, type)
        and issubclass(element_type, ElementComponent)
        and hilbert_state(element_type) is not None
    )


@dataclass(frozen=True)
class HilbertDomain(BaseDomain):
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
        >>> from skcomplex.formalisms.quantum import HilbertDomain
        >>> space = HilbertDomain(2, {'z': (np.diag([1.0, -1.0]), True)}, basis={'up': 0, 'down': 1})
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

    @property
    def domain_kind(self) -> DomainKind:
        """Kind of value the domain holds."""
        return DomainKind.HILBERT

    @property
    def sage_domain(self) -> str:
        """Sage domain of the symbols of the domain, complex for amplitudes."""
        return 'complex'

    def encode(self, value: object) -> npt.NDArray[Any]:
        """Return a value as complex numbers.

        Args:
            value:
                A vector of amplitudes.

        Returns:
            array:
                The complex values.
        """
        return np.asarray(value, dtype=complex)

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
        """Check that a value is a basis label or index, or a vector of unit norm of the space's dimension.

        Args:
            value:
                The value.

        Returns:
            is_contained:
                Whether the value is a state of the space.
        """
        if isinstance(value, str | int | np.integer) and not isinstance(value, bool):
            try:
                self.index(value)
            except ValueError:
                return False
            return True
        try:
            vector = np.asarray(value, dtype=complex).ravel()
        except (TypeError, ValueError):
            return False
        return vector.size == self.dimension and abs(float(np.linalg.norm(vector)) - 1) <= NORM_TOLERANCE

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
    def family(function: Callable[[Any], 'HilbertDomain']) -> 'HilbertFamilyDomain':
        """Return the Hilbert spaces of an element type that depend on the values of each element.

        Args:
            function:
                A function of an element that returns its Hilbert space, such as the space of a spin of quantum
                number `element.j`.

        Returns:
            family:
                The family, which a state declares as its domain.
        """
        return HilbertFamilyDomain(function)


@dataclass(frozen=True)
class HilbertFamilyDomain(BaseDomain):
    """Hilbert spaces of the elements of a type, each given by a function of the element.

    Args:
        function:
            A function of an element that returns its Hilbert space.
    """

    function: Callable[[Any], HilbertDomain]

    @property
    def value(self) -> str:
        """Name of the domain, used in error messages."""
        return 'Hilbert space'

    @property
    def domain_kind(self) -> DomainKind:
        """Kind of value the domain holds."""
        return DomainKind.HILBERT

    @property
    def sage_domain(self) -> str:
        """Sage domain of the symbols of the domain, complex for amplitudes."""
        return 'complex'

    def encode(self, value: object) -> npt.NDArray[Any]:
        """Return a value as complex numbers.

        Args:
            value:
                A vector of amplitudes.

        Returns:
            array:
                The complex values.
        """
        return np.asarray(value, dtype=complex)

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
        """Check that a value is a basis label or index, or a finite vector, since the space depends on the element.

        Args:
            value:
                The value.

        Returns:
            is_contained:
                Whether the value can be a state of a space of the family.
        """
        if isinstance(value, str | int | np.integer) and not isinstance(value, bool):
            return True
        try:
            return bool(np.all(np.isfinite(np.asarray(value, dtype=complex))))
        except (TypeError, ValueError):
            return False

    def of(self, element: object) -> HilbertDomain:
        """Return the Hilbert space of one element.

        Args:
            element:
                The element.

        Returns:
            space:
                Its Hilbert space.
        """
        return self.function(element)


@dataclass(frozen=True)
class QuantumView:
    """View of an element with a state in a Hilbert space, as the quantum compiler reads it.

    Args:
        element:
            The element.

    Examples:
        >>> from skcomplex.formalisms.quantum import QuantumView
        >>> from skcomplex.models.physics.realizations.quantum import QubitComponent
        >>> QuantumView(QubitComponent('q', initial='down')).initial_vector().tolist()
        [0j, (1+0j)]
    """

    element: ElementComponent

    @property
    def state(self) -> str:
        """Name of the state in the Hilbert space."""
        name = hilbert_state(type(self.element))
        if name is None:
            error_msg = f'{type(self.element).__name__} {self.element.label!r} has no state in a Hilbert space.'
            raise ValidationError(error_msg)
        return name

    @property
    def space(self) -> HilbertDomain:
        """Hilbert space of the element."""
        domain = declarations_of(type(self.element), VariableKind.STATE)[self.state].domain
        space = domain.of(self.element) if isinstance(domain, HilbertFamilyDomain) else domain
        if not isinstance(space, HilbertDomain):
            error_msg = f'The state {self.state} of {self.label!r} lies in {space.value}, not in a Hilbert space.'
            raise ValidationError(error_msg)
        return space

    @property
    def label(self) -> str:
        """Label of the element."""
        return self.element.label

    @property
    def dimension(self) -> int:
        """Dimension of the Hilbert space."""
        return self.space.dimension

    @property
    def display(self) -> str | None:
        """Operator whose expectation visuals show for a group, or `None` for the probabilities of the basis."""
        return self.space.display

    def local_operators(self) -> dict[str, LocalOperator]:
        """Return the local operators of the element by name.

        Returns:
            operators:
                A mapping from name to local operator.
        """
        return {
            name: LocalOperator(name, None if matrix is None else sparse_matrix(matrix), hermitian=hermitian)
            for name, (matrix, hermitian) in self.space.operators.items()
        }

    def basis_index(self, label: object) -> int:
        """Return the index of the basis state of a label or an index.

        Args:
            label:
                The label of a basis state, or its index.

        Returns:
            index:
                The index of the basis state.

        Raises:
            ValidationError:
                The label names no basis state.
        """
        try:
            return self.space.index(label)
        except ValueError as error:
            error_msg = f'{type(self.element).__name__} {self.label!r}: {error}'
            raise ValidationError(error_msg) from error

    def initial_vector(self) -> npt.NDArray[np.complex128]:
        """Return the initial state of the element as a normalised vector.

        Returns:
            vector:
                The initial state.

        Raises:
            ValidationError:
                The initial state names no basis state, or is a vector of the wrong size or norm.
        """
        initial = getattr(self.element, self.state)
        if isinstance(initial, str | int | np.integer):
            vector = np.zeros(self.dimension, dtype=complex)
            vector[self.basis_index(initial)] = 1.0
            return vector
        vector = np.asarray(initial, dtype=complex).ravel()
        norm = float(np.linalg.norm(vector))
        if vector.size != self.dimension or abs(norm - 1) > NORM_TOLERANCE:
            error_msg = (
                f'The initial vector of {self.label!r} has {vector.size} components and norm {norm:.6g}, but its '
                f'space has dimension {self.dimension} and states have norm 1. Normalise the vector.'
            )
            raise ValidationError(error_msg)
        return vector

    def namespace(self, key: str) -> dict[str, Any]:
        """Return the operators of the subsystem by name, with the names the element adds through `operator_namespace`.

        Args:
            key:
                The key of the subsystem, such as `chain[3]`.

        Returns:
            names:
                A mapping from the name of every local operator to the operator acting on the subsystem.
        """
        names: dict[str, Any] = {name: Operator({((key, local),): 1}) for name, local in self.local_operators().items()}
        additional = getattr(self.element, 'operator_namespace', None)
        return {**names, **(additional(key, self) if additional is not None else {})}

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the laws of the element for a namespace of its operators.

        Args:
            s:
                The namespace.

        Returns:
            laws:
                The laws of the element.
        """
        return self.element.laws(s)


def view_of(component: ElementComponent | GroupComponent) -> QuantumView:
    """Return the quantum view of an element, or of the first member of a group.

    Args:
        component:
            A quantum element or a group of them.

    Returns:
        view:
            The view, from whose space the display and the local operators of the component are read.
    """
    if not isinstance(component, GroupComponent):
        return QuantumView(component)
    values = {
        name: value if isinstance(value, str) or np.ndim(value) == 0 or len(value) != component.size else value[0]
        for name, value in (component.values or {}).items()
    }
    return QuantumView(component.element(component.label, **values))
