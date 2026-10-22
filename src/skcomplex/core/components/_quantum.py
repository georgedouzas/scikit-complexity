"""Define views of quantum elements, whose states lie in Hilbert spaces, as the quantum compiler reads them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..base import HilbertFamily, HilbertSpace, ValidationError, VariableKind, declarations_of
from ..laws._laws import Law
from ..laws._operators import LocalOperator, Operator, sparse_matrix
from ..laws._symbols import Symbols
from ._components import Element, Group

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
        if isinstance(declaration.domain, HilbertSpace | HilbertFamily):
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
        isinstance(element_type, type) and issubclass(element_type, Element) and hilbert_state(element_type) is not None
    )


@dataclass(frozen=True)
class QuantumView:
    """View of an element with a state in a Hilbert space, as the quantum compiler reads it.

    Args:
        element:
            The element.

    Examples:
        >>> from skcomplex.core.components import QuantumView
        >>> from skcomplex.models.physics.realizations import Qubit
        >>> QuantumView(Qubit('q', initial='down')).initial_vector().tolist()
        [0j, (1+0j)]
    """

    element: Element

    @property
    def state(self) -> str:
        """Name of the state in the Hilbert space."""
        name = hilbert_state(type(self.element))
        if name is None:
            error_msg = f'{type(self.element).__name__} {self.element.label!r} has no state in a Hilbert space.'
            raise ValidationError(error_msg)
        return name

    @property
    def space(self) -> HilbertSpace:
        """Hilbert space of the element."""
        domain = declarations_of(type(self.element), VariableKind.STATE)[self.state].domain
        space = domain.of(self.element) if isinstance(domain, HilbertFamily) else domain
        if not isinstance(space, HilbertSpace):
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

    def laws(self, s: Symbols) -> list[Law]:
        """Return the laws of the element for a namespace of its operators.

        Args:
            s:
                The namespace.

        Returns:
            laws:
                The laws of the element.
        """
        return self.element.laws(s)


def view_of(component: Element | Group) -> QuantumView:
    """Return the quantum view of an element, or of the first member of a group.

    Args:
        component:
            A quantum element or a group of them.

    Returns:
        view:
            The view, from whose space the display and the local operators of the component are read.
    """
    if not isinstance(component, Group):
        return QuantumView(component)
    values = {
        name: value if isinstance(value, str) or np.ndim(value) == 0 or len(value) != component.size else value[0]
        for name, value in (component.values or {}).items()
    }
    return QuantumView(component.element(component.label, **values))
