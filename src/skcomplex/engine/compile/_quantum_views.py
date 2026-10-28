"""Define views of quantum elements, whose states lie in Hilbert spaces, as the quantum compiler reads them.

An element is a quantum subsystem when it takes a `HilbertSpec` as an argument of its constructor and declares a complex
state, over `CC` or over `CC^dimension`, such as `psi: Annotated[object, StateSpec(domain=CC)] = 'up'`. A group of
such elements takes the spec of its members, which `values` may replace like any other argument.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import CC

from ...core.elements import BaseElement, Element, GroupElement
from ...core.errors import ValidationError
from ...core.specs import HilbertSpec
from ..problems import VariableKind, declarations_of
from ._quantum_operators import LocalOperator, sparse_matrix

NORM_TOLERANCE = 1e-8


def hilbert_state(element_type: type) -> str | None:
    """Return the name of the complex state of an element type, over `CC` or `CC^dimension`, or `None` for none.

    Args:
        element_type:
            The element type.

    Returns:
        name:
            The name of the state.

    Examples:
        >>> from typing import Annotated
        >>> from sage.all__sagemath_symbolics import CC
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.engine.compile._quantum_views import hilbert_state
        >>> class QubitElement(Element):
        ...     def __init__(self, label, psi: Annotated[object, StateSpec(domain=CC**2)] = 'up'):
        ...         super().__init__(label)
        ...         self.psi = psi
        >>> hilbert_state(QubitElement)
        'psi'
    """
    for name, declaration in declarations_of(element_type, VariableKind.STATE).items():
        base_ring = getattr(declaration.domain, 'base_ring', None)
        if declaration.domain is CC or (base_ring is not None and base_ring() is CC):
            return name
    return None


def hilbert_of(element: BaseElement) -> HilbertSpec | None:
    """Return the Hilbert space an element takes, of a member for a group, or `None` for a classical element.

    Args:
        element:
            An element or a group.

    Returns:
        space:
            The first argument of the element that is a `HilbertSpec`, or `None`.
    """
    member = member_of(element)
    if member is None:
        return None
    return next((value for value in member.get_params(deep=False).values() if isinstance(value, HilbertSpec)), None)


def member_of(element: BaseElement) -> Element | None:
    """Return an element itself, the first member of a group, or `None` for any other part.

    Args:
        element:
            An element or a group.

    Returns:
        member:
            The element, built with the values of the first member for a group.
    """
    if isinstance(element, Element):
        return element
    if not isinstance(element, GroupElement) or not isinstance(element.element, type):
        return None
    values = {
        name: value if isinstance(value, str | HilbertSpec) or np.ndim(value) == 0 else value[0]
        for name, value in (element.values or {}).items()
    }
    return element.element(element.label).set_params(**values)


def is_quantum(element: object) -> bool:
    """Check whether an element or a group is a quantum subsystem, which takes a `HilbertSpec`.

    Args:
        element:
            An element, a group, or any other part.

    Returns:
        quantum:
            Whether the element, or a member of the group, takes a Hilbert space.
    """
    return isinstance(element, BaseElement) and hilbert_of(element) is not None


@dataclass(frozen=True)
class QuantumView:
    """View of an element with a state in a Hilbert space, as the quantum compiler reads it.

    Args:
        element:
            The element.

    Examples:
        >>> import numpy as np
        >>> from typing import Annotated
        >>> from sage.all__sagemath_symbolics import CC
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.specs import HilbertSpec, StateSpec
        >>> from skcomplex.engine.compile import QuantumView
        >>> QUBIT = HilbertSpec(2, {'z': (np.diag([1.0, -1.0]), True)}, basis={'up': 0, 'down': 1})
        >>> class QubitElement(Element):
        ...     def __init__(self, label, psi: Annotated[object, StateSpec(domain=CC**2)] = 'up', hilbert=QUBIT):
        ...         super().__init__(label)
        ...         self.psi = psi
        ...         self.hilbert = hilbert
        >>> QuantumView(QubitElement('q', psi='down')).initial_vector().tolist()
        [0j, (1+0j)]
    """

    element: Element

    @property
    def state(self) -> str:
        """Name of the state in the Hilbert space."""
        name = hilbert_state(type(self.element))
        if name is None:
            error_msg = (
                f'{type(self.element).__name__} {self.element.label!r} takes a HilbertSpec but declares no complex '
                'state. Declare its state as `StateSpec(domain=CC)`.'
            )
            raise ValidationError(error_msg)
        return name

    @property
    def space(self) -> HilbertSpec:
        """Hilbert space of the element, checked against the domain of its state."""
        space = hilbert_of(self.element)
        if space is None:
            error_msg = f'{type(self.element).__name__} {self.label!r} takes no HilbertSpec.'
            raise ValidationError(error_msg)
        space.check()
        domain = declarations_of(type(self.element), VariableKind.STATE)[self.state].domain
        dimension = space.dimension if domain is CC else domain.dimension()
        if dimension != space.dimension:
            error_msg = (
                f'{type(self.element).__name__} {self.label!r} has its state {self.state} over CC^{dimension}, but its '
                f'HilbertSpec has dimension {space.dimension}.'
            )
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
            name: LocalOperator(name, sparse_matrix(matrix), hermitian=hermitian)
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
        space = self.space
        if isinstance(label, str) and label in space.basis:
            return space.basis[label]
        if isinstance(label, int | np.integer) and not isinstance(label, bool) and 0 <= label < space.dimension:
            return int(label)
        labels = ', '.join(map(repr, space.basis)) or 'none'
        error_msg = (
            f'{type(self.element).__name__} {self.label!r}: {label!r} is no basis state. Use a label ({labels}) or an '
            f'index from 0 to {space.dimension - 1}.'
        )
        raise ValidationError(error_msg)

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
