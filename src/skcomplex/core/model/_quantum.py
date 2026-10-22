"""Define quantum subsystems, operators with symbolic coefficients, and the Hamiltonian and dissipator laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..base import HilbertFamily, HilbertSpace, ValidationError, VariableKind, declarations_of
from ._components import Element, Group, Interaction
from ._lattices import LatticeSpace
from ._laws import Law
from ._observables import Observable
from ._operators import LocalOperator, Operator, sparse_matrix
from ._spaces import NetworkSpace, Space
from ._symbols import Symbols
from ._systems import System
from ._time import Time

DEFAULT_MAX_DIMENSION = 2**20
NORM_TOLERANCE = 1e-8
DIGEST_SIZE = 8


def neighbour_pairs(space: Space | None, size: int) -> list[tuple[int, int]]:
    """Return the pairs of neighbouring members of a lattice or a network, each pair once.

    Args:
        space:
            The lattice or the network.

        size:
            The number of members, one per site or node.

    Returns:
        pairs:
            The member indices of every pair of neighbours.

    Raises:
        ValidationError:
            The space is neither a lattice nor a network.

    Examples:
        >>> from skcomplex.core.model import LatticeSpace, neighbour_pairs
        >>> neighbour_pairs(LatticeSpace(3, boundary='fixed'), 3)
        [(0, 1), (1, 2)]
    """
    if isinstance(space, NetworkSpace):
        index = {node: number for number, node in enumerate(space.graph.nodes)}
        return [(index[first], index[second]) for first, second in space.graph.edges if first != second]
    if not isinstance(space, LatticeSpace):
        error_msg = f'Neighbours are defined on a LatticeSpace or a NetworkSpace, not on {space!r}.'
        raise ValidationError(error_msg)
    dimensions = space.dimensions
    sites = np.arange(size).reshape(dimensions)
    found: dict[frozenset[int], tuple[int, int]] = {}
    for axis, length in enumerate(dimensions):
        shifted = np.roll(sites, -1, axis=axis)
        pairs = np.stack([sites.ravel(), shifted.ravel()], axis=1)
        if space.boundary != 'periodic':
            edge = np.indices(dimensions)[axis].ravel() == length - 1
            pairs = pairs[~edge]
        for first, second in pairs.tolist():
            if first != second:
                found.setdefault(frozenset((first, second)), (first, second))
    return list(found.values())


def _members(values: Any) -> Any:  # noqa: ANN401
    """Return the values of the members of a group unreduced."""
    return np.asarray(values)


def expectation(
    label: str,
    expression: Callable[..., Any],
    *,
    group: str | None = None,
    reduce: str | Callable[[Any], Any] | None = None,
) -> Observable:
    """Return the observable of the expectation value of an operator.

    For a system, `expression(s)` returns an operator of the system namespace, where `s.<element>` is the namespace of
    a quantum element and `s.<group>` is the tuple of the namespaces of its members. For a group, `expression(m)`
    returns an operator of one member, and the observable holds one value per member unless it is reduced. The value is
    `<psi|O|psi>` for a pure state and `Tr(rho O)` for a density matrix, real for a Hermitian operator.

    Args:
        label:
            The label of the observable.

        expression:
            A module-level function of the namespace that returns an operator.

        group:
            The path of the quantum group to evaluate member by member, or `None` for the whole system.

        reduce:
            The reduction of the member values, such as `'sum'` or `'mean'`, or `None` to keep one value per member.

    Returns:
        observable:
            The observable.

    Examples:
        >>> from skcomplex.core.model import expectation
        >>> expectation('sz', abs, group='chain').group
        'chain'
    """
    return Observable(label, expression, group=group, reduce=_members if reduce is None else reduce)


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
        >>> from skcomplex.core.model import QuantumView
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
        return domain.of(self.element) if isinstance(domain, HilbertFamily) else domain

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
        extra = getattr(self.element, 'operator_namespace', None)
        return {**names, **(extra(key, self) if extra is not None else {})}

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


class QuantumSystem(System):
    """System of quantum elements with a whole-system initial state and a limit on the dimension of its space.

    Args:
        label:
            The label of the system.

        components:
            The quantum elements and groups, and nested systems.

        interactions:
            The interactions, such as `Exchange` subclasses.

        space:
            The space of groups that have none of their own.

        time:
            The time of the system, `ContinuousTime` or `None` to infer it.

        initial:
            The initial state of the whole system, a vector, a density matrix, `'ground'` for the lowest eigenstate of
            the Hamiltonian at the initial parameters, or `None` for the product of the initial states of the elements.

        max_dimension:
            The largest dimension of the Hilbert space the system compiles to.

    Examples:
        >>> from skcomplex.core.model import QuantumSystem
        >>> from skcomplex.models.physics.realizations import Qubit
        >>> QuantumSystem(components=[Qubit('q')], initial='ground').max_dimension
        1048576
    """

    def __init__(
        self,
        label: str = 'system',
        *,
        components: Sequence[Element | Group | System] = (),
        interactions: Sequence[Interaction] = (),
        space: Space | None = None,
        time: Time | None = None,
        initial: object = None,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
    ) -> None:
        super().__init__(label, components=components, interactions=interactions, space=space, time=time)
        self.initial = initial
        self.max_dimension = max_dimension

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the label and the number of children of each kind."""
        return super().__repr__(N_CHAR_MAX).replace('System(', 'QuantumSystem(', 1)
