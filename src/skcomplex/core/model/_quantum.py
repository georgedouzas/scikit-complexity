"""Define quantum subsystems, operators with symbolic coefficients, and the Hamiltonian and dissipator laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt
import qutip
import scipy.sparse as sp

from ..base import ValidationError, VisualSpec, role
from ._components import Element, Group, Interaction
from ._lattices import LatticeSpace
from ._observables import Observable
from ._operators import LocalOperator, Operator, sparse_matrix
from ._spaces import NetworkSpace, Space
from ._systems import System

DEFAULT_MAX_DIMENSION = 2**20
NORM_TOLERANCE = 1e-8
DIGEST_SIZE = 8


def _bonds(space: Space | None, size: int) -> list[tuple[int, int]]:
    """Return the pairs of neighbouring members of a lattice or a network, each pair once."""
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


class QuantumElement(Element):
    """Base class of quantum subsystems, elements whose state is a vector of a Hilbert space.

    Subclasses implement `dimension`, `local_operators`, and `basis_index`. Laws written for an element receive the
    namespace of its local operators and parameters, and return `Hamiltonian` and `Dissipator` laws. A group of quantum
    elements is the tensor product of its members' spaces, in member order.

    Args:
        label:
            The label of the element.

        initial:
            The initial state, a basis label, a basis index, or a vector of the element's Hilbert space.
    """

    visual: ClassVar[VisualSpec] = VisualSpec('expectations')
    display: ClassVar[str | None] = 'z'
    initial: Any = 0

    @classmethod
    def lattice(
        cls,
        label: str,
        space: LatticeSpace,
        /,
        *,
        visual: VisualSpec | None = None,
        **values: Any,  # noqa: ANN401
    ) -> Group:
        """Return a group of subsystems, one member per site of a lattice, drawn by their local expectations.

        Args:
            label:
                The label of the group.

            space:
                The lattice.

            visual:
                The visual of the group, or `None` for the visual of the element type.

            **values:
                Arrays of the lattice shape, or values shared by every member.

        Returns:
            group:
                The group.

        Examples:
            >>> from skcomplex.core.model import LatticeSpace, Spin
            >>> Spin.lattice('chain', LatticeSpace(4)).size
            4
        """
        return super().lattice(label, space, visual=cls.visual if visual is None else visual, **values)

    @property
    def dimension(self) -> int:
        """Dimension of the Hilbert space of the element."""
        raise NotImplementedError

    def local_operators(self) -> dict[str, LocalOperator]:
        """Return the local operators of the element by name.

        Returns:
            operators:
                A mapping from name to local operator.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def basis_index(self, label: object) -> int:
        """Return the index of the basis state of a label.

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
        if isinstance(label, int | np.integer) and not isinstance(label, bool) and 0 <= label < self.dimension:
            return int(label)
        error_msg = (
            f'{type(self).__name__} {self.label!r} has no basis state {label!r}. Use an index from 0 to '
            f'{self.dimension - 1}.'
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
        initial = self.initial
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
        """Return the operators of a subsystem by name, as the namespace of laws and observables holds them.

        Args:
            key:
                The key of the subsystem, such as `chain[3]`.

        Returns:
            names:
                A mapping from the name of every local operator to the operator acting on the subsystem.
        """
        return {name: Operator({((key, local),): 1}) for name, local in self.local_operators().items()}


class Spin(QuantumElement):
    """Spin of quantum number `j`, with the spin components `x`, `y`, `z` and the ladder operators `plus` and `minus`.

    The basis is ordered from `m = j` to `m = -j`, so `'up'` is index 0 and `'down'` is the last index.

    Args:
        label:
            The label of the spin.

        j:
            The spin quantum number, a positive multiple of one half.

        initial:
            `'up'`, `'down'`, a basis index, or a vector.

    Examples:
        >>> from skcomplex.core.model import Spin
        >>> Spin('s', j=1).dimension
        3
    """

    j: float = 0.5
    initial: Any = 'up'

    @property
    def dimension(self) -> int:
        """Dimension `2 j + 1` of the Hilbert space."""
        size = 2 * self.j + 1
        if not float(size).is_integer() or size < 2:  # noqa: PLR2004
            error_msg = f'Spin {self.label!r} has j = {self.j!r}, which is not a positive multiple of one half.'
            raise ValidationError(error_msg)
        return int(size)

    def local_operators(self) -> dict[str, LocalOperator]:
        """Return the spin components and the ladder operators.

        Returns:
            operators:
                The operators `x`, `y`, `z`, `plus`, and `minus`.
        """
        self.dimension  # noqa: B018
        return {
            'x': LocalOperator('x', sparse_matrix(qutip.jmat(self.j, 'x')), hermitian=True),
            'y': LocalOperator('y', sparse_matrix(qutip.jmat(self.j, 'y')), hermitian=True),
            'z': LocalOperator('z', sparse_matrix(qutip.jmat(self.j, 'z')), hermitian=True),
            'plus': LocalOperator('plus', sparse_matrix(qutip.jmat(self.j, '+'))),
            'minus': LocalOperator('minus', sparse_matrix(qutip.jmat(self.j, '-'))),
        }

    def basis_index(self, label: object) -> int:
        """Return the index of `'up'`, `'down'`, or a basis index.

        Args:
            label:
                The label of a basis state, or its index.

        Returns:
            index:
                The index of the basis state.
        """
        named = {'up': 0, 'down': self.dimension - 1}
        return named[label] if isinstance(label, str) and label in named else super().basis_index(label)


class Qubit(QuantumElement):
    """Two-level system with the Pauli matrices `x`, `y`, `z` and the ladder operators `plus` and `minus`.

    The state `'up'`, also `'excited'`, is index 0 with `z = 1`, and `'down'`, also `'ground'`, is index 1.

    Args:
        label:
            The label of the qubit.

        initial:
            `'up'`, `'down'`, `'excited'`, `'ground'`, a basis index, or a vector.

    Examples:
        >>> from skcomplex.core.model import Qubit
        >>> sorted(Qubit('q').local_operators())
        ['minus', 'plus', 'x', 'y', 'z']
    """

    initial: Any = 'up'

    @property
    def dimension(self) -> int:
        """Dimension 2 of the Hilbert space."""
        return 2

    def local_operators(self) -> dict[str, LocalOperator]:
        """Return the Pauli matrices and the ladder operators.

        Returns:
            operators:
                The operators `x`, `y`, `z`, `plus`, and `minus`.
        """
        return {
            'x': LocalOperator('x', sparse_matrix(qutip.sigmax()), hermitian=True),
            'y': LocalOperator('y', sparse_matrix(qutip.sigmay()), hermitian=True),
            'z': LocalOperator('z', sparse_matrix(qutip.sigmaz()), hermitian=True),
            'plus': LocalOperator('plus', sparse_matrix(qutip.sigmap())),
            'minus': LocalOperator('minus', sparse_matrix(qutip.sigmam())),
        }

    def basis_index(self, label: object) -> int:
        """Return the index of `'up'`, `'down'`, `'excited'`, `'ground'`, or a basis index.

        Args:
            label:
                The label of a basis state, or its index.

        Returns:
            index:
                The index of the basis state.
        """
        named = {'up': 0, 'excited': 0, 'down': 1, 'ground': 1}
        return named[label] if isinstance(label, str) and label in named else super().basis_index(label)


class Mode(QuantumElement):
    """Bosonic mode truncated at a number of quanta, with `annihilate`, `create`, and `number`.

    Args:
        label:
            The label of the mode.

        quanta:
            The largest number of quanta, so the dimension is `quanta + 1`.

        initial:
            The initial number of quanta, or a vector.

    Examples:
        >>> from skcomplex.core.model import Mode
        >>> Mode('cavity', quanta=4).dimension
        5
    """

    display: ClassVar[str | None] = 'number'
    quanta: int = 4
    initial: Any = 0

    @property
    def dimension(self) -> int:
        """Dimension `quanta + 1` of the Hilbert space."""
        if isinstance(self.quanta, bool) or not isinstance(self.quanta, int | np.integer) or self.quanta < 1:
            error_msg = f'Mode {self.label!r} has quanta = {self.quanta!r}, which is not a positive integer.'
            raise ValidationError(error_msg)
        return int(self.quanta) + 1

    def local_operators(self) -> dict[str, LocalOperator]:
        """Return the ladder operators and the number operator.

        Returns:
            operators:
                The operators `annihilate`, `create`, and `number`.
        """
        size = self.dimension
        return {
            'annihilate': LocalOperator('annihilate', sparse_matrix(qutip.destroy(size))),
            'create': LocalOperator('create', sparse_matrix(qutip.create(size))),
            'number': LocalOperator('number', sparse_matrix(qutip.num(size)), hermitian=True),
        }


class Site(QuantumElement):
    """Single particle on the sites of a lattice or a network, whose basis state `k` is the particle at site `k`.

    Its namespace holds the Laplacian of the space, `laplacian`, spectral on a periodic lattice and of finite
    differences otherwise, the adjacency of neighbouring sites, `adjacency`, the site indices, `sites`, and the
    functions `projector(k)` and `diagonal(values)` that return the projector on a site and the operator with given
    values on the diagonal.

    Args:
        label:
            The label of the particle.

        space:
            The lattice or the network.

        initial:
            The initial site, or a vector of amplitudes over the sites.

    Examples:
        >>> from skcomplex.core.model import LatticeSpace, Site
        >>> Site('particle', space=LatticeSpace(16)).dimension
        16
    """

    display: ClassVar[str | None] = None
    space: Space | None = None
    initial: Any = 0

    @property
    def dimension(self) -> int:
        """Number of sites of the space."""
        if isinstance(self.space, LatticeSpace):
            return self.space.size
        if isinstance(self.space, NetworkSpace):
            return int(self.space.graph.number_of_nodes())
        error_msg = f'Site {self.label!r} has space {self.space!r}. Give it a LatticeSpace or a NetworkSpace.'
        raise ValidationError(error_msg)

    def local_operators(self) -> dict[str, LocalOperator]:
        """Return the Laplacian and the adjacency of the space.

        Returns:
            operators:
                The operators `laplacian` and `adjacency`.
        """
        size = self.dimension
        rows, columns = zip(*_bonds(self.space, size), strict=True) if size > 1 else ((), ())
        adjacency = sp.coo_array((np.ones(len(rows)), (rows, columns)), shape=(size, size))
        laplacian = None
        if isinstance(self.space, NetworkSpace):
            laplacian = -sparse_matrix(nx.laplacian_matrix(self.space.graph))
        return {
            'laplacian': LocalOperator('laplacian', laplacian, hermitian=True),
            'adjacency': LocalOperator('adjacency', sparse_matrix(adjacency + adjacency.T), hermitian=True),
        }

    def namespace(self, key: str) -> dict[str, Any]:
        """Return the operators of the particle with its site indices and the functions `projector` and `diagonal`.

        Args:
            key:
                The key of the subsystem.

        Returns:
            names:
                A mapping from name to operator or function.
        """
        size = self.dimension

        def projector(site: int) -> Operator:
            values = np.zeros(size)
            values[self.basis_index(site)] = 1.0
            matrix = sp.csr_array(sp.diags_array(values.astype(complex)))
            return Operator({((key, LocalOperator(f'projector({site})', matrix, hermitian=True)),): 1})

        def diagonal(values: npt.ArrayLike) -> Operator:
            array = np.broadcast_to(np.asarray(values, dtype=float), (size,))
            digest = hashlib.blake2b(array.tobytes(), digest_size=DIGEST_SIZE).hexdigest()
            matrix = sp.csr_array(sp.diags_array(array.astype(complex)))
            return Operator({((key, LocalOperator(f'diagonal({digest})', matrix, hermitian=True)),): 1})

        sites = tuple(range(size))
        return {**super().namespace(key), 'projector': projector, 'diagonal': diagonal, 'sites': sites}


class Exchange(Interaction):
    """Base class of interactions between the neighbouring members of a quantum group on a lattice or a network.

    Subclasses return `Hamiltonian` and `Dissipator` laws from `laws`, written once for a bond, where `s.first` and
    `s.second` are the namespaces of its two members. The compiled system adds the laws of every bond of the space of
    the group, each pair of neighbours once.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the quantum group.

    Examples:
        >>> from skcomplex.core.model import Exchange, Hamiltonian
        >>> class Ising(Exchange):
        ...     def laws(self, s):
        ...         return [Hamiltonian(s.first.z * s.second.z)]
        >>> Ising('coupling', target='chain').target
        'chain'
    """

    target: str = role()

    def pairs(self, space: Space | None, size: int) -> list[tuple[int, int]]:
        """Return the pairs of members the laws act on, the neighbours of the space.

        Args:
            space:
                The space of the group.

            size:
                The number of members.

        Returns:
            pairs:
                The member indices of every bond, each bond once.

        Raises:
            ValidationError:
                The space is neither a lattice nor a network.
        """
        return _bonds(space, size)


class QuantumSystem(System):
    """System of quantum elements with a whole-system initial state and a limit on the dimension of its space.

    Args:
        label:
            The label of the system.

        elements:
            The quantum elements and groups.

        interactions:
            The interactions, such as `Exchange` subclasses.

        systems:
            The child systems.

        space:
            The space of groups that have none of their own.

        observables:
            The observables, such as those `expectation` returns.

        initial:
            The initial state of the whole system, a vector, a density matrix, `'ground'` for the lowest eigenstate of
            the Hamiltonian at the initial parameters, or `None` for the product of the initial states of the elements.

        max_dimension:
            The largest dimension of the Hilbert space the system compiles to.

    Examples:
        >>> from skcomplex.core.model import QuantumSystem, Qubit
        >>> QuantumSystem(elements=[Qubit('q')], initial='ground').max_dimension
        1048576
    """

    def __init__(
        self,
        label: str = 'system',
        *,
        elements: Sequence[Element | Group] = (),
        interactions: Sequence[Interaction] = (),
        systems: Sequence[System] = (),
        space: Space | None = None,
        observables: Sequence[Observable] = (),
        initial: object = None,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
    ) -> None:
        super().__init__(
            label,
            elements=elements,
            interactions=interactions,
            systems=systems,
            space=space,
            observables=observables,
        )
        self.initial = initial
        self.max_dimension = max_dimension

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the label and the number of children of each kind."""
        return super().__repr__(N_CHAR_MAX).replace('System(', 'QuantumSystem(', 1)
