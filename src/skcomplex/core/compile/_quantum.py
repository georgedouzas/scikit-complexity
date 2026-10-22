"""Compile quantum systems into problems whose state is a pure state or a density matrix."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import functools
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import qutip
import scipy.sparse as sp
from sage.all__sagemath_symbolics import SR

from ..base import (
    Domain,
    Expression,
    Form,
    HilbertSpaceError,
    IncompatibleLawsError,
    ValidationError,
    VariableKind,
    declarations_of,
    symbol,
    vector,
)
from ..model import (
    Dissipator,
    Generator,
    Group,
    Interaction,
    LatticeSpace,
    Observable,
    Operator,
    QuantumView,
    Symbols,
    System,
    hilbert_state,
    is_quantum,
    neighbour_pairs,
    reduce_values,
    view_of,
)
from ._fields import spectral_multiplier
from ._layout import Entry, Layout
from ._printing import numpy_function
from ._problem import CompiledElement, CompiledInteraction, Problem, Vector
from ._tree import Node, coerce_value, flatten_system

DENSE_LIMIT = 1024
HERMITIAN_TOLERANCE = 1e-10
NORM_TOLERANCE = 1e-8
SAMPLE_TIME = 0.37
REAL_TOLERANCE = 1e-10

type Getter = Callable[[float, Vector, Vector, Mapping[str, Any]], Any]

DEFAULT_MAX_DIMENSION = 2**20


@dataclass(frozen=True)
class _Subsystem:
    """Quantum subsystem with its key, its element, and its index in the tensor product."""

    key: str
    element: QuantumView
    index: int


@dataclass
class _Laws:
    """Hamiltonian and dissipator laws collected from the components, with the description of their owner."""

    hamiltonians: list[tuple[str, Operator]] = field(default_factory=list)
    dissipators: list[tuple[str, Operator, object]] = field(default_factory=list)


def _is_quantum_node(node: Node) -> bool:
    """Check whether a node is a quantum element or a group of quantum elements."""
    return not isinstance(node.component, Interaction) and is_quantum(node.component_type)


def _sum_matrices(matrices: Iterable[sp.csr_array], size: int) -> sp.csr_array:
    """Return the sum of sparse matrices, the zero matrix without any."""
    total = sp.csr_array((size, size), dtype=complex)
    for matrix in matrices:
        total = total + matrix
    return sp.csr_array(total)


def _laplacian_of(space: Any) -> sp.csr_array:  # noqa: ANN401
    """Return the Laplacian of a lattice, spectral when it is periodic and of finite differences otherwise."""
    dimensions, spacings = space.dimensions, space.spacings
    size = math.prod(dimensions)
    if space.boundary == 'periodic':
        multiplier = spectral_multiplier('laplacian', 0, dimensions, spacings).real
        basis = np.eye(size).reshape(size, *dimensions)
        axes = tuple(range(1, len(dimensions) + 1))
        transformed = multiplier * np.fft.rfftn(basis, axes=axes)
        columns = np.fft.irfftn(transformed, s=dimensions, axes=axes).reshape(size, size)
        return sp.csr_array(columns.T.astype(complex))
    matrix = sp.csr_array((size, size), dtype=complex)
    for axis, (length, spacing) in enumerate(zip(dimensions, spacings, strict=True)):
        line = sp.diags_array([np.ones(length - 1), -2 * np.ones(length), np.ones(length - 1)], offsets=[-1, 0, 1])
        identities = [sp.identity(item) for item in dimensions]
        identities[axis] = line / spacing**2
        matrix = matrix + functools.reduce(lambda left, right: sp.kron(left, right, format='csr'), identities)
    return sp.csr_array(matrix.astype(complex))


def _member_value(value: Any, member: int, size: int) -> Any:  # noqa: ANN401
    """Return the value of one member, from a per-member sequence or a value shared by every member."""
    if isinstance(value, str) or value is None or np.ndim(value) == 0:
        return value
    return value[member] if len(value) == size else value


def _members_of(node: Node) -> list[tuple[str, QuantumView]]:
    """Return the key and the element of every member of a quantum element or group."""
    component = node.component
    if not isinstance(component, Group):
        return [(node.path, QuantumView(component))]
    names = [item.name for item in dataclasses.fields(component.element) if item.init and item.name != 'label']
    members = []
    for member in range(component.size):
        values = {name: _member_value(node.value_of(name), member, component.size) for name in names}
        element = component.element(label=f'{component.label}_{member}', **values)
        members.append((f'{node.path}[{member}]', QuantumView(element)))
    return members


def _check_classical(nodes: list[Node]) -> None:
    """Raise an error when a quantum system holds classical elements or quantum elements with states."""
    classical = [
        node.path for node in nodes if not isinstance(node.component, Interaction) and not _is_quantum_node(node)
    ]
    if classical:
        error_msg = (
            f'The system mixes quantum elements with the classical elements {", ".join(classical)}, which no single '
            'form runs. Model the quantum and the classical parts as separate systems.'
        )
        raise IncompatibleLawsError(error_msg)
    for node in nodes:
        states = declarations_of(node.component_type, VariableKind.STATE) if _is_quantum_node(node) else {}
        states = {name: item for name, item in states.items() if name != hilbert_state(node.component_type)}
        if states:
            error_msg = (
                f'Quantum element {node.path} declares the states {", ".join(states)} besides its state in a Hilbert '
                'space, which is its only state, so declare parameters instead.'
            )
            raise ValidationError(error_msg)


def _check_dimension(nodes: list[Node], dimensions: list[int], limit: int) -> None:
    """Raise an error when the dimension of the Hilbert space exceeds the limit of the solver."""
    total = math.prod(dimensions)
    if total <= limit:
        return
    parts = []
    for node in nodes:
        if _is_quantum_node(node):
            members = _members_of(node)
            parts.append(f'{node.path} of dimension {math.prod(element.dimension for _, element in members)}')
    error_msg = (
        f'The Hilbert space of {", ".join(parts)} has dimension {total}, above the limit max_dimension = {limit}. Use '
        'fewer or smaller subsystems, or raise max_dimension of the quantum solver.'
    )
    raise HilbertSpaceError(error_msg)


@dataclass(frozen=True)
class QuantumTerm:
    """Sparse matrix on the whole Hilbert space with a symbolic coefficient and its lowered function.

    Args:
        matrix:
            The sparse matrix.

        coefficient:
            The symbolic coefficient, 1 for a constant matrix.

        function:
            The function `c(t, p)` of the coefficient, or `None` for a constant matrix.
    """

    matrix: sp.csr_array
    coefficient: Expression
    function: Callable[[float, Vector], complex] | None = None

    @property
    def time_dependent(self) -> bool:
        """Whether the coefficient depends on time."""
        return bool(self.coefficient.has(symbol('t')))

    def value(self, t: float, p: Vector) -> complex:
        """Return the coefficient at a time and a parameter vector.

        Args:
            t:
                The time.

            p:
                The parameter vector.

        Returns:
            value:
                The coefficient.
        """
        return 1.0 if self.function is None else self.function(t, p)


@dataclass(frozen=True)
class _Lowering:
    """Lowering of operators to sparse matrices on the whole space and of coefficients to functions."""

    time: Expression
    subsystems: Mapping[str, _Subsystem]
    dimensions: tuple[int, ...]
    arguments: Mapping[Expression, int]
    spaces: Mapping[str, Any]

    def _local(self, subsystem: _Subsystem, local: Any) -> sp.csr_array:  # noqa: ANN401
        """Return the matrix of a local operator, building the Laplacian of a space when it has none."""
        if local.matrix is not None:
            return sp.csr_array(local.matrix)
        space = self.spaces.get(subsystem.key)
        if not isinstance(space, LatticeSpace):
            error_msg = f'The local operator {local.name} of {subsystem.key} needs a lattice to be built.'
            raise ValidationError(error_msg)
        laplacian = _laplacian_of(space)
        return laplacian if not local.name.endswith('.dag') else sp.csr_array(laplacian.conj().T)

    def matrix(self, key: tuple[Any, ...], owner: str) -> sp.csr_array:
        """Return the matrix of a product of local operators on the whole space.

        Args:
            key:
                The factors of the product, each the key of a subsystem and a local operator.

            owner:
                The description of the law or observable, for error messages.

        Returns:
            matrix:
                The sparse matrix.

        Raises:
            ValidationError:
                A factor acts on a subsystem that is not part of the system.
        """
        products: dict[int, sp.csr_array] = {}
        for name, local in key:
            if name not in self.subsystems:
                error_msg = f'The operator of {owner} acts on {name}, which is not a quantum subsystem of the system.'
                raise ValidationError(error_msg)
            subsystem = self.subsystems[name]
            matrix = self._local(subsystem, local)
            previous = products.get(subsystem.index)
            products[subsystem.index] = matrix if previous is None else sp.csr_array(previous @ matrix)
        blocks: list[Any] = []
        pending = 1
        for index, size in enumerate(self.dimensions):
            if index not in products:
                pending *= size
                continue
            if pending > 1:
                blocks.append(sp.identity(pending, dtype=complex, format='csr'))
            blocks.append(products[index])
            pending = 1
        if pending > 1 or not blocks:
            blocks.append(sp.identity(pending, dtype=complex, format='csr'))
        return sp.csr_array(functools.reduce(lambda left, right: sp.kron(left, right, format='csr'), blocks))

    def function(self, coefficient: Expression, owner: str) -> Callable[[float, Vector], complex]:
        """Return the function of a coefficient after checking that it reads parameters and time only.

        Args:
            coefficient:
                The symbolic coefficient.

            owner:
                The description of the law or observable, for error messages.

        Returns:
            function:
                The function `c(t, p)`.

        Raises:
            ValidationError:
                The coefficient uses a symbol that is neither a parameter nor time.
        """
        variables = sorted(set(coefficient.variables()) - {self.time}, key=str)
        unknown = [item for item in variables if item not in self.arguments]
        if unknown:
            error_msg = (
                f'The operator of {owner} has a coefficient that uses {", ".join(map(str, unknown))}, which is not a '
                'parameter of the quantum system. Coefficients may read parameters and time only.'
            )
            raise ValidationError(error_msg)
        lowered = numpy_function(self.time, variables, [coefficient])
        indices = np.asarray([self.arguments[item] for item in variables], dtype=np.intp)

        def evaluate(t: float, p: Vector) -> complex:
            return complex(lowered(t, *p[indices])[0])

        return evaluate

    def terms(self, operator: Operator, owner: str) -> tuple[QuantumTerm, ...]:
        """Return the terms of an operator, with constant terms folded and terms of equal coefficients summed.

        Args:
            operator:
                The operator.

            owner:
                The description of the law or observable, for error messages.

        Returns:
            terms:
                The constant term first, if any, then one term per distinct coefficient.
        """
        size = math.prod(self.dimensions)
        constant: list[sp.csr_array] = []
        grouped: dict[Expression, list[sp.csr_array]] = {}
        for key, value in operator.terms.items():
            coefficient = SR(value)
            matrix = self.matrix(key, owner)
            if coefficient.variables():
                grouped.setdefault(coefficient, []).append(matrix)
            else:
                constant.append(complex(coefficient) * matrix)
        terms = [QuantumTerm(_sum_matrices(constant, size), SR(1))] if constant else []
        for coefficient, matrices in grouped.items():
            terms.append(QuantumTerm(_sum_matrices(matrices, size), coefficient, self.function(coefficient, owner)))
        return tuple(terms)


def _matrix_at(terms: Iterable[QuantumTerm], t: float, p: Vector, size: int) -> sp.csr_array:
    """Return the sum of the terms at a time and a parameter vector."""
    return _sum_matrices((term.value(t, p) * term.matrix for term in terms), size)


@dataclass(frozen=True)
class QuantumProblem(Problem):
    """Problem of the quantum form, whose state is a pure state or a density matrix stored as complex pairs.

    Args:
        dimensions:
            The Hilbert space dimensions of the subsystems, in tensor product order.

        subsystems:
            The keys of the subsystems, such as `chain[3]`, in tensor product order.

        hamiltonian:
            The terms of the Hamiltonian.

        dissipators:
            The terms of every collapse operator, each with the square root of its rate in the coefficients.

        pure:
            Whether the state is a pure state vector rather than a density matrix.

        namespaces:
            The namespace of every system, by its path, which observables receive.

        members:
            The namespaces of the members of every quantum element or group, by path.

        lowering:
            The lowering of operators to matrices and functions.

        hamiltonian_operator:
            The sum of the Hamiltonian laws.

        cache:
            The matrices of local operators computed so far.
    """

    dimensions: tuple[int, ...] = ()
    subsystems: tuple[str, ...] = ()
    hamiltonian: tuple[QuantumTerm, ...] = field(default=(), repr=False)
    dissipators: tuple[tuple[QuantumTerm, ...], ...] = field(default=(), repr=False)
    pure: bool = True
    namespaces: Mapping[str, Symbols] = field(default_factory=dict, repr=False)
    members: Mapping[str, tuple[Symbols, ...]] = field(default_factory=dict, repr=False)
    lowering: Any = field(default=None, repr=False)
    hamiltonian_operator: Operator = field(default_factory=lambda: Operator({}), repr=False)
    cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def dimension(self) -> int:
        """Dimension of the Hilbert space."""
        return math.prod(self.dimensions)

    @property
    def state_name(self) -> str:
        """Dotted name of the state in the state vector."""
        return self.states.names[0]

    def state_of(self, x: Vector) -> npt.NDArray[np.complex128]:
        """Return the complex state, a vector or a density matrix, from a state vector.

        Args:
            x:
                The state vector.

        Returns:
            state:
                The pure state of shape `(dimension,)` or the density matrix of shape `(dimension, dimension)`.
        """
        return np.asarray(self.states.read(x, self.state_name))

    def encode(self, state: npt.ArrayLike) -> Vector:
        """Return a complex state as a state vector.

        Args:
            state:
                The pure state or the density matrix.

        Returns:
            x:
                The state vector.
        """
        return self.states.entry(self.state_name).encode(state)

    def hamiltonian_at(self, t: float, p: Vector) -> sp.csr_array:
        """Return the Hamiltonian matrix at a time and a parameter vector.

        Args:
            t:
                The time.

            p:
                The parameter vector.

        Returns:
            matrix:
                The sparse Hamiltonian.
        """
        return _matrix_at(self.hamiltonian, t, p, self.dimension)

    def operator_terms(self, operator: Operator, owner: str = 'an observable') -> tuple[QuantumTerm, ...]:
        """Return the terms of an operator of the system.

        Args:
            operator:
                The operator, built from the namespaces of the system.

            owner:
                The description of the operator, for error messages.

        Returns:
            terms:
                The terms of the operator.
        """
        return self.lowering.terms(operator, owner)

    def expect(self, terms: Iterable[QuantumTerm], t: float, x: Vector, p: Vector) -> float | complex:
        """Return the expectation value of an operator at a state.

        Args:
            terms:
                The terms of the operator.

            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            value:
                `<psi|O|psi>` or `Tr(rho O)`, a float when its imaginary part vanishes.
        """
        state = self.state_of(x)
        total = 0j
        for term in terms:
            value = np.vdot(state, term.matrix @ state) if self.pure else term.matrix.multiply(state.T).sum()
            total += term.value(t, p) * complex(value)
        if abs(total.imag) <= REAL_TOLERANCE * (1 + abs(total.real)):
            return float(total.real)
        return total

    def lowest_states(
        self,
        p: Vector,
        k: int = 1,
        basis: npt.NDArray[np.intp] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
        """Return the lowest eigenvalues and eigenstates of the Hamiltonian at time 0, through QuTiP.

        Args:
            p:
                The parameter vector.

            k:
                The number of eigenvalues.

            basis:
                The indices of the basis states that span a sector to diagonalise in, or `None` for the whole space.

        Returns:
            spectrum:
                The eigenvalues in increasing order and the eigenstates as columns over the whole space.
        """
        matrix = self.hamiltonian_at(0.0, p)
        indices = np.arange(self.dimension) if basis is None else np.asarray(basis, dtype=np.intp)
        matrix = sp.csr_array(matrix[indices][:, indices])
        count = min(k, indices.size)
        sparse = indices.size > DENSE_LIMIT and count < indices.size - 1
        energies, kets = qutip.Qobj(matrix).eigenstates(sparse=sparse, eigvals=count)
        states = np.zeros((self.dimension, count), dtype=complex)
        for column, ket in enumerate(kets):
            states[indices, column] = ket.full().ravel()
        return np.real(np.asarray(energies)), states

    def _local_matrices(self, path: str, name: str) -> list[sp.csr_array]:
        """Return the matrices on the whole space of one local operator of every member of an element or group."""
        cached = self.cache.get(f'{path}:{name}')
        if cached is None:
            cached = [self.lowering.terms(getattr(member, name), path)[0].matrix for member in self.members[path]]
            self.cache[f'{path}:{name}'] = cached
        return cached

    def _probabilities(self, x: Vector) -> npt.NDArray[np.float64]:
        """Return the probability of every basis state of the whole space."""
        state = self.state_of(x)
        return np.abs(state) ** 2 if self.pure else np.real(np.diagonal(state))

    def local_values(self, path: str, x: Vector) -> dict[str, Any]:
        """Return the local expectations of a quantum element or group at a state, which visuals draw.

        A group gives the expectation of the displayed local operator of every member. A single element without a
        displayed operator, such as a particle on sites, gives the probability of every basis state, and one with it
        gives the expectation of each of its Hermitian local operators.

        Args:
            path:
                The path of the element or group.

            x:
                The state vector.

        Returns:
            values:
                A mapping from the name of a local operator, or `'probability'`, to an array or a number.
        """
        element = next(item for item in self.elements if item.path == path)
        display = view_of(element.element).display
        if element.members is None and display is None:
            index = self.subsystems.index(path)
            shaped = self._probabilities(x).reshape(self.dimensions)
            others = tuple(axis for axis in range(len(self.dimensions)) if axis != index)
            return {'probability': shaped.sum(axis=others)}
        if element.members is not None and display is not None:
            matrices = self._local_matrices(path, display)
            values = [self.expect([QuantumTerm(matrix, SR(1))], 0.0, x, np.zeros(0)) for matrix in matrices]
            return {display: np.real(np.asarray(values))}
        names = [name for name, local in view_of(element.element).local_operators().items() if local.hermitian]
        return {
            name: float(np.real(self.expect([QuantumTerm(matrix, SR(1))], 0.0, x, np.zeros(0))))
            for name in names
            for matrix in self._local_matrices(path, name)
        }


def _parameters(nodes: list[Node]) -> tuple[list[Entry], list[Vector], dict[str, Any]]:
    """Return the parameter entries, the initial values, and the symbols of every component, per member for groups."""
    entries: list[Entry] = []
    values: list[Vector] = []
    symbols: dict[str, Any] = {}
    for node in nodes:
        for variable, declaration in declarations_of(node.component_type, VariableKind.PARAMETER).items():
            name = f'{node.path}.{variable}'
            if declaration.vector:
                error_msg = f'{name} is a vector parameter, which quantum systems do not hold. Use scalar parameters.'
                raise ValidationError(error_msg)
            members = node.members
            shape = () if members is None else (members,)
            values.append(coerce_value(name, node.value_of(variable), (), declaration.domain, members))
            entries.append(Entry(name, sum(item.size for item in entries), shape, declaration.domain, members))
            base = f'{variable}_{"".join(char if char.isalnum() else "_" for char in node.path)}'
            if members is None:
                symbols[name] = symbol(base, declaration.domain)
            else:
                symbols[name] = vector([symbol(f'{base}_{member}', declaration.domain) for member in range(members)])
    return entries, values, symbols


def _parameter_symbols(node: Node, symbols: Mapping[str, Any], member: int | None) -> dict[str, Any]:
    """Return the parameter symbols of a component, of one member for a group."""
    found = {}
    for variable in declarations_of(node.component_type, VariableKind.PARAMETER):
        value = symbols[f'{node.path}.{variable}']
        found[variable] = value if member is None else value[member]
    return found


def _role_namespace(
    node: Node,
    label: str,
    members: Mapping[str, tuple[Symbols, ...]],
    groups: set[str],
) -> Symbols | tuple[Symbols, ...]:
    """Return the namespace an interaction role binds to, one member's for an element and every member's for a group."""
    path = f'{node.scope}{label}'
    if path not in members:
        error_msg = f'Interaction {node.path} binds a role to {path}, which is not a quantum element or group.'
        raise ValidationError(error_msg)
    return members[path] if path in groups else members[path][0]


def _collect_laws(laws: _Laws, owner: str, items: Iterable[Any]) -> None:
    """Add the Hamiltonian and dissipator laws of a component, rejecting every other law."""
    for law in items:
        if isinstance(law, Generator):
            laws.hamiltonians.append((owner, law.operator))
        elif isinstance(law, Dissipator):
            laws.dissipators.append((owner, law.operator, law.rate))
        else:
            error_msg = (
                f'{owner} returns the law {type(law).__name__}, which a quantum system cannot hold. Quantum systems '
                'take Hamiltonian and Dissipator laws only.'
            )
            raise IncompatibleLawsError(error_msg)


def _interaction_laws(
    node: Node,
    symbols: Mapping[str, Any],
    members: Mapping[str, tuple[Symbols, ...]],
    groups: dict[str, Node],
    laws: _Laws,
) -> None:
    """Add the laws of an interaction, once per bond of its group for an exchange interaction."""
    interaction = node.component
    assert isinstance(interaction, Interaction)  # nosec B101: the caller passes interactions only
    parameters = {**_parameter_symbols(node, symbols, None), 't': symbol('t')}
    if type(interaction).over == 'neighbours':
        roles = list(declarations_of(interaction, VariableKind.ROLE))
        if len(roles) != 1:
            error_msg = (
                f'Interaction {node.path} over neighbours declares the roles {roles}. Declare one role, bound to the '
                'group whose neighbours it couples.'
            )
            raise ValidationError(error_msg)
        path = f'{node.scope}{getattr(interaction, roles[0])}'
        if path not in groups:
            error_msg = f'Interaction {node.path} over neighbours targets {path}, which is not a quantum group.'
            raise ValidationError(error_msg)
        group = groups[path]
        for first, second in neighbour_pairs(group.space, group.members or 1):
            namespace = Symbols(first=members[path][first], second=members[path][second], **parameters)
            _collect_laws(laws, f'{node.path} on the bond ({first}, {second})', interaction.laws(namespace))
        return
    roles = {
        name: _role_namespace(node, getattr(interaction, name), members, set(groups))
        for name in declarations_of(interaction, VariableKind.ROLE)
    }
    _collect_laws(laws, node.path, interaction.laws(Symbols(**roles, **parameters)))


def _check_hermitian(laws: _Laws, lowering: _Lowering, p: Vector) -> None:
    """Raise an error when the Hamiltonian is not Hermitian at sampled parameter values."""
    size = math.prod(lowering.dimensions)
    rng = np.random.default_rng(0)
    sample = p * (1 + 0.1 * rng.random(p.size)) + 0.01 * rng.random(p.size)

    def asymmetry(terms: Iterable[QuantumTerm]) -> float:
        matrix = _matrix_at(terms, SAMPLE_TIME, sample, size)
        difference = matrix - matrix.conj().T
        return float(abs(difference).max()) if difference.nnz else 0.0

    total = [term for owner, operator in laws.hamiltonians for term in lowering.terms(operator, owner)]
    if asymmetry(total) <= HERMITIAN_TOLERANCE:
        return
    owners = [owner for owner, operator in laws.hamiltonians if asymmetry(lowering.terms(operator, owner))]
    error_msg = (
        f'The Hamiltonian of {", ".join(owners)} is not Hermitian. Add the adjoint of every non-Hermitian '
        'term, such as `s.minus` for `s.plus`, to the law.'
    )
    raise ValidationError(error_msg)


def _dissipator_terms(laws: _Laws, lowering: _Lowering, p: Vector) -> tuple[tuple[QuantumTerm, ...], ...]:
    """Return the terms of every collapse operator scaled by the square root of its rate."""
    collapses = []
    for owner, operator, rate in laws.dissipators:
        value = SR(rate)
        initial = lowering.function(value, owner)(0.0, p) if value.variables() else complex(value)
        if initial.real < 0 or abs(initial.imag) > REAL_TOLERANCE:
            error_msg = f'The dissipator of {owner} has the rate {initial}, which is not a non-negative number.'
            raise ValidationError(error_msg)
        collapses.append(lowering.terms(operator * value.sqrt(), owner))
    return tuple(collapses)


def _system_initial(
    system: System,
    initial: object,
    product: npt.NDArray[np.complex128],
    ground: Callable[[], npt.NDArray[np.complex128]],
) -> npt.NDArray[np.complex128]:
    """Return the initial state of the whole system, the product of the elements' states unless one is given."""
    if initial is None:
        return product
    if isinstance(initial, str):
        if initial != 'ground':
            error_msg = f"System {system.label!r} has initial = {initial!r}. Use 'ground', a vector, or a matrix."
            raise ValidationError(error_msg)
        return ground()
    state = np.asarray(initial, dtype=complex)
    size = product.size
    if state.shape not in {(size,), (size, size)}:
        error_msg = (
            f'The initial state of {system.label!r} has shape {state.shape}, but its space has dimension {size}.'
        )
        raise ValidationError(error_msg)
    norm = float(np.linalg.norm(state)) if state.ndim == 1 else float(np.real(np.trace(state)))
    if abs(norm - 1) > NORM_TOLERANCE:
        quantity = 'norm' if state.ndim == 1 else 'trace'
        error_msg = (
            f'The initial state of system {system.label!r} has {quantity} {norm:.6g}, but states have {quantity} 1. '
            'Normalise it.'
        )
        raise ValidationError(error_msg)
    if state.ndim == 2 and not np.allclose(state, state.conj().T, atol=NORM_TOLERANCE):  # noqa: PLR2004
        error_msg = f'The initial density matrix of system {system.label!r} is not Hermitian.'
        raise ValidationError(error_msg)
    return state


def _namespaces(system: System, members: Mapping[str, tuple[Symbols, ...]], groups: set[str]) -> dict[str, Symbols]:
    """Return the namespace of every system of the tree by path, with its quantum elements and child systems."""
    namespaces: dict[str, Symbols] = {}

    def build(current: System, scope: str) -> Symbols:
        names: dict[str, Any] = {'t': symbol('t')}
        for element in current.elements:
            path = f'{scope}{element.label}'
            names[element.label] = members[path] if path in groups else members[path][0]
        for child in current.systems:
            names[child.label] = build(child, f'{scope}{child.label}/')
        namespace = Symbols(**names)
        namespaces[scope.removesuffix('/')] = namespace
        return namespace

    build(system, '')
    return namespaces


def _subsystems_of(quantum: list[Node]) -> tuple[dict[str, _Subsystem], dict[str, Any]]:
    """Return the subsystems of the quantum elements and groups in tensor product order, with their spaces."""
    subsystems: dict[str, _Subsystem] = {}
    spaces: dict[str, Any] = {}
    for node in quantum:
        for key, element in _members_of(node):
            subsystems[key] = _Subsystem(key, element, len(subsystems))
            spaces[key] = getattr(element.element, 'space', None) or node.space
    return subsystems, spaces


def _arguments_of(entries: list[Entry], symbols: Mapping[str, Any]) -> dict[Expression, int]:
    """Return the index in the parameter vector of every parameter symbol."""
    arguments = {}
    for entry in entries:
        components = symbols[entry.name]
        for offset, item in enumerate(components if entry.members is not None else [components]):
            arguments[item] = entry.start + offset
    return arguments


def _element_laws(
    quantum: list[Node],
    symbols: Mapping[str, Any],
    time: Expression,
    laws: _Laws,
) -> dict[str, tuple[Symbols, ...]]:
    """Add the laws of every quantum element and member, and return the member namespaces by path."""
    members: dict[str, tuple[Symbols, ...]] = {}
    for node in quantum:
        namespaces = []
        for member, (key, element) in enumerate(_members_of(node)):
            index = member if node.members is not None else None
            namespace = Symbols(**element.namespace(key), **_parameter_symbols(node, symbols, index), t=time)
            namespaces.append(namespace)
            _collect_laws(laws, key, element.laws(namespace))
        members[node.path] = tuple(namespaces)
    return members


def _stored_state(  # noqa: PLR0917
    system: System,
    initial: object,
    subsystems: Mapping[str, _Subsystem],
    hamiltonian: tuple[QuantumTerm, ...],
    p: Vector,
    is_open: bool,
) -> npt.NDArray[np.complex128]:
    """Return the initial state as stored, a vector for a closed pure state and a density matrix otherwise."""
    product = np.ones(1, dtype=complex)
    for item in subsystems.values():
        product = np.kron(product, item.element.initial_vector())
    size = product.size

    def ground() -> npt.NDArray[np.complex128]:
        matrix = _matrix_at(hamiltonian, 0.0, p, size)
        return np.asarray(qutip.Qobj(matrix).groundstate(sparse=size > DENSE_LIMIT)[1].full().ravel())

    state = _system_initial(system, initial, product, ground)
    return np.outer(state, state.conj()) if is_open and state.ndim == 1 else state


def _compiled_element(node: Node) -> CompiledElement:
    """Return the compiled element of a quantum element or group, with no states in the state vector of its own."""
    component = node.component
    visual = component.visual if isinstance(component, Group) and component.visual else node.component_type.visual
    parameters = {name: f'{node.path}.{name}' for name in declarations_of(node.component_type, VariableKind.PARAMETER)}
    return CompiledElement(node.path, component, {}, parameters, visual, node.members)


def _compiled_interaction(node: Node) -> CompiledInteraction:
    """Return the compiled interaction of a quantum system with the paths its roles bind to."""
    component = node.component
    roles = {name: f'{node.scope}{getattr(component, name)}' for name in declarations_of(component, VariableKind.ROLE)}
    return CompiledInteraction(node.path, component, roles)


def contains_quantum(system: System) -> bool:
    """Check whether a system tree holds a quantum element or group.

    Args:
        system:
            The root system.

    Returns:
        found:
            Whether any element of the tree is quantum.
    """
    for element in system.elements:
        kind = element.element if isinstance(element, Group) else type(element)
        if is_quantum(kind):
            return True
    return any(contains_quantum(child) for child in system.systems)


def compile_quantum(
    system: System,
    *,
    initial: object = None,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> QuantumProblem:
    """Compile a system of quantum elements into a problem of the quantum form.

    Args:
        system:
            The system.

        initial:
            The initial state of the whole system, a vector, a density matrix, `'ground'` for the lowest eigenstate of
            the Hamiltonian at the initial parameters, or `None` for the product of the initial states of the elements.

        max_dimension:
            The largest dimension of the Hilbert space, the limit of the quantum solver.

    Returns:
        problem:
            The quantum problem.

    Raises:
        IncompatibleLawsError:
            The system holds classical elements or laws other than Hamiltonians and dissipators.

        HilbertSpaceError:
            The dimension of the Hilbert space exceeds the limit.

        ValidationError:
            A coefficient reads a symbol other than parameters and time, the Hamiltonian is not Hermitian, a rate is
            negative, or an initial state is not normalised.
    """
    time = symbol('t')
    nodes = flatten_system(system)
    _check_classical(nodes)
    quantum = [node for node in nodes if _is_quantum_node(node)]
    subsystems, spaces = _subsystems_of(quantum)
    dimensions = tuple(item.element.dimension for item in subsystems.values())
    _check_dimension(nodes, list(dimensions), max_dimension)
    entries, values, symbols = _parameters(nodes)
    p = np.concatenate(values) if values else np.zeros(0)
    lowering = _Lowering(time, subsystems, dimensions, _arguments_of(entries, symbols), spaces)
    groups = {node.path: node for node in quantum if isinstance(node.component, Group)}
    laws = _Laws()
    members = _element_laws(quantum, symbols, time, laws)
    for node in nodes:
        if isinstance(node.component, Interaction):
            _interaction_laws(node, symbols, members, groups, laws)
    _check_hermitian(laws, lowering, p)
    total = functools.reduce(Operator.__add__, [item for _, item in laws.hamiltonians], Operator({}))
    hamiltonian = lowering.terms(total, 'the Hamiltonian')
    dissipators = _dissipator_terms(laws, lowering, p)
    state = _stored_state(system, initial, subsystems, hamiltonian, p, bool(dissipators))
    pure = state.ndim == 1
    state_entry = Entry(f'{system.label}.state', 0, state.shape, Domain.COMPLEX, None, is_complex=True)
    elements = tuple(_compiled_element(node) for node in quantum)
    interactions = tuple(_compiled_interaction(node) for node in nodes if isinstance(node.component, Interaction))
    return QuantumProblem(
        form=Form.QUANTUM,
        states=Layout([state_entry]),
        parameters=Layout(entries),
        initial_state=state_entry.encode(state),
        initial_parameters=p,
        time=time,
        symbols=symbols,
        equations={},
        equation_symbols={},
        has_callables=False,
        has_group_sums=False,
        elements=elements,
        interactions=interactions,
        system=system,
        dimensions=dimensions,
        subsystems=tuple(subsystems),
        hamiltonian=hamiltonian,
        dissipators=dissipators,
        pure=pure,
        namespaces=_namespaces(system, members, set(groups)),
        members=members,
        lowering=lowering,
        hamiltonian_operator=total,
    )


def _member_getter(problem: QuantumProblem, path: str, observable: Observable, group: str) -> Getter:
    """Return the getter of a quantum observable evaluated member by member and reduced."""
    if group not in problem.members:
        error_msg = f'Observable {path} reduces {group}, which is not a quantum group of the system.'
        raise ValidationError(error_msg)
    expression = observable.expression
    if not callable(expression):
        error_msg = f'Observable {path} of a quantum group needs a function of a member namespace.'
        raise ValidationError(error_msg)
    terms = [problem.operator_terms(_operator_of(path, expression(member)), path) for member in problem.members[group]]

    def get(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
        del done
        values = np.asarray([problem.expect(item, t, x, p) for item in terms])
        return reduce_values(values, observable.reduce)

    return get


def _operator_of(path: str, value: object) -> Operator:
    """Return the operator an observable expression gives, after checking that it is one."""
    if not isinstance(value, Operator):
        error_msg = f'Observable {path} of a quantum system gives {value!r}, but it must give an operator.'
        raise ValidationError(error_msg)
    return value


def _expectation_getter(problem: QuantumProblem, terms: tuple[QuantumTerm, ...]) -> Getter:
    """Return the getter of the expectation value of an operator."""

    def get(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
        del done
        return problem.expect(terms, t, x, p)

    return get


def _purity_getter(problem: QuantumProblem) -> Getter:
    """Return the getter of the purity `Tr(rho^2)` of the state."""

    def get(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
        del t, p, done
        state = problem.state_of(x)
        return 1.0 if problem.pure else float(np.sum(np.abs(state) ** 2))

    return get


def quantum_getters(problem: QuantumProblem, observables: Iterable[tuple[str, Observable]]) -> dict[str, Getter]:
    """Return the getters of the observables of a quantum problem, with its energy, and its purity when it is open.

    Args:
        problem:
            The quantum problem.

        observables:
            The path of the declaring system and the observable, for every observable of the system tree.

    Returns:
        getters:
            A mapping from observable path to getter.

    Raises:
        ValidationError:
            Two observables share a path, or an observable does not give an operator.
    """
    getters: dict[str, Getter] = {}
    for scope, observable in observables:
        path = f'{scope}/{observable.label}' if scope else observable.label
        if path in getters:
            error_msg = f'Two observables have the path {path}. Give observables of one system distinct labels.'
            raise ValidationError(error_msg)
        if observable.group is not None:
            group = f'{scope}/{observable.group}' if scope else observable.group
            getters[path] = _member_getter(problem, path, observable, group)
            continue
        expression = observable.expression
        value = expression(problem.namespaces[scope]) if callable(expression) else expression
        getters[path] = _expectation_getter(problem, problem.operator_terms(_operator_of(path, value), path))
    getters.setdefault('energy', _expectation_getter(problem, problem.hamiltonian))
    if not problem.pure:
        getters.setdefault('purity', _purity_getter(problem))
    return getters
