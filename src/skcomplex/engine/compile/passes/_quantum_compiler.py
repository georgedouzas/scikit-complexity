"""Pass that compiles a system holding quantum elements into a problem of the quantum form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
import operator
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import qutip
import scipy.sparse as sp
from sage.all__sagemath_symbolics import CC, SR, ZZ, Expression
from sage.symbolic.operators import add_vararg, mul_vararg

from ....core.elements import GroupElement
from ....core.errors import IncompatibleLawsError, SkcomplexError, ValidationError
from ....core.interactions import BaseInteraction, NeighborInteraction
from ....core.laws import DissipatorLaw, GeneratorLaw
from ....core.laws.utils import dagger, has_operator, operator_product
from ....core.laws.utils._quantum import LOCAL_OPERATOR
from ....core.specs import HilbertSpec, ObservableSpec
from ....core.system import System
from ....core.types import Vector
from ...backends import BaseBackend
from ...problems import (
    BaseProblem,
    CompiledElement,
    CompiledInteraction,
    Entry,
    ExpressionPrinter,
    Form,
    Layout,
    VariableKind,
    declarations_of,
    symbol,
    vector,
)
from ...problems._local_operator import LocalOperator
from ...problems._operator import Operator
from ...problems._quantum_kernel import QuantumKernel
from ...problems._quantum_problem import REAL_TOLERANCE, QuantumProblem
from ...problems._quantum_view import QuantumView
from .._base import BaseCompiler
from .._declarations import Declarations
from .._issue import Issue
from .._tree import Node
from ._flatten import FlattenPass
from ._hilbert_space_error import HilbertSpaceError

DENSE_LIMIT = 1024


_HERMITIAN_TOLERANCE = 1e-10


NORM_TOLERANCE = 1e-8


_SAMPLE_TIME = 0.37


DEFAULT_MAX_DIMENSION = 2**20


@dataclass
class _Laws:
    """Hamiltonian and dissipator laws collected from the components, with the description of their owner."""

    hamiltonians: list[tuple[str, Expression]] = field(default_factory=list)
    dissipators: list[tuple[str, Expression, object]] = field(default_factory=list)


@dataclass(frozen=True)
class _Subsystem:
    """Quantum subsystem with its key, its element, its index in the tensor product, its state symbol, and operators."""

    key: str
    element: QuantumView
    index: int
    symbol: Expression
    operators: Mapping[str, LocalOperator]


@dataclass(frozen=True)
class _Lowering:
    """Lowering of operators to sparse matrices on the whole space and of coefficients to functions."""

    time: Expression
    subsystems: Mapping[str, _Subsystem]
    dimensions: tuple[int, ...]
    arguments: Mapping[Expression, int]

    def operator(self, expression: object, owner: str) -> Operator:
        """Return the operator of a Sage expression of local operators of the subsystems.

        Args:
            expression:
                The expression, built with `local_operator`, `operator_product`, and `dagger`.

            owner:
                The description of the law or observable, for error messages.

        Returns:
            operator:
                The operator.

        Raises:
            ValidationError:
                The expression is not an operator of the subsystems.
        """
        subsystems = {item.symbol: (item.key, item.operators) for item in self.subsystems.values()}
        try:
            return self._operator_of(expression, subsystems)
        except ValidationError as error:
            error_msg = f'The operator of {owner} is invalid: {error}'
            raise ValidationError(error_msg) from error

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
            matrix = sp.csr_array(local.matrix)
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
        lowered = ExpressionPrinter(time=self.time, compiled=True).check().prepare(variables).run([coefficient])
        indices = np.asarray([self.arguments[item] for item in variables], dtype=np.intp)

        def evaluate(t: float, p: Vector) -> complex:
            return complex(lowered(t, *p[indices])[0])

        return evaluate

    def terms(self, expression: object, owner: str) -> tuple[QuantumKernel, ...]:
        """Return the terms of an operator, with constant terms folded and terms of equal coefficients summed.

        Args:
            expression:
                The Sage expression of the operator, built with `local_operator`, `operator_product`, and `dagger`.

            owner:
                The description of the law or observable, for error messages.

        Returns:
            terms:
                The constant term first, if any, then one term per distinct coefficient.
        """
        size = math.prod(self.dimensions)
        constant: list[sp.csr_array] = []
        grouped: dict[Expression, list[sp.csr_array]] = {}
        for key, value in self.operator(expression, owner).terms.items():
            coefficient = SR(value)
            matrix = self.matrix(key, owner)
            if coefficient.variables():
                grouped.setdefault(coefficient, []).append(matrix)
            else:
                constant.append(complex(coefficient) * matrix)
        terms = [QuantumKernel(QuantumProblem._sum_matrices(constant, size), SR(1)).check()] if constant else []
        for coefficient, matrices in grouped.items():
            terms.append(
                QuantumKernel(
                    QuantumProblem._sum_matrices(matrices, size), coefficient, self.function(coefficient, owner),
                ).check(),
            )
        return tuple(terms)

    @staticmethod
    def _acted_on(operator: Operator) -> set[str]:
        """Return the keys of the subsystems any term of an operator acts on."""
        return {subsystem for key in operator.terms for subsystem, _ in key}

    @staticmethod
    def _local_of(
        expression: Expression,
        subsystems: Mapping[Expression, tuple[str, Mapping[str, LocalOperator]]],
    ) -> Operator:
        """Return the operator of a call `local_operator(subsystem, name)`."""
        subsystem, name = expression.operands()
        if subsystem not in subsystems:
            error_msg = f'{expression} acts on {subsystem}, which is not the state of a quantum element of the system.'
            raise ValidationError(error_msg)
        key, locals_ = subsystems[subsystem]
        if str(name) not in locals_:
            error_msg = f'{expression} names the operator {name}, but {key} has the operators {", ".join(locals_)}.'
            raise ValidationError(error_msg)
        return Operator({((key, locals_[str(name)]),): 1})

    @staticmethod
    def _product_of(factors: list[Operator], expression: Expression) -> Operator:
        """Return the product of the factors of a Sage product, which commute only on distinct subsystems."""
        acting = [_Lowering._acted_on(factor) for factor in factors if any(key for key in factor.terms)]
        if sum(len(item) for item in acting) != len(set().union(*acting)):
            error_msg = (
                f'{expression} multiplies operators that act on the same subsystem with `*`, which commutes. Write the '
                'product in its order with `operator_product`.'
            )
            raise ValidationError(error_msg)
        return functools.reduce(Operator.__mul__, factors, Operator({(): 1}))

    @staticmethod
    def _operator_of(
        expression: object,
        subsystems: Mapping[Expression, tuple[str, Mapping[str, LocalOperator]]],
    ) -> Operator:
        """Return the operator of a Sage expression of local operators, products, and adjoints.

        Args:
            expression:
                A Sage expression built with `local_operator`, `operator_product`, and `dagger`, or a number or a scalar
                expression, which stands for that multiple of the identity.

            subsystems:
                A mapping from the state symbol of every quantum subsystem to its key and its local operators by name.

        Returns:
            operator:
                The operator.

        Raises:
            ValidationError:
                A local operator acts on an unknown subsystem or names an unknown operator, or a Sage product multiplies
                operators of the same subsystem.

        Examples:
            >>> import qutip
            >>> from sage.all__sagemath_symbolics import SR, var
            >>> from skcomplex.core.laws.utils import local_operator
            >>> from skcomplex.engine.problems import LocalOperator
            >>> from skcomplex.engine.compile.passes._quantum_compiler import _Lowering
            >>> q = SR.symbol('q')
            >>> z = LocalOperator('z', qutip.sigmaz(), hermitian=True)
            >>> _Lowering._operator_of(var('h') * local_operator(q, SR.symbol('z')), {q: ('q', {'z': z})}).text
            'h*q.z'
        """
        value = SR(expression)
        if not has_operator(value):
            return Operator({(): value})
        function, operands = value.operator(), value.operands()
        if function == LOCAL_OPERATOR:
            return _Lowering._local_of(value, subsystems)
        if function is operator.pow and operands[1] in ZZ and operands[1] > 0:
            function, operands = operator_product, [operands[0]] * int(operands[1])
        parts = [_Lowering._operator_of(item, subsystems) for item in operands]
        if function == dagger:
            return parts[0].adjoint()
        if function == operator_product:
            return functools.reduce(Operator.__mul__, parts)
        if function is add_vararg:
            return functools.reduce(Operator.__add__, parts)
        if function is mul_vararg:
            return _Lowering._product_of(parts, value)
        error_msg = f'{value} is not a sum of products of local operators with scalar coefficients.'
        raise ValidationError(error_msg)


class QuantumCompiler(BaseCompiler):
    """Compiler of a system holding quantum elements into a problem of the quantum form.

    Args:
        backend:
            The backend requested for the problem, which the quantum form does not use.

        solver:
            The solver the problem runs on. A solver with a largest dimension of its own limits the Hilbert space.

        observables:
            The observables to compute at every frame, labelled by their path from the root.

        max_dimension:
            The largest dimension of the Hilbert space, for a solver without a largest dimension of its own.

    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.compile import QuantumCompiler
        >>> from skcomplex.models.biology.ecology import PopulationElement
        >>> QuantumCompiler().check().prepare(System(elements=[PopulationElement('rabbits')])).claims_
        False
    """

    def __init__(
        self,
        backend: BaseBackend | None = None,
        solver: object = None,
        observables: Sequence[ObservableSpec] = (),
        max_dimension: int = DEFAULT_MAX_DIMENSION,
    ) -> None:
        super().__init__(backend, solver, observables)
        self.max_dimension = max_dimension

    def _check(self) -> Self:
        """Raise an error for a largest dimension that is not a positive integer.

        Returns:
            self:
                The compiler.

        Raises:
            ValidationError:
                A setting is invalid.
        """
        super()._check()
        if not isinstance(self.max_dimension, int) or isinstance(self.max_dimension, bool) or self.max_dimension < 1:
            error_msg = f'The largest dimension is {self.max_dimension!r}. Give a positive integer.'
            raise ValidationError(error_msg)
        return self

    def _claims(self, system: System) -> bool:
        """Check whether a system holds a quantum element or group.

        Args:
            system:
                The system.

        Returns:
            claims:
                Whether the compiler compiles the system.
        """
        return self._contains_quantum(system)

    def _issues(self, system: System) -> list[Issue]:
        """Return the issue of a claimed system whose tree cannot be flattened.

        Args:
            system:
                The system, with the observables of the run.

        Returns:
            issues:
                The issues, none for a system the compiler does not claim.
        """
        if not self.claims_:
            return []
        try:
            FlattenPass._flatten_system(system)
        except SkcomplexError as error:
            fix = 'Pass elements, groups, interactions, and systems as children.'
            return [self._issue_of(error, system.label, 'structure', fix)]
        return []

    def _compile(self, initial_state: Mapping[str, object]) -> BaseProblem:
        """Compile the prepared quantum system into a problem of the quantum form.

        Args:
            initial_state:
                The initial state of the whole system under `<system label>.state`.

        Returns:
            problem:
                The problem of the quantum form.

        Raises:
            ValidationError:
                An initial value names another state than the whole state of the system.
        """
        system = self.system_
        whole = f'{system.label}.state'
        unknown = set(initial_state) - {whole}
        if unknown:
            error_msg = (
                f'The initial state names {", ".join(sorted(unknown))}, which are not states of the system. Use the '
                f'dotted names of states, such as {whole}.'
            )
            raise ValidationError(error_msg)
        limit = getattr(self.solver, 'max_dimension', self.max_dimension)
        return self._compile_quantum(system, initial=initial_state.get(whole), max_dimension=limit)

    @staticmethod
    def _is_quantum_node(node: Node) -> bool:
        """Check whether a node is a quantum element or a group of quantum elements."""
        return not isinstance(node.component, BaseInteraction) and QuantumView._is_quantum(node.component)

    @staticmethod
    def _member_value(value: object, member: int, size: int) -> object:
        """Return the value of one member, from a per-member sequence or a value shared by every member."""
        if isinstance(value, str | HilbertSpec) or value is None or np.ndim(value) == 0:
            return value
        return value[member] if len(value) == size else value

    @staticmethod
    def _members_of(node: Node) -> list[tuple[str, QuantumView]]:
        """Return the key and the element of every member of a quantum element or group."""
        component = node.component
        if not isinstance(component, GroupElement):
            return [(node.path, QuantumView(component))]
        size = node.members or 0
        members = []
        for member in range(size):
            values = {
                name: QuantumCompiler._member_value(value, member, size)
                for name, value in (component.values or {}).items()
            }
            element = component.element(f'{component.label}_{member}').set_params(**values).check()
            members.append((f'{node.path}[{member}]', QuantumView(element)))
        return members

    @staticmethod
    def _check_classical(nodes: list[Node]) -> None:
        """Raise an error when a quantum system holds classical elements or quantum elements with states."""
        classical = [
            node.path
            for node in nodes
            if not isinstance(node.component, BaseInteraction) and not QuantumCompiler._is_quantum_node(node)
        ]
        if classical:
            error_msg = (
                f'The system mixes quantum elements with the classical elements {", ".join(classical)}, which no '
                'single form runs. Model the quantum and the classical parts as separate systems.'
            )
            raise IncompatibleLawsError(error_msg)
        for node in nodes:
            states = (
                declarations_of(node.component_type, VariableKind.STATE)
                if QuantumCompiler._is_quantum_node(node)
                else {}
            )
            states = {
                name: item for name, item in states.items() if name != QuantumView._hilbert_state(node.component_type)
            }
            if states:
                error_msg = (
                    f'Quantum element {node.path} declares the states {", ".join(states)} besides its state in a '
                    'Hilbert space, which is its only state, so declare parameters instead.'
                )
                raise ValidationError(error_msg)

    @staticmethod
    def _check_dimension(nodes: list[Node], dimensions: list[int], limit: int) -> None:
        """Raise an error when the dimension of the Hilbert space exceeds the limit of the solver."""
        total = math.prod(dimensions)
        if total <= limit:
            return
        parts = []
        for node in nodes:
            if QuantumCompiler._is_quantum_node(node):
                members = QuantumCompiler._members_of(node)
                parts.append(f'{node.path} of dimension {math.prod(element.dimension for _, element in members)}')
        error_msg = (
            f'The Hilbert space of {", ".join(parts)} has dimension {total}, above the limit max_dimension = {limit}. '
            'Use fewer or smaller subsystems, or raise max_dimension of the quantum solver.'
        )
        raise HilbertSpaceError(error_msg)

    @staticmethod
    def _parameter_symbols(node: Node, symbols: Mapping[str, Any], member: int | None) -> dict[str, Any]:
        """Return the parameter symbols of a component, of one member for a group."""
        found = {}
        for variable in declarations_of(node.component_type, VariableKind.PARAMETER):
            value = symbols[f'{node.path}.{variable}']
            found[variable] = value if member is None else value[member]
        return found

    @staticmethod
    def _role_namespace(
        node: Node,
        label: str,
        members: Mapping[str, tuple[SimpleNamespace, ...]],
        groups: set[str],
    ) -> SimpleNamespace | tuple[SimpleNamespace, ...]:
        """Return the namespace a role binds to, one member's for an element and every member's for a group."""
        path = f'{node.scope}{label}'
        if path not in members:
            error_msg = f'Interaction {node.path} binds a role to {path}, which is not a quantum element or group.'
            raise ValidationError(error_msg)
        return members[path] if path in groups else members[path][0]

    @staticmethod
    def _collect_laws(laws: _Laws, owner: str, items: Iterable[Any]) -> None:
        """Add the Hamiltonian and dissipator laws of a component, rejecting every other law."""
        for law in items:
            if isinstance(law, GeneratorLaw):
                laws.hamiltonians.append((owner, law.check().operator))
            elif isinstance(law, DissipatorLaw):
                laws.dissipators.append((owner, law.check().operator, law.rate))
            else:
                error_msg = (
                    f'{owner} returns the law {type(law).__name__}, which a quantum system cannot hold. Quantum '
                    'systems take generator and dissipator laws only.'
                )
                raise IncompatibleLawsError(error_msg)

    @staticmethod
    def _interaction_laws(
        node: Node,
        symbols: Mapping[str, Any],
        members: Mapping[str, tuple[SimpleNamespace, ...]],
        groups: dict[str, Node],
        laws: _Laws,
    ) -> None:
        """Add the laws of an interaction, once per pair of neighbours of its group for a neighbour interaction."""
        interaction = node.component
        assert isinstance(interaction, BaseInteraction)  # nosec B101: the caller passes interactions only
        parameters = {**QuantumCompiler._parameter_symbols(node, symbols, None), 't': symbol('t')}
        if isinstance(interaction, NeighborInteraction):
            role_names = list(declarations_of(interaction, VariableKind.ROLE))
            if len(role_names) != 1:
                error_msg = (
                    f'Neighbour interaction {node.path} declares the roles {role_names}. Declare one role, bound to '
                    'the group whose neighbours it couples.'
                )
                raise ValidationError(error_msg)
            path = f'{node.scope}{getattr(interaction, role_names[0])}'
            if path not in groups:
                error_msg = f'Neighbour interaction {node.path} binds {path}, which is not a quantum group.'
                raise ValidationError(error_msg)
            space = groups[path].space
            if space is None:
                error_msg = f'Neighbour interaction {node.path} binds {path}, which lives in no space.'
                raise ValidationError(error_msg)
            for source, target in space.check().prepare()._neighbor_pairs(interaction.layer):
                namespace = SimpleNamespace(source=members[path][source], target=members[path][target], **parameters)
                QuantumCompiler._collect_laws(
                    laws,
                    f'{node.path} on the pair ({source}, {target})',
                    interaction.run(namespace),
                )
            return
        roles = {
            name: QuantumCompiler._role_namespace(node, getattr(interaction, name), members, set(groups))
            for name in declarations_of(interaction, VariableKind.ROLE)
        }
        QuantumCompiler._collect_laws(laws, node.path, interaction.run(SimpleNamespace(**roles, **parameters)))

    @staticmethod
    def _check_hermitian(laws: _Laws, lowering: _Lowering, p: Vector) -> None:
        """Raise an error when the Hamiltonian is not Hermitian at sampled parameter values."""
        size = math.prod(lowering.dimensions)
        rng = np.random.default_rng(0)
        sample = p * (1 + 0.1 * rng.random(p.size)) + 0.01 * rng.random(p.size)

        def asymmetry(terms: Iterable[QuantumKernel]) -> float:
            matrix = QuantumProblem._matrix_at(terms, _SAMPLE_TIME, sample, size)
            difference = matrix - matrix.conj().T
            return float(abs(difference).max()) if difference.nnz else 0.0

        total = [term for owner, operator in laws.hamiltonians for term in lowering.terms(operator, owner)]
        if asymmetry(total) <= _HERMITIAN_TOLERANCE:
            return
        owners = [owner for owner, operator in laws.hamiltonians if asymmetry(lowering.terms(operator, owner))]
        error_msg = (
            f'The Hamiltonian of {", ".join(owners)} is not Hermitian. Add the adjoint of every non-Hermitian '
            'term, `dagger(term)`, to the law.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _dissipator_terms(laws: _Laws, lowering: _Lowering, p: Vector) -> tuple[tuple[QuantumKernel, ...], ...]:
        """Return the terms of every collapse operator scaled by the square root of its rate."""
        collapses = []
        for owner, collapse, rate in laws.dissipators:
            value = SR(rate)
            initial = lowering.function(value, owner)(0.0, p) if value.variables() else complex(value)
            if initial.real < 0 or abs(initial.imag) > REAL_TOLERANCE:
                error_msg = f'The dissipator of {owner} has the rate {initial}, which is not a non-negative number.'
                raise ValidationError(error_msg)
            collapses.append(lowering.terms(collapse * value.sqrt(), owner))
        return tuple(collapses)

    @staticmethod
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
                f'The initial state of system {system.label!r} has {quantity} {norm:.6g}, but states have {quantity} '
                '1. Normalise it.'
            )
            raise ValidationError(error_msg)
        if state.ndim == 2 and not np.allclose(state, state.conj().T, atol=NORM_TOLERANCE):  # noqa: PLR2004
            error_msg = f'The initial density matrix of system {system.label!r} is not Hermitian.'
            raise ValidationError(error_msg)
        return state

    @staticmethod
    def _namespaces(
        system: System,
        members: Mapping[str, tuple[SimpleNamespace, ...]],
        groups: set[str],
    ) -> dict[str, SimpleNamespace]:
        """Return the namespace of every system of the tree by path, with its quantum elements and child systems."""
        namespaces: dict[str, SimpleNamespace] = {}

        def build(current: System, scope: str) -> SimpleNamespace:
            names: dict[str, Any] = {'t': symbol('t')}
            for element in current.elements:
                path = f'{scope}{element.label}'
                names[element.label] = members[path] if path in groups else members[path][0]
            for child in current.systems:
                names[child.label] = build(child, f'{scope}{child.label}/')
            namespace = SimpleNamespace(**names)
            namespaces[scope.removesuffix('/')] = namespace
            return namespace

        build(system, '')
        return namespaces

    @staticmethod
    def _subsystems_of(quantum: list[Node]) -> dict[str, _Subsystem]:
        """Return the subsystems of the quantum elements and groups in tensor order, each with its state symbol."""
        subsystems: dict[str, _Subsystem] = {}
        for node in quantum:
            for key, element in QuantumCompiler._members_of(node):
                state = symbol(f'{element.state}_{QuantumCompiler._name_of(key)}', CC)
                subsystems[key] = _Subsystem(key, element, len(subsystems), state, element.local_operators())
        return subsystems

    @staticmethod
    def _arguments_of(entries: list[Entry], symbols: Mapping[str, Any]) -> dict[Expression, int]:
        """Return the index in the parameter vector of every parameter symbol."""
        arguments = {}
        for entry in entries:
            components = symbols[entry.name]
            for offset, item in enumerate(components if entry.members is not None else [components]):
                arguments[item] = entry.start + offset
        return arguments

    @staticmethod
    def _element_laws(
        quantum: list[Node],
        subsystems: Mapping[str, _Subsystem],
        symbols: Mapping[str, Any],
        time: Expression,
        laws: _Laws,
    ) -> dict[str, tuple[SimpleNamespace, ...]]:
        """Add the laws of every quantum element and member, and return the member namespaces by path.

        The namespace of a member holds the symbol of its state, which names its subsystem in `local_operator`, and the
        symbols of its parameters. The laws of the member read time too.
        """
        members: dict[str, tuple[SimpleNamespace, ...]] = {}
        for node in quantum:
            namespaces = []
            for member, (key, element) in enumerate(QuantumCompiler._members_of(node)):
                index = member if node.members is not None else None
                state = {element.state: subsystems[key].symbol}
                namespace = SimpleNamespace(**state, **QuantumCompiler._parameter_symbols(node, symbols, index))
                namespaces.append(namespace)
                QuantumCompiler._collect_laws(
                    laws,
                    key,
                    element.element.check().run(SimpleNamespace(**vars(namespace), t=time)),
                )
            members[node.path] = tuple(namespaces)
        return members

    @staticmethod
    def _stored_state(  # noqa: PLR0917
        system: System,
        initial: object,
        subsystems: Mapping[str, _Subsystem],
        hamiltonian: tuple[QuantumKernel, ...],
        p: Vector,
        is_open: bool,
    ) -> npt.NDArray[np.complex128]:
        """Return the initial state as stored, a vector for a closed pure state and a density matrix otherwise."""
        product = np.ones(1, dtype=complex)
        for item in subsystems.values():
            product = np.kron(product, item.element.initial_vector())
        size = product.size

        def ground() -> npt.NDArray[np.complex128]:
            matrix = QuantumProblem._matrix_at(hamiltonian, 0.0, p, size)
            return np.asarray(qutip.Qobj(matrix).groundstate(sparse=size > DENSE_LIMIT)[1].full().ravel())

        state = QuantumCompiler._system_initial(system, initial, product, ground)
        return np.outer(state, state.conj()) if is_open and state.ndim == 1 else state

    @staticmethod
    def _compiled_element(node: Node) -> CompiledElement:
        """Return the compiled element of a quantum element or group, with no states in the state vector of its own."""
        component = node.component
        visual = component.visual if component.visual is not None else component.default_visual_
        parameters = {
            name: f'{node.path}.{name}' for name in declarations_of(node.component_type, VariableKind.PARAMETER)
        }
        return CompiledElement(node.path, component, {}, parameters, visual, node.members)

    @staticmethod
    def _compiled_interaction(node: Node) -> CompiledInteraction:
        """Return the compiled interaction of a quantum system with the paths its roles bind to."""
        component = node.component
        roles = {
            name: f'{node.scope}{getattr(component, name)}' for name in declarations_of(component, VariableKind.ROLE)
        }
        return CompiledInteraction(node.path, component, roles)

    @staticmethod
    def _compile_quantum(
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
                The initial state of the whole system, a vector, a density matrix, `'ground'` for the lowest eigenstate
                of the Hamiltonian at the initial parameters, or `None` for the product of the initial states of the
                elements.

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
        nodes = FlattenPass._flatten_system(system)
        QuantumCompiler._check_classical(nodes)
        quantum = [node for node in nodes if QuantumCompiler._is_quantum_node(node)]
        subsystems = QuantumCompiler._subsystems_of(quantum)
        dimensions = tuple(item.element.dimension for item in subsystems.values())
        QuantumCompiler._check_dimension(nodes, list(dimensions), max_dimension)
        entries, values, symbols = QuantumCompiler._parameters(nodes)
        p = np.concatenate(values) if values else np.zeros(0)
        lowering = _Lowering(time, subsystems, dimensions, QuantumCompiler._arguments_of(entries, symbols))
        groups = {node.path: node for node in quantum if isinstance(node.component, GroupElement)}
        laws = _Laws()
        members = QuantumCompiler._element_laws(quantum, subsystems, symbols, time, laws)
        for node in nodes:
            if isinstance(node.component, BaseInteraction):
                QuantumCompiler._interaction_laws(node, symbols, members, groups, laws)
        QuantumCompiler._check_hermitian(laws, lowering, p)
        total = sum((item for _, item in laws.hamiltonians), SR(0))
        hamiltonian = lowering.terms(total, 'the Hamiltonian')
        dissipators = QuantumCompiler._dissipator_terms(laws, lowering, p)
        state = QuantumCompiler._stored_state(system, initial, subsystems, hamiltonian, p, bool(dissipators))
        pure = state.ndim == 1
        state_entry = Entry(f'{system.label}.state', 0, state.shape, CC, None, is_complex=True)
        elements = tuple(QuantumCompiler._compiled_element(node) for node in quantum)
        interactions = tuple(
            QuantumCompiler._compiled_interaction(node) for node in nodes if isinstance(node.component, BaseInteraction)
        )
        return QuantumProblem(
            form=Form.QUANTUM,
            states=Layout([state_entry]),
            parameters=Layout(entries),
            initial_state=state_entry.encode(state),
            initial_parameters=p,
            time=time,
            symbols=symbols,
            elements=elements,
            interactions=interactions,
            system=system,
            dimensions=dimensions,
            subsystems=tuple(subsystems),
            hamiltonian=hamiltonian,
            dissipators=dissipators,
            pure=pure,
            namespaces=QuantumCompiler._namespaces(system, members, set(groups)),
            members=members,
            lowering=lowering,
            hamiltonian_operator=total,
        )

    @staticmethod
    def _name_of(path: str) -> str:
        """Return a path, such as `chain[3]`, as a valid part of a symbol name, such as `chain_3`."""
        return '_'.join(part for part in re.split(r'\W+', path) if part)

    @staticmethod
    def _parameters(nodes: list[Node]) -> tuple[list[Entry], list[Vector], dict[str, Any]]:
        """Return the parameter entries, initial values, and symbols of every component, per member for groups."""
        entries: list[Entry] = []
        values: list[Vector] = []
        symbols: dict[str, Any] = {}
        for node in nodes:
            for variable, declaration in declarations_of(node.component_type, VariableKind.PARAMETER).items():
                name = f'{node.path}.{variable}'
                if declaration.vector:
                    error_msg = (
                        f'{name} is a vector parameter, which quantum systems do not hold. Use scalar parameters.'
                    )
                    raise ValidationError(error_msg)
                members = node.members
                shape = () if members is None else (members,)
                values.append(
                    Declarations._coerce_value(name, node.value_of(variable), (), declaration.domain, members),
                )
                entries.append(Entry(name, sum(item.size for item in entries), shape, declaration.domain, members))
                base = f'{variable}_{QuantumCompiler._name_of(node.path)}'
                if members is None:
                    symbols[name] = symbol(base, declaration.domain)
                else:
                    symbols[name] = vector(
                        [symbol(f'{base}_{member}', declaration.domain) for member in range(members)],
                    )
        return entries, values, symbols

    @staticmethod
    def _contains_quantum(system: System) -> bool:
        """Check whether a system tree holds a quantum element or group.

        Args:
            system:
                The root system.

        Returns:
            found:
                Whether any element of the tree is quantum.
        """
        if any(QuantumView._is_quantum(element) for element in system.elements):
            return True
        return any(QuantumCompiler._contains_quantum(child) for child in system.systems)
