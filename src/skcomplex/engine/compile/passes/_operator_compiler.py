"""Pass that compiles a system holding operator elements into a problem of the operator form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import qutip
from sage.all__sagemath_symbolics import CC, SR, Expression
from sage.modules.free_module import FreeModule_generic

from ....core.elements import BaseElement, Element, GroupElement
from ....core.interactions import BaseInteraction, NeighborInteraction
from ....core.laws import DissipatorLaw, GeneratorLaw
from ....core.specs import CompiledElementSpec, CompiledInteractionSpec, NodeSpec, OperatorViewSpec
from ....core.specs._subsystem_spec import _SubsystemSpec
from ....core.system import System
from ....errors import IncompatibleLawsError, SkcomplexError, ValidationError
from ....specs import EntrySpec, HilbertSpec, IssueSpec, LayoutSpec, ObservableSpec
from ....types import Array, FormKind, VariableKind
from ..._utils import declarations_of, symbol, vector
from ...backends import BaseBackend
from ...problems import BaseProblemSpec, OperatorProblemSpec
from ...problems._operator_kernel import OperatorKernel
from ...problems._operator_lowering import OperatorLowering
from ...problems._operator_problem_spec import REAL_TOLERANCE
from ...types import Dissipators, Hamiltonians
from .._base import BaseCompiler
from ._flatten import FlattenPass
from ._hilbert_space_error import HilbertSpaceError
from ._layout import LayoutPass

DENSE_LIMIT = 1024


_HERMITIAN_TOLERANCE = 1e-10


NORM_TOLERANCE = 1e-8


_MATRIX_NDIM = 2


_SAMPLE_TIME = 0.37


DEFAULT_MAX_DIMENSION = 2**20


class OperatorCompiler(BaseCompiler):
    """Compiler of a system holding operator elements into a problem of the operator form.

    Args:
        backend:
            The backend requested for the problem, which the operator form does not use.

        solver:
            The solver the problem runs on. A solver with a largest dimension of its own limits the Hilbert space.

        observables:
            The observables to compute at every frame, labelled by their path from the root.

        max_dimension:
            The largest dimension of the Hilbert space, for a solver without a largest dimension of its own.

    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.compile import OperatorCompiler
        >>> from skcomplex.models.biology.common.populations import PopulationElement
        >>> OperatorCompiler().check().prepare(System(elements=[PopulationElement('rabbits')])).report_.ok
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

    def _check(self) -> None:
        """Raise an error for a largest dimension that is not a positive integer.

        Raises:
            ValidationError:
                A setting is invalid.
        """
        super()._check()
        if not isinstance(self.max_dimension, int) or isinstance(self.max_dimension, bool) or self.max_dimension < 1:
            error_msg = f'The largest dimension is {self.max_dimension!r}. Give a positive integer.'
            raise ValidationError(error_msg)

    def _issues(self, system: System) -> list[IssueSpec]:
        """Return the issue of a system the compiler does not compile, or whose tree cannot be flattened.

        Args:
            system:
                The system, with the observables of the run.

        Returns:
            issues:
                The issues.
        """
        if not self._contains_operator(system):
            message = f'The system {system.label} holds no operator element or group for OperatorCompiler to compile.'
            fix = 'Pass the compiler of the system, or none for PassCompiler.'
            return [IssueSpec('compiler', system.label, message, fix)]
        try:
            FlattenPass._flatten_system(system)
        except SkcomplexError as error:
            fix = 'Pass elements, groups, interactions, and systems as children.'
            return [self._issue_of(error, system.label, 'structure', fix)]
        return []

    def _compile(self, initial_state: Mapping[str, object]) -> BaseProblemSpec:
        """Compile the prepared operator system into a problem of the operator form.

        Args:
            initial_state:
                The initial state of the whole system under `<system label>.state`.

        Returns:
            problem:
                The problem of the operator form.

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
        return self._compile_operator(system, initial=initial_state.get(whole), max_dimension=limit)

    @staticmethod
    def _view_of(element: Element) -> OperatorViewSpec:
        """Return the operator view of an element, with the name of its state and its Hilbert space.

        Args:
            element:
                The element.

        Returns:
            view:
                The view.

        Raises:
            ValidationError:
                The element takes no `HilbertSpec`, declares no complex state, or has a state whose dimension differs
                from the dimension of its space.
        """
        space = OperatorCompiler._hilbert_of(element)
        if space is None:
            error_msg = f'{type(element).__name__} {element.label!r} takes no HilbertSpec.'
            raise ValidationError(error_msg)
        state = OperatorCompiler._hilbert_state(type(element))
        if state is None:
            error_msg = (
                f'{type(element).__name__} {element.label!r} takes a HilbertSpec but declares no complex state. '
                'Declare its state as `StateSpec(domain=CC)`.'
            )
            raise ValidationError(error_msg)
        space.check()
        domain = declarations_of(type(element), VariableKind.STATE)[state].domain
        dimension = domain.dimension() if isinstance(domain, FreeModule_generic) else space.dimension
        if dimension != space.dimension:
            error_msg = (
                f'{type(element).__name__} {element.label!r} has its state {state} over CC^{dimension}, but its '
                f'HilbertSpec has dimension {space.dimension}.'
            )
            raise ValidationError(error_msg)
        return OperatorViewSpec(element, state, space)

    @staticmethod
    def _hilbert_of(element: BaseElement) -> HilbertSpec | None:
        """Return the Hilbert space an element takes, of a member for a group, or `None` for a classical element.

        Args:
            element:
                An element or a group.

        Returns:
            space:
                The first argument of the element that is a `HilbertSpec`, or `None`.
        """
        member = OperatorCompiler._member_of(element)
        if member is None:
            return None
        return next((value for value in member.get_params(deep=False).values() if isinstance(value, HilbertSpec)), None)

    @staticmethod
    def _is_operator(element: object) -> bool:
        """Check whether an element or a group is a subsystem, which takes a `HilbertSpec`.

        Args:
            element:
                An element, a group, or any other part.

        Returns:
            operator:
                Whether the element, or a member of the group, takes a Hilbert space.
        """
        return isinstance(element, BaseElement) and OperatorCompiler._hilbert_of(element) is not None

    @staticmethod
    def _hilbert_state(element_type: type) -> str | None:
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
            >>> from skcomplex.engine.compile import OperatorCompiler
            >>> from skcomplex.specs import StateSpec
            >>> class QubitElement(Element):
            ...     def __init__(self, label, psi: Annotated[object, StateSpec(domain=CC**2)] = 'up'):
            ...         super().__init__(label)
            ...         self.psi = psi
            >>> OperatorCompiler._hilbert_state(QubitElement)
            'psi'
        """
        for name, declaration in declarations_of(element_type, VariableKind.STATE).items():
            base_ring = getattr(declaration.domain, 'base_ring', None)
            if declaration.domain is CC or (base_ring is not None and base_ring() is CC):
                return name
        return None

    @staticmethod
    def _member_of(element: BaseElement) -> Element | None:
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
            name: OperatorCompiler._member_value(value, 0, None) for name, value in (element.values or {}).items()
        }
        return element.element(element.label).set_params(**values)

    @staticmethod
    def _is_operator_node(node: NodeSpec) -> bool:
        """Check whether a node is an operator element or a group of operator elements."""
        return not isinstance(node.component, BaseInteraction) and OperatorCompiler._is_operator(node.component)

    @staticmethod
    def _member_value(value: object, member: int, size: int | None) -> object:
        """Return the value of one member, from a per-member sequence or a value shared by every member.

        Args:
            value:
                The value of a group, one per member or shared by every member.

            member:
                The index of the member.

            size:
                The number of members, or `None` to take the member from any sequence.

        Returns:
            value:
                The value of the member.
        """
        if (
            isinstance(value, str | HilbertSpec)
            or not isinstance(value, Iterable)
            or np.asarray(value, dtype=object).ndim == 0
        ):
            return value
        items = list(value)
        return items[member] if size is None or len(items) == size else value

    @staticmethod
    def _members_of(node: NodeSpec) -> list[tuple[str, OperatorViewSpec]]:
        """Return the key and the element of every member of an operator element or group."""
        component = node.component
        if not isinstance(component, GroupElement):
            return [(node.path, OperatorCompiler._view_of(component))]
        size = node.members or 0
        members = []
        for member in range(size):
            values = {
                name: OperatorCompiler._member_value(value, member, size)
                for name, value in (component.values or {}).items()
            }
            element = component.element(f'{component.label}_{member}').set_params(**values).check()
            members.append((f'{node.path}[{member}]', OperatorCompiler._view_of(element)))
        return members

    @staticmethod
    def _check_classical(nodes: list[NodeSpec]) -> None:
        """Raise an error when an operator system holds classical elements or operator elements with states."""
        classical = [
            node.path
            for node in nodes
            if not isinstance(node.component, BaseInteraction) and not OperatorCompiler._is_operator_node(node)
        ]
        if classical:
            error_msg = (
                f'The system mixes operator elements with the classical elements {", ".join(classical)}, which no '
                'single form runs. Model the operator and the classical parts as separate systems.'
            )
            raise IncompatibleLawsError(error_msg)
        for node in nodes:
            states = (
                declarations_of(node.component_type, VariableKind.STATE)
                if OperatorCompiler._is_operator_node(node)
                else {}
            )
            states = {
                name: item
                for name, item in states.items()
                if name != OperatorCompiler._hilbert_state(node.component_type)
            }
            if states:
                error_msg = (
                    f'Operator element {node.path} declares the states {", ".join(states)} besides its state in a '
                    'Hilbert space, which is its only state, so declare parameters instead.'
                )
                raise ValidationError(error_msg)

    @staticmethod
    def _check_dimension(nodes: list[NodeSpec], dimensions: list[int], limit: int) -> None:
        """Raise an error when the dimension of the Hilbert space exceeds the limit of the solver."""
        total = math.prod(dimensions)
        if total <= limit:
            return
        parts = []
        for node in nodes:
            if OperatorCompiler._is_operator_node(node):
                members = OperatorCompiler._members_of(node)
                parts.append(f'{node.path} of dimension {math.prod(element.dimension for _, element in members)}')
        error_msg = (
            f'The Hilbert space of {", ".join(parts)} has dimension {total}, above the limit max_dimension = {limit}. '
            'Use fewer or smaller subsystems, or raise max_dimension of the operator solver.'
        )
        raise HilbertSpaceError(error_msg)

    @staticmethod
    def _parameter_symbols(
        node: NodeSpec,
        symbols: Mapping[str, Expression],
        member: int | None,
    ) -> dict[str, Expression]:
        """Return the parameter symbols of a component, of one member for a group."""
        found = {}
        for variable in declarations_of(node.component_type, VariableKind.PARAMETER):
            value = symbols[f'{node.path}.{variable}']
            found[variable] = value if member is None else value[member]
        return found

    @staticmethod
    def _role_namespace(
        node: NodeSpec,
        label: str,
        members: Mapping[str, tuple[SimpleNamespace, ...]],
        groups: set[str],
    ) -> SimpleNamespace | tuple[SimpleNamespace, ...]:
        """Return the namespace a role binds to, one member's for an element and every member's for a group."""
        path = f'{node.scope}{label}'
        if path not in members:
            error_msg = f'Interaction {node.path} binds a role to {path}, which is not an operator element or group.'
            raise ValidationError(error_msg)
        return members[path] if path in groups else members[path][0]

    @staticmethod
    def _collect_laws(hamiltonians: Hamiltonians, dissipators: Dissipators, owner: str, items: object) -> None:
        """Add the Hamiltonian and dissipator laws a component returns, rejecting every other law."""
        if not isinstance(items, Iterable):
            error_msg = f'{owner} returns a {type(items).__name__}, not a list of laws.'
            raise TypeError(error_msg)
        for law in items:
            if isinstance(law, GeneratorLaw):
                hamiltonians.append((owner, law.check().operator))
            elif isinstance(law, DissipatorLaw):
                dissipators.append((owner, law.check().operator, law.rate))
            else:
                error_msg = (
                    f'{owner} returns the law {type(law).__name__}, which an operator system cannot hold. Operator '
                    'systems take generator and dissipator laws only.'
                )
                raise IncompatibleLawsError(error_msg)

    @staticmethod
    def _interaction_laws(
        node: NodeSpec,
        interaction: BaseInteraction,
        *,
        symbols: Mapping[str, object],
        members: Mapping[str, tuple[SimpleNamespace, ...]],
        groups: dict[str, NodeSpec],
        hamiltonians: Hamiltonians,
        dissipators: Dissipators,
    ) -> None:
        """Add the laws of an interaction of a node, once per pair of neighbours of its group for a neighbour one."""
        parameters = {**OperatorCompiler._parameter_symbols(node, symbols, None), 't': symbol('t')}
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
                error_msg = f'Neighbour interaction {node.path} binds {path}, which is not an operator group.'
                raise ValidationError(error_msg)
            space = groups[path].space
            if space is None:
                error_msg = f'Neighbour interaction {node.path} binds {path}, which lives in no space.'
                raise ValidationError(error_msg)
            for source, target in space.check().prepare()._neighbor_pairs(interaction.layer):
                namespace = SimpleNamespace(source=members[path][source], target=members[path][target], **parameters)
                OperatorCompiler._collect_laws(
                    hamiltonians,
                    dissipators,
                    f'{node.path} on the pair ({source}, {target})',
                    interaction.run(namespace).laws_,
                )
            return
        roles = {
            name: OperatorCompiler._role_namespace(node, getattr(interaction, name), members, set(groups))
            for name in declarations_of(interaction, VariableKind.ROLE)
        }
        OperatorCompiler._collect_laws(
            hamiltonians,
            dissipators,
            node.path,
            interaction.run(SimpleNamespace(**roles, **parameters)).laws_,
        )

    @staticmethod
    def _terms(lowering: OperatorLowering, operator: Expression, owner: str) -> tuple[OperatorKernel, ...]:
        """Return the terms an operator lowers to.

        Args:
            lowering:
                The lowering of operators.

            operator:
                The operator.

            owner:
                The description of its law or observable.

        Returns:
            terms:
                The terms.
        """
        return lowering.run((operator, owner)).kernels_

    @staticmethod
    def _check_hermitian(hamiltonians: Hamiltonians, lowering: OperatorLowering, p: Array) -> None:
        """Raise an error when the Hamiltonian is not Hermitian at sampled parameter values."""
        size = math.prod(lowering.dimensions)
        rng = np.random.default_rng(0)
        sample = p * (1 + 0.1 * rng.random(p.size)) + 0.01 * rng.random(p.size)

        def asymmetry(terms: Iterable[OperatorKernel]) -> float:
            matrix = OperatorProblemSpec._matrix_at(terms, _SAMPLE_TIME, sample, size)
            difference = matrix - matrix.conj().T
            return float(abs(difference).max()) if difference.nnz else 0.0

        total = [
            term for owner, operator in hamiltonians for term in OperatorCompiler._terms(lowering, operator, owner)
        ]
        if asymmetry(total) <= _HERMITIAN_TOLERANCE:
            return
        owners = [
            owner for owner, operator in hamiltonians if asymmetry(OperatorCompiler._terms(lowering, operator, owner))
        ]
        error_msg = (
            f'The Hamiltonian of {", ".join(owners)} is not Hermitian. Add the adjoint of every non-Hermitian '
            'term, `dagger(term)`, to the law.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _dissipator_terms(
        dissipators: Dissipators,
        lowering: OperatorLowering,
        p: Array,
    ) -> tuple[tuple[OperatorKernel, ...], ...]:
        """Return the terms of every collapse operator scaled by the square root of its rate."""
        collapses = []
        for owner, collapse, rate in dissipators:
            value = SR(rate)
            initial = lowering._function(value, owner)(0.0, p) if value.variables() else complex(value)
            if initial.real < 0 or abs(initial.imag) > REAL_TOLERANCE:
                error_msg = f'The dissipator of {owner} has the rate {initial}, which is not a non-negative number.'
                raise ValidationError(error_msg)
            collapses.append(OperatorCompiler._terms(lowering, collapse * value.sqrt(), owner))
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
        if state.ndim == _MATRIX_NDIM and not np.allclose(state, state.conj().T, atol=NORM_TOLERANCE):
            error_msg = f'The initial density matrix of system {system.label!r} is not Hermitian.'
            raise ValidationError(error_msg)
        return state

    @staticmethod
    def _namespaces(
        system: System,
        members: Mapping[str, tuple[SimpleNamespace, ...]],
        groups: set[str],
    ) -> dict[str, SimpleNamespace]:
        """Return the namespace of every system of the tree by path, with its operator elements and child systems."""
        namespaces: dict[str, SimpleNamespace] = {}

        def build(current: System, scope: str) -> SimpleNamespace:
            names: dict[str, object] = {'t': symbol('t')}
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
    def _subsystems_of(operator_nodes: list[NodeSpec]) -> dict[str, _SubsystemSpec]:
        """Return the subsystems of the operator elements and groups in tensor order, each with its state symbol."""
        subsystems: dict[str, _SubsystemSpec] = {}
        for node in operator_nodes:
            for key, element in OperatorCompiler._members_of(node):
                state = symbol(f'{element.state}_{OperatorCompiler._name_of(key)}', CC)
                subsystems[key] = _SubsystemSpec(key, element, len(subsystems), state, element.local_operators())
        return subsystems

    @staticmethod
    def _arguments_of(entries: list[EntrySpec], symbols: Mapping[str, Expression]) -> dict[Expression, int]:
        """Return the index in the parameter vector of every parameter symbol."""
        arguments = {}
        for entry in entries:
            components = symbols[entry.name]
            for offset, item in enumerate(components if entry.members is not None else [components]):
                arguments[item] = entry.start + offset
        return arguments

    @staticmethod
    def _element_laws(
        operator_nodes: list[NodeSpec],
        subsystems: Mapping[str, _SubsystemSpec],
        *,
        symbols: Mapping[str, object],
        time: Expression,
        hamiltonians: Hamiltonians,
        dissipators: Dissipators,
    ) -> dict[str, tuple[SimpleNamespace, ...]]:
        """Add the laws of every operator element and member, and return the member namespaces by path.

        The namespace of a member holds the symbol of its state, which names its subsystem in `local_operator`, and the
        symbols of its parameters. The laws of the member read time too.
        """
        members: dict[str, tuple[SimpleNamespace, ...]] = {}
        for node in operator_nodes:
            namespaces = []
            for member, (key, element) in enumerate(OperatorCompiler._members_of(node)):
                index = member if node.members is not None else None
                state = {element.state: subsystems[key].symbol}
                namespace = SimpleNamespace(**state, **OperatorCompiler._parameter_symbols(node, symbols, index))
                namespaces.append(namespace)
                OperatorCompiler._collect_laws(
                    hamiltonians,
                    dissipators,
                    key,
                    element.element.check().prepare().run(SimpleNamespace(**vars(namespace), t=time)).laws_,
                )
            members[node.path] = tuple(namespaces)
        return members

    @staticmethod
    def _stored_state(
        system: System,
        initial: object,
        *,
        subsystems: Mapping[str, _SubsystemSpec],
        hamiltonian: tuple[OperatorKernel, ...],
        p: Array,
        is_open: bool,
    ) -> npt.NDArray[np.complex128]:
        """Return the initial state as stored, a vector for a closed pure state and a density matrix otherwise."""
        product = np.ones(1, dtype=complex)
        for item in subsystems.values():
            product = np.kron(product, item.element.initial_vector())
        size = product.size

        def ground() -> npt.NDArray[np.complex128]:
            matrix = OperatorProblemSpec._matrix_at(hamiltonian, 0.0, p, size)
            return np.asarray(qutip.Qobj(matrix).groundstate(sparse=size > DENSE_LIMIT)[1].full().ravel())

        state = OperatorCompiler._system_initial(system, initial, product, ground)
        return np.outer(state, state.conj()) if is_open and state.ndim == 1 else state

    @staticmethod
    def _compiled_element(node: NodeSpec) -> CompiledElementSpec:
        """Return the compiled element of an operator element or group, with no states in the state vector."""
        component = node.component
        visual = component.visual if component.visual is not None else component.default_visual_
        parameters = {
            name: f'{node.path}.{name}' for name in declarations_of(node.component_type, VariableKind.PARAMETER)
        }
        return CompiledElementSpec(node.path, component, {}, parameters, visual, node.members)

    @staticmethod
    def _compiled_interaction(node: NodeSpec) -> CompiledInteractionSpec:
        """Return the compiled interaction of an operator system with the paths its roles bind to."""
        component = node.component
        roles = {
            name: f'{node.scope}{getattr(component, name)}' for name in declarations_of(component, VariableKind.ROLE)
        }
        return CompiledInteractionSpec(node.path, component, roles)

    @staticmethod
    def _compile_operator(
        system: System,
        *,
        initial: object = None,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
    ) -> OperatorProblemSpec:
        """Compile a system of operator elements into a problem of the operator form.

        Args:
            system:
                The system.

            initial:
                The initial state of the whole system, a vector, a density matrix, `'ground'` for the lowest eigenstate
                of the Hamiltonian at the initial parameters, or `None` for the product of the initial states of the
                elements.

            max_dimension:
                The largest dimension of the Hilbert space, the limit of the operator solver.

        Returns:
            problem:
                The operator problem.

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
        OperatorCompiler._check_classical(nodes)
        operator_nodes = [node for node in nodes if OperatorCompiler._is_operator_node(node)]
        subsystems = OperatorCompiler._subsystems_of(operator_nodes)
        dimensions = tuple(item.element.dimension for item in subsystems.values())
        OperatorCompiler._check_dimension(nodes, list(dimensions), max_dimension)
        entries, values, symbols = OperatorCompiler._parameters(nodes)
        p = np.concatenate(values) if values else np.zeros(0)
        arguments = OperatorCompiler._arguments_of(entries, symbols)
        lowering = OperatorLowering(time, subsystems, dimensions, arguments).check().prepare()
        groups = {node.path: node for node in operator_nodes if isinstance(node.component, GroupElement)}
        hamiltonians: Hamiltonians = []
        dissipators: Dissipators = []
        members = OperatorCompiler._element_laws(
            operator_nodes,
            subsystems,
            symbols=symbols,
            time=time,
            hamiltonians=hamiltonians,
            dissipators=dissipators,
        )
        for node in nodes:
            if isinstance(node.component, BaseInteraction):
                OperatorCompiler._interaction_laws(
                    node,
                    node.component,
                    symbols=symbols,
                    members=members,
                    groups=groups,
                    hamiltonians=hamiltonians,
                    dissipators=dissipators,
                )
        OperatorCompiler._check_hermitian(hamiltonians, lowering, p)
        total = sum((item for _, item in hamiltonians), SR(0))
        hamiltonian = OperatorCompiler._terms(lowering, total, 'the Hamiltonian')
        collapses = OperatorCompiler._dissipator_terms(dissipators, lowering, p)
        state = OperatorCompiler._stored_state(
            system,
            initial,
            subsystems=subsystems,
            hamiltonian=hamiltonian,
            p=p,
            is_open=bool(collapses),
        )
        pure = state.ndim == 1
        state_entry = EntrySpec(f'{system.label}.state', 0, state.shape, CC, None, is_complex=True)
        elements = tuple(OperatorCompiler._compiled_element(node) for node in operator_nodes)
        interactions = tuple(
            OperatorCompiler._compiled_interaction(node)
            for node in nodes
            if isinstance(node.component, BaseInteraction)
        )
        return OperatorProblemSpec(
            form=FormKind.OPERATOR,
            states=LayoutSpec((state_entry,)),
            parameters=LayoutSpec(tuple(entries)),
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
            dissipators=collapses,
            pure=pure,
            namespaces=OperatorCompiler._namespaces(system, members, set(groups)),
            members=members,
            lowering=lowering,
            hamiltonian_operator=total,
        )

    @staticmethod
    def _name_of(path: str) -> str:
        """Return a path, such as `chain[3]`, as a valid part of a symbol name, such as `chain_3`."""
        return '_'.join(part for part in re.split(r'\W+', path) if part)

    @staticmethod
    def _parameters(nodes: list[NodeSpec]) -> tuple[list[EntrySpec], list[Array], dict[str, Expression]]:
        """Return the parameter entries, initial values, and symbols of every component, per member for groups."""
        entries: list[EntrySpec] = []
        values: list[Array] = []
        symbols: dict[str, Expression] = {}
        for node in nodes:
            for variable, declaration in declarations_of(node.component_type, VariableKind.PARAMETER).items():
                name = f'{node.path}.{variable}'
                if declaration.vector:
                    error_msg = (
                        f'{name} is a vector parameter, which operator systems do not hold. Use scalar parameters.'
                    )
                    raise ValidationError(error_msg)
                members = node.members
                shape = () if members is None else (members,)
                values.append(
                    LayoutPass._coerce_value(
                        name,
                        LayoutPass._value_of(node, variable),
                        (),
                        declaration.domain,
                        members,
                    ),
                )
                entries.append(EntrySpec(name, sum(item.size for item in entries), shape, declaration.domain, members))
                base = f'{variable}_{OperatorCompiler._name_of(node.path)}'
                if members is None:
                    symbols[name] = symbol(base, declaration.domain)
                else:
                    symbols[name] = vector(
                        [symbol(f'{base}_{member}', declaration.domain) for member in range(members)],
                    )
        return entries, values, symbols

    @staticmethod
    def _contains_operator(system: System) -> bool:
        """Check whether a system tree holds an operator element or group.

        Args:
            system:
                The root system.

        Returns:
            found:
                Whether any element of the tree is operator.
        """
        if any(OperatorCompiler._is_operator(element) for element in system.elements):
            return True
        return any(OperatorCompiler._contains_operator(child) for child in system.systems)
