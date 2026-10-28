"""Pass that compiles a system holding quantum elements into a problem of the quantum form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
import qutip
from sage.all__sagemath_symbolics import CC, SR, Expression

from ....core.elements import BaseElement, Element, GroupElement
from ....core.interactions import BaseInteraction, NeighborInteraction
from ....core.laws import DissipatorLaw, GeneratorLaw
from ....core.system import System
from ....errors import IncompatibleLawsError, SkcomplexError, ValidationError
from ....specs import (
    BaseProblemSpec,
    CompiledElementSpec,
    CompiledInteractionSpec,
    EntrySpec,
    HilbertSpec,
    IssueSpec,
    LayoutSpec,
    NodeSpec,
    ObservableSpec,
    QuantumProblemSpec,
    QuantumViewSpec,
)
from ....specs._quantum_problem_spec import REAL_TOLERANCE
from ....specs._subsystem_spec import _SubsystemSpec
from ....types import Array, FormKind, VariableKind
from ..._utils import declarations_of, symbol, vector
from ...backends import BaseBackend
from ...problems._quantum_kernel import QuantumKernel
from ...problems._quantum_lowering import QuantumLowering
from .._base import BaseCompiler
from ._flatten import FlattenPass
from ._hilbert_space_error import HilbertSpaceError
from ._layout import LayoutPass

DENSE_LIMIT = 1024


_HERMITIAN_TOLERANCE = 1e-10


NORM_TOLERANCE = 1e-8


_SAMPLE_TIME = 0.37


DEFAULT_MAX_DIMENSION = 2**20


type Hamiltonians = list[tuple[str, Expression]]
type Dissipators = list[tuple[str, Expression, object]]


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

    def _issues(self, system: System) -> list[IssueSpec]:
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

    def _compile(self, initial_state: Mapping[str, object]) -> BaseProblemSpec:
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
    def _view_of(element: Element) -> QuantumViewSpec:
        """Return the quantum view of an element, with the name of its state and its Hilbert space.

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
        space = QuantumCompiler._hilbert_of(element)
        if space is None:
            error_msg = f'{type(element).__name__} {element.label!r} takes no HilbertSpec.'
            raise ValidationError(error_msg)
        state = QuantumCompiler._hilbert_state(type(element))
        if state is None:
            error_msg = (
                f'{type(element).__name__} {element.label!r} takes a HilbertSpec but declares no complex state. '
                'Declare its state as `StateSpec(domain=CC)`.'
            )
            raise ValidationError(error_msg)
        space.check()
        domain = declarations_of(type(element), VariableKind.STATE)[state].domain
        dimension = space.dimension if domain is CC else domain.dimension()
        if dimension != space.dimension:
            error_msg = (
                f'{type(element).__name__} {element.label!r} has its state {state} over CC^{dimension}, but its '
                f'HilbertSpec has dimension {space.dimension}.'
            )
            raise ValidationError(error_msg)
        return QuantumViewSpec(element, state, space)

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
        member = QuantumCompiler._member_of(element)
        if member is None:
            return None
        return next((value for value in member.get_params(deep=False).values() if isinstance(value, HilbertSpec)), None)

    @staticmethod
    def _is_quantum(element: object) -> bool:
        """Check whether an element or a group is a quantum subsystem, which takes a `HilbertSpec`.

        Args:
            element:
                An element, a group, or any other part.

        Returns:
            quantum:
                Whether the element, or a member of the group, takes a Hilbert space.
        """
        return isinstance(element, BaseElement) and QuantumCompiler._hilbert_of(element) is not None

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
            >>> from skcomplex.engine.compile import QuantumCompiler
            >>> from skcomplex.specs import StateSpec
            >>> class QubitElement(Element):
            ...     def __init__(self, label, psi: Annotated[object, StateSpec(domain=CC**2)] = 'up'):
            ...         super().__init__(label)
            ...         self.psi = psi
            >>> QuantumCompiler._hilbert_state(QubitElement)
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
            name: value if isinstance(value, str | HilbertSpec) or np.ndim(value) == 0 else value[0]
            for name, value in (element.values or {}).items()
        }
        return element.element(element.label).set_params(**values)

    @staticmethod
    def _is_quantum_node(node: NodeSpec) -> bool:
        """Check whether a node is a quantum element or a group of quantum elements."""
        return not isinstance(node.component, BaseInteraction) and QuantumCompiler._is_quantum(node.component)

    @staticmethod
    def _member_value(value: object, member: int, size: int) -> object:
        """Return the value of one member, from a per-member sequence or a value shared by every member."""
        if isinstance(value, str | HilbertSpec) or value is None or np.ndim(value) == 0:
            return value
        return value[member] if len(value) == size else value

    @staticmethod
    def _members_of(node: NodeSpec) -> list[tuple[str, QuantumViewSpec]]:
        """Return the key and the element of every member of a quantum element or group."""
        component = node.component
        if not isinstance(component, GroupElement):
            return [(node.path, QuantumCompiler._view_of(component))]
        size = node.members or 0
        members = []
        for member in range(size):
            values = {
                name: QuantumCompiler._member_value(value, member, size)
                for name, value in (component.values or {}).items()
            }
            element = component.element(f'{component.label}_{member}').set_params(**values).check()
            members.append((f'{node.path}[{member}]', QuantumCompiler._view_of(element)))
        return members

    @staticmethod
    def _check_classical(nodes: list[NodeSpec]) -> None:
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
                name: item
                for name, item in states.items()
                if name != QuantumCompiler._hilbert_state(node.component_type)
            }
            if states:
                error_msg = (
                    f'Quantum element {node.path} declares the states {", ".join(states)} besides its state in a '
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
            if QuantumCompiler._is_quantum_node(node):
                members = QuantumCompiler._members_of(node)
                parts.append(f'{node.path} of dimension {math.prod(element.dimension for _, element in members)}')
        error_msg = (
            f'The Hilbert space of {", ".join(parts)} has dimension {total}, above the limit max_dimension = {limit}. '
            'Use fewer or smaller subsystems, or raise max_dimension of the quantum solver.'
        )
        raise HilbertSpaceError(error_msg)

    @staticmethod
    def _parameter_symbols(node: NodeSpec, symbols: Mapping[str, Any], member: int | None) -> dict[str, Any]:
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
            error_msg = f'Interaction {node.path} binds a role to {path}, which is not a quantum element or group.'
            raise ValidationError(error_msg)
        return members[path] if path in groups else members[path][0]

    @staticmethod
    def _collect_laws(hamiltonians: Hamiltonians, dissipators: Dissipators, owner: str, items: Iterable[Any]) -> None:
        """Add the Hamiltonian and dissipator laws of a component, rejecting every other law."""
        for law in items:
            if isinstance(law, GeneratorLaw):
                hamiltonians.append((owner, law.check().operator))
            elif isinstance(law, DissipatorLaw):
                dissipators.append((owner, law.check().operator, law.rate))
            else:
                error_msg = (
                    f'{owner} returns the law {type(law).__name__}, which a quantum system cannot hold. Quantum '
                    'systems take generator and dissipator laws only.'
                )
                raise IncompatibleLawsError(error_msg)

    @staticmethod
    def _interaction_laws(  # noqa: PLR0917
        node: NodeSpec,
        symbols: Mapping[str, Any],
        members: Mapping[str, tuple[SimpleNamespace, ...]],
        groups: dict[str, NodeSpec],
        hamiltonians: Hamiltonians,
        dissipators: Dissipators,
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
                    hamiltonians,
                    dissipators,
                    f'{node.path} on the pair ({source}, {target})',
                    interaction.run(namespace),
                )
            return
        roles = {
            name: QuantumCompiler._role_namespace(node, getattr(interaction, name), members, set(groups))
            for name in declarations_of(interaction, VariableKind.ROLE)
        }
        QuantumCompiler._collect_laws(
            hamiltonians, dissipators, node.path, interaction.run(SimpleNamespace(**roles, **parameters)),
        )

    @staticmethod
    def _check_hermitian(hamiltonians: Hamiltonians, lowering: QuantumLowering, p: Array) -> None:
        """Raise an error when the Hamiltonian is not Hermitian at sampled parameter values."""
        size = math.prod(lowering.dimensions)
        rng = np.random.default_rng(0)
        sample = p * (1 + 0.1 * rng.random(p.size)) + 0.01 * rng.random(p.size)

        def asymmetry(terms: Iterable[QuantumKernel]) -> float:
            matrix = QuantumProblemSpec._matrix_at(terms, _SAMPLE_TIME, sample, size)
            difference = matrix - matrix.conj().T
            return float(abs(difference).max()) if difference.nnz else 0.0

        total = [term for owner, operator in hamiltonians for term in lowering.run((operator, owner))]
        if asymmetry(total) <= _HERMITIAN_TOLERANCE:
            return
        owners = [owner for owner, operator in hamiltonians if asymmetry(lowering.run((operator, owner)))]
        error_msg = (
            f'The Hamiltonian of {", ".join(owners)} is not Hermitian. Add the adjoint of every non-Hermitian '
            'term, `dagger(term)`, to the law.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _dissipator_terms(
        dissipators: Dissipators, lowering: QuantumLowering, p: Array,
    ) -> tuple[tuple[QuantumKernel, ...], ...]:
        """Return the terms of every collapse operator scaled by the square root of its rate."""
        collapses = []
        for owner, collapse, rate in dissipators:
            value = SR(rate)
            initial = lowering._function(value, owner)(0.0, p) if value.variables() else complex(value)
            if initial.real < 0 or abs(initial.imag) > REAL_TOLERANCE:
                error_msg = f'The dissipator of {owner} has the rate {initial}, which is not a non-negative number.'
                raise ValidationError(error_msg)
            collapses.append(lowering.run((collapse * value.sqrt(), owner)))
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
    def _subsystems_of(quantum: list[NodeSpec]) -> dict[str, _SubsystemSpec]:
        """Return the subsystems of the quantum elements and groups in tensor order, each with its state symbol."""
        subsystems: dict[str, _SubsystemSpec] = {}
        for node in quantum:
            for key, element in QuantumCompiler._members_of(node):
                state = symbol(f'{element.state}_{QuantumCompiler._name_of(key)}', CC)
                subsystems[key] = _SubsystemSpec(key, element, len(subsystems), state, element.local_operators())
        return subsystems

    @staticmethod
    def _arguments_of(entries: list[EntrySpec], symbols: Mapping[str, Any]) -> dict[Expression, int]:
        """Return the index in the parameter vector of every parameter symbol."""
        arguments = {}
        for entry in entries:
            components = symbols[entry.name]
            for offset, item in enumerate(components if entry.members is not None else [components]):
                arguments[item] = entry.start + offset
        return arguments

    @staticmethod
    def _element_laws(  # noqa: PLR0917
        quantum: list[NodeSpec],
        subsystems: Mapping[str, _SubsystemSpec],
        symbols: Mapping[str, Any],
        time: Expression,
        hamiltonians: Hamiltonians,
        dissipators: Dissipators,
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
                    hamiltonians,
                    dissipators,
                    key,
                    element.element.check().run(SimpleNamespace(**vars(namespace), t=time)),
                )
            members[node.path] = tuple(namespaces)
        return members

    @staticmethod
    def _stored_state(  # noqa: PLR0917
        system: System,
        initial: object,
        subsystems: Mapping[str, _SubsystemSpec],
        hamiltonian: tuple[QuantumKernel, ...],
        p: Array,
        is_open: bool,
    ) -> npt.NDArray[np.complex128]:
        """Return the initial state as stored, a vector for a closed pure state and a density matrix otherwise."""
        product = np.ones(1, dtype=complex)
        for item in subsystems.values():
            product = np.kron(product, item.element.initial_vector())
        size = product.size

        def ground() -> npt.NDArray[np.complex128]:
            matrix = QuantumProblemSpec._matrix_at(hamiltonian, 0.0, p, size)
            return np.asarray(qutip.Qobj(matrix).groundstate(sparse=size > DENSE_LIMIT)[1].full().ravel())

        state = QuantumCompiler._system_initial(system, initial, product, ground)
        return np.outer(state, state.conj()) if is_open and state.ndim == 1 else state

    @staticmethod
    def _compiled_element(node: NodeSpec) -> CompiledElementSpec:
        """Return the compiled element of a quantum element or group, with no states in the state vector of its own."""
        component = node.component
        visual = component.visual if component.visual is not None else component.default_visual_
        parameters = {
            name: f'{node.path}.{name}' for name in declarations_of(node.component_type, VariableKind.PARAMETER)
        }
        return CompiledElementSpec(node.path, component, {}, parameters, visual, node.members)

    @staticmethod
    def _compiled_interaction(node: NodeSpec) -> CompiledInteractionSpec:
        """Return the compiled interaction of a quantum system with the paths its roles bind to."""
        component = node.component
        roles = {
            name: f'{node.scope}{getattr(component, name)}' for name in declarations_of(component, VariableKind.ROLE)
        }
        return CompiledInteractionSpec(node.path, component, roles)

    @staticmethod
    def _compile_quantum(
        system: System,
        *,
        initial: object = None,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
    ) -> QuantumProblemSpec:
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
        arguments = QuantumCompiler._arguments_of(entries, symbols)
        lowering = QuantumLowering(time, subsystems, dimensions, arguments).check()
        groups = {node.path: node for node in quantum if isinstance(node.component, GroupElement)}
        hamiltonians: Hamiltonians = []
        dissipators: Dissipators = []
        members = QuantumCompiler._element_laws(quantum, subsystems, symbols, time, hamiltonians, dissipators)
        for node in nodes:
            if isinstance(node.component, BaseInteraction):
                QuantumCompiler._interaction_laws(node, symbols, members, groups, hamiltonians, dissipators)
        QuantumCompiler._check_hermitian(hamiltonians, lowering, p)
        total = sum((item for _, item in hamiltonians), SR(0))
        hamiltonian = cast('tuple[QuantumKernel, ...]', lowering.run((total, 'the Hamiltonian')))
        collapses = QuantumCompiler._dissipator_terms(dissipators, lowering, p)
        state = QuantumCompiler._stored_state(system, initial, subsystems, hamiltonian, p, bool(collapses))
        pure = state.ndim == 1
        state_entry = EntrySpec(f'{system.label}.state', 0, state.shape, CC, None, is_complex=True)
        elements = tuple(QuantumCompiler._compiled_element(node) for node in quantum)
        interactions = tuple(
            QuantumCompiler._compiled_interaction(node) for node in nodes if isinstance(node.component, BaseInteraction)
        )
        return QuantumProblemSpec(
            form=FormKind.QUANTUM,
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
    def _parameters(nodes: list[NodeSpec]) -> tuple[list[EntrySpec], list[Array], dict[str, Any]]:
        """Return the parameter entries, initial values, and symbols of every component, per member for groups."""
        entries: list[EntrySpec] = []
        values: list[Array] = []
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
                    LayoutPass._coerce_value(
                        name, LayoutPass._value_of(node, variable), (), declaration.domain, members,
                    ),
                )
                entries.append(EntrySpec(name, sum(item.size for item in entries), shape, declaration.domain, members))
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
        if any(QuantumCompiler._is_quantum(element) for element in system.elements):
            return True
        return any(QuantumCompiler._contains_quantum(child) for child in system.systems)
