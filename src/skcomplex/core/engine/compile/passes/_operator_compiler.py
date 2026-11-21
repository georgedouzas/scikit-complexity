"""Pass that compiles a system of operator elements into an operator problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import inspect
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import qutip
from sage.all__sagemath_symbolics import CC, SR, Expression
from sage.modules.free_module import FreeModule_generic
from sklearn.base import clone

from ....components import EntrySpec, Observable
from ....components.elements import BaseElement, Element
from ....components.elements._element import RESERVED
from ....components.interactions import BaseInteraction
from ....components.laws import BaseLaw
from ....enums import FormKind, VariableKind
from ....errors import IncompatibleLawsError, SkcomplexError, ValidationError
from ....types import Array
from ..._utils import declarations_of, symbol, vector
from ...backends import BaseBackend
from ...problems import BaseProblemSpec, CompiledElementSpec, LayoutSpec
from ...problems._compiled_interaction_spec import _CompiledInteractionSpec
from ...problems._composite_system import _CompositeSystem
from ...problems._declared_element import _DeclaredElement
from ...problems._group_element import _GroupElement
from ...problems._node_spec import _NodeSpec
from ...problems._operator_kernel import OperatorKernel
from ...problems._operator_lowering import OperatorLowering
from ...problems._operator_problem_spec import REAL_TOLERANCE, OperatorProblemSpec
from ...problems._operator_view_spec import _OperatorViewSpec
from ...problems._subsystem_spec import _SubsystemSpec
from ...types import Dissipators, Generators, HilbertSpaceProtocol
from .._base import BaseCompiler
from .._issue_spec import IssueSpec
from .._state_inference import _StateInference
from ._flatten import FlattenPass
from ._hilbert_space_error import HilbertSpaceError
from ._layout import LayoutPass
from ._problem import ProblemPass

DENSE_LIMIT = 1024
HERMITIAN_TOLERANCE = 1e-10
NORM_TOLERANCE = 1e-8
MATRIX_NDIM = 2
SAMPLE_TIME = 0.37
DEFAULT_MAX_DIMENSION = 2**20


class OperatorCompiler(BaseCompiler):
    """A compiler of a system holding operator elements into an operator problem.

    An operator element takes the space of its states, a value whose class names the compiler of the field that runs
    it. A subclass in that field adds the laws of its elements to the generator or the dissipators in `_add_law`, and it
    names the default solver of its problems.

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
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.core.engine.compile import OperatorCompiler
        >>> from skcomplex.systems.biology.common.populations import PopulationElement
        >>> OperatorCompiler().prepare(System(elements=[PopulationElement('rabbits')])).report_.ok
        False
    """

    def __init__(
        self,
        backend: BaseBackend | None = None,
        solver: object = None,
        observables: Sequence[Observable] = (),
        max_dimension: int = DEFAULT_MAX_DIMENSION,
    ) -> None:
        super().__init__(backend, solver, observables)
        self.max_dimension = max_dimension

    def _prepare(self, context: _CompositeSystem | None) -> None:
        """Raise an error for a largest dimension that is not a positive integer.

        Args:
            context:
                The system.

        Raises:
            ValidationError:
                A setting is invalid.
        """
        if not isinstance(self.max_dimension, int) or isinstance(self.max_dimension, bool) or self.max_dimension < 1:
            error_msg = f'The largest dimension is {self.max_dimension!r}. Give a positive integer.'
            raise ValidationError(error_msg)
        super()._prepare(context)

    def _get_issues(self, system: _CompositeSystem) -> list[IssueSpec]:
        """Return the issue of a system the compiler cannot compile or flatten.

        The tree of the system cannot be flattened, or the compiler does not compile it.

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
            return [self._get_issue_of(error, system.label, 'structure', fix)]
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
    def _get_view_of(element: Element) -> _OperatorViewSpec:
        """Return the operator view of an element.

        It holds the name of its state and its Hilbert space.

        Args:
            element:
                The element.

        Returns:
            view:
                The view.

        Raises:
            ValidationError:
                The element takes no Hilbert space, declares no complex state, or has a state whose dimension differs
                from the dimension of its space.
        """
        space = OperatorCompiler._get_hilbert_of(element)
        if space is None:
            error_msg = f'{type(element).__name__} {element.label!r} takes no Hilbert space.'
            raise ValidationError(error_msg)
        state = OperatorCompiler._get_hilbert_state(element)
        if state is None:
            error_msg = (
                f'{type(element).__name__} {element.label!r} takes a Hilbert space but has no complex state. Give its '
                "state complex allowed values, such as `psi: Annotated[object, CC] = 'up'`."
            )
            raise ValidationError(error_msg)
        space._check()
        domain = (
            CC if OperatorCompiler._is_plain(element) else declarations_of(element, VariableKind.STATE)[state].domain
        )
        dimension = domain.dimension() if isinstance(domain, FreeModule_generic) else space.dimension
        if dimension != space.dimension:
            error_msg = (
                f'{type(element).__name__} {element.label!r} has its state {state} over CC^{dimension}, but its '
                f'Hilbert space has dimension {space.dimension}.'
            )
            raise ValidationError(error_msg)
        return _OperatorViewSpec(element, state, space)

    @staticmethod
    def _get_hilbert_of(element: BaseElement) -> HilbertSpaceProtocol | None:
        """Return the Hilbert space an element takes.

        For a group, it is that of a member, and it is `None` for a classical element.

        Args:
            element:
                An element or a group.

        Returns:
            space:
                The first argument of the element that is a Hilbert space, or `None`.
        """
        member = OperatorCompiler._get_member_of(element)
        if member is None:
            return None
        values = [*member.get_params(deep=False).values(), *(getattr(member, 'values', None) or {}).values()]
        return next((value for value in values if isinstance(value, HilbertSpaceProtocol)), None)

    @staticmethod
    def _is_operator(element: object) -> bool:
        """Check whether an element or a group is a subsystem.

        A subsystem takes a Hilbert space.

        Args:
            element:
                An element, a group, or any other component.

        Returns:
            operator:
                Whether the element, or a member of the group, takes a Hilbert space.
        """
        return isinstance(element, BaseElement) and OperatorCompiler._get_hilbert_of(element) is not None

    @staticmethod
    def _get_hilbert_state(element: object) -> str | None:
        """Return the name of the complex state of an element or an element class.

        The state is over `CC` or `CC^dimension`, and the result is `None` for none. A plain element class names it
        with complex allowed values, such as `psi: Annotated[object, CC] = 'up'`, and a plain element may also hold it
        as its only value besides the Hilbert space.

        Args:
            element:
                The element or its class.

        Returns:
            name:
                The name of the state.
        """
        if isinstance(element, Element) and OperatorCompiler._is_plain(element):
            return OperatorCompiler._get_plain_state(element)
        element_type = element if isinstance(element, type) else type(element)
        for name, declaration in declarations_of(element_type, VariableKind.STATE).items():
            if OperatorCompiler._is_complex(declaration.domain):
                return name
        if not issubclass(element_type, Element) or issubclass(element_type, _DeclaredElement):
            return None
        parameters = inspect.signature(element_type, eval_str=True).parameters
        return next(
            (
                name
                for name, parameter in parameters.items()
                if name not in RESERVED
                and OperatorCompiler._is_complex(Element._get_domain_of(parameter.annotation, 0.0))
            ),
            None,
        )

    @staticmethod
    def _is_complex(domain: object) -> bool:
        """Return whether a domain is `CC` or `CC^dimension`."""
        base_ring = getattr(domain, 'base_ring', None)
        return domain is CC or (callable(base_ring) and base_ring() is CC)

    @staticmethod
    def _get_plain_state(element: Element) -> str | None:
        """Return the name of the state in the Hilbert space of a plain element, or `None`.

        It is the value whose allowed values in `Annotated` are complex, such as `psi: Annotated[object, CC] = 'up'`,
        or else the only value besides the Hilbert space, such as `psi` of
        `Element('qubit', values={'psi': 'up', 'hilbert': QUBIT})`.
        """
        values = {
            name: value
            for name, value in _StateInference._get_values(element).items()
            if not isinstance(value, HilbertSpaceProtocol)
        }
        hints = element._get_hints()
        for name, value in values.items():
            annotation = hints.get(name, inspect.Parameter.empty)
            if OperatorCompiler._is_complex(Element._get_domain_of(annotation, value)):
                return name
        return next(iter(values)) if len(values) == 1 else None

    @staticmethod
    def _is_plain(element: object) -> bool:
        """Return whether an element declares no states, so its values say what it holds."""
        if not isinstance(element, Element) or isinstance(element, _DeclaredElement):
            return False
        return not declarations_of(type(element), VariableKind.STATE)

    @staticmethod
    def _get_count(element: object) -> int | None:
        """Return the number of members of a plain element, or `None`."""
        if isinstance(element, Element) and OperatorCompiler._is_plain(element):
            return _StateInference._get_count(element)
        return None

    @staticmethod
    def _get_plain_member(element: Element, member: int, size: int) -> Element:
        """Return one member of a plain element with many members.

        A value given as one entry per member, such as the labels `['up', 'down']` of two spins, gives the member its
        entry, and any other value, such as the Hilbert space, is shared.
        """
        params = element.get_params(deep=False)
        settings: dict[str, object] = {
            'label': f'{element.label}_{member}',
            'size': None,
            'space': None,
            'values': None,
        }
        if element.values:
            settings['values'] = {
                name: OperatorCompiler._get_member_value(value, member, size) for name, value in element.values.items()
            }
        settings |= {
            name: OperatorCompiler._get_member_value(params[name], member, size)
            for name in element._get_hints()
            if name in params
        }
        return clone(element).set_params(**{name: value for name, value in settings.items() if name in params})

    @staticmethod
    def _get_member_of(element: BaseElement) -> Element | None:
        """Return an element itself, or the first member of a group, or `None`.

        It is `None` for any other component.

        Args:
            element:
                An element or a group.

        Returns:
            member:
                The element, built with the values of the first member for a group or a plain element with many
                members.
        """
        count = OperatorCompiler._get_count(element)
        if isinstance(element, Element) and count is not None:
            return OperatorCompiler._get_plain_member(element, 0, count)
        if isinstance(element, Element):
            return element
        if not isinstance(element, _GroupElement) or not isinstance(element.element, type):
            return None
        values = {
            name: OperatorCompiler._get_member_value(value, 0, None) for name, value in (element.values or {}).items()
        }
        return element.element(element.label).set_params(**values)

    @staticmethod
    def _is_operator_node(node: _NodeSpec) -> bool:
        """Check whether a node is an operator element or a group of operator elements."""
        return not isinstance(node.component, BaseInteraction) and OperatorCompiler._is_operator(node.component)

    @staticmethod
    def _get_member_value(value: object, member: int, size: int | None) -> object:
        """Return the value of one member.

        It comes from a per-member sequence or a value shared by every member.

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
            isinstance(value, str | HilbertSpaceProtocol)
            or not isinstance(value, Iterable)
            or np.asarray(value, dtype=object).ndim == 0
        ):
            return value
        items = list(value)
        return items[member] if size is None or len(items) == size else value

    @staticmethod
    def _get_members_of(node: _NodeSpec) -> list[tuple[str, _OperatorViewSpec]]:
        """Return the key and the element of every member of an operator element or group."""
        component = node.as_element
        if isinstance(component, Element) and node.members is None:
            return [(node.path, OperatorCompiler._get_view_of(component))]
        size = node.members or 0
        members = []
        for member in range(size):
            if isinstance(component, Element):
                element = OperatorCompiler._get_plain_member(component, member, size).prepare()
            else:
                values = {
                    name: OperatorCompiler._get_member_value(value, member, size)
                    for name, value in (component.values or {}).items()
                }
                element = component.element(f'{component.label}_{member}').set_params(**values).prepare()
            members.append((f'{node.path}[{member}]', OperatorCompiler._get_view_of(element)))
        return members

    @staticmethod
    def _get_counted(node: _NodeSpec) -> _NodeSpec:
        """Return a node with the number of members of its plain element."""
        count = None if node.interaction else OperatorCompiler._get_count(node.component)
        return node if count is None else dataclasses.replace(node, members=count)

    @staticmethod
    def _check_classical(nodes: list[_NodeSpec]) -> None:
        """Raise an error for an operator system with classical components or states.

        The components are classical elements, or operator elements with states.
        """
        classical = [
            node.path
            for node in nodes
            if not isinstance(node.component, BaseInteraction) and not OperatorCompiler._is_operator_node(node)
        ]
        if classical:
            error_msg = (
                f'The system mixes operator elements with the classical elements {", ".join(classical)}, which no '
                'single form runs. Write the operator and the classical components as separate systems.'
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
                if name != OperatorCompiler._get_hilbert_state(node.component_type)
            }
            if states:
                error_msg = (
                    f'Operator element {node.path} declares the states {", ".join(states)} besides its state in a '
                    'Hilbert space, which is its only state, so declare parameters instead.'
                )
                raise ValidationError(error_msg)

    @staticmethod
    def _check_dimension(nodes: list[_NodeSpec], dimensions: list[int], limit: int) -> None:
        """Raise an error when the Hilbert space dimension exceeds the solver limit."""
        total = math.prod(dimensions)
        if total <= limit:
            return
        parts = []
        for node in nodes:
            if OperatorCompiler._is_operator_node(node):
                members = OperatorCompiler._get_members_of(node)
                parts.append(f'{node.path} of dimension {math.prod(element.dimension for _, element in members)}')
        error_msg = (
            f'The Hilbert space of {", ".join(parts)} has dimension {total}, above the limit max_dimension = {limit}. '
            'Use fewer or smaller subsystems, or raise max_dimension of the operator solver.'
        )
        raise HilbertSpaceError(error_msg)

    @staticmethod
    def _get_parameter_symbols(
        node: _NodeSpec,
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
    def _get_role_namespace(
        node: _NodeSpec,
        label: str,
        members: Mapping[str, tuple[SimpleNamespace, ...]],
        groups: set[str],
    ) -> SimpleNamespace | tuple[SimpleNamespace, ...]:
        """Return the namespace a role binds to.

        It is one member's for an element and every member's for a group.
        """
        path = f'{node.scope}{label}'
        if path not in members:
            error_msg = f'Interaction {node.path} binds a role to {path}, which is not an operator element or group.'
            raise ValidationError(error_msg)
        return members[path] if path in groups else members[path][0]

    def _collect_laws(self, generators: Generators, dissipators: Dissipators, owner: str, items: object) -> None:
        """Add every law a component returns to the generator or the dissipators.

        Args:
            generators:
                The owner and the operator of every generator term found so far.

            dissipators:
                The owner, the collapse operator, and the rate of every dissipator found so far.

            owner:
                The description of the component that returns the laws.

            items:
                The laws the component returns.

        Raises:
            TypeError:
                The component returns something other than a sequence of laws.
        """
        if not isinstance(items, Iterable):
            error_msg = f'{owner} returns a {type(items).__name__}, not a list of laws.'
            raise TypeError(error_msg)
        for law in items:
            self._add_law(law, owner, generators, dissipators)

    def _add_law(self, law: BaseLaw, owner: str, generators: Generators, dissipators: Dissipators) -> None:
        """Add one law to the generator or the dissipators.

        The compiler of a field adds the laws of its elements and passes every other law here.

        Args:
            law:
                The law.

            owner:
                The description of the component that returns the law.

            generators:
                The owner and the Hermitian operator of every generator term found so far.

            dissipators:
                The owner, the collapse operator, and the rate of every dissipator found so far.

        Raises:
            IncompatibleLawsError:
                The compiler adds no law of this class.
        """
        del generators, dissipators
        error_msg = f'{owner} returns the law {law._get_name()}, which {type(self).__name__} cannot compile.'
        raise IncompatibleLawsError(error_msg)

    def _get_interaction_laws(
        self,
        node: _NodeSpec,
        interaction: BaseInteraction,
        *,
        symbols: Mapping[str, object],
        members: Mapping[str, tuple[SimpleNamespace, ...]],
        groups: dict[str, _NodeSpec],
        generators: Generators,
        dissipators: Dissipators,
    ) -> None:
        """Add the laws of an interaction of a node.

        A neighbour interaction adds them once per pair of neighbours of its group.
        """
        parameters = {**OperatorCompiler._get_parameter_symbols(node, symbols, None), 't': symbol('t')}
        if interaction._runs_on_edges():
            role_names = list(declarations_of(interaction, VariableKind.ROLE))
            if len(role_names) != 1:
                error_msg = (
                    f'Neighbour interaction {node.path} declares the roles {role_names}. Declare one role, bound to '
                    'the group whose neighbours it couples.'
                )
                raise ValidationError(error_msg)
            path = f"{node.scope}{interaction._get_binding(role_names[0])}"
            if path not in groups:
                error_msg = f'Neighbour interaction {node.path} binds {path}, which is not an operator group.'
                raise ValidationError(error_msg)
            space = groups[path].space
            if space is None:
                error_msg = f'Neighbour interaction {node.path} binds {path}, which lives in no space.'
                raise ValidationError(error_msg)
            for source, target in space.prepare()._get_neighbor_pairs(interaction._get_layer()):
                namespace = SimpleNamespace(source=members[path][source], target=members[path][target], **parameters)
                self._collect_laws(
                    generators,
                    dissipators,
                    f'{node.path} on the pair ({source}, {target})',
                    interaction._get_laws_of(namespace),
                )
            return
        roles = {
            name: OperatorCompiler._get_role_namespace(node, str(interaction._get_binding(name)), members, set(groups))
            for name in declarations_of(interaction, VariableKind.ROLE)
        }
        self._collect_laws(
            generators,
            dissipators,
            node.path,
            interaction._get_laws_of(SimpleNamespace(**roles, **parameters)),
        )

    @staticmethod
    def _get_terms(lowering: OperatorLowering, operator: Expression, owner: str) -> tuple[OperatorKernel, ...]:
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
    def _check_hermitian(generators: Generators, lowering: OperatorLowering, p: Array) -> None:
        """Raise an error when the generator is not Hermitian at sampled parameter values."""
        size = math.prod(lowering.dimensions)
        rng = np.random.default_rng(0)
        sample = p * (1 + 0.1 * rng.random(p.size)) + 0.01 * rng.random(p.size)

        def asymmetry(terms: Iterable[OperatorKernel]) -> float:
            matrix = OperatorProblemSpec._get_matrix_at(terms, SAMPLE_TIME, sample, size)
            difference = matrix - matrix.conj().T
            return float(abs(difference).max()) if difference.nnz else 0.0

        total = [
            term for owner, operator in generators for term in OperatorCompiler._get_terms(lowering, operator, owner)
        ]
        if asymmetry(total) <= HERMITIAN_TOLERANCE:
            return
        owners = [
            owner for owner, operator in generators if asymmetry(OperatorCompiler._get_terms(lowering, operator, owner))
        ]
        error_msg = (
            f'The generator of {", ".join(owners)} is not Hermitian. Add the adjoint of every non-Hermitian '
            'term, `dagger(term)`, to the law.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _get_dissipator_terms(
        dissipators: Dissipators,
        lowering: OperatorLowering,
        p: Array,
    ) -> tuple[tuple[OperatorKernel, ...], ...]:
        """Return the terms of every collapse operator scaled by the root of its rate.

        The root is the square root.
        """
        collapses = []
        for owner, collapse, rate in dissipators:
            value = SR(rate)
            initial = lowering._get_function(value, owner)(0.0, p) if value.variables() else complex(value)
            if initial.real < 0 or abs(initial.imag) > REAL_TOLERANCE:
                error_msg = f'The dissipator of {owner} has the rate {initial}, which is not a non-negative number.'
                raise ValidationError(error_msg)
            collapses.append(OperatorCompiler._get_terms(lowering, collapse * value.sqrt(), owner))
        return tuple(collapses)

    @staticmethod
    def _get_system_initial(
        system: _CompositeSystem,
        initial: object,
        product: npt.NDArray[np.complex128],
        ground: Callable[[], npt.NDArray[np.complex128]],
    ) -> npt.NDArray[np.complex128]:
        """Return the initial state of the whole system.

        It is the product of the states of the elements unless one is given.
        """
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
        if state.ndim == MATRIX_NDIM and not np.allclose(state, state.conj().T, atol=NORM_TOLERANCE):
            error_msg = f'The initial density matrix of system {system.label!r} is not Hermitian.'
            raise ValidationError(error_msg)
        return state

    @staticmethod
    def _get_namespaces(
        system: _CompositeSystem,
        members: Mapping[str, tuple[SimpleNamespace, ...]],
        groups: set[str],
    ) -> dict[str, SimpleNamespace]:
        """Return the namespace of every system of the tree by path.

        It holds its operator elements and child systems.
        """
        namespaces: dict[str, SimpleNamespace] = {}

        def build(current: _CompositeSystem, scope: str) -> SimpleNamespace:
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
    def _get_subsystems_of(operator_nodes: list[_NodeSpec]) -> dict[str, _SubsystemSpec]:
        """Return the subsystems of the operator elements and groups in tensor order.

        Each comes with its state symbol.
        """
        subsystems: dict[str, _SubsystemSpec] = {}
        for node in operator_nodes:
            for key, element in OperatorCompiler._get_members_of(node):
                state = symbol(f'{element.state}_{OperatorCompiler._get_name_of(key)}', CC)
                subsystems[key] = _SubsystemSpec(key, element, len(subsystems), state, element.get_local_operators())
        return subsystems

    @staticmethod
    def _get_arguments_of(entries: list[EntrySpec], symbols: Mapping[str, Expression]) -> dict[Expression, int]:
        """Return the index in the parameter vector of every parameter symbol."""
        arguments = {}
        for entry in entries:
            components = symbols[entry.name]
            for offset, item in enumerate(components if entry.members is not None else [components]):
                arguments[item] = entry.start + offset
        return arguments

    @staticmethod
    def _get_member_namespaces(
        operator_nodes: list[_NodeSpec],
        subsystems: Mapping[str, _SubsystemSpec],
        symbols: Mapping[str, object],
    ) -> dict[str, tuple[SimpleNamespace, ...]]:
        """Return the namespaces of every operator element and member, by path.

        The namespace of a member holds the symbol of its state, which names its subsystem in `local_operator`, and the
        symbols of its parameters.
        """
        members: dict[str, tuple[SimpleNamespace, ...]] = {}
        for node in operator_nodes:
            namespaces = []
            for member, (key, element) in enumerate(OperatorCompiler._get_members_of(node)):
                index = member if node.members is not None else None
                state = {element.state: subsystems[key].symbol}
                namespace = SimpleNamespace(**state, **OperatorCompiler._get_parameter_symbols(node, symbols, index))
                namespaces.append(namespace)
            members[node.path] = tuple(namespaces)
        return members

    @staticmethod
    def _get_stored_state(
        system: _CompositeSystem,
        initial: object,
        *,
        subsystems: Mapping[str, _SubsystemSpec],
        generator: tuple[OperatorKernel, ...],
        p: Array,
        is_open: bool,
    ) -> npt.NDArray[np.complex128]:
        """Return the initial state as stored.

        It is a vector for a closed pure state and a density matrix otherwise.
        """
        product = np.ones(1, dtype=complex)
        for item in subsystems.values():
            product = np.kron(product, item.element.get_initial_vector())
        size = product.size

        def ground() -> npt.NDArray[np.complex128]:
            matrix = OperatorProblemSpec._get_matrix_at(generator, 0.0, p, size)
            return np.asarray(qutip.Qobj(matrix).groundstate(sparse=size > DENSE_LIMIT)[1].full().ravel())

        state = OperatorCompiler._get_system_initial(system, initial, product, ground)
        return np.outer(state, state.conj()) if is_open and state.ndim == 1 else state

    @staticmethod
    def _get_compiled_element(node: _NodeSpec) -> CompiledElementSpec:
        """Return the compiled element of an operator element or group.

        It has no states in the state vector.
        """
        component = node.as_element
        visual = ProblemPass._get_visual_of(node)
        parameters = {
            name: f'{node.path}.{name}' for name in declarations_of(node.component_type, VariableKind.PARAMETER)
        }
        return CompiledElementSpec(node.path, component, {}, parameters, visual, node.members)

    @staticmethod
    def _get_compiled_interaction(node: _NodeSpec) -> _CompiledInteractionSpec:
        """Return the compiled interaction of an operator system.

        It holds the paths its roles bind to.
        """
        component = node.as_interaction
        roles = {
            name: f'{node.scope}{component._get_binding(name)}'
            for name in declarations_of(component, VariableKind.ROLE)
        }
        return _CompiledInteractionSpec(node.path, component, roles)

    def _compile_operator(
        self,
        system: _CompositeSystem,
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
                of the generator at the initial parameters, or `None` for the product of the initial states of the
                elements.

            max_dimension:
                The largest dimension of the Hilbert space, the limit of the operator solver.

        Returns:
            problem:
                The operator problem.

        Raises:
            IncompatibleLawsError:
                The system holds classical elements or laws the compiler does not add.

            HilbertSpaceError:
                The dimension of the Hilbert space exceeds the limit.

            ValidationError:
                A coefficient reads a symbol other than parameters and time, the generator is not Hermitian, a rate is
                negative, or an initial state is not normalised.
        """
        time = symbol('t')
        nodes = [OperatorCompiler._get_counted(node) for node in FlattenPass._flatten_system(system)]
        OperatorCompiler._check_classical(nodes)
        operator_nodes = [node for node in nodes if OperatorCompiler._is_operator_node(node)]
        subsystems = OperatorCompiler._get_subsystems_of(operator_nodes)
        dimensions = tuple(item.element.dimension for item in subsystems.values())
        OperatorCompiler._check_dimension(nodes, list(dimensions), max_dimension)
        entries, values, symbols = OperatorCompiler._get_parameters(nodes)
        p = np.concatenate(values) if values else np.zeros(0)
        arguments = OperatorCompiler._get_arguments_of(entries, symbols)
        lowering = OperatorLowering(time, subsystems, dimensions, arguments).prepare()
        groups = {node.path: node for node in operator_nodes if node.members is not None}
        generators: Generators = []
        dissipators: Dissipators = []
        members = OperatorCompiler._get_member_namespaces(operator_nodes, subsystems, symbols)
        for node in nodes:
            if isinstance(node.component, BaseInteraction):
                self._get_interaction_laws(
                    node,
                    node.component,
                    symbols=symbols,
                    members=members,
                    groups=groups,
                    generators=generators,
                    dissipators=dissipators,
                )
        OperatorCompiler._check_hermitian(generators, lowering, p)
        total = sum((item for _, item in generators), SR(0))
        generator = OperatorCompiler._get_terms(lowering, total, 'the generator')
        collapses = OperatorCompiler._get_dissipator_terms(dissipators, lowering, p)
        state = OperatorCompiler._get_stored_state(
            system,
            initial,
            subsystems=subsystems,
            generator=generator,
            p=p,
            is_open=bool(collapses),
        )
        pure = state.ndim == 1
        state_entry = EntrySpec(f'{system.label}.state', 0, state.shape, CC, None, is_complex=True)
        elements = tuple(OperatorCompiler._get_compiled_element(node) for node in operator_nodes)
        interactions = tuple(
            OperatorCompiler._get_compiled_interaction(node)
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
            generator=generator,
            dissipators=collapses,
            pure=pure,
            namespaces=OperatorCompiler._get_namespaces(system, members, set(groups)),
            members=members,
            lowering=lowering,
            generator_operator=total,
        )

    @staticmethod
    def _get_name_of(path: str) -> str:
        """Return a path, such as `chain[3]`, as a valid part of a symbol name.

        An example is `chain_3`.
        """
        return '_'.join(part for part in re.split(r'\W+', path) if part)

    @staticmethod
    def _get_parameters(nodes: list[_NodeSpec]) -> tuple[list[EntrySpec], list[Array], dict[str, Expression]]:
        """Return the parameter entries, initial values, and symbols of every component.

        For groups, they are given per member.
        """
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
                        LayoutPass._get_value_of(node, variable),
                        (),
                        declaration.domain,
                        members,
                    ),
                )
                entries.append(EntrySpec(name, sum(item.size for item in entries), shape, declaration.domain, members))
                base = f'{variable}_{OperatorCompiler._get_name_of(node.path)}'
                if members is None:
                    symbols[name] = symbol(base, declaration.domain)
                else:
                    symbols[name] = vector(
                        [symbol(f'{base}_{member}', declaration.domain) for member in range(members)],
                    )
        return entries, values, symbols

    @staticmethod
    def _contains_operator(system: _CompositeSystem) -> bool:
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
