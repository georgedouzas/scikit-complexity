"""Compile a system into a problem of one mathematical form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import itertools
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import SR, I, matrix

from ..base import (
    BackendError,
    ComponentKind,
    Domain,
    Expression,
    Form,
    IncompatibleLawsError,
    LabelError,
    RoleError,
    SkcomplexError,
    SymbolicVector,
    ValidationError,
    Variable,
    VariableKind,
    VisualSpec,
    components_of,
    declarations_of,
    domain_of,
    real_components_of,
    symbol,
    symbols_of,
    vector,
    wrap_angle,
)
from ..model import (
    Aggregate,
    Auxiliary,
    Boundary,
    Call,
    ContinuousSpace,
    Drive,
    Element,
    Emission,
    Energy,
    Equation,
    Event,
    Extremal,
    Field,
    Flux,
    FractionalNoise,
    Group,
    Interaction,
    Invariant,
    Jump,
    LatticeSpace,
    Law,
    LocalRule,
    Match,
    Mesh,
    Move,
    NetworkSpace,
    Noise,
    Observable,
    Rate,
    Reduction,
    Relaxation,
    Rule,
    Sample,
    Sampler,
    Space,
    Symbols,
    System,
    Transfer,
    Transition,
    Update,
    count,
    either,
    has_reductions,
    maximum,
    mean,
    minimum,
    mod,
    reductions_in,
    total,
    variance,
    where,
)
from ..model._laws import NOISE_CONVENTIONS, Renewal
from ._backends import Backend, resolve_backend
from ._complex import modulus_form, split_law
from ._fields import DiffusiveLimit, FieldLowering, has_operators
from ._flows import StockFlowPass, quantities_of
from ._hybrid import (
    Breakpoints,
    CompiledEvent,
    Delay,
    Scalars,
    breakpoints_of,
    flat_sources,
    scalar_function,
    state_conditions_of,
)
from ._jacobians import ExactJacobian
from ._kernels import Kernel, Output, PairSearch, Reference, Source
from ._lattices import (
    LatticeContext,
    LatticeState,
    LatticeStep,
    compile_drive,
    compile_extremal,
    compile_monte_carlo,
    compile_relaxation,
    compile_rule,
)
from ._layout import Entry, Layout
from ._lowering import symbolic_components
from ._members import MemberShifts
from ._observables import _join, _tree, observables_of, with_observables
from ._printing import numpy_function
from ._problem import (
    Accelerated,
    CompiledAgents,
    CompiledElement,
    CompiledInteraction,
    FractionalRows,
    MarkovMatrix,
    Partition,
    Problem,
    Vector,
    VectorFunction,
)
from ._quantities import StockFlow
from ._quantum import DEFAULT_MAX_DIMENSION, compile_quantum, contains_quantum
from ._reductions import CompiledReduction, Read, ReductionPass
from ._renewals import CompiledRenewal, compile_renewal
from ._rewriting import compile_rewriting, is_rewriting
from ._splitting import LinearTerm, linear_terms
from ._stochastic import DrawLowering, MatrixLaw, check_draw_placement, compile_matrix, markov_laws, stochastic_update
from ._tree import Node, coerce_value, flatten_system, shape_of
from ._volumes import FiniteVolume, compile_volumes

FORMS: dict[frozenset[type], Form] = {
    frozenset(): Form.ODE,
    frozenset({Rate}): Form.ODE,
    frozenset({Noise}): Form.SDE,
    frozenset({Rate, Noise}): Form.SDE,
    frozenset({Jump}): Form.JUMP,
    frozenset({Update}): Form.DISCRETE,
    frozenset({Rule}): Form.AGENT,
    **{
        frozenset(kinds): Form.LATTICE
        for count in (1, 2, 3, 4)
        for kinds in itertools.combinations((LocalRule, Drive, Relaxation, Extremal), count)
    },
    frozenset({Energy, Move}): Form.MONTE_CARLO,
    frozenset({Energy, Move, Sampler}): Form.MONTE_CARLO,
    frozenset({Renewal}): Form.RENEWAL,
}
FORMLESS = frozenset({Boundary, Event, Flux})
CONTINUOUS = frozenset({Rate, Noise})
ASSIGNED_KINDS = (Update, Boundary)
MEMBER_REFERENCE = re.compile(r'^(?P<path>.+)\[(?P<index>-?\d+)\]$')
LATTICE_LAWS = (LocalRule, Drive, Relaxation, Extremal, Energy, Move, Sampler)
KERNEL_FREE = (*LATTICE_LAWS, Flux, Transfer, Auxiliary, Invariant)
DECLARED_FREE = (LocalRule, Drive, Relaxation, Extremal, Flux, Transfer, Auxiliary, Invariant)
AGENT_VIEW_NAMES = frozenset(
    {
        'id',
        'group',
        'group_label',
        'location',
        'alive',
        'neighbor_cells',
        'neighbors',
        'agents_here',
        'move_to',
        'remove',
        'create',
        'displacement_to',
        'distance_to',
    },
)
MESA_AGENT_NAMES = frozenset({'position', 'pos', 'cell', 'unique_id', 'model'})
AGENT_RESERVED_NAMES = AGENT_VIEW_NAMES | MESA_AGENT_NAMES


def _renamed(value: Expression, mapping: dict[Expression, Expression]) -> Expression:
    """Return the renamed symbol of a variable, or a complex value with its real and imaginary symbols renamed."""
    return mapping[value] if value.operator() is None else value.subs(mapping)


@dataclass
class _Block:
    """Variables of a component with the symbols of one member and the components they stand for."""

    node: Node
    symbols: dict[str, Expression | SymbolicVector] = field(default_factory=dict)
    entries: dict[str, Entry] = field(default_factory=dict)
    references: dict[Expression, Reference] = field(default_factory=dict)
    per: set[str] = field(default_factory=set)

    @property
    def rows(self) -> int:
        """Number of members, 1 for a single element or an interaction."""
        return self.node.members or 1

    def _namespace(self, mapping: dict[Expression, Expression]) -> Symbols:
        """Return the namespace of the block's variables in renamed symbols, with tuples for per-entry parameters."""
        symbols: dict[str, Any] = {}
        for name, value in self.symbols.items():
            if not isinstance(value, SymbolicVector):
                symbols[name] = _renamed(value, mapping)
            elif name in self.per:
                symbols[name] = tuple(_renamed(item, mapping) for item in value)
            else:
                symbols[name] = vector(_renamed(item, mapping) for item in value)
        return Symbols(**symbols)

    def namespace(self, source: Source, alias: str = '') -> tuple[Symbols, dict[Expression, Reference]]:
        """Return the namespace of the block's symbols and their references for one way of reading them.

        Args:
            source:
                How the block's components are read.

            alias:
                A suffix that gives the block's symbols distinct names, used for the second member of a pair.

        Returns:
            namespace:
                The namespace and the references of its symbols.
        """
        mapping = {item: symbol(f'{item}{alias}', domain_of(item)) for item in self.references}
        references = {
            mapping[item]: Reference(reference.entry, reference.is_state, reference.column, source)
            for item, reference in self.references.items()
        }
        return self._namespace(mapping), references

    def member_namespace(self, index: int) -> tuple[Symbols, dict[Expression, Reference]]:
        """Return the namespace of one member of a group, read like a single element, and the references of its symbols.

        Args:
            index:
                The index of the member.

        Returns:
            namespace:
                The namespace and the references of its symbols, whose columns count from the group's first component.
        """
        mapping = {item: symbol(f'{item}__m{index}', domain_of(item)) for item in self.references}
        references = {
            mapping[item]: Reference(
                reference.entry,
                reference.is_state,
                index * reference.entry.columns + reference.column,
                Source.SINGLE,
            )
            for item, reference in self.references.items()
        }
        return self._namespace(mapping), references


def _per_entry_shape(node: Node, variable: str, declaration: Variable) -> tuple[int, ...]:
    """Return the shape of a parameter with one value per entry of a variadic role, after checking its length."""
    roles = declarations_of(node.component_type, VariableKind.ROLE)
    role = declaration.per
    if role not in roles or not roles[role].variadic:
        error_msg = (
            f'{node.path}.{variable} is declared per entry of {role!r}, which is not a variadic role of '
            f'{node.component_type.__name__}. Declare the role with `role(variadic=True)`.'
        )
        raise ValidationError(error_msg)
    entries = len(getattr(node.component, role))
    value = node.value_of(variable)
    given = len(value) if isinstance(value, list | tuple | np.ndarray) and np.ndim(value) else None
    if given is not None and given != entries:
        error_msg = (
            f'Interaction {node.path} gives its parameter {variable!r} {given} values, but its role {role!r} binds '
            f'{entries} entries. Give one value per entry.'
        )
        raise ValidationError(error_msg)
    return (entries,)


@dataclass
class _Declarations:
    """Entries, initial values, and blocks collected from the components."""

    time: Expression
    states: list[Entry] = field(default_factory=list)
    parameters: list[Entry] = field(default_factory=list)
    initial_states: list[Vector] = field(default_factory=list)
    initial_parameters: list[Vector] = field(default_factory=list)
    blocks: dict[str, _Block] = field(default_factory=dict)
    ends: dict[bool, int] = field(default_factory=lambda: {True: 0, False: 0})

    def _declare(self, block: _Block, variable: str, declaration: Variable, is_state: bool) -> None:
        """Add one variable of a component to its block."""
        node = block.node
        entries, values = (self.states, self.initial_states) if is_state else (self.parameters, self.initial_parameters)
        name = f'{node.path}.{variable}'
        if declaration.per is None:
            member_shape = shape_of(name, declaration, node.space)
        else:
            member_shape = _per_entry_shape(node, variable, declaration)
            block.per.add(variable)
        members = None if node.is_agent else node.members
        shape = member_shape if members is None else (members, *member_shape)
        values.append(coerce_value(name, node.value_of(variable), member_shape, declaration.domain, members))
        is_complex = declaration.domain is Domain.COMPLEX
        entry = Entry(name, self.ends[is_state], shape, declaration.domain, members, is_complex)
        entries.append(entry)
        self.ends[is_state] += entry.size
        base = f'{variable}_{re.sub(r"[^0-9A-Za-z]", "_", node.path)}'
        names = [f'{base}_{index}' for index in range(member_shape[0])] if member_shape else [base]
        columns = [f'{item}_{part}' for item in names for part in ('re', 'im')] if is_complex else names
        domain = Domain.REAL if is_complex else declaration.domain
        components = [symbol(item, domain) for item in columns]
        values_of = (
            [real + I * imaginary for real, imaginary in itertools.batched(components, 2)] if is_complex else components
        )
        block.symbols[variable] = vector(values_of) if member_shape else values_of[0]
        block.entries[variable] = entry
        for column, item in enumerate(components):
            block.references[item] = Reference(entry, is_state, column)

    def declare_state(
        self,
        name: str,
        initial: Vector,
        domain: Domain,
        shape: tuple[int, ...] = (),
        members: int | None = None,
    ) -> Entry:
        """Add a hidden state after every state declared so far and return its entry.

        Args:
            name:
                The path of the state.

            initial:
                The flat initial value.

            domain:
                The domain of the state.

            shape:
                The shape of the state.

            members:
                The number of members of a group state, or `None`.

        Returns:
            entry:
                The entry of the state.
        """
        entry = Entry(name, self.ends[True], shape, domain, members)
        self.states.append(entry)
        self.ends[True] += entry.size
        self.initial_states.append(initial)
        return entry

    @cached_property
    def groups(self) -> frozenset[str]:
        """Paths of the declared groups, indexed once every component is declared."""
        return frozenset(path for path, block in self.blocks.items() if block.node.members is not None)

    @cached_property
    def places(self) -> dict[str, Field]:
        """Fields by the name of each of their entries, indexed once every component is declared."""
        return {
            entry.name: block.node.component
            for block in self.blocks.values()
            if isinstance(block.node.component, Field)
            for entry in block.entries.values()
        }

    def declare_component(self, node: Node) -> _Block:
        """Add the states and parameters of a component and return its block.

        Args:
            node:
                The component with its place in the tree.

        Returns:
            block:
                The block of the component.
        """
        block = _Block(node)
        states = {} if node.is_agent else declarations_of(node.component_type, VariableKind.STATE)
        for variable, declaration in states.items():
            self._declare(block, variable, declaration, is_state=True)
        for variable, declaration in declarations_of(node.component_type, VariableKind.PARAMETER).items():
            self._declare(block, variable, declaration, is_state=False)
        self.blocks[node.path] = block
        return block


@dataclass(frozen=True)
class _SetSymbol:
    """Symbol of one element of an element set, with the same symbol of every element of the set."""

    canonical: Expression
    references: tuple[Reference, ...]
    label: str


@dataclass
class _KernelSpec:
    """Namespace, references, and evaluation indices that the laws of one component are lowered with."""

    owner: str
    namespace: Symbols
    references: dict[Expression, Reference]
    rows: int
    pairs: PairSearch | None = None
    roles: dict[str, str | tuple[str, ...]] = field(default_factory=dict)
    group: str | None = None
    groups: dict[str, int] = field(default_factory=dict)
    sets: dict[Expression, _SetSymbol] = field(default_factory=dict)
    set_index: int = 0
    aligned: tuple[str, ...] = ()


@dataclass
class _Lowered:
    """Kernels of every law kind with the extra data of jumps."""

    kernels: dict[type, list[Kernel]] = field(default_factory=lambda: defaultdict(list))
    jump_changes: dict[int, list[tuple[Reference, float]]] = field(default_factory=dict)
    hurst: dict[int, tuple[str, object]] = field(default_factory=dict)


def _check_symbol_names(blocks: dict[str, _Block]) -> None:
    """Raise an error when two variables produce the same symbol name."""
    seen: dict[str, str] = {}
    for block in blocks.values():
        for item, reference in block.references.items():
            name = str(item)
            if name in seen and seen[name] != reference.entry.name:
                error_msg = (
                    f'{reference.entry.name} and {seen[name]} produce the same symbol {name}. Rename one '
                    'of their labels or variables.'
                )
                raise LabelError(error_msg)
            seen[name] = reference.entry.name


def _resolve_path(node: Node, role: str, label: object, elements: dict[str, _Block]) -> _Block:
    """Return the block of an element or group that a role of an interaction binds to by its label."""
    path = f'{node.scope}{label}' if isinstance(label, str) else None
    if path not in elements:
        where = node.scope.rstrip('/') or 'the root system'
        error_msg = (
            f'Interaction {node.path} binds the role {role!r} to {label!r}, which is not an element or group path '
            f'relative to {where}.'
        )
        raise RoleError(error_msg)
    return elements[path]


def _resolve_role(node: Node, role: str, elements: dict[str, _Block]) -> _Block:
    """Return the block of the element or group bound to a role of an interaction."""
    return _resolve_path(node, role, getattr(node.component, role), elements)


def _resolve_member(node: Node, role: str, reference: str, elements: dict[str, _Block]) -> tuple[_Block, int] | None:
    """Return the group block and the member index of a member reference such as `sites[3]`, or `None` for a path."""
    match = MEMBER_REFERENCE.match(reference)
    if match is None or f'{node.scope}{reference}' in elements:
        return None
    block = _resolve_path(node, role, match['path'], elements)
    index, size = int(match['index']), block.node.members
    if size is None or not 0 <= index < size:
        error_msg = (
            f'Interaction {node.path} binds {reference!r} to its role {role!r}, but {block.node.path} holds '
            f'{size or "no"} members. Give a member index from 0 to the group size minus 1.'
        )
        raise RoleError(error_msg)
    return block, index


def _resolve_entry(node: Node, role: str, entry: object, elements: dict[str, _Block]) -> tuple[_Block, int | None]:
    """Return the block of one entry of a variadic role, with the member index of a member reference."""
    member = _resolve_member(node, role, entry, elements) if isinstance(entry, str) else None
    if member is not None:
        return member
    block = _resolve_path(node, role, entry, elements)
    if block.node.members is not None:
        error_msg = (
            f'Interaction {node.path} binds the group {block.node.path} as an entry of its variadic role {role!r}. '
            f'Bind single elements or members of the group, such as {entry}[0].'
        )
        raise RoleError(error_msg)
    return block, None


def _resolve_entries(
    node: Node,
    role: str,
    entries: object,
    elements: dict[str, _Block],
) -> list[tuple[_Block, int | None]]:
    """Return the blocks and member indices of the entries of a variadic role, in the given order."""
    if not isinstance(entries, list | tuple):
        error_msg = (
            f'Interaction {node.path} binds its variadic role {role!r} to {entries!r}. Bind a list of element paths '
            'and group member references.'
        )
        raise RoleError(error_msg)
    return [_resolve_entry(node, role, entry, elements) for entry in entries]


def _entry_path(block: _Block, index: int | None) -> str:
    """Return the path of an entry of a variadic role, with the member index of a member reference."""
    return block.node.path if index is None else f'{block.node.path}[{index}]'


def _entry_namespace(block: _Block, index: int | None) -> tuple[Symbols, dict[Expression, Reference]]:
    """Return the namespace of an entry of a variadic role, an element or one member of a group."""
    return block.namespace(Source.SINGLE) if index is None else block.member_namespace(index)


def _resolve_set(node: Node, role: str, labels: Sequence[str], elements: dict[str, _Block]) -> list[_Block]:
    """Return the blocks of the single elements of one type that a role binds to as an element set."""
    blocks = [_resolve_path(node, role, label, elements) for label in labels]
    grouped = [block.node.path for block in blocks if block.node.members is not None]
    if grouped or not blocks:
        error_msg = (
            f'Interaction {node.path} binds the role {role!r} to a set that holds the groups {", ".join(grouped)}. '
            'An element set lists single elements of one type, and a role binds to a group by its path.'
        )
        raise RoleError(error_msg)
    types = list(dict.fromkeys(block.node.component_type.__name__ for block in blocks))
    if len(types) > 1:
        error_msg = (
            f'Interaction {node.path} binds the role {role!r} to elements of the types {" and ".join(types)}. An '
            'element set holds elements of one type.'
        )
        raise RoleError(error_msg)
    return blocks


def _member_spec(block: _Block, time: Expression) -> _KernelSpec:
    """Return the kernel spec of the intrinsic laws of an element or group."""
    namespace, references = block.namespace(Source.MEMBER)
    group = block.node.path if block.node.members is not None else None
    return _KernelSpec(block.node.path, Symbols(t=time, **vars(namespace)), references, block.rows, group=group)


@dataclass(frozen=True)
class _Observed:
    """Value of an observable that a role reads at the current state, with the references of its symbols."""

    namespace: Symbols
    references: dict[Expression, Reference]


type Observe = Callable[[Node, object], _Observed | None]


def _set_symbols(members: list[_Block], index: int) -> dict[Expression, _SetSymbol]:
    """Return the symbols of one element of a set with the same symbol of every element."""
    label = ', '.join(member.node.path for member in members)
    symbols = {}
    for variable, value in members[index].symbols.items():
        for column, item in enumerate(real_components_of(value)):
            references = tuple(
                member.references[real_components_of(member.symbols[variable])[column]] for member in members
            )
            symbols[item] = _SetSymbol(real_components_of(members[0].symbols[variable])[column], references, label)
    return symbols


REDUCERS: dict[str, Callable[..., Expression]] = {
    'sum': total,
    'mean': mean,
    'max': maximum,
    'min': minimum,
    'var': variance,
}

type Declared = dict[str, tuple[str, Observable]]


def _is_lowerable(scope: str, observable: Observable, nodes: dict[str, Node]) -> bool:
    """Check whether laws can read an observable at the current state, as an expression or a symbolic reduction."""
    if observable.group is None:
        return True
    node = nodes.get(_join(scope, observable.group))
    return node is not None and not node.is_agent and isinstance(observable.reduce, str)


def _source_of(block: _Block) -> Source:
    """Return how laws read the variables of a block from outside it, per member for a group."""
    return Source.MEMBER if block.node.members is not None else Source.SINGLE


def _reduced_value(path: str, observable: Observable, block: _Block) -> object:
    """Return an observable that reduces a group as a reduction over the group's member symbols."""
    namespace, _ = block.namespace(_source_of(block))
    if observable.reduce == 'count':
        return count(over=namespace)
    if observable.variable is not None:
        if observable.variable not in block.symbols:
            error_msg = f'Observable {path} reduces {observable.variable}, which {block.node.path} does not declare.'
            raise ValidationError(error_msg)
        value: Any = block.symbols[observable.variable]
    elif observable.expression is None:
        error_msg = f'Observable {path} reduces {block.node.path} with {observable.reduce!r} but names no variable.'
        raise ValidationError(error_msg)
    else:
        value = observable.expression(namespace) if callable(observable.expression) else observable.expression
    reducer = REDUCERS[str(observable.reduce)]
    if isinstance(value, SymbolicVector):
        return vector(reducer(item, over=namespace) for item in value)
    return reducer(value, over=namespace)


def _lowerable_observables(system: System, nodes: list[Node]) -> Declared:
    """Return every observable of a system tree that laws can read at the current state, by path.

    An expression of a system's symbols is lowerable when every observable declared before it in its system is.
    """
    by_path = {node.path: node for node in nodes}
    lowerable: Declared = {}
    blocked: set[str] = set()
    for scope, observable in observables_of(system):
        if scope in blocked or not _is_lowerable(scope, observable, by_path):
            blocked.add(scope)
            continue
        lowerable[_join(scope, observable.label)] = (scope, observable)
    return lowerable


@dataclass
class _Observer:
    """Values of the observables that laws read at the current state, built from the blocks of a system."""

    declarations: _Declarations
    live: Declared
    values: dict[str, tuple[object, dict[Expression, Reference]]] = field(default_factory=dict)

    def _system_value(
        self,
        path: str,
        scope: str,
        observable: Observable,
    ) -> tuple[object, dict[Expression, Reference]]:
        """Return an expression of the symbols of a system's elements, with earlier observables by label."""
        prefix = f'{scope}/' if scope else ''
        items: dict[str, Any] = {}
        references: dict[Expression, Reference] = {}
        for block in self.declarations.blocks.values():
            for variable, entry in block.entries.items():
                if entry.members is None and entry.name.startswith(prefix):
                    items[entry.name[len(prefix) :]] = block.symbols[variable]
                    references.update(
                        {item: block.references[item] for item in real_components_of(block.symbols[variable])},
                    )
        namespace = _tree(items)
        for other, (other_scope, earlier) in self.live.items():
            if other == path:
                break
            if other_scope == scope:
                value, other_references = self.value(other, other_scope, earlier)
                setattr(namespace, earlier.label, value)
                references.update(other_references)
        expression = observable.expression(namespace) if callable(observable.expression) else observable.expression
        return expression, references

    def value(self, path: str, scope: str, observable: Observable) -> tuple[object, dict[Expression, Reference]]:
        """Return the value of an observable with the references of the symbols it reads."""
        if path in self.values:
            return self.values[path]
        if observable.group is None:
            self.values[path] = self._system_value(path, scope, observable)
            return self.values[path]
        block = self.declarations.blocks[_join(scope, observable.group)]
        _, references = block.namespace(_source_of(block))
        self.values[path] = (_reduced_value(path, observable, block), references)
        return self.values[path]

    def __call__(self, node: Node, label: object) -> _Observed | None:
        """Return the namespace of the observable a role binds to, or `None` for a role bound to no live observable."""
        path = f'{node.scope}{label}' if isinstance(label, str) else None
        if path not in self.live:
            return None
        scope, observable = self.live[path]
        value, references = self.value(path, scope, observable)
        return _Observed(Symbols(value=value), dict(references))


def _interaction_specs(
    block: _Block,
    elements: dict[str, _Block],
    time: Expression,
    observe: Observe,
) -> list[_KernelSpec]:
    """Return the kernel specs of an interaction, one per element of its element sets, or else one."""
    node = block.node
    declared = declarations_of(node.component, VariableKind.ROLE)
    labels = {role: getattr(node.component, role) for role in declared}
    variadic = {
        role: _resolve_entries(node, role, labels[role], elements)
        for role, declaration in declared.items()
        if declaration.variadic
    }
    sets = {
        role: _resolve_set(node, role, label, elements)
        for role, label in labels.items()
        if role not in variadic and isinstance(label, list | tuple)
    }
    observed = {
        role: found
        for role, label in labels.items()
        if role not in sets and role not in variadic and (found := observe(node, label)) is not None
    }
    single_members = {
        role: member
        for role, label in labels.items()
        if role not in sets and role not in observed and role not in variadic and isinstance(label, str)
        if (member := _resolve_member(node, role, label, elements)) is not None
    }
    bound = {
        role: _resolve_role(node, role, elements)
        for role in labels
        if role not in sets and role not in observed and role not in variadic and role not in single_members
    }
    sizes = {len(members) for members in sets.values()}
    if len(sizes) > 1:
        error_msg = f'Interaction {node.path} binds roles to element sets of different sizes. Give the sets one size.'
        raise RoleError(error_msg)
    groups = {item.node.path: item.rows for item in bound.values() if item.node.members is not None}
    group = next(iter(groups)) if len(groups) == 1 else None
    paths: dict[str, str | tuple[str, ...]] = {role: item.node.path for role, item in bound.items()}
    paths.update({role: tuple(member.node.path for member in members) for role, members in sets.items()})
    paths.update({role: f'{node.scope}{labels[role]}' for role in observed})
    paths.update({role: tuple(_entry_path(*entry) for entry in entries) for role, entries in variadic.items()})
    paths.update({role: _entry_path(*member) for role, member in single_members.items()})
    specs = []
    for index in range(next(iter(sizes), 1)):
        parameters, references = block.namespace(Source.SINGLE)
        namespaces: dict[str, Symbols | tuple[Symbols, ...]] = {}
        set_symbols: dict[Expression, _SetSymbol] = {}
        for role, members in sets.items():
            namespaces[role], role_references = members[index].namespace(Source.SINGLE)
            references.update(role_references)
            set_symbols.update(_set_symbols(members, index))
        for role, value in observed.items():
            namespaces[role] = value.namespace
            references.update(value.references)
        for role, entries in variadic.items():
            read = [_entry_namespace(*entry) for entry in entries]
            namespaces[role] = tuple(namespace for namespace, _ in read)
            for _, entry_references in read:
                references.update(entry_references)
        for role, (member_block, position) in single_members.items():
            namespaces[role], role_references = member_block.member_namespace(position)
            references.update(role_references)
        for role, item in bound.items():
            namespaces[role], role_references = item.namespace(
                Source.MEMBER if item.node.members is not None else Source.SINGLE,
            )
            references.update(role_references)
        specs.append(
            _KernelSpec(
                node.path,
                Symbols(t=time, **vars(parameters), **namespaces),
                references,
                groups[group] if group is not None else 1,
                roles=paths,
                group=group,
                groups=groups,
                sets=set_symbols,
                set_index=index,
            ),
        )
    return specs


def _outside_reductions(value: object) -> set[Expression]:
    """Return the symbols of a law value that occur outside its reductions."""
    expressions = [item for item in components_of(value) if isinstance(item, Expression)]
    found = [reduction.call for item in expressions for reduction in reductions_in(item)]
    calls, operators = {str(call) for call in found}, {call.operator() for call in found}
    symbols: set[Expression] = set()
    stack = list(expressions)
    while stack:
        item = stack.pop()
        if item.operator() is None:
            symbols |= set(item.variables())
        elif item.operator() not in operators or str(item) not in calls:
            stack.extend(item.operands())
    return symbols


def _check_aligned(spec: _KernelSpec, groups: list[str]) -> None:
    """Raise an error when the groups an interaction reads member by member differ in size."""
    sizes = {group: spec.groups[group] for group in groups}
    if len(set(sizes.values())) <= 1:
        return
    described = ', '.join(f'{group} of {size} members' for group, size in sizes.items())
    error_msg = (
        f'Interaction {spec.owner} reads the groups {described} member by member, which needs groups of one size. '
        "Read the other groups through reductions, or couple every pair of members with an interaction over 'pairs'."
    )
    raise RoleError(error_msg)


def _settle_rows(spec: _KernelSpec, laws: list[Law]) -> None:
    """Set the groups an interaction is evaluated over from the member symbols its laws read outside reductions.

    Laws without reductions keep the group their one group role binds to, so a single element bound next to it
    receives the sum over the members. Groups of one size that laws read outside reductions are read member by
    member, row `i` of each together.
    """
    values = [law.rate if isinstance(law, Jump) else getattr(law, 'value', None) for law in laws]
    if not spec.groups or (len(spec.groups) == 1 and not any(has_reductions(value) for value in values)):
        return
    read: set[Expression] = set()
    for law, value in zip(laws, values, strict=True):
        read |= _outside_reductions(value)
        read |= set(components_of(getattr(law, 'target', None)))
    owners = {
        reference.entry.name.rpartition('.')[0]
        for item in read
        if (reference := spec.references.get(item)) is not None and reference.source is Source.MEMBER
    }
    groups = sorted(owners & set(spec.groups))
    _check_aligned(spec, groups)
    spec.group = next(iter(groups), None)
    spec.aligned = tuple(groups)
    spec.rows = spec.groups[spec.group] if spec.group is not None else 1


def _space_period(node: Node) -> float | None:
    """Return the side of the periodic box that a pair interaction measures minimum image distances in, or `None`."""
    space = node.space
    if not isinstance(space, ContinuousSpace) or space.boundary != 'periodic':
        return None
    if len(set(space.extent)) > 1:
        error_msg = (
            f'Pair interaction {node.path} lives in the periodic box {space.extent}, whose sides differ. Pair searches '
            'measure minimum image distances in boxes with equal sides.'
        )
        raise ValidationError(error_msg)
    return space.extent[0]


def _pair_spec(block: _Block, elements: dict[str, _Block], time: Expression) -> _KernelSpec:
    """Return the kernel spec of a pair interaction that is evaluated once per pair within its cutoff."""
    node = block.node
    interaction: Interaction = node.component
    first, second = _resolve_role(node, 'first', elements), _resolve_role(node, 'second', elements)
    position = getattr(type(interaction), 'position', 'x')
    for bound in (first, second):
        entry = bound.entries.get(position)
        if entry is None or not entry.member_shape:
            error_msg = f'Pair interaction {node.path} needs the vector state {position!r} in {bound.node.path}.'
            raise ValidationError(error_msg)
    parameters, references = block.namespace(Source.SINGLE)
    first_namespace, first_references = first.namespace(Source.FIRST)
    second_namespace, second_references = second.namespace(Source.SECOND, alias='_pair')
    references.update(first_references)
    references.update(second_references)
    period = getattr(interaction, 'period', None)
    period = period if period is not None else _space_period(node)
    pairs = PairSearch(first.entries[position], second.entries[position], block.entries['cutoff'], period)
    namespace = Symbols(t=time, **vars(parameters), first=first_namespace, second=second_namespace)
    paths: dict[str, str | tuple[str, ...]] = {'first': first.node.path, 'second': second.node.path}
    return _KernelSpec(node.path, namespace, references, 0, pairs, roles=paths)


def _state_name(spec: _KernelSpec, target: Expression) -> str:
    """Return the dotted name of the state a target symbol of a namespace stands for, or the symbol itself."""
    reference = spec.references.get(target)
    return reference.entry.name if reference is not None else str(target)


def _unpacked(laws: object) -> object:
    """Return laws with jumps on clocks of waiting times as renewals and samples as their energy, move, and sampler."""
    if not isinstance(laws, list | tuple):
        return laws
    unpacked: list[object] = []
    for law in laws:
        if isinstance(law, Jump) and law.renewal is not None:
            unpacked.append(law.renewal)
        elif isinstance(law, Sample):
            unpacked += law.parts
        elif isinstance(law, Match):
            unpacked.append(law.rule)
        else:
            unpacked.append(law)
    return unpacked


def _collect_laws(spec: _KernelSpec, component: object, flows: StockFlowPass, *, expand: bool = True) -> list[Law]:
    """Return the laws a component returns for its namespace after checking their kinds and expanding its flows.

    Jumps on clocks of waiting times become renewals and samples their energy, move, and sampler, as the steppers run
    them. Without `expand`, the laws are kept as declared, as the census of validation reads them.
    """
    declared = component.laws(spec.namespace)  # type: ignore[attr-defined]
    kinds = (Rate, Noise, Jump, Update, Transition, Emission, Boundary, Rule, Equation, Event, Sample, Match)
    kinds = (*kinds, *DECLARED_FREE)
    if not isinstance(declared, list | tuple) or not all(isinstance(law, kinds) for law in declared):
        error_msg = (
            f'`laws` of {spec.owner} must return a list of Rate, Noise, FractionalNoise, Jump, Update, Transition, '
            'Emission, Boundary, Rule, Equation, Event, LocalRule, Drive, Relaxation, Extremal, Sample, Flux, '
            'Transfer, or Auxiliary laws.'
        )
        raise ValidationError(error_msg)
    if spec.pairs is not None and not all(isinstance(law, Rate) for law in declared):
        error_msg = f'Pair interaction {spec.owner} returns laws other than Rate, which pair interactions do not take.'
        raise ValidationError(error_msg)
    if not expand:
        return list(declared)
    laws = flows.expand(spec, list(_unpacked(declared)))
    return [item for law in laws for item in split_law(law, spec.owner, lambda target: _state_name(spec, target))]


@dataclass(frozen=True)
class _Orders:
    """Orders and lags of the states of a system by entry name."""

    orders: dict[str, int]
    lags: dict[str, int]

    @classmethod
    def of(cls, blocks: list[_Block]) -> '_Orders':
        """Return the orders above 1 and the lags of every state of the blocks."""
        orders, lags = {}, {}
        for block in blocks:
            if _is_interaction(block.node) or block.node.is_agent:
                continue
            declarations = declarations_of(block.node.component_type, VariableKind.STATE)
            for variable, entry in block.entries.items():
                declaration = declarations.get(variable)
                if declaration is not None and declaration.order > 1:
                    orders[entry.name] = declaration.order
                if declaration is not None and declaration.lags:
                    lags[entry.name] = declaration.lags
        return cls(orders, lags)


def _companion_symbol(spec: _KernelSpec, item: Expression, suffix: str) -> Expression:
    """Return the symbol of a companion state of the state a symbol reads, in the same namespace and column."""
    reference = spec.references[item]
    name = f'{reference.entry.name}{suffix}'
    for other, candidate in spec.references.items():
        if (
            candidate.entry.name == name
            and candidate.column == reference.column
            and candidate.source == reference.source
        ):
            return other
    error_msg = f'A law of {spec.owner} reads {name}, which the namespace does not hold.'
    raise ValidationError(error_msg)


def _calls(expression: Expression, name: str) -> list[Expression]:
    """Return the calls of a symbolic function by name inside an expression."""
    found, stack = [], [expression]
    while stack:
        item = stack.pop()
        if item.operator() is None:
            continue
        if getattr(item.operator(), 'name', lambda: '')() == name:
            found.append(item)
        stack.extend(item.operands())
    return found


@dataclass(frozen=True)
class _Relation:
    """Equation of a component as `expression == 0`, with the placeholders of its highest derivatives."""

    spec: _KernelSpec
    expression: Expression
    highest: dict[Expression, Expression]
    description: str


def _substitute(
    value: object,
    spec: _KernelSpec,
    orders: _Orders,
    owner: str,
    highest: dict[Expression, Expression] | None = None,
) -> object:
    """Return a law value with derivatives and lags of states replaced by their companion symbols.

    When `highest` is given, derivatives of a state's own order become placeholders that map to the symbol of the
    state's highest derivative, which an equation solves for.
    """
    if isinstance(value, SymbolicVector):
        return vector(_substitute(item, spec, orders, owner, highest) for item in value)
    if not isinstance(value, Expression):
        return value
    replacements = {}
    for call in _calls(value, 'derivative'):
        item, degree = call.operands()
        reference = spec.references.get(item)
        order = orders.orders.get(reference.entry.name, 1) if reference is not None else 1
        if highest is not None and reference is not None and degree.is_integer() and int(degree) == order:
            top = item if order == 1 else _companion_symbol(spec, item, f'_d{order - 1}')
            placeholder = symbol(f'_highest_{top}')
            highest[placeholder] = top
            replacements[call] = placeholder
            continue
        if reference is None or not degree.is_integer() or not 0 < int(degree) < order:
            error_msg = (
                f'{owner} uses {call}, but laws read derivatives of a state only below its declared order, here '
                f'{order}. Declare a higher order with `state(order=...)`, or state the highest derivative with an '
                '`Equation`.'
            )
            raise ValidationError(error_msg)
        replacements[call] = _companion_symbol(spec, item, f'_d{int(degree)}')
    for call in _calls(value, 'lag'):
        item, steps = call.operands()
        reference = spec.references.get(item)
        kept = orders.lags.get(reference.entry.name, 0) if reference is not None else 0
        if reference is None or not steps.is_integer() or not 0 < int(steps) <= kept:
            error_msg = (
                f'{owner} uses {call}, but the state keeps {kept} past values. Declare more with `state(lags=...)`.'
            )
            raise ValidationError(error_msg)
        replacements[call] = _companion_symbol(spec, item, f'_lag{int(steps)}')
    return value.subs(replacements) if replacements else value


def _retarget(law: Rate | Noise, spec: _KernelSpec, orders: _Orders) -> Rate | Noise:
    """Return a rate or noise law that targets the highest derivative of its state, after checking its order."""
    items = components_of(law.target)
    references = [spec.references.get(item) for item in items]
    declared = {orders.orders.get(reference.entry.name, 1) for reference in references if reference is not None}
    order = max(declared, default=1)
    if law.order is not None and law.order != order:
        names = ', '.join(sorted({reference.entry.name for reference in references if reference is not None}))
        error_msg = (
            f'A {type(law).__name__} law of {spec.owner} on {names} gives the derivative of order {law.order}, but the '
            f'state has order {order}. Match the law to the order declared with `state(order=...)`.'
        )
        raise ValidationError(error_msg)
    if order == 1:
        return law
    tops = [_companion_symbol(spec, item, f'_d{order - 1}') for item in items]
    target = vector(tops) if isinstance(law.target, SymbolicVector) else tops[0]
    return replace(law, target=target)


def _intrinsic_laws(spec: _KernelSpec, block: _Block, orders: _Orders) -> list[Law]:
    """Return the chain `x' = x_d1, ...` of every higher-order state and the shifts of every lagged state."""
    laws: list[Law] = []
    for variable, entry in block.entries.items():
        names = [variable, *(f'{variable}_d{index}' for index in range(1, orders.orders.get(entry.name, 1)))]
        laws += [
            Rate(getattr(spec.namespace, lower), getattr(spec.namespace, upper))
            for lower, upper in itertools.pairwise(names)
        ]
        shifted = [variable, *(f'{variable}_lag{index}' for index in range(1, orders.lags.get(entry.name, 0) + 1))]
        laws += [
            Update(getattr(spec.namespace, older), getattr(spec.namespace, newer))
            for newer, older in itertools.pairwise(shifted)
        ]
    return laws


def _relation(spec: _KernelSpec, law: Equation, orders: _Orders) -> _Relation:
    """Return an equation law as a relation in placeholders of the highest derivatives it contains."""
    description = f'the equation {law.lhs} == {law.rhs} of {spec.owner}'
    if spec.pairs is not None:
        error_msg = f'Pair interaction {spec.owner} returns {description}. Pair interactions take only rate laws.'
        raise ValidationError(error_msg)
    highest: dict[Expression, Expression] = {}
    lhs = _substitute(SR(law.lhs), spec, orders, f'In {description}, it', highest)
    rhs = _substitute(SR(law.rhs), spec, orders, f'In {description}, it', highest)
    if not highest:
        error_msg = f'{description[0].upper()}{description[1:]} contains no highest derivative of a state to solve for.'
        raise ValidationError(error_msg)
    return _Relation(spec, SR(lhs) - SR(rhs), highest, description)


def _with_orders(spec: _KernelSpec, laws: list[Law], block: _Block, orders: _Orders) -> list[Any]:
    """Return laws with derivatives and lags substituted, rates retargeted, equations as relations, and chains added."""
    if not orders.orders and not orders.lags and not any(isinstance(law, Equation) for law in laws):
        return laws
    result: list[Any] = []
    for law in laws:
        if isinstance(law, (Rule, Transition, Emission, *KERNEL_FREE)):
            result.append(law)
            continue
        if isinstance(law, Equation):
            result.append(_relation(spec, law, orders))
            continue
        owner = f'A {type(law).__name__} law of {spec.owner}'
        if isinstance(law, Jump):
            result.append(Jump(_substitute(law.rate, spec, orders, owner), law.changes))
            continue
        if isinstance(law, Event | Renewal):
            reset = {
                _substitute(SR(target), spec, orders, owner): _substitute(SR(value), spec, orders, owner)
                for target, value in (law.reset or {}).items()
            }
            reset_value = reset if law.reset is not None else None
            if isinstance(law, Renewal):
                result.append(
                    replace(law, waiting=_substitute(SR(law.waiting), spec, orders, owner), reset=reset_value),
                )
                continue
            guard = _substitute(SR(law.guard), spec, orders, owner)
            result.append(replace(law, guard=guard, reset=reset_value))
            continue
        value = _substitute(law.value, spec, orders, owner)
        if isinstance(law, Rate | Noise):
            result.append(_retarget(replace(law, value=value), spec, orders))
        else:
            result.append(type(law)(law.target, value))
    if not _is_interaction(block.node):
        result += [
            item
            for law in _intrinsic_laws(spec, block, orders)
            for item in split_law(law, spec.owner, lambda target: _state_name(spec, target))
        ]
    return result


def _companions(blocks: list[_Block]) -> dict[str, tuple[str, str, int]]:
    """Return every companion state by name, with its kind, the name of its state, and its index."""
    companions = {}
    for block in blocks:
        if _is_interaction(block.node) or block.node.is_agent:
            continue
        declarations = declarations_of(block.node.component_type, VariableKind.STATE)
        for variable, entry in block.entries.items():
            declaration = declarations.get(variable)
            if declaration is None or (declaration.derivative_of is None and declaration.lag_of is None):
                continue
            kind, base = (
                ('derivative', declaration.derivative_of) if declaration.derivative_of else ('lag', declaration.lag_of)
            )
            companions[entry.name] = (kind, f'{block.node.path}.{base}', declaration.index)
    return companions


@dataclass(frozen=True)
class _Renewal:
    """Renewal law of a component with its label, the group it renews, and the hidden state that counts it."""

    spec: _KernelSpec
    law: Renewal
    label: str
    group: str
    entry: Entry
    symbol: Expression


def _renewal_group(spec: _KernelSpec, law: Renewal) -> str:
    """Return the path of the element or group whose states a renewal law resets, or the law's component."""
    targets = [item for target in (law.reset or {}) for item in components_of(target)]
    reference = spec.references.get(targets[0]) if targets else None
    return spec.owner if reference is None else reference.entry.name.rpartition('.')[0]


def _declare_renewal(spec: _KernelSpec, law: Renewal, declarations: _Declarations, label: str) -> _Renewal:
    """Return a renewal law with the hidden state that counts the renewals of every member, after declaring it."""
    group = _renewal_group(spec, law)
    rows = spec.rows
    sanitized = re.sub(r'[^0-9A-Za-z]', '_', label)
    entry = declarations.declare_state(
        f'{group}.{sanitized}~renewals',
        np.zeros(rows),
        Domain.NONNEGATIVE_INTEGER,
        (rows,) if rows > 1 else (),
        rows if rows > 1 else None,
    )
    return _Renewal(spec, law, label, group, entry, symbol(f'renewals_{sanitized}', Domain.NONNEGATIVE_INTEGER))


@dataclass
class _Hybrid:
    """Delays, events, and corrected noise collected from the laws of a system."""

    delays: list[dict[str, Any]] = field(default_factory=list)
    placeholders: dict[tuple[int, str, int, str], Expression] = field(default_factory=dict)
    events: list[tuple[_KernelSpec, Event]] = field(default_factory=list)
    noisy: set[tuple[str, int]] = field(default_factory=set)
    renewals: list[_Renewal] = field(default_factory=list)


def _delay_minimum(
    spec: _KernelSpec,
    tau: Expression,
    minimum: Expression | None,
    declarations: _Declarations,
) -> float:
    """Return the smallest value of a delay: the delay itself, its declared minimum, or its value at the parameters."""
    if SR(tau).is_numeric():
        return float(tau)
    if minimum is not None and SR(minimum).is_numeric():
        return float(minimum)
    symbols = sorted(SR(tau).variables(), key=str)
    references = [spec.references.get(item) for item in symbols]
    if symbols and all(reference is not None and not reference.is_state for reference in references):
        values = np.concatenate([np.zeros(0), *declarations.initial_parameters])
        columns = [reference.column_of(np.zeros(0), values) for reference in references if reference is not None]
        delays = numpy_function(declarations.time, symbols, [SR(tau)])(0.0, *columns)[0]
        return float(np.min(delays))
    error_msg = (
        f'A delay of {spec.owner} is the expression {tau}, whose smallest value is unknown. Give it as the third '
        'argument, `delayed(x, tau, minimum)`.'
    )
    raise ValidationError(error_msg)


def _substitute_delays(value: object, spec: _KernelSpec, declarations: _Declarations, hybrid: _Hybrid) -> object:
    """Return a law value with every delayed state replaced by a hidden parameter that solvers fill from history."""
    if isinstance(value, SymbolicVector):
        return vector(_substitute_delays(item, spec, declarations, hybrid) for item in value)
    if not isinstance(value, Expression):
        return value
    replacements = {}
    for call in _calls(value, 'delayed'):
        arguments = call.operands()
        item, tau = arguments[0], arguments[1]
        reference = spec.references.get(item)
        if reference is None or not reference.is_state or reference.source not in {Source.SINGLE, Source.MEMBER}:
            error_msg = f'{spec.owner} delays {item}, which is not a state of an element or a group member.'
            raise ValidationError(error_msg)
        key = (id(spec), reference.entry.name, reference.column, str(tau))
        if key in hybrid.placeholders:
            replacements[call] = hybrid.placeholders[key]
            continue
        entry = reference.entry
        members = entry.members if reference.source is Source.MEMBER else None
        indices = (
            entry.start + np.arange(members) * entry.columns + reference.column
            if members is not None
            else np.asarray([entry.start + reference.column])
        )
        initial = np.concatenate([np.zeros(0), *declarations.initial_states])[indices]
        index = len(hybrid.delays)
        hidden = Entry(
            f'__delayed_{index}',
            declarations.ends[False],
            (members,) if members is not None else (),
            Domain.REAL,
            members,
        )
        declarations.parameters.append(hidden)
        declarations.ends[False] += hidden.size
        declarations.initial_parameters.append(np.asarray(initial, dtype=float))
        placeholder = symbol(f'_delayed_{index}')
        spec.references[placeholder] = Reference(hidden, False, 0, reference.source)
        hybrid.delays.append(
            {
                'entry': hidden,
                'states': indices,
                'tau': tau,
                'minimum': _delay_minimum(
                    spec,
                    tau,
                    next(iter(arguments[2:]), None),
                    declarations,
                ),
                'spec': spec,
                'symbol': placeholder,
                'state': item,
            },
        )
        hybrid.placeholders[key] = placeholder
        replacements[call] = placeholder
    return value.subs(replacements) if replacements else value


def _noise_corrections(spec: _KernelSpec, law: Noise, alpha: float, hybrid: _Hybrid) -> list[Law]:
    """Return the drift corrections `alpha g dg/dx` that turn a noise law of a convention into the Itô form."""
    if not alpha:
        return []
    targets = components_of(law.target)
    values = (
        components_of(law.value) if isinstance(law.value, SymbolicVector | list | tuple) else [law.value] * len(targets)
    )
    corrections: list[Law] = []
    for target, value in zip(targets, values, strict=True):
        reference = spec.references[target]
        key = (reference.entry.name, reference.column)
        if key in hybrid.noisy:
            error_msg = (
                f'{reference.entry.name} receives noise from several laws, so its drift correction for the '
                'Stratonovich or anti-Itô convention is ambiguous. Join the noise into one law.'
            )
            raise ValidationError(error_msg)
        hybrid.noisy.add(key)
        if isinstance(value, Expression | int | float):
            correction = alpha * SR(value) * SR(value).diff(target)
            if not correction.is_zero():
                corrections.append(Rate(target, correction))
    return corrections


def _with_hybrid(
    spec: _KernelSpec,
    laws: list[Law],
    declarations: _Declarations,
    hybrid: _Hybrid,
) -> list[Law]:
    """Return laws with delays substituted and noise corrected, keeping events aside and declaring renewal counts."""
    result: list[Law] = []
    renewals = [law for law in laws if isinstance(law, Renewal)]
    for law in laws:
        if isinstance(law, Rule):
            result.append(law)
        elif isinstance(law, Renewal):
            position = renewals.index(law)
            label = law.label or (spec.owner if len(renewals) == 1 else f'{spec.owner}:{position}')
            hybrid.renewals.append(_declare_renewal(spec, law, declarations, label))
            result.append(law)
        elif isinstance(law, Event):
            guard = _substitute_delays(SR(law.guard), spec, declarations, hybrid)
            reset = {
                target: _substitute_delays(SR(value), spec, declarations, hybrid)
                for target, value in (law.reset or {}).items()
            }
            hybrid.events.append((spec, Event(guard, reset, law.direction, law.terminal, law.label)))
        elif isinstance(law, Jump):
            result.append(Jump(_substitute_delays(law.rate, spec, declarations, hybrid), law.changes))
        elif isinstance(law, Rate | Noise):
            rebuilt = replace(law, value=_substitute_delays(law.value, spec, declarations, hybrid))
            result.append(rebuilt)
            if isinstance(rebuilt, Noise):
                result += _noise_corrections(spec, rebuilt, NOISE_CONVENTIONS[law.convention], hybrid)
        elif isinstance(law, Update | Boundary):
            result.append(type(law)(law.target, _substitute_delays(law.value, spec, declarations, hybrid)))
        else:
            result.append(law)
    return result


@dataclass
class _Fields:
    """Stencils lowered from the operators of a system, with the laws as written and the stable step."""

    written: dict[tuple[str, int], list[tuple[Expression, Expression]]] = field(
        default_factory=lambda: defaultdict(list),
    )
    limits: list[float] = field(default_factory=list)
    volume_limits: list[DiffusiveLimit] = field(default_factory=list)
    linear: list[LinearTerm] = field(default_factory=list)


def _initial_values(spec: _KernelSpec, declarations: _Declarations) -> dict[Expression, float]:
    """Return the initial value of every symbol of a component, the first member's for groups."""
    states = np.concatenate([np.zeros(0), *declarations.initial_states])
    parameters = np.concatenate([np.zeros(0), *declarations.initial_parameters])
    return {
        item: float((states if reference.is_state else parameters)[reference.entry.start + reference.column])
        for item, reference in spec.references.items()
        if reference.entry.start + reference.column < (states if reference.is_state else parameters).size
    }


def _linear_terms_of(spec: _KernelSpec, laws: list[Law], places: dict[str, Field]) -> list[LinearTerm]:
    """Return the linear terms with constant coefficients of the rates of field states."""
    terms = []
    for law in laws:
        if not isinstance(law, Rate) or not isinstance(law.value, Expression):
            continue
        for reference in _target_references(law.target, spec):
            if reference.entry.name in places and reference.source is Source.MEMBER:
                terms += linear_terms(law.value, reference, spec.references, places)
    return terms


def _with_fields(spec: _KernelSpec, laws: list[Law], declarations: _Declarations, fields: _Fields) -> list[Law]:
    """Return laws whose differential operators are lowered, noting stable steps and the linear terms of rates."""
    places = declarations.places
    fields.linear += _linear_terms_of(spec, laws, places)
    if not any(has_operators(getattr(law, 'value', None)) for law in laws):
        return laws
    lowering = FieldLowering(spec.owner, spec.references, places, _initial_values(spec, declarations))
    result: list[Law] = []
    for law in laws:
        if not isinstance(law, Rate | Noise | Update | Boundary) or not has_operators(law.value):
            result.append(law)
            continue
        if isinstance(law, Rate):
            lowering.note_limit(law.value)
        lowered = lowering.lower(law.value)
        if isinstance(law.value, Expression):
            for reference in _target_references(law.target, spec):
                fields.written[(reference.entry.name, reference.column)].append((lowered, law.value))
        if isinstance(law, Rate | Noise):
            result.append(replace(law, value=lowered))
        else:
            result.append(type(law)(law.target, lowered))
    fields.limits += lowering.limits
    fields.volume_limits += lowering.volume_limits
    return result


def _with_member_shifts(spec: _KernelSpec, laws: list[Law], time: Expression) -> list[Law]:
    """Return laws whose member offset reads are lowered to neighbour reads of the group's index table."""
    shifts = MemberShifts(spec.owner, spec.references, time)
    result: list[Law] = []
    for law in laws:
        if isinstance(law, Jump):
            changes: dict[Expression, float] = defaultdict(float)
            for target, increment in law.changes.items():
                changes[SR(shifts.lower(target))] += increment
            result.append(Jump(shifts.lower(law.rate), dict(changes)))
        elif isinstance(law, Rate | Noise):
            result.append(replace(law, value=shifts.lower(law.value)))
        elif isinstance(law, Update | Boundary):
            result.append(type(law)(law.target, shifts.lower(law.value)))
        else:
            result.append(law)
    return result


def _with_neighbor_reads(spec: _KernelSpec, laws: list[Law], declarations: _Declarations, fields: _Fields) -> list[Law]:
    """Return laws whose differential operators and member offset reads are lowered to neighbour reads."""
    return _with_member_shifts(spec, _with_fields(spec, laws, declarations, fields), declarations.time)


def _written_equations(equations: dict[str, Expression], states: Layout, fields: _Fields) -> dict[str, Expression]:
    """Return equations with the stencils of operators replaced by the operators as written."""
    written = dict(equations)
    for (name, column), pairs in fields.written.items():
        key = states.entry(name).generic_names[column]
        if key in written:
            written[key] = SR(written[key]) - sum(SR(item) for item, _ in pairs) + sum(SR(item) for _, item in pairs)
    return written


def _reset_indices(reference: Reference) -> npt.NDArray[np.intp]:
    """Return the flat indices a reset of a single element's state or of every group member's state writes."""
    if reference.entry.members is not None and reference.source in {Source.MEMBER, Source.REDUCED}:
        return reference.flat_indices.astype(np.intp)
    return np.asarray([reference.entry.start + reference.column], dtype=np.intp)


def _reset_function(
    spec: _KernelSpec,
    event: Event,
    time: Expression,
    reductions: ReductionPass,
) -> VectorFunction | None:
    """Return a function that sets the states an event resets, all from the state before the event."""
    reset = event.reset or {}
    targets = [item for target in reset for item in components_of(target)]
    if not targets:
        return None
    values = [
        SR(item)
        for target, value in reset.items()
        for item in (
            components_of(value) if isinstance(value, SymbolicVector) else [value] * len(components_of(target))
        )
    ]
    references = [spec.references.get(item) for item in targets]
    if any(
        item is None or not item.is_state or item.source not in {Source.SINGLE, Source.MEMBER} for item in references
    ):
        error_msg = f'An event of {spec.owner} resets a symbol that is not a state of an element or a group member.'
        raise ValidationError(error_msg)
    arguments = sorted(symbols_of(values) - {time}, key=str)
    unknown = [item for item in arguments if spec.references.get(item) is None]
    if unknown:
        error_msg = f'An event of {spec.owner} uses {", ".join(map(str, unknown))}, which the namespace does not hold.'
        raise ValidationError(error_msg)
    reads = [(spec.references[item].is_state, _reset_indices(spec.references[item])) for item in arguments]
    places = [_reset_indices(reference) for reference in references]  # type: ignore[arg-type]
    function = numpy_function(time, arguments, values)

    def apply(t: float, x: Vector, p: Vector) -> Vector:
        q = reductions.extend(t, x, p)
        results = function(t, *((x if is_state else q)[index] for is_state, index in reads))
        state = x.copy()
        for place, value in zip(places, results, strict=True):
            state[place] = np.broadcast_to(np.asarray(value, dtype=float), place.shape)
        return state

    return apply


def _compile_events(hybrid: _Hybrid, time: Expression, reductions: ReductionPass) -> tuple[CompiledEvent, ...]:
    """Return the events of a system compiled into guard and reset functions of the state vector."""
    compiled = []
    counts = Counter(spec.owner for spec, _ in hybrid.events)
    positions: Counter[str] = Counter()
    for spec, event in hybrid.events:
        position = positions[spec.owner]
        positions[spec.owner] += 1
        sources = flat_sources(spec.references, spec.owner)
        guard = scalar_function([event.guard], sources, time, spec.owner)
        compiled.append(
            CompiledEvent(
                event.label or (spec.owner if counts[spec.owner] == 1 else f'{spec.owner}:{position}'),
                reductions.wrap(_first_value(guard)),  # type: ignore[arg-type]
                _reset_function(spec, event, time, reductions),
                int(event.direction),
                bool(event.terminal),
            ),
        )
    return tuple(compiled)


def _law_values(laws: list[Law]) -> list[Any]:
    """Return the expressions of the laws that integrators follow, leaving out the rates of jumps."""
    values: list[Any] = []
    for law in laws:
        if isinstance(law, Jump):
            continue
        sides = [law.lhs, law.rhs] if isinstance(law, Equation) else [getattr(law, 'value', None)]
        for side in sides:
            if side is not None:
                values += [item for item in components_of(side) if isinstance(item, Expression)]
    return values


def _hidden_events(
    specs: list[tuple[_KernelSpec, list[Law]]],
    time: Expression,
    reductions: ReductionPass,
) -> tuple[CompiledEvent, ...]:
    """Return one hidden event per condition on the states of single components that switches a law of a flow."""
    compiled = []
    for spec, laws in specs:
        sources = flat_sources(spec.references, spec.owner)
        for index, condition in enumerate(state_conditions_of(_law_values(laws), time)):
            if not set(condition.variables()) - {time} <= set(sources):
                continue
            guard = scalar_function([condition], sources, time, spec.owner)
            label = f'{spec.owner}:switch:{index}'
            compiled.append(CompiledEvent(label, reductions.wrap(_first_value(guard)), None, 0, False, hidden=True))  # type: ignore[arg-type]
    return tuple(compiled)


def _first_value(function: Scalars) -> Callable[[float, Vector, Vector], float]:
    """Return a function that gives the single value of a scalar function."""

    def first(t: float, x: Vector, p: Vector) -> float:
        return function(t, x, p)[0]

    return first


def _compile_delays(hybrid: _Hybrid, time: Expression) -> tuple[Delay, ...]:
    """Return the delays of a system with their delay functions compiled."""
    delays = []
    for item in hybrid.delays:
        spec = item['spec']
        tau = scalar_function([item['tau']], flat_sources(spec.references, spec.owner), time, spec.owner)
        hidden = item['entry']
        delays.append(
            Delay(
                parameters=hidden.start + np.arange(hidden.size),
                states=np.asarray(item['states'], dtype=np.intp),
                tau=_first_value(tau),
                minimum=float(item['minimum']),
                symbol=item['symbol'],
                state=item['state'],
                delay=SR(item['tau']),
            ),
        )
    return tuple(delays)


def _breakpoints(specs: list[tuple[_KernelSpec, list[Law]]], hybrid: _Hybrid, time: Expression) -> Breakpoints:
    """Return the breakpoints of the time-dependent inputs in every law and event guard."""
    expressions: list[Any] = []
    for _, laws in specs:
        for law in laws:
            value = law.rate if isinstance(law, Jump) else getattr(law, 'value', None)
            expressions += (
                [item for item in components_of(value) if isinstance(item, Expression)] if value is not None else []
            )
    expressions += [event.guard for _, event in hybrid.events]
    return breakpoints_of(expressions, time)


EQUATION_UNKNOWNS = 6


def _numeric_solution(rows: int, size: int, index: int) -> Callable[..., Any]:
    """Return a function of the matrix and right side entries that solves the linear system for one unknown."""

    def solve(t: float, *entries: Any) -> Any:  # noqa: ANN401
        del t
        values = [np.broadcast_to(np.asarray(entry, dtype=float), (rows,)) for entry in entries]
        matrix_ = np.stack(values[: size * size], axis=-1).reshape(rows, size, size)
        right = np.stack(values[size * size :], axis=-1)
        return np.linalg.solve(matrix_, right[..., None])[..., 0][:, index]

    return solve


def _solve_equations(specs: list[tuple[_KernelSpec, list[Any]]]) -> list[tuple[_KernelSpec, list[Law]]]:
    """Return specs whose equations are replaced by rate laws on the highest derivatives they solve for jointly."""
    relations = [item for _, laws in specs for item in laws if isinstance(item, _Relation)]
    kept = [(spec, [law for law in laws if not isinstance(law, _Relation)]) for spec, laws in specs]
    if not relations:
        return kept
    highest: dict[Expression, Expression] = {}
    for relation in relations:
        highest.update(relation.highest)
    unknowns = list(highest)
    for relation in relations:
        for first in unknowns:
            for second in unknowns:
                if not relation.expression.diff(first).diff(second).is_trivial_zero():
                    error_msg = (
                        f'{relation.description[0].upper()}{relation.description[1:]} is not linear in the highest '
                        f'derivative {highest[first]}. Write the equation so that highest derivatives appear linearly.'
                    )
                    raise ValidationError(error_msg)
    descriptions = '; '.join(relation.description for relation in relations)
    if len(relations) != len(unknowns):
        error_msg = (
            f'{len(relations)} equations give {len(unknowns)} highest derivatives, '
            f'{", ".join(str(highest[item]) for item in unknowns)}, in {descriptions}. Give one equation per '
            'derivative.'
        )
        raise ValidationError(error_msg)
    zero = dict.fromkeys(unknowns, 0)
    coefficients = [[relation.expression.diff(item) for item in unknowns] for relation in relations]
    right = [-relation.expression.subs(zero) for relation in relations]
    references = {key: value for relation in relations for key, value in relation.spec.references.items()}
    rows = max(relation.spec.rows for relation in relations)
    owner = ', '.join(dict.fromkeys(relation.spec.owner for relation in relations))
    merged = _KernelSpec(owner, Symbols(), references, rows)
    laws: list[Law]
    if len(unknowns) <= EQUATION_UNKNOWNS:
        system = matrix(SR, coefficients)
        if system.det().simplify_full().is_trivial_zero():
            error_msg = f'The highest derivatives of {descriptions} cannot be solved for: the system is singular.'
            raise ValidationError(error_msg)
        solution = system.solve_right(vector([SR(item) for item in right]), check=False)
        laws = [Rate(highest[item], value) for item, value in zip(unknowns, solution, strict=True)]
    else:
        entries = [entry for row in coefficients for entry in row] + right
        laws = [
            Rate(highest[item], Call(_numeric_solution(rows, len(unknowns), index), *entries))
            for index, item in enumerate(unknowns)
        ]
    return [*kept, (merged, laws)]


def _component_of(node: Node) -> object:
    """Return the object whose `laws` method describes a node, a representative member for a group."""
    if isinstance(node.component, Group):
        return node.component.element(node.component.label, **(node.component.values or {}))
    return node.component


def _target_references(target: object, spec: _KernelSpec) -> list[Reference]:
    """Return the references of the state components a law targets."""
    components = real_components_of(target)
    references = [spec.references.get(component) for component in components]
    if not all(reference is not None and reference.is_state for reference in references):
        error_msg = f'A law of {spec.owner} targets {target!r}, which is not a state symbol of the namespace `s`.'
        raise ValidationError(error_msg)
    return references  # type: ignore[return-value]


def _outputs_of(value: object, targets: list[Reference | None], owner: str) -> list[Output]:
    """Return the outputs of one law value for its target components, with the law and component as owner."""
    if isinstance(value, Call):
        arguments = tuple(
            item for argument in value.arguments for item in symbolic_components(argument, 1, owner) or []
        )
        if len(arguments) != len(value.arguments):
            error_msg = f'A Call in a {owner} has an argument that is not a symbolic expression.'
            raise ValidationError(error_msg)
        return [
            Output(target, function=value.function, arguments=arguments, component=index, components=len(targets))  # type: ignore[arg-type]
            for index, target in enumerate(targets)
        ]
    expressions = symbolic_components(value, len(targets), owner)
    if expressions is None:
        return [
            Output(target, function=value, component=index, components=len(targets))  # type: ignore[arg-type]
            for index, target in enumerate(targets)
        ]
    return [Output(target, expression) for target, expression in zip(targets, expressions, strict=True)]


def _check_assignments(
    spec: _KernelSpec,
    law: Law,
    targets: list[Reference],
    assigned: set[tuple[str, int]],
    *,
    once: bool,
) -> None:
    """Raise an error for an update or boundary law that sums over members or assigns a component twice."""
    kind = type(law).__name__
    for target in targets:
        if target.source is Source.SINGLE and spec.rows > 1 and not once:
            error_msg = f'A {kind} law of {spec.owner} targets a single element from a group, which has no meaning.'
            raise ValidationError(error_msg)
        key = (target.entry.name, target.column)
        if key in assigned:
            error_msg = f'A {kind} law of {spec.owner} sets {target.entry.name} a second time. Each takes one law.'
            raise ValidationError(error_msg)
        assigned.add(key)


@dataclass
class _Reductions:
    """Reductions collected from the laws of a system, deduplicated, with their placeholders and hidden entries."""

    declarations: _Declarations
    drawn: int = 0
    found: dict[tuple[Any, ...], CompiledReduction] = field(default_factory=dict)
    by_symbol: dict[Expression, CompiledReduction] = field(default_factory=dict)
    readers: dict[Expression, str | None] = field(default_factory=dict)

    @property
    def parameters(self) -> int:
        """Size of the parameter vector and the drawn values, where the hidden entries of the reductions start."""
        return self.declarations.ends[False] + self.drawn

    @property
    def extent(self) -> int:
        """Size of the extended parameter vector."""
        return max((item.entry.start + item.entry.size for item in self.found.values()), default=self.parameters)

    def lower(self, value: object, spec: _KernelSpec) -> object:
        """Return a law value with every reduction replaced by the placeholder that kernels read."""
        if isinstance(value, SymbolicVector):
            return vector(self.lower(item, spec) for item in value)
        if not isinstance(value, Expression):
            return value
        found = reductions_in(value)
        if not found:
            return value
        placeholders = {item.call: self._placeholder(item, spec) for item in found}
        lowered: Expression = value
        for call in sorted(placeholders, key=lambda item: -len(str(item))):
            lowered = lowered.subs({call: placeholders[call]})
        return modulus_form(lowered)

    def _group_of(self, item: Expression, spec: _KernelSpec) -> str | None:
        """Return the group, element set, or element whose members a symbol of a reduction stands for."""
        if item in spec.sets:
            return spec.sets[item].label
        if item in self.by_symbol:
            return self.readers[item]
        reference = spec.references.get(item)
        if reference is None or reference.source not in {Source.MEMBER, Source.SINGLE}:
            return None
        owner = reference.entry.name.rpartition('.')[0]
        return owner if reference.source is Source.MEMBER and owner in self.declarations.blocks else None

    def _group(self, reduction: Reduction, spec: _KernelSpec, symbols: set[Expression]) -> str | None:
        """Return the one group a reduction reduces over, after checking that it reads no second group."""
        groups = {group for item in symbols if (group := self._group_of(item, spec)) is not None}
        over = None
        if reduction.over is not None:
            over = self._group_of(reduction.over, spec)
            reference = spec.references.get(reduction.over)
            if over is None and reference is not None:
                over = reference.entry.name.rpartition('.')[0]
            if over is not None:
                groups.add(over)
        if len(groups) > 1:
            names = ' and '.join(sorted(groups))
            error_msg = (
                f'The reduction {reduction.call} of {spec.owner} reads the groups {names}. A reduction reduces over '
                'one group, so write one reduction per group.'
            )
            raise RoleError(error_msg)
        if groups:
            return next(iter(groups))
        if spec.group is not None:
            return spec.group
        return next((item.label for item in spec.sets.values()), None)

    def _read(self, item: Expression, spec: _KernelSpec, group: str | None, size: int) -> Read:
        """Return how the reduction pass reads one symbol of a reduction."""
        if item in spec.sets:
            references = spec.sets[item].references
            indices = np.asarray([reference.entry.start + reference.column for reference in references], dtype=np.intp)
            return Read(references[0].is_state, indices)
        if item in self.by_symbol:
            nested = self.by_symbol[item]
            if nested.weights is None:
                return Read(False, nested.entry.start, nested.entry)
            if self.readers[item] != group:
                error_msg = (
                    f'A reduction of {spec.owner} over {group} reads the weighted reduction {nested.symbol} of the '
                    f'members of {self.readers[item]}. Read a weighted reduction inside reductions over its readers.'
                )
                raise ValidationError(error_msg)
            return Read(False, (nested.entry.start + np.arange(nested.entry.size)).astype(np.intp), nested.entry)
        reference = spec.references.get(item)
        if reference is None or reference.source not in {Source.MEMBER, Source.SINGLE}:
            error_msg = (
                f'A reduction of {spec.owner} reads {item}, which is not a state or parameter of an element, a group '
                'member, or another reduction. Reductions read the namespace `s` of the law.'
            )
            raise ValidationError(error_msg)
        entry = reference.entry
        if reference.source is Source.MEMBER and entry.members is not None:
            indices = entry.start + np.arange(entry.rows) * entry.columns + reference.column
            return Read(reference.is_state, indices.astype(np.intp), entry, reference.column, entry.columns)
        return Read(reference.is_state, entry.start + reference.column, entry, reference.column)

    def _members(self, group: str | None, spec: _KernelSpec) -> tuple[int, Entry | None, npt.NDArray[np.intp] | None]:
        """Return the size of a group, an entry that gives its size, and the state indices of an element set."""
        if group is None:
            return 1, None, None
        block = self.declarations.blocks.get(group)
        if block is not None:
            entry = next(iter(block.entries.values()), None)
            return block.rows, entry, None
        references = next(item.references for item in spec.sets.values() if item.label == group)
        indices = np.asarray([reference.entry.start + reference.column for reference in references], dtype=np.intp)
        return len(references), None, indices

    def _weights(self, reduction: Reduction, spec: _KernelSpec, group: str | None, size: int) -> Any:  # noqa: ANN401
        """Return the weights of a reduction as a CSR array after checking their shape."""
        if reduction.weights is None:
            return None
        readers = len(next(iter(spec.sets.values())).references) if spec.sets else spec.rows
        shape = tuple(reduction.weights.shape)
        if shape != (readers, size):
            error_msg = (
                f'The weights of the reduction {reduction.call} of {spec.owner} over {group or "no group"} have the '
                f'shape {shape}, but the reduction needs the shape {(readers, size)}, one row per reading member and '
                'one column per reduced member.'
            )
            raise ValidationError(error_msg)
        return scipy.sparse.csr_array(reduction.weights, dtype=float)

    def _placeholder(self, reduction: Reduction, spec: _KernelSpec) -> Expression:
        """Return the placeholder of a reduction, or the placeholders of its real and imaginary parts."""
        if not SR(reduction.argument).has(I):
            return self._compiled(reduction, spec)
        if reduction.kind not in {'sum', 'mean'}:
            error_msg = (
                f'The reduction {reduction.call} of {spec.owner} reduces a complex expression, which only total and '
                'mean reduce. Reduce its real or imaginary part.'
            )
            raise ValidationError(error_msg)
        argument = SR(reduction.argument)
        real = self._compiled(replace(reduction, argument=argument.real_part()), spec)
        imaginary = self._compiled(replace(reduction, argument=argument.imag_part()), spec)
        return real + I * imaginary

    def _compiled(self, reduction: Reduction, spec: _KernelSpec) -> Expression:
        """Return the placeholder of a reduction of a real expression, compiling it the first time it is found."""
        argument = SR(self.lower(reduction.argument, spec))
        condition = None if reduction.condition is None else SR(self.lower(reduction.condition, spec))
        symbols = symbols_of([argument, *([condition] if condition is not None else [])]) - {self.declarations.time}
        group = self._group(reduction, spec, symbols)
        size, members, set_indices = self._members(group, spec)
        weights = self._weights(reduction, spec, group, size)
        canonical = {item: spec.sets[item].canonical for item in symbols if item in spec.sets}
        argument_key = argument.subs(canonical) if canonical else argument
        condition_key = condition.subs(canonical) if canonical and condition is not None else condition
        key = (reduction.kind, str(argument_key), str(condition_key), str(reduction.call.operands()[2]), group)
        if key not in self.found:
            index = len(self.found)
            readers = weights.shape[0] if weights is not None else None
            shape = (readers,) if readers is not None else ()
            members_of = readers if readers is not None and not spec.sets else None
            entry = Entry(f'__reduced_{index}', self.extent, shape, Domain.REAL, members_of)
            placeholder = symbol(f'_reduced_{index}')
            reads = {
                canonical.get(item, item): self._read(item, spec, group, size) for item in sorted(symbols, key=str)
            }
            compiled = CompiledReduction(
                placeholder,
                reduction.kind,
                argument_key,
                condition_key,
                weights,
                entry,
                size,
                group,
                reads,
                self.declarations.time,
                members,
                set_indices,
            )
            self.found[key] = compiled
            self.by_symbol[placeholder] = compiled
            reading = spec.group or next((item.label for item in spec.sets.values()), None)
            self.readers[placeholder] = reading if weights is not None else None
        compiled = self.found[key]
        column = spec.set_index if compiled.weights is not None and spec.sets else 0
        spec.references[compiled.symbol] = Reference(compiled.entry, False, column, Source.REDUCED)
        return compiled.symbol


def _lower_law(law: Law, spec: _KernelSpec, reductions: _Reductions) -> Law:
    """Return a law with every reduction in its values replaced by its placeholder."""
    if isinstance(law, Jump):
        return Jump(reductions.lower(law.rate, spec), law.changes)
    if not isinstance(law, Rate | Noise | Update | Boundary):
        return law
    value = law.value
    if isinstance(value, Call):
        value = Call(value.function, *(reductions.lower(argument, spec) for argument in value.arguments))
    else:
        value = reductions.lower(value, spec)
    if isinstance(law, Rate | Noise):
        return replace(law, value=value)
    return type(law)(law.target, value)


def _prune_foreign(spec: _KernelSpec, declarations: _Declarations) -> None:
    """Remove the references of groups that a spec reads only through reductions, so kernels loop over its group."""
    if spec.pairs is not None:
        return
    owners = {
        reference.entry.name.rpartition('.')[0]
        for reference in spec.references.values()
        if reference.source is Source.MEMBER
    }
    foreign = (owners & declarations.groups) - {spec.group, *spec.aligned}
    if not foreign:
        return
    spec.references = {
        item: reference
        for item, reference in spec.references.items()
        if reference.source is not Source.MEMBER or reference.entry.name.rpartition('.')[0] not in foreign
    }


def _with_reductions(
    specs: list[tuple[_KernelSpec, list[Law]]],
    hybrid: _Hybrid,
    declarations: _Declarations,
    drawn: int,
) -> tuple[list[tuple[_KernelSpec, list[Law]]], ReductionPass]:
    """Return laws and events whose reductions read placeholders, and the pass that evaluates the reductions."""
    reductions = _Reductions(declarations, drawn)
    lowered = [(spec, [_lower_law(law, spec, reductions) for law in laws]) for spec, laws in specs]
    events = []
    for spec, event in hybrid.events:
        reset = {target: reductions.lower(SR(value), spec) for target, value in (event.reset or {}).items()}
        guard = reductions.lower(SR(event.guard), spec)
        events.append((spec, replace(event, guard=guard, reset=reset if event.reset is not None else None)))
    hybrid.events = events
    for spec, _ in lowered:
        _prune_foreign(spec, declarations)
    return lowered, ReductionPass(tuple(reductions.found.values()), reductions.parameters)


def _is_once(spec: _KernelSpec, law: Law) -> bool:
    """Check whether a law of a group's interaction sets single elements from reductions alone, so it runs once."""
    if spec.rows <= 1 or spec.pairs is not None or not isinstance(law, Rate | Noise | Update | Boundary):
        return False
    if not all(reference.source is Source.SINGLE for reference in _target_references(law.target, spec)):
        return False
    expressions = [item for item in components_of(law.value) if isinstance(item, Expression)]
    sources = {spec.references[item].source for item in symbols_of(expressions) if item in spec.references}
    return Source.REDUCED in sources and Source.MEMBER not in sources


def _single_references(spec: _KernelSpec) -> dict[Expression, Reference]:
    """Return the references of a spec without its group members, for laws that run once."""
    return {
        item: reference
        for item, reference in spec.references.items()
        if reference.source is not Source.MEMBER and reference.entry.members is None
    }


def _lower_laws(specs: list[tuple[_KernelSpec, list[Law]]], declarations: _Declarations) -> _Lowered:
    """Return the kernels of every law kind."""
    states, parameters = Layout(declarations.states), Layout(declarations.parameters)
    lowered = _Lowered()
    assigned: dict[type, set[tuple[str, int]]] = defaultdict(set)
    for spec, laws in specs:
        outputs: dict[tuple[type, bool], list[Output]] = defaultdict(list)
        for law in laws:
            if isinstance(law, (Rule, Event, Renewal, *KERNEL_FREE)):
                continue
            if isinstance(law, Jump):
                output = _outputs_of(law.rate, [None], f'Jump law of {spec.owner}')[0]
                changes = [
                    (_target_references(target, spec)[0], float(increment)) for target, increment in law.changes.items()
                ]
                lowered.jump_changes[id(output)] = changes
                outputs[(Jump, False)].append(output)
                continue
            if isinstance(law, Equation | Transition | Emission):
                error_msg = f'A {type(law).__name__} law of {spec.owner} reached lowering before it was rewritten.'
                raise ValidationError(error_msg)
            targets = _target_references(law.target, spec)
            once = _is_once(spec, law)
            if isinstance(law, ASSIGNED_KINDS):
                _check_assignments(spec, law, targets, assigned[type(law)], once=once)
            law_outputs = _outputs_of(law.value, targets, f'{type(law).__name__} law of {spec.owner}')  # type: ignore[arg-type]
            if isinstance(law, FractionalNoise):
                description = _fractional_description(spec, law)
                lowered.hurst.update({id(item): (description, law.hurst) for item in law_outputs})
            outputs[(_kind(law), once)].extend(law_outputs)
        for (kind, once), kind_outputs in outputs.items():
            lowered.kernels[kind].append(
                Kernel(
                    spec.owner,
                    declarations.time,
                    _single_references(spec) if once else spec.references,
                    1 if once else spec.rows,
                    states,
                    parameters,
                    spec.pairs,
                    kind_outputs,
                ),
            )
    return lowered


def _kind(law: Law) -> type:
    """Return the kind of a law that decides its form, with fractional noise counted as noise."""
    return Noise if isinstance(law, Noise) else type(law)


def _form_kinds(owners: Iterable[set[type]]) -> frozenset[type]:
    """Return the law kinds that decide the form of components with their own kinds.

    Renewal laws combine with the rate and noise laws of the same components, so they drop out of the kinds then. A
    component with renewal laws alone next to components with other laws keeps them, so the system does not fit one
    form and runs as partitions.
    """
    owned = [set(kinds) - FORMLESS for kinds in owners]
    kinds: set[type] = set().union(*owned)
    rest = kinds - {Renewal}
    if Renewal not in kinds or not rest:
        return frozenset(kinds)
    alone = any(item == {Renewal} for item in owned)
    return frozenset(kinds if alone or not rest <= CONTINUOUS else rest)


def _infer_form(owners: Iterable[set[type]]) -> Form:
    """Return the form that the law kinds of the components of a system determine."""
    kinds = _form_kinds(owners)
    if kinds not in FORMS:
        found = ', '.join(sorted(kind.__name__ for kind in kinds))
        supported = '; '.join(
            ', '.join(sorted(kind.__name__ for kind in combination)) for combination in FORMS if combination
        )
        error_msg = f'The system mixes the law kinds {found}. Supported combinations are: {supported}.'
        raise IncompatibleLawsError(error_msg)
    return FORMS[kinds]


def _sums_over_members(kernel: Kernel, item: Output) -> bool:
    """Check whether an output sums over the pairs or the members of a group."""
    if kernel.pairs is not None:
        return True
    return item.target is not None and item.target.source is Source.SINGLE and kernel.rows > 1


def _equations_of(
    kernels: list[Kernel],
    declarations: _Declarations,
    form: Form,
) -> tuple[dict[str, Expression], dict[str, Expression]]:
    """Return the symbolic equations and their symbols, leaving out components with callables or group sums."""
    contributions: dict[tuple[str, int], list[Expression]] = defaultdict(list)
    excluded: set[tuple[str, int]] = set()
    for kernel in kernels:
        for item in kernel.outputs:
            key = (item.target.entry.name, item.target.column)  # type: ignore[union-attr]
            if item.expression is None or _sums_over_members(kernel, item):
                excluded.add(key)
                continue
            contributions[key].append(item.expression)
    equations, symbols = {}, {}
    for block in declarations.blocks.values():
        for variable, entry in block.entries.items():
            if entry not in declarations.states:
                continue
            components = block.symbols[variable]
            for column, item in enumerate(real_components_of(components)):
                key = (entry.name, column)
                if key in excluded:
                    continue
                default = item if form is Form.DISCRETE else SR(0)
                terms = contributions.get(key)
                name = entry.generic_names[column]
                equations[name] = sum(terms[1:], terms[0]) if terms else default
                symbols[name] = item
    return equations, symbols


def _lower(
    form: Form,
    lowered: _Lowered,
    declarations: _Declarations,
    backend: Backend,
    reductions: ReductionPass,
    *,
    drawing: DrawLowering,
) -> dict[str, object]:
    """Return the numeric functions and equations of a form, with the update of a stochastic map taking a generator."""
    kernels, size = lowered.kernels, declarations.ends[True]
    draws = drawing.draws
    extra = {} if draws is None else {'draws': draws}
    functions: dict[str, object] = vars(
        backend.lower(form, kernels, lowered.jump_changes, size, reductions=reductions, **extra),
    ).copy()
    if draws is not None and functions.get('update') is not None:
        functions['update'] = stochastic_update(cast('Callable[..., Vector]', functions['update']), draws)
    accelerated = cast('Accelerated | None', functions.get('accelerated'))
    if draws is not None and draws.sampled and accelerated is not None:
        functions['accelerated'] = replace(accelerated, iterate=None)
    functions['backend'] = backend.name
    functions['draws'] = draws
    main = {Form.ODE: kernels[Rate], Form.SDE: kernels[Rate] + kernels[Noise], Form.JUMP: kernels[Jump]}.get(
        form,
        kernels[Update],
    )
    equation_kernels = {Form.DISCRETE: kernels[Update], Form.JUMP: []}.get(form, kernels[Rate])
    equations, symbols = _equations_of(equation_kernels, declarations, form)
    functions.update(
        equations=equations,
        equation_symbols=symbols,
        has_callables=any(item.expression is None for kernel in main for item in kernel.outputs),
        has_group_sums=any(_sums_over_members(kernel, item) for kernel in main for item in kernel.outputs),
    )
    return functions


def _checked_matrices(
    markov: list[tuple[_KernelSpec, MatrixLaw]],
    declarations: _Declarations,
) -> tuple[MarkovMatrix, ...]:
    """Return the matrices of the Markov laws after checking them at the initial parameter values."""
    initial = np.concatenate([np.zeros(0), *declarations.initial_parameters])
    matrices = []
    for spec, law in markov:
        matrix = compile_matrix(law, spec.references, max(spec.rows, 1), declarations.time)
        violation = matrix.violation(initial)
        if violation is not None:
            error_msg = (
                f'At the initial parameter values, {violation}. Give a matrix with nonnegative entries whose rows sum '
                'to 1.'
            )
            raise ValidationError(error_msg)
        matrices.append(matrix)
    return tuple(matrices)


def _check_agent_names(node: Node) -> None:
    """Raise an error for a state or parameter of an agent named like an attribute of the agent view or of Mesa."""
    declared = [
        *declarations_of(node.component_type, VariableKind.STATE),
        *declarations_of(node.component_type, VariableKind.PARAMETER),
    ]
    clashes = sorted(name for name in declared if name in AGENT_RESERVED_NAMES)
    if clashes:
        error_msg = (
            f'Agents of {node.path} declare the variables {", ".join(clashes)}, whose names the agent view or Mesa '
            f'reserves. Rename them, since {", ".join(sorted(AGENT_RESERVED_NAMES))} are reserved names of agents.'
        )
        raise LabelError(error_msg)


def _compiled_agents(blocks: list[_Block], specs: list[tuple[_KernelSpec, list[Law]]]) -> tuple[CompiledAgents, ...]:
    """Return the agent elements and groups with their states, initial values, parameters, and rules."""
    rules = {spec.owner: tuple(law for law in laws if isinstance(law, Rule)) for spec, laws in specs}
    invariants = {
        spec.owner: tuple(law for law in laws if isinstance(law, Invariant) and law.is_member) for spec, laws in specs
    }
    agents = []
    for block in blocks:
        node = block.node
        if not node.is_agent:
            continue
        _check_agent_names(node)
        declared = declarations_of(node.component_type, VariableKind.STATE)
        complex_states = [name for name, declaration in declared.items() if declaration.domain is Domain.COMPLEX]
        if complex_states:
            error_msg = (
                f'Agent group {node.path} declares the complex state {complex_states[0]}, which the Mesa backend '
                'cannot run. Agent states take real domains, so keep the real and imaginary parts as two states.'
            )
            raise BackendError(error_msg)
        states = tuple(declared)
        agents.append(
            CompiledAgents(
                path=node.path,
                element=node.component,
                size=node.members or 1,
                states=states,
                values={name: node.value_of(name) for name in states},
                parameters={name: entry.name for name, entry in block.entries.items()},
                rules=rules.get(node.path, ()),
                visual=_agent_visual(node),
                invariants=invariants.get(node.path, ()),
            ),
        )
    return tuple(agents)


def _agent_visual(node: Node) -> VisualSpec:
    """Return the visual of agents, their own or their element type's, and the agents' points without one."""
    visual = _visual_of(node)
    return VisualSpec('agents') if visual == VisualSpec() else visual


def _agent_space(blocks: list[_Block]) -> Space | None:
    """Return the one space every agent group lives in, or raise an error when groups live in different spaces."""
    spaces = {id(block.node.space): block.node.space for block in blocks if block.node.is_agent}
    if len(spaces) > 1:
        error_msg = 'Agent groups of one system must live in the same space.'
        raise ValidationError(error_msg)
    return next(iter(spaces.values()), None)


def _is_interaction(node: Node) -> bool:
    """Check whether a node is an interaction."""
    return not isinstance(node.component, Element | Group)


def _visual_of(node: Node) -> VisualSpec:
    """Return the visual of an element, a group's own visual when it sets one, or the default of a field's space."""
    if isinstance(node.component, Group) and node.component.visual is not None:
        return node.component.visual
    if isinstance(node.component, Field) and isinstance(node.component.space, LatticeSpace):
        return VisualSpec('image')
    if isinstance(node.component, Field) and isinstance(node.component.space, Mesh):
        return VisualSpec('mesh')
    declared: VisualSpec = node.component_type.visual
    if (
        isinstance(node.component, Field)
        and isinstance(node.component.space, NetworkSpace)
        and declared == VisualSpec()
    ):
        return VisualSpec('sites')
    return declared


def _law_owner(spec: _KernelSpec, law: Law, agents: set[str]) -> str:
    """Return the path of the element or group whose state a law changes, or that a rule acts for.

    A rule acts for the first agent group among its roles, or for its first role when none is an agent group.
    """
    if isinstance(law, Jump) and law.renewal is not None:
        law = law.renewal
    if isinstance(law, Sample):
        law = law.energy
    if isinstance(law, Match):
        law = law.rule
    if isinstance(law, Rule):
        paths = [path for value in spec.roles.values() for path in ([value] if isinstance(value, str) else value)]
        return next((path for path in paths if path in agents), next(iter(paths), spec.owner))
    if isinstance(law, Equation | Event | Energy | Sampler | Renewal | Transfer | Auxiliary | Invariant):
        return _renewal_group(spec, law) if isinstance(law, Renewal) else spec.owner
    if isinstance(law, Relaxation):
        target = next(iter(law.targets), None)
        return spec.owner if target is None else _target_references(target, spec)[0].entry.name.rpartition('.')[0]
    if isinstance(law, LocalRule | Extremal):
        return _target_references(law.targets[0], spec)[0].entry.name.rpartition('.')[0]
    if isinstance(law, Flux):
        return _target_references(law.target, spec)[0].entry.name.rpartition('.')[0]
    target = next(iter(law.changes)) if isinstance(law, Jump) else law.target
    return _target_references(target, spec)[0].entry.name.rpartition('.')[0]


def _fits(owners: Iterable[set[type]]) -> bool:
    """Check whether the law kinds of components fit one form."""
    return _form_kinds(owners) in FORMS


def _mixed_error(name: str, kinds: set[type]) -> IncompatibleLawsError:
    """Return the error of a subsystem whose own laws fit no form."""
    found = ', '.join(sorted(kind.__name__ for kind in kinds if kind not in {Boundary, Event, Flux}))
    supported = '; '.join(
        ', '.join(sorted(kind.__name__ for kind in combination)) for combination in FORMS if combination
    )
    error_msg = (
        f'Subsystem {name} mixes the law kinds {found}. Supported combinations are: {supported}. Move the laws of '
        'different kinds into separate subsystems, which then run as partitions of a co-simulation.'
    )
    return IncompatibleLawsError(error_msg)


def _plan_partitions(system: System, kinds: dict[str, set[type]], scope: str = '') -> list[tuple[str, set[str]]]:
    """Return the partitions of a system tree as names with the element paths whose laws they hold."""
    name = scope.rstrip('/') or system.label
    under = {owner: owner_kinds for owner, owner_kinds in kinds.items() if owner.startswith(scope)}
    if _fits(under.values()):
        return [(name, set(under))] if under else []
    own = {owner: owner_kinds for owner, owner_kinds in under.items() if '/' not in owner[len(scope) :]}
    own_kinds: set[type] = set().union(*own.values())
    if not system.systems or not _fits(own.values()):
        raise _mixed_error(name, own_kinds if system.systems else set().union(*under.values()))
    plan = [(name, set(own))] if own else []
    for child in system.systems:
        plan += _plan_partitions(child, kinds, f'{scope}{child.label}/')
    return plan


def _compile_partitions(system: System, backend: str | Backend, plan: list[tuple[str, set[str]]]) -> Problem:
    """Return the problem of a system split into partitions, each compiled with the laws of its elements."""
    partitions = []
    for name, owners in plan:
        problem = _compile(system, backend, frozenset(owners))
        owned = [entry.indices for entry in problem.states if entry.name.rpartition('.')[0] in owners]
        indices = np.concatenate([np.arange(item.start, item.stop) for item in owned]) if owned else np.zeros(0)
        partitions.append(Partition(name, problem, indices.astype(np.intp), frozenset(owners)))
    first = partitions[0].problem
    equations = {
        key: value
        for item in partitions
        for key, value in item.problem.equations.items()
        if key.partition('.')[0] in item.owners
    }
    symbols = {
        key: item.problem.equation_symbols[key]
        for item in partitions
        for key in item.problem.equations
        if key in equations
    }
    return Problem(
        form=Form.COUPLED,
        states=first.states,
        parameters=first.parameters,
        initial_state=first.initial_state,
        initial_parameters=first.initial_parameters,
        time=first.time,
        symbols=first.symbols,
        equations=equations,
        equation_symbols=symbols,
        has_callables=any(item.problem.has_callables for item in partitions),
        has_group_sums=any(item.problem.has_group_sums for item in partitions),
        elements=first.elements,
        interactions=first.interactions,
        system=system,
        agents=tuple(agents for item in partitions for agents in item.problem.agents),
        agent_space=next((item.problem.agent_space for item in partitions if item.problem.agents), None),
        partitions=tuple(partitions),
    )


def _specs_of(block: _Block, elements: dict[str, _Block], time: Expression, observe: Observe) -> list[_KernelSpec]:
    """Return the kernel specs of a component: its intrinsic laws, its pairs, or its interaction laws."""
    if not _is_interaction(block.node):
        return [_member_spec(block, time)]
    if type(block.node.component).over == 'pairs':
        return [_pair_spec(block, elements, time)]
    return _interaction_specs(block, elements, time, observe)


def _with_draws(
    specs: list[tuple[_KernelSpec, list[Law]]],
    declarations: _Declarations,
) -> tuple[list[tuple[_KernelSpec, list[Law]]], DrawLowering]:
    """Return the laws with their draws replaced by values drawn after the parameters, and the lowering of the draws."""
    drawing = DrawLowering(declarations.ends[False], declarations.time)
    lowered = [(spec, drawing.lower(spec.owner, spec.rows, laws, spec.references)) for spec, laws in specs]
    drawing.allocate()
    return lowered, drawing


def _fractional_description(spec: _KernelSpec, law: FractionalNoise) -> str:
    """Return the description of a fractional noise law by the states it drives."""
    states = dict.fromkeys(_state_name(spec, item) for item in components_of(law.target))
    return f'fractional noise law of {", ".join(states)}'


def _checked_hurst(spec: _KernelSpec, law: FractionalNoise, declarations: _Declarations, description: str) -> None:
    """Raise an error when the Hurst index of a fractional noise law is not a number or a parameter in `(0, 1)`."""
    hurst = SR(law.hurst)
    if hurst.is_numeric():
        values = np.asarray([float(hurst)])
    else:
        reference = spec.references.get(hurst)
        if reference is None or reference.is_state:
            error_msg = f'The {description} of {spec.owner} has the Hurst index {hurst}. Give a number or a parameter.'
            raise ValidationError(error_msg)
        parameters = np.concatenate([np.zeros(0), *declarations.initial_parameters])
        values = np.asarray(reference.column_of(np.zeros(0), parameters), dtype=float)
    if not np.all((values > 0) & (values < 1)):
        error_msg = (
            f'The {description} of {spec.owner} has the Hurst index {hurst} = {values[0]}, outside the open interval '
            '(0, 1). Give a Hurst index between 0 and 1.'
        )
        raise ValidationError(error_msg)


def _checked_fractional(spec: _KernelSpec, laws: list[Law], declarations: _Declarations) -> list[Law]:
    """Return laws with every fractional noise law of Hurst index 1/2 as a noise law, after validating the others."""
    result: list[Law] = []
    for law in laws:
        if not isinstance(law, FractionalNoise):
            result.append(law)
            continue
        description = _fractional_description(spec, law)
        _checked_hurst(spec, law, declarations, description)
        if SR(law.hurst).is_numeric() and float(SR(law.hurst)) == 0.5:  # noqa: PLR2004
            result.append(Noise(law.target, law.value, law.order))
            continue
        values = [item for item in components_of(law.value) if isinstance(item, Expression)]
        if not isinstance(law.value, Expression | SymbolicVector) and not SR(law.value).is_numeric():
            error_msg = f'The {description} of {spec.owner} has an amplitude that is not symbolic. Give an expression.'
            raise ValidationError(error_msg)
        read = [
            item
            for item in sorted(symbols_of(values), key=str)
            if (reference := spec.references.get(item)) is not None and reference.is_state
        ]
        if read:
            error_msg = (
                f'The {description} of {spec.owner} has an amplitude that depends on the state '
                f'{_state_name(spec, read[0])}. Only state-independent amplitudes are supported for a Hurst index '
                'other than 1/2.'
            )
            raise ValidationError(error_msg)
        result.append(law)
    return result


def _space_boundary_laws(block: _Block, spec: _KernelSpec) -> list[Law]:
    """Return the laws by which the boundary of a continuous space acts on the position states of a component.

    A periodic boundary wraps every position component into the box. A reflecting boundary mirrors a position that
    crossed a wall back into the box and reverses the velocity component across the wall, where the velocity is the
    state declared as the rate of the position, or the first derivative of a second-order position.
    """
    space = block.node.space
    if _is_interaction(block.node) or not isinstance(space, ContinuousSpace) or space.boundary == 'open':
        return []
    declarations = declarations_of(block.node.component_type, VariableKind.STATE)
    laws: list[Law] = []
    for name, declaration in declarations.items():
        if not declaration.position:
            continue
        velocity_name = next(
            (other for other, item in declarations.items() if item.velocity_of == name),
            f'{name}_d1' if declaration.order > 1 else None,
        )
        positions = components_of(getattr(spec.namespace, name))
        velocities = components_of(getattr(spec.namespace, velocity_name)) if velocity_name else [None] * len(positions)
        for position, velocity, side in zip(positions, velocities, space.extent, strict=True):
            if space.boundary == 'periodic':
                laws.append(Boundary(position, mod(position, side)))
                continue
            below, above = position < 0, position > side
            laws.append(Boundary(position, where(below, -position, where(above, 2 * side - position, position))))
            if velocity is not None:
                laws.append(Boundary(velocity, where(either(below, above), -velocity, velocity)))
    return laws


def _collected_laws(
    blocks: list[_Block],
    elements: dict[str, _Block],
    time: Expression,
    observe: Observe,
    declarations: _Declarations,
) -> tuple[list[tuple[_KernelSpec, list[Law], _Block]], list[tuple[_KernelSpec, MatrixLaw]], StockFlow]:
    """Return the laws of every kernel spec with flows and Markov laws rewritten, the Markov matrices, and the flows."""
    flows = StockFlowPass(quantities_of(declarations.blocks))
    collected: list[tuple[_KernelSpec, list[Law], _Block]] = []
    markov: list[tuple[_KernelSpec, MatrixLaw]] = []
    for block in blocks:
        for spec in _specs_of(block, elements, time, observe):
            laws, matrices = markov_laws(
                spec.owner,
                [*_collect_laws(spec, _component_of(block.node), flows), *_space_boundary_laws(block, spec)],
                spec.references,
            )
            laws = _checked_fractional(spec, laws, declarations)
            check_draw_placement(spec.owner, laws)
            _settle_rows(spec, laws)
            collected.append((spec, laws, block))
            markov.extend((spec, matrix) for matrix in matrices)
    stock_flow = flows.lower([(spec, laws) for spec, laws, _ in collected], declarations)
    return collected, markov, stock_flow


@dataclass(frozen=True)
class CensusRecord:
    """Law of a component with the element it changes and the states it targets.

    Args:
        path:
            The path of the component that returns the law.

        owner:
            The path of the element or group whose state the law changes, or that a rule acts for.

        law:
            The law.

        targets:
            The layout entries of the states the law targets.

        vectors:
            The names of the targets declared as vector states.

        component:
            The kind of the owner: an element, a fixed or variable group, or a group on the sites of a space.

        space:
            The nearest space enclosing the component.
    """

    path: str
    owner: str
    law: Law
    targets: tuple[Entry, ...]
    vectors: frozenset[str]
    component: ComponentKind
    space: Space | None


def _component_kind(node: Node) -> ComponentKind:
    """Return the kind of the component of a node."""
    if isinstance(node.component, Group) and node.component.variable:
        return ComponentKind.VARIABLE_GROUP
    if isinstance(node.component, Group):
        parts = {'sites': ComponentKind.SITE_GROUP, 'edges': ComponentKind.EDGE_GROUP}
        parts['hyperedges'] = ComponentKind.HYPEREDGE_GROUP
        return parts.get(node.component.on, node.component.component_kind)
    return ComponentKind.ELEMENT


def _is_vector(entry: Entry, elements: dict[str, _Block]) -> bool:
    """Check whether the state of a layout entry is declared as a vector of the space's dimension."""
    path, _, name = entry.name.rpartition('.')
    block = elements.get(path)
    if block is None:
        return False
    declaration = declarations_of(block.node.component_type, VariableKind.STATE).get(name)
    return declaration is not None and declaration.vector


def law_census(system: System) -> tuple[list[CensusRecord], list[tuple[str, SkcomplexError]]]:
    """Return every law of a system with its owner and targets, and the errors of the components that fail.

    The census declares every component and collects its laws as compilation does, without lowering them, and keeps
    going past a failing component, so validation reports every problem at once.

    Args:
        system:
            The system.

    Returns:
        records:
            The laws of the components that declare and collect without error.

        errors:
            The path of every failing component with its error.
    """
    nodes = flatten_system(system)
    declarations = _Declarations(time=symbol('t'))
    errors: list[tuple[str, SkcomplexError]] = []
    blocks = []
    for node in nodes:
        try:
            blocks.append(declarations.declare_component(node))
        except SkcomplexError as error:
            errors.append((node.path, error))
    elements = {block.node.path: block for block in blocks if not _is_interaction(block.node)}
    observe = _Observer(declarations, _lowerable_observables(system, nodes))
    flows = StockFlowPass(quantities_of(declarations.blocks))
    agents = {block.node.path for block in blocks if block.node.is_agent}
    records = []
    for block in blocks:
        try:
            for spec in _specs_of(block, elements, declarations.time, observe):
                for law in _collect_laws(spec, _component_of(block.node), flows, expand=False):
                    if isinstance(law, Auxiliary | Invariant):
                        continue
                    target = getattr(law, 'target', None)
                    items = components_of(target) if target is not None else []
                    references = [spec.references.get(item) for item in items if isinstance(item, Expression)]
                    targets = tuple(dict.fromkeys(item.entry for item in references if item is not None))
                    owner = _law_owner(spec, law, agents)
                    vectors = frozenset(entry.name for entry in targets if _is_vector(entry, elements))
                    kind = _component_kind(elements.get(owner, block).node)
                    records.append(CensusRecord(block.node.path, owner, law, targets, vectors, kind, block.node.space))
        except SkcomplexError as error:
            errors.append((block.node.path, error))
    return records, errors


def _compile(system: System, backend: str | Backend, owners: frozenset[str] | None) -> Problem:
    """Return the problem of a system, keeping only the laws of the elements in `owners` when it is given."""
    nodes = flatten_system(system)
    declarations = _Declarations(time=symbol('t'))
    blocks = [declarations.declare_component(node) for node in nodes]
    _check_symbol_names(declarations.blocks)
    elements = {block.node.path: block for block in blocks if not _is_interaction(block.node)}
    observe = _Observer(declarations, _lowerable_observables(system, nodes))
    collected, markov, stock_flow = _collected_laws(blocks, elements, declarations.time, observe, declarations)
    checked = _checked_matrices(markov, declarations)
    orders = _Orders.of(blocks)
    specs = _solve_equations([(spec, _with_orders(spec, laws, block, orders)) for spec, laws, block in collected])
    hybrid = _Hybrid()
    specs = [(spec, _with_hybrid(spec, laws, declarations, hybrid)) for spec, laws in specs]
    fields = _Fields()
    specs = [(spec, _with_neighbor_reads(spec, laws, declarations, fields)) for spec, laws in specs]
    agents = {block.node.path for block in blocks if block.node.is_agent}
    if owners is not None:
        specs = [(spec, [law for law in laws if _law_owner(spec, law, agents) in owners]) for spec, laws in specs]
    specs, drawing = _with_draws(specs, declarations)
    kinds: dict[str, set[type]] = defaultdict(set)
    for spec, laws in specs:
        for law in laws:
            if not (isinstance(law, Invariant) and law.is_member):
                kinds[_law_owner(spec, law, agents)].add(_kind(law))
    try:
        form = _infer_form(kinds.values())
    except IncompatibleLawsError:
        if owners is not None:
            raise
        plan = _plan_partitions(system, kinds)
        return _compile_partitions(system, backend, plan)
    specs, reductions = _with_reductions(specs, hybrid, declarations, drawing.draws.size if drawing.draws else 0)
    lowered = _lower_laws(specs, declarations)
    members = sum(block.node.members or 0 for block in blocks if not block.node.is_agent)
    stepwise = form in {Form.AGENT, Form.LATTICE, Form.MONTE_CARLO, Form.RENEWAL}
    chosen = resolve_backend('numpy' if stepwise else backend, members)
    fused = resolve_backend(backend, members).name == 'numba'
    functions = _lower(form, lowered, declarations, chosen, reductions, drawing=drawing)
    states, parameters = Layout(declarations.states), Layout(declarations.parameters)
    functions['boundary'] = _wrapping_angles(cast('VectorFunction | None', functions.get('boundary')), states)
    functions['equations'] = _written_equations(cast('dict[str, Expression]', functions['equations']), states, fields)
    volumes, constrained = _with_discretizations(functions, specs, declarations, blocks)
    jacobian = (
        ExactJacobian(
            tuple(lowered.kernels[Rate]),
            reductions,
            states.size,
            declarations.time,
            volumes=volumes,
            symmetries=tuple(_symmetries(blocks)),
            constrained=constrained[0],
        )
        if form is Form.ODE
        else None
    )
    initial_state = np.concatenate([np.zeros(0), *declarations.initial_states])
    initial_state[constrained[0]] = constrained[1]
    compiled_elements = tuple(
        CompiledElement(
            path=block.node.path,
            element=block.node.component,
            states={
                name: f'{block.node.path}.{name}'
                for name in declarations_of(block.node.component_type, VariableKind.STATE)
            },
            parameters={
                name: f'{block.node.path}.{name}'
                for name in declarations_of(block.node.component_type, VariableKind.PARAMETER)
            },
            visual=_visual_of(block.node),
            members=block.node.members,
        )
        for block in elements.values()
        if not block.node.is_agent
    )
    symbols = {entry.name: block.symbols[variable] for block in blocks for variable, entry in block.entries.items()}
    symbols.update({item.entry.name: item.symbol for item in hybrid.renewals})
    compiled_interactions = tuple(
        CompiledInteraction(
            path=block.node.path,
            interaction=block.node.component,
            roles=next(spec.roles for spec, _ in specs if spec.owner == block.node.path),
        )
        for block in blocks
        if _is_interaction(block.node)
    )
    return Problem(
        form=form,
        states=states,
        parameters=parameters,
        initial_state=initial_state,
        initial_parameters=np.concatenate([np.zeros(0), *declarations.initial_parameters]),
        time=declarations.time,
        symbols=symbols,
        elements=compiled_elements,
        interactions=compiled_interactions,
        system=system,
        agents=_compiled_agents(blocks, specs),
        model_rules=tuple(
            (spec.owner, law)
            for spec, laws in specs
            for law in laws
            if isinstance(law, Rule) and spec.owner not in {block.node.path for block in blocks if block.node.is_agent}
        ),
        agent_space=_agent_space(blocks),
        companions=_companions(blocks),
        delays=_compile_delays(hybrid, declarations.time),
        events=(
            *_compile_events(hybrid, declarations.time, reductions),
            *(_hidden_events(specs, declarations.time, reductions) if form is Form.ODE else ()),
        ),
        reductions=reductions.reductions,
        jacobian=jacobian,
        breakpoints=_breakpoints(specs, hybrid, declarations.time),
        history=_history(blocks),
        lattice=_compile_lattice(specs, declarations, fused),
        stable_step=min(fields.limits, default=None),
        diffusive_limits=tuple(fields.volume_limits),
        linear=tuple(fields.linear),
        volumes=volumes,
        constrained=constrained[0],
        matrices=checked,
        renewals=_compile_renewals(hybrid, declarations.time, owners),
        fractional=_fractional_rows(lowered) if form is Form.SDE else (),
        stock_flow=stock_flow,
        **functions,  # type: ignore[arg-type]
    )


def _compile_renewals(
    hybrid: _Hybrid,
    time: Expression,
    owners: frozenset[str] | None,
) -> tuple[CompiledRenewal, ...]:
    """Return the renewal laws of the components a problem holds, compiled for their members."""
    return tuple(
        compile_renewal(
            item.spec.owner,
            item.law,
            item.spec.references,
            time=time,
            rows=item.spec.rows,
            label=item.label,
            group=item.group,
            counts=item.entry.start + np.arange(item.entry.size, dtype=np.intp),
        )
        for item in hybrid.renewals
        if owners is None or item.group in owners
    )


def _hurst_function(kernel: Kernel, hurst: object) -> Callable[[Vector], Vector]:
    """Return the function of the parameter vector that gives the Hurst index of every row of a noise kernel."""
    rows = kernel.rows
    if SR(hurst).is_numeric():
        value = float(SR(hurst))
        return lambda p: np.full(rows, value)
    reference = kernel.references[SR(hurst)]
    empty = np.zeros(0)
    return lambda p: np.broadcast_to(np.asarray(reference.column_of(empty, p), dtype=float), (rows,)).copy()


def _fractional_rows(lowered: _Lowered) -> tuple[FractionalRows, ...]:
    """Return the noise rows that fractional noise laws drive, in the order of the noise rows."""
    rows, position = [], 0
    for kernel in lowered.kernels[Noise]:
        for item in kernel.outputs:
            found = lowered.hurst.get(id(item))
            if found is not None:
                positions = position + np.arange(kernel.rows, dtype=np.intp)
                rows.append(FractionalRows(found[0], positions, _hurst_function(kernel, found[1])))
            position += kernel.rows
    return tuple(rows)


def _lattice_context(spec: _KernelSpec, declarations: _Declarations, fused: bool) -> LatticeContext:
    """Return the lattice states and the arguments of the symbols a component's laws read."""
    states, arguments = {}, {}
    for item, reference in spec.references.items():
        entry = reference.entry
        owner, _, variable = entry.name.rpartition('.')
        block = declarations.blocks.get(owner)
        space = block.node.space if block is not None else None
        is_lattice = (
            isinstance(space, LatticeSpace | NetworkSpace)
            and block is not None
            and block.node.members is not None
            and not block.node.is_agent
        )
        if reference.is_state and is_lattice and not entry.member_shape:
            states[item] = LatticeState(entry.name, entry.start, space)  # type: ignore[arg-type]
        if not entry.member_shape:
            arguments[item] = (variable, reference.is_state, entry.start, entry.size)
    return LatticeContext(spec.owner, states, arguments, fused)


def _monte_carlo_step(spec: _KernelSpec, laws: list[Law], context: LatticeContext) -> LatticeStep | None:
    """Return the Monte Carlo sampling of a component's energy, move, and sampler, or `None` without them."""
    energies = [law for law in laws if isinstance(law, Energy)]
    moves = [law for law in laws if isinstance(law, Move)]
    samplers = [law for law in laws if isinstance(law, Sampler)]
    if not (energies or moves or samplers):
        return None
    if len(energies) != 1 or len(moves) != 1 or len(samplers) > 1:
        error_msg = (
            f'{spec.owner} declares {len(energies)} energies, {len(moves)} moves, and {len(samplers)} samplers. '
            'Monte Carlo sampling takes one energy, one move, and at most one sampler.'
        )
        raise ValidationError(error_msg)
    if samplers:
        sampler = samplers[0]
    elif hasattr(spec.namespace, 'temperature'):
        sampler = Sampler(temperature=spec.namespace.temperature)
    else:
        error_msg = f'{spec.owner} declares no sampler and no parameter `temperature` for the default sampler.'
        raise ValidationError(error_msg)
    return compile_monte_carlo(energies[0], moves[0], sampler, context)


def _compile_lattice(
    specs: list[tuple[_KernelSpec, list[Law]]],
    declarations: _Declarations,
    fused: bool,
) -> tuple[LatticeStep, ...]:
    """Return the steps of local rules, drives, relaxations, and Monte Carlo sampling in declaration order."""
    steps: list[LatticeStep] = []
    compilers = {
        LocalRule: compile_rule,
        Drive: compile_drive,
        Relaxation: compile_relaxation,
        Extremal: compile_extremal,
    }
    for spec, laws in specs:
        if not any(isinstance(law, LATTICE_LAWS) for law in laws):
            continue
        context = _lattice_context(spec, declarations, fused)
        steps += [compilers[type(law)](law, context) for law in laws if type(law) in compilers]  # type: ignore[operator]
        sampling = _monte_carlo_step(spec, laws, context)
        if sampling is not None:
            steps.append(sampling)
    return tuple(steps)


def _compile_fluxes(
    specs: list[tuple[_KernelSpec, list[Law]]],
    declarations: _Declarations,
) -> tuple[FiniteVolume, ...]:
    """Return the conservation laws of every finite-volume field from the flux laws of its component."""
    volumes = []
    for spec, laws in specs:
        fluxes = [law for law in laws if isinstance(law, Flux)]
        if not fluxes:
            continue
        targets = [(law, _target_references(law.target, spec)[0]) for law in fluxes]
        owners = {reference.entry.name.rpartition('.')[0] for _, reference in targets}
        for owner in sorted(owners):
            block = declarations.blocks[owner]
            place = block.node.component
            if not isinstance(place, Field):
                error_msg = f'{spec.owner} declares fluxes of {owner}, which is not a field.'
                raise ValidationError(error_msg)
            mine = [(law, reference) for law, reference in targets if reference.entry.name.startswith(f'{owner}.')]
            volumes.append(compile_volumes(mine, spec.references, place, owner))
    return tuple(volumes)


def _mesh_constraints(blocks: list[_Block]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Return the state indices and values of the boundary nodes of finite element fields with fixed boundaries."""
    indices: list[npt.NDArray[np.intp]] = []
    values: list[npt.NDArray[np.float64]] = []
    for block in blocks:
        place = block.node.component
        if not isinstance(place, Field) or not isinstance(place.space, Mesh):
            continue
        if place.discretization != 'finite-element':
            error_msg = f'Field {block.node.path} lives on a mesh, which needs the finite-element discretization.'
            raise ValidationError(error_msg)
        nodes = place.space.boundary_nodes()
        coordinates = place.space.nodes[nodes]
        for variable, entry in block.entries.items():
            if variable not in declarations_of(place.element, VariableKind.STATE):
                continue
            kind, value = place.boundary_of(variable)
            if kind != 'fixed':
                continue
            fixed = value(*coordinates.T) if callable(value) else value
            indices.append(entry.start + nodes * entry.columns)
            values.append(np.broadcast_to(np.asarray(fixed, dtype=float), nodes.shape).copy())
    if not indices:
        return np.zeros(0, dtype=np.intp), np.zeros(0)
    return np.concatenate(indices).astype(np.intp), np.concatenate(values)


def _symmetries(blocks: list[_Block]) -> list[tuple[npt.NDArray[np.intp], tuple[int, ...], list[tuple[int, int]]]]:
    """Return the state indices, the lattice shape, and the axes with their signs of every symmetric field state."""
    found = []
    for block in blocks:
        place = block.node.component
        if not isinstance(place, Field) or not place.symmetry:
            continue
        space = place.space
        if not isinstance(space, LatticeSpace) or space.boundary != 'periodic':
            error_msg = f'Field {block.node.path} declares symmetries, which need a periodic lattice.'
            raise ValidationError(error_msg)
        for variable, parities in place.symmetry.items():
            entry = block.entries.get(variable)
            if entry is None or len(parities) != space.n_dim:
                error_msg = f'The symmetry of {block.node.path}.{variable} needs a state and one parity per axis.'
                raise ValidationError(error_msg)
            signs = []
            for axis, parity in enumerate(parities):
                if parity not in {'odd', 'even', None}:
                    error_msg = f"The parity {parity!r} of {block.node.path}.{variable} is not 'odd', 'even', or None."
                    raise ValidationError(error_msg)
                if parity is not None:
                    signs.append((axis, -1 if parity == 'odd' else 1))
            indices = entry.start + np.arange(place.size) * entry.columns
            found.append((indices, space.dimensions, signs))
    return found


def _with_symmetries(drift: Any, symmetries: list[Any]) -> Any:  # noqa: ANN401
    """Return the drift with the rates of symmetric states projected onto their odd or even parts."""
    if not symmetries or drift is None:
        return drift

    def symmetric(t: float, x: Vector, p: Vector) -> Vector:
        rates = np.array(drift(t, x, p), dtype=float)
        for indices, dimensions, signs in symmetries:
            values = np.reshape(rates[indices], dimensions)
            for axis, sign in signs:
                mirrored = np.roll(np.flip(values, axis=axis), 1, axis=axis)
                values = (values + sign * mirrored) / 2
            rates[indices] = values.ravel()
        return rates

    return symmetric


def _with_constraints(drift: Any, indices: npt.NDArray[np.intp]) -> Any:  # noqa: ANN401
    """Return the drift with the rates of fixed boundary nodes held at zero."""
    if not indices.size or drift is None:
        return drift

    def constrained(t: float, x: Vector, p: Vector) -> Vector:
        rates = np.array(drift(t, x, p), dtype=float)
        rates[indices] = 0.0
        return rates

    return constrained


def _with_discretizations(
    functions: dict[str, Any],
    specs: list[tuple[_KernelSpec, list[Law]]],
    declarations: _Declarations,
    blocks: list[_Block],
) -> tuple[tuple[FiniteVolume, ...], tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]]:
    """Add flux differences, symmetries, and fixed boundary nodes to the drift, returning the laws and fixed nodes."""
    volumes = _compile_fluxes(specs, declarations)
    symmetries = _symmetries(blocks)
    constrained, fixed = _mesh_constraints(blocks)
    drift = _with_volumes(functions.get('drift'), volumes)
    drift = _with_symmetries(drift, symmetries)
    drift = _with_constraints(drift, constrained)
    if volumes or symmetries or constrained.size:
        functions['drift'] = drift
        functions['accelerated'] = None
    return volumes, (constrained, fixed)


def _with_volumes(drift: Any, volumes: tuple[FiniteVolume, ...]) -> Any:  # noqa: ANN401
    """Return the drift with the flux differences of finite-volume fields added."""
    if not volumes or drift is None:
        return drift

    def conserved(t: float, x: Vector, p: Vector) -> Vector:
        rates = drift(t, x, p)
        for volume in volumes:
            rates = rates + volume.rates(x, p)
        return rates

    return conserved


def _history(blocks: list[_Block]) -> dict[str, Any]:
    """Return the history every state declares, by the name of its layout entry."""
    history = {}
    for block in blocks:
        if _is_interaction(block.node) or block.node.is_agent:
            continue
        declarations = declarations_of(block.node.component_type, VariableKind.STATE)
        for variable, entry in block.entries.items():
            declaration = declarations.get(variable)
            if declaration is not None and declaration.history is not None:
                history[entry.name] = declaration.history
    return history


def _wrapping_angles(boundary: VectorFunction | None, states: Layout) -> VectorFunction | None:
    """Return a boundary function that also wraps every angle state into `(-pi, pi]`."""
    indices = np.concatenate(
        [np.zeros(0, dtype=np.intp)]
        + [np.arange(entry.start, entry.start + entry.size) for entry in states if entry.domain is Domain.ANGLE],
    )
    if not indices.size:
        return boundary

    def wrapped(t: float, x: Vector, p: Vector) -> Vector:
        """Apply the boundary laws, then wrap the angle states."""
        state = x if boundary is None else boundary(t, x, p)
        state = state.copy()
        state[indices] = wrap_angle(state[indices])
        return state

    return wrapped


def _role_paths(system: System, scope: str = '') -> set[str]:
    """Return the paths from the root of everything that roles of a system tree bind to."""
    paths = set()
    for interaction in system.interactions:
        for name in declarations_of(interaction, VariableKind.ROLE):
            value = getattr(interaction, name)
            labels = [value] if isinstance(value, str) else list(value) if isinstance(value, list | tuple) else []
            paths |= {f'{scope}/{label}' if scope else label for label in labels if isinstance(label, str)}
    for child in system.systems:
        paths |= _role_paths(child, f'{scope}/{child.label}' if scope else child.label)
    return paths


def _with_held_observables(system: System, read: set[str], scope: str = '') -> tuple[System, list[str]]:
    """Return a system tree in which every observable in `read` is also an aggregate, and their paths."""
    held: list[str] = []
    aggregates = []
    for observable in getattr(system, 'observables', ()):
        path = f'{scope}/{observable.label}' if scope else observable.label
        if path not in read:
            continue
        if observable.group is None or observable.expression is not None:
            error_msg = (
                f'A role binds to the observable {path}, which reads agents through an expression or an earlier '
                'observable. Laws read observables of agents only when they reduce a variable or count the agents. '
                'Declare it with `group` and `variable`.'
            )
            raise ValidationError(error_msg)
        aggregates.append(
            Aggregate(observable.label, group=observable.group, reduce=observable.reduce, variable=observable.variable),
        )
        held.append(path)
    children = []
    for child in system.systems:
        new_child, child_held = _with_held_observables(child, read, f'{scope}/{child.label}' if scope else child.label)
        children.append(new_child)
        held += child_held
    if not held:
        return system, held
    copied = copy.copy(system)
    copied.components = [*system.elements, *aggregates, *children]
    return copied, held


def _held_observables(system: System) -> set[str]:
    """Return the paths of the observables that roles bind to and laws read only at synchronisation points."""
    nodes = flatten_system(system)
    lowerable = _lowerable_observables(system, nodes)
    declared = {_join(scope, observable.label) for scope, observable in observables_of(system)}
    return (_role_paths(system) & declared) - set(lowerable)


def _readers_of(problem: Problem, path: str) -> set[str]:
    """Return the paths of the elements and groups that interactions reading an observable bind their other roles to."""
    readers: set[str] = set()
    for interaction in problem.interactions:
        bound = [
            item for value in interaction.roles.values() for item in ([value] if isinstance(value, str) else value)
        ]
        if path in bound:
            readers |= set(bound) - {path}
    return readers


def _check_held_reads(problem: Problem, held: list[str]) -> None:
    """Raise an error when a law reads an observable that cannot be lowered from the partition that computes it."""
    groups = {
        _join(scope, observable.label): _join(scope, observable.group or '')
        for scope, observable in observables_of(problem.system)  # type: ignore[arg-type]
    }
    for path in held:
        readers = _readers_of(problem, path)
        shared = [item.name for item in problem.partitions if groups[path] in item.owners and readers & item.owners]
        if problem.form is not Form.COUPLED or shared:
            error_msg = (
                f'A law reads the observable {path} in the partition that computes it, but the observable is a Python '
                'function of the values or reduces agents, which laws cannot read at the current state. Write it '
                'symbolically with a variable or an expression of the members, or move the reading law into another '
                'subsystem of a different form.'
            )
            raise ValidationError(error_msg)


def compile_system(
    system: System,
    backend: str | Backend = 'auto',
    observables: Sequence[Observable] = (),
    *,
    solver: object = 'auto',
    initial_state: Mapping[str, object] | None = None,
) -> Problem:
    """Compile a system into a problem of the form its law kinds determine.

    Elements and groups are compiled once each, whatever the size of a group. Laws written for one member are
    evaluated for every member, and pair interactions are evaluated for every pair within their cutoff.

    Args:
        system:
            The system to compile.

        backend:
            `'auto'`, a registered backend name such as `'numpy'` or `'numba'`, or a backend instance. `'auto'` picks
            Numba for systems whose groups hold at least 1000 members in total and NumPy otherwise.

        observables:
            The observables to compute at every frame, labelled by their path from the root, such as
            `village/mean_wealth` for an observable of the nested system `village`.

        solver:
            The solver the problem runs on, `'auto'` or a solver instance, whose limits compiling checks, such as the
            largest dimension of a Hilbert space.

        initial_state:
            Initial values that replace those of the components, by the dotted name of the state, such as
            `{'rabbits.n': 20}`. The whole state of a quantum system is `<system label>.state`, which takes a vector, a
            density matrix, or `'ground'` for the lowest eigenstate of the Hamiltonian at the initial parameters.

    Returns:
        problem:
            The compiled problem.

    Raises:
        ValidationError:
            A child has the wrong type, a value is outside its domain or has the wrong shape, or a law is malformed.

        LabelError:
            A label is not valid, duplicated, or produces a symbol collision.

        RoleError:
            A role refers to an element that does not exist, or an interaction reads groups of different sizes member
            by member.

        SpaceError:
            A vector variable has no enclosing space.

        IncompatibleLawsError:
            The system mixes law kinds that no form supports.

        BackendError:
            The backend is not registered, or an agent group declares a complex state.

        ValidationError:
            Validation finds problems of more than one kind, all listed in the message.

    Examples:
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowth, Population
        >>> problem = compile_system(
        ...     System(
        ...         components=[Population('rabbits', n=10)],
        ...         interactions=[ExponentialGrowth('growth', target='rabbits', r=0.5)],
        ...     ),
        ... )
        >>> problem.form.name
        'ODE'
        >>> problem.equations
        {'rabbits.n': n_rabbits*r_growth}
    """
    from ._validation import validate  # noqa: PLC0415  # validation reads the compiler's census

    validate(system, backend=backend, observables=observables).raise_errors()
    system = with_observables(system, observables)
    if is_rewriting(system):
        return compile_rewriting(system)
    initial = dict(initial_state or {})
    if contains_quantum(system):
        whole = f'{system.label}.state'
        _check_initial_names(set(initial) - {whole}, [whole])
        limit = getattr(solver, 'max_dimension', DEFAULT_MAX_DIMENSION)
        return compile_quantum(system, initial=initial.get(whole), max_dimension=limit)
    system, held = _with_held_observables(system, _held_observables(system))
    problem = _compile(system, backend, None)
    _check_held_reads(problem, held)
    return with_initial_state(problem, initial)


def _check_initial_names(unknown: set[str], names: Sequence[str]) -> None:
    """Raise an error for initial values of states the problem does not have."""
    if unknown:
        error_msg = (
            f'The initial state names {", ".join(sorted(unknown))}, which are not states of the system. Use the dotted '
            f'names of states, such as {", ".join(list(names)[:3])}.'
        )
        raise ValidationError(error_msg)


def with_initial_state(problem: Problem, initial_state: Mapping[str, object]) -> Problem:
    """Return a problem whose initial state takes given values of some states.

    Args:
        problem:
            The compiled problem.

        initial_state:
            The initial values by dotted state name.

    Returns:
        problem:
            The problem with the new initial state, or the same problem when no value is given.

    Raises:
        ValidationError:
            A name is not a state of the problem, or a value has the wrong shape.
    """
    if not initial_state:
        return problem
    _check_initial_names(set(initial_state) - set(problem.states.names), problem.states.names)
    state = problem.initial_state.copy()
    for name, value in initial_state.items():
        entry = problem.states.entry(name)
        try:
            state[entry.indices] = entry.encode(value)
        except ValueError as error:
            error_msg = f'The initial value of {name} does not fit its shape {entry.shape}: {error}'
            raise ValidationError(error_msg) from error
    return replace(problem, initial_state=state)


def partition_system(system: System, backend: str | Backend = 'auto') -> Problem:
    """Compile a system as one partition per subsystem, also when the whole system fits one form.

    The elements of the root system form one partition and every child subsystem forms another, so a co-simulation
    can advance them with separate solvers.

    Args:
        system:
            The system to compile.

        backend:
            The backend of every partition.

    Returns:
        problem:
            The partitioned problem.

    Raises:
        IncompatibleLawsError:
            A partition mixes law kinds that no form supports.

    Examples:
        >>> from skcomplex.core.compile import partition_system
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowth, Population
        >>> system = System(
        ...     components=[
        ...         System('a', components=[Population('x')], interactions=[ExponentialGrowth('g', target='x')]),
        ...         System('b', components=[Population('y')], interactions=[ExponentialGrowth('g', target='y')]),
        ...     ],
        ... )
        >>> [partition.name for partition in partition_system(system).partitions]
        ['a', 'b']
    """
    nodes = flatten_system(system)
    elements = {node.path for node in nodes if not _is_interaction(node)}
    own = {path for path in elements if '/' not in path}
    plan = [(system.label, own)] if own else []
    for child in system.systems:
        plan.append((child.label, {path for path in elements if path.startswith(f'{child.label}/')}))
    return _compile_partitions(system, backend, plan)
