"""Compile a system into a problem of one mathematical form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import itertools
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, matrix

from ..core import (
    Domain,
    Expression,
    Form,
    IncompatibleLawsError,
    LabelError,
    RoleError,
    SymbolicVector,
    ValidationError,
    Variable,
    VariableKind,
    VisualSpec,
    components_of,
    declarations_of,
    domain_of,
    symbol,
    vector,
)
from ..model import (
    Aggregate,
    Boundary,
    Call,
    Drive,
    Element,
    Energy,
    Equation,
    Event,
    Field,
    Flux,
    Group,
    Jump,
    LatticeSpace,
    Law,
    LocalRule,
    MesaSystem,
    Mesh,
    Move,
    Noise,
    PairInteraction,
    Rate,
    Relaxation,
    Rule,
    Sampler,
    Space,
    Symbols,
    System,
    Update,
)
from ._backends import Backend, resolve_backend
from ._fields import FieldLowering, has_operators
from ._hybrid import Breakpoints, CompiledEvent, Delay, Scalars, breakpoints_of, flat_sources, scalar_function
from ._kernels import Kernel, Output, PairSearch, Reference, Source
from ._lattices import (
    LatticeContext,
    LatticeState,
    LatticeStep,
    compile_drive,
    compile_monte_carlo,
    compile_relaxation,
    compile_rule,
)
from ._layout import Entry, Layout
from ._lowering import symbolic_components
from ._problem import CompiledAgents, CompiledElement, CompiledInteraction, Partition, Problem, Vector, VectorFunction
from ._splitting import LinearTerm, linear_terms
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
        for count in (1, 2, 3)
        for kinds in itertools.combinations((LocalRule, Drive, Relaxation), count)
    },
    frozenset({Energy, Move}): Form.MONTE_CARLO,
    frozenset({Energy, Move, Sampler}): Form.MONTE_CARLO,
}
ASSIGNED_KINDS = (Update, Boundary)
LATTICE_LAWS = (LocalRule, Drive, Relaxation, Energy, Move, Sampler)
KERNEL_FREE = (*LATTICE_LAWS, Flux)


@dataclass
class _Block:
    """Variables of a component with the symbols of one member and the components they stand for."""

    node: Node
    symbols: dict[str, Expression | SymbolicVector] = field(default_factory=dict)
    entries: dict[str, Entry] = field(default_factory=dict)
    references: dict[Expression, Reference] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        """Number of members, 1 for a single element or an interaction."""
        return self.node.members or 1

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
        symbols = {
            name: (vector(mapping[item] for item in value) if isinstance(value, SymbolicVector) else mapping[value])
            for name, value in self.symbols.items()
        }
        return Symbols(**symbols), references


@dataclass
class _Declarations:
    """Entries, initial values, and blocks collected from the components."""

    time: Expression
    states: list[Entry] = field(default_factory=list)
    parameters: list[Entry] = field(default_factory=list)
    initial_states: list[Vector] = field(default_factory=list)
    initial_parameters: list[Vector] = field(default_factory=list)
    blocks: dict[str, _Block] = field(default_factory=dict)

    def _declare(self, block: _Block, variable: str, declaration: Variable, is_state: bool) -> None:
        """Add one variable of a component to its block."""
        node = block.node
        entries, values = (self.states, self.initial_states) if is_state else (self.parameters, self.initial_parameters)
        name = f'{node.path}.{variable}'
        member_shape = shape_of(name, declaration, node.space)
        members = None if node.is_agent else node.members
        shape = member_shape if members is None else (members, *member_shape)
        values.append(coerce_value(name, node.value_of(variable), member_shape, declaration.domain, members))
        entry = Entry(name, sum(item.size for item in entries), shape, declaration.domain, members)
        entries.append(entry)
        base = f'{variable}_{re.sub(r"[^0-9A-Za-z]", "_", node.path)}'
        domain = declaration.domain
        components = (
            [symbol(f'{base}_{index}', domain) for index in range(member_shape[0])]
            if member_shape
            else [symbol(base, domain)]
        )
        block.symbols[variable] = vector(components) if member_shape else components[0]
        block.entries[variable] = entry
        for column, item in enumerate(components):
            block.references[item] = Reference(entry, is_state, column)

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


@dataclass
class _KernelSpec:
    """Namespace, references, and evaluation indices that the laws of one component are lowered with."""

    owner: str
    namespace: Symbols
    references: dict[Expression, Reference]
    rows: int
    pairs: PairSearch | None = None
    roles: dict[str, str] = field(default_factory=dict)


@dataclass
class _Lowered:
    """Kernels of every law kind with the extra data of jumps."""

    kernels: dict[type, list[Kernel]] = field(default_factory=lambda: defaultdict(list))
    jump_changes: dict[int, list[tuple[Reference, float]]] = field(default_factory=dict)


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


def _resolve_role(node: Node, role: str, elements: dict[str, _Block]) -> _Block:
    """Return the block of the element or group bound to a role of an interaction."""
    label = getattr(node.component, role)
    path = f'{node.scope}{label}' if isinstance(label, str) else None
    if path not in elements:
        where = node.scope.rstrip('/') or 'the root system'
        error_msg = (
            f'Interaction {node.path} binds the role {role!r} to {label!r}, which is not an element or group path '
            f'relative to {where}.'
        )
        raise RoleError(error_msg)
    return elements[path]


def _member_spec(block: _Block, time: Expression) -> _KernelSpec:
    """Return the kernel spec of the intrinsic laws of an element or group."""
    namespace, references = block.namespace(Source.MEMBER)
    return _KernelSpec(block.node.path, Symbols(t=time, **vars(namespace)), references, block.rows)


def _interaction_spec(block: _Block, elements: dict[str, _Block], time: Expression) -> _KernelSpec:
    """Return the kernel spec of an interaction that is evaluated once per member of its group."""
    node = block.node
    roles = {role: _resolve_role(node, role, elements) for role in declarations_of(node.component, VariableKind.ROLE)}
    groups = {bound.node.path for bound in roles.values() if bound.node.members is not None}
    if len(groups) > 1:
        error_msg = (
            f'Interaction {node.path} binds roles to the groups {", ".join(sorted(groups))}. An interaction acts on '
            'at most one group, and a PairInteraction couples members of two groups.'
        )
        raise RoleError(error_msg)
    parameters, references = block.namespace(Source.SINGLE)
    namespaces = {}
    for role, bound in roles.items():
        namespaces[role], role_references = bound.namespace(
            Source.MEMBER if bound.node.members is not None else Source.SINGLE,
        )
        references.update(role_references)
    rows = next((bound.rows for bound in roles.values() if bound.node.members is not None), 1)
    paths = {role: bound.node.path for role, bound in roles.items()}
    return _KernelSpec(node.path, Symbols(t=time, **vars(parameters), **namespaces), references, rows, roles=paths)


def _pair_spec(block: _Block, elements: dict[str, _Block], time: Expression) -> _KernelSpec:
    """Return the kernel spec of a pair interaction that is evaluated once per pair within its cutoff."""
    node = block.node
    interaction: PairInteraction = node.component
    first, second = _resolve_role(node, 'first', elements), _resolve_role(node, 'second', elements)
    position = type(interaction).position
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
    pairs = PairSearch(first.entries[position], second.entries[position], block.entries['cutoff'], interaction.period)
    namespace = Symbols(t=time, **vars(parameters), first=first_namespace, second=second_namespace)
    paths = {'first': first.node.path, 'second': second.node.path}
    return _KernelSpec(node.path, namespace, references, 0, pairs, roles=paths)


def _collect_laws(spec: _KernelSpec, component: object) -> list[Law]:
    """Return the laws a component returns for its namespace after checking their kinds."""
    laws = component.laws(spec.namespace)  # type: ignore[attr-defined]
    kinds = (Rate, Noise, Jump, Update, Boundary, Rule, Equation, Event, *KERNEL_FREE)
    if not isinstance(laws, list | tuple) or not all(isinstance(law, kinds) for law in laws):
        error_msg = (
            f'`laws` of {spec.owner} must return a list of Rate, Noise, Jump, Update, Boundary, Rule, Equation, Event, '
            'LocalRule, Drive, Relaxation, Energy, Move, Sampler, or Flux laws.'
        )
        raise ValidationError(error_msg)
    if spec.pairs is not None and not all(isinstance(law, Rate) for law in laws):
        error_msg = f'Pair interaction {spec.owner} returns laws other than Rate, which pair interactions do not take.'
        raise ValidationError(error_msg)
    return list(laws)


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
    return type(law)(target, law.value, law.order)


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
        if isinstance(law, (Rule, *KERNEL_FREE)):
            result.append(law)
            continue
        if isinstance(law, Equation):
            result.append(_relation(spec, law, orders))
            continue
        owner = f'A {type(law).__name__} law of {spec.owner}'
        if isinstance(law, Jump):
            result.append(Jump(_substitute(law.rate, spec, orders, owner), law.changes))
            continue
        if isinstance(law, Event):
            reset = {
                _substitute(SR(target), spec, orders, owner): _substitute(SR(value), spec, orders, owner)
                for target, value in (law.reset or {}).items()
            }
            guard = _substitute(SR(law.guard), spec, orders, owner)
            result.append(replace(law, guard=guard, reset=reset if law.reset is not None else None))
            continue
        value = _substitute(law.value, spec, orders, owner)
        if isinstance(law, Rate | Noise):
            result.append(_retarget(type(law)(law.target, value, law.order), spec, orders))
        else:
            result.append(type(law)(law.target, value))
    if not _is_interaction(block.node):
        result += _intrinsic_laws(spec, block, orders)
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


NOISE_CONVENTIONS = {'ito': 0.0, 'stratonovich': 0.5, 'anti-ito': 1.0}


@dataclass
class _Hybrid:
    """Delays, events, and corrected noise collected from the laws of a system."""

    delays: list[dict[str, Any]] = field(default_factory=list)
    placeholders: dict[tuple[int, str, int, str], Expression] = field(default_factory=dict)
    events: list[tuple[_KernelSpec, Event]] = field(default_factory=list)
    noisy: set[tuple[str, int]] = field(default_factory=set)


def _delay_minimum(
    spec: _KernelSpec,
    tau: Expression,
    minimum: Expression | None,
    declarations: _Declarations,
) -> float:
    """Return the smallest value of a delay: the delay itself, its declared minimum, or a parameter's value."""
    if SR(tau).is_numeric():
        return float(tau)
    if minimum is not None and SR(minimum).is_numeric():
        return float(minimum)
    reference = spec.references.get(SR(tau))
    if reference is not None and not reference.is_state:
        values = np.concatenate([np.zeros(0), *declarations.initial_parameters])
        return float(values[reference.entry.start + reference.column])
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
        members = entry.members
        indices = (
            entry.start + np.arange(members) * entry.columns + reference.column
            if members is not None
            else np.asarray([entry.start + reference.column])
        )
        initial = np.concatenate([np.zeros(0), *declarations.initial_states])[indices]
        index = len(hybrid.delays)
        hidden = Entry(
            f'__delayed_{index}',
            sum(item.size for item in declarations.parameters),
            (members,) if members is not None else (),
            Domain.REAL,
            members,
        )
        declarations.parameters.append(hidden)
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
    alpha: float,
) -> list[Law]:
    """Return laws with delays substituted and noise corrected, keeping events aside."""
    result: list[Law] = []
    for law in laws:
        if isinstance(law, Rule):
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
            rebuilt = type(law)(law.target, _substitute_delays(law.value, spec, declarations, hybrid), law.order)
            result.append(rebuilt)
            if isinstance(rebuilt, Noise):
                result += _noise_corrections(spec, rebuilt, alpha, hybrid)
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
    places = {
        entry.name: block.node.component
        for block in declarations.blocks.values()
        if isinstance(block.node.component, Field)
        for entry in block.entries.values()
    }
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
            result.append(type(law)(law.target, lowered, law.order))
        else:
            result.append(type(law)(law.target, lowered))
    fields.limits += lowering.limits
    return result


def _written_equations(equations: dict[str, Expression], states: Layout, fields: _Fields) -> dict[str, Expression]:
    """Return equations with the stencils of operators replaced by the operators as written."""
    written = dict(equations)
    for (name, column), pairs in fields.written.items():
        key = states.entry(name).generic_names[column]
        if key in written:
            written[key] = SR(written[key]) - sum(SR(item) for item, _ in pairs) + sum(SR(item) for _, item in pairs)
    return written


def _compile_events(hybrid: _Hybrid, time: Expression) -> tuple[CompiledEvent, ...]:
    """Return the events of a system compiled into guard and reset functions of the state vector."""
    compiled = []
    counts = Counter(spec.owner for spec, _ in hybrid.events)
    positions: Counter[str] = Counter()
    for spec, event in hybrid.events:
        position = positions[spec.owner]
        positions[spec.owner] += 1
        sources = flat_sources(spec.references, spec.owner)
        guard = scalar_function([event.guard], sources, time, spec.owner)
        targets = [item for target in (event.reset or {}) for item in components_of(target)]
        values = [
            item
            for target, value in (event.reset or {}).items()
            for item in (
                components_of(value) if isinstance(value, SymbolicVector) else [value] * len(components_of(target))
            )
        ]
        if any(item not in sources or not sources[item][0] for item in targets):
            error_msg = f'An event of {spec.owner} resets a symbol that is not a state of a single element.'
            raise ValidationError(error_msg)
        indices = np.asarray([sources[item][1] for item in targets], dtype=np.intp)
        reset = _reset_function(scalar_function(values, sources, time, spec.owner), indices) if targets else None
        compiled.append(
            CompiledEvent(
                event.label or (spec.owner if counts[spec.owner] == 1 else f'{spec.owner}:{position}'),
                _first_value(guard),
                reset,
                int(event.direction),
                bool(event.terminal),
            ),
        )
    return tuple(compiled)


def _first_value(function: Scalars) -> Callable[[float, Vector, Vector], float]:
    """Return a function that gives the single value of a scalar function."""

    def first(t: float, x: Vector, p: Vector) -> float:
        return function(t, x, p)[0]

    return first


def _reset_function(evaluate: Scalars, indices: npt.NDArray[np.intp]) -> VectorFunction:
    """Return a function that sets state components to the values of an event's reset."""

    def reset(t: float, x: Vector, p: Vector) -> Vector:
        state = x.copy()
        state[indices] = evaluate(t, x, p)
        return state

    return reset


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
        try:
            solution = matrix(SR, coefficients).solve_right(vector(right))
        except (ValueError, ZeroDivisionError) as error:
            error_msg = f'The highest derivatives of {descriptions} cannot be solved for: {error}.'
            raise ValidationError(error_msg) from error
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
    components = components_of(target)
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


def _check_assignments(spec: _KernelSpec, law: Law, targets: list[Reference], assigned: set[tuple[str, int]]) -> None:
    """Raise an error for an update or boundary law that sums over members or assigns a component twice."""
    kind = type(law).__name__
    for target in targets:
        if target.source is Source.SINGLE and spec.rows > 1:
            error_msg = f'A {kind} law of {spec.owner} targets a single element from a group, which has no meaning.'
            raise ValidationError(error_msg)
        key = (target.entry.name, target.column)
        if key in assigned:
            error_msg = f'A {kind} law of {spec.owner} sets {target.entry.name} a second time. Each takes one law.'
            raise ValidationError(error_msg)
        assigned.add(key)


def _lower_laws(specs: list[tuple[_KernelSpec, list[Law]]], declarations: _Declarations) -> _Lowered:
    """Return the kernels of every law kind."""
    states, parameters = Layout(declarations.states), Layout(declarations.parameters)
    lowered = _Lowered()
    assigned: dict[type, set[tuple[str, int]]] = defaultdict(set)
    for spec, laws in specs:
        outputs: dict[type, list[Output]] = defaultdict(list)
        for law in laws:
            if isinstance(law, (Rule, Event, *KERNEL_FREE)):
                continue
            if isinstance(law, Jump):
                output = _outputs_of(law.rate, [None], f'Jump law of {spec.owner}')[0]
                changes = [
                    (_target_references(target, spec)[0], float(increment)) for target, increment in law.changes.items()
                ]
                lowered.jump_changes[id(output)] = changes
                outputs[Jump].append(output)
                continue
            if isinstance(law, Equation):
                error_msg = f'An equation of {spec.owner} reached lowering before it was solved.'
                raise ValidationError(error_msg)
            targets = _target_references(law.target, spec)
            if isinstance(law, ASSIGNED_KINDS):
                _check_assignments(spec, law, targets, assigned[type(law)])
            outputs[type(law)].extend(_outputs_of(law.value, targets, f'{type(law).__name__} law of {spec.owner}'))  # type: ignore[arg-type]
        for kind, kind_outputs in outputs.items():
            lowered.kernels[kind].append(
                Kernel(
                    spec.owner,
                    declarations.time,
                    spec.references,
                    spec.rows,
                    states,
                    parameters,
                    spec.pairs,
                    kind_outputs,
                ),
            )
    return lowered


def _infer_form(laws: list[Law]) -> Form:
    """Return the form that the law kinds of a system determine."""
    kinds = frozenset(type(law) for law in laws if not isinstance(law, Boundary | Event | Flux))
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
            for column, item in enumerate(components_of(components)):
                key = (entry.name, column)
                if key in excluded:
                    continue
                default = item if form is Form.DISCRETE else SR(0)
                terms = contributions.get(key)
                name = entry.generic_names[column]
                equations[name] = sum(terms[1:], terms[0]) if terms else default
                symbols[name] = item
    return equations, symbols


def _lower(form: Form, lowered: _Lowered, declarations: _Declarations, backend: Backend) -> dict[str, object]:
    """Return the numeric functions and equations of a form."""
    kernels, size = lowered.kernels, sum(entry.size for entry in declarations.states)
    functions: dict[str, object] = vars(backend.lower(form, kernels, lowered.jump_changes, size)).copy()
    functions['backend'] = backend.name
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


def _compile_mesa(system: MesaSystem) -> Problem:
    """Return the agent-based problem of a wrapped Mesa model, whose numeric constructor arguments are parameters."""
    entries: list[Entry] = []
    values: list[float] = []
    for name, value in (system.parameters or {}).items():
        if isinstance(value, bool) or not Domain.REAL.contains(value) or np.ndim(value):
            continue
        entries.append(Entry(f'{system.label}.{name}', len(entries), (), Domain.REAL))
        values.append(float(value))
    return Problem(
        form=Form.AGENT,
        states=Layout([]),
        parameters=Layout(entries),
        initial_state=np.zeros(0),
        initial_parameters=np.asarray(values, dtype=float),
        time=symbol('t'),
        symbols={},
        equations={},
        equation_symbols={},
        has_callables=False,
        has_group_sums=False,
        elements=(),
        system=system,
    )


def _compiled_agents(blocks: list[_Block], specs: list[tuple[_KernelSpec, list[Law]]]) -> tuple[CompiledAgents, ...]:
    """Return the agent types and groups with their states, initial values, parameters, and rules."""
    rules = {spec.owner: tuple(law for law in laws if isinstance(law, Rule)) for spec, laws in specs}
    agents = []
    for block in blocks:
        node = block.node
        if not node.is_agent:
            continue
        states = tuple(declarations_of(node.component_type, VariableKind.STATE))
        agents.append(
            CompiledAgents(
                path=node.path,
                element=node.component,
                size=node.members or 1,
                states=states,
                values={name: node.value_of(name) for name in states},
                parameters={name: entry.name for name, entry in block.entries.items()},
                rules=rules.get(node.path, ()),
                visual=_visual_of(node),
            ),
        )
    return tuple(agents)


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
    """Return the visual of an element, or a group's own visual when it sets one."""
    if isinstance(node.component, Group) and node.component.visual is not None:
        return node.component.visual
    if isinstance(node.component, Field) and isinstance(node.component.space, LatticeSpace):
        return VisualSpec('image')
    if isinstance(node.component, Field) and isinstance(node.component.space, Mesh):
        return VisualSpec('mesh')
    return node.component_type.visual


def _law_owner(spec: _KernelSpec, law: Law, agents: set[str]) -> str:
    """Return the path of the element or group whose state a law changes, or that a rule acts for.

    A rule acts for the first agent group among its roles, or for its first role when none is an agent group.
    """
    if isinstance(law, Rule):
        paths = list(spec.roles.values())
        return next((path for path in paths if path in agents), next(iter(paths), spec.owner))
    if isinstance(law, Equation | Event | Energy | Sampler):
        return spec.owner
    if isinstance(law, Relaxation):
        target = next(iter(law.targets), None)
        return spec.owner if target is None else _target_references(target, spec)[0].entry.name.rpartition('.')[0]
    if isinstance(law, LocalRule):
        return _target_references(law.targets[0], spec)[0].entry.name.rpartition('.')[0]
    if isinstance(law, Flux):
        return _target_references(law.target, spec)[0].entry.name.rpartition('.')[0]
    target = next(iter(law.changes)) if isinstance(law, Jump) else law.target
    return _target_references(target, spec)[0].entry.name.rpartition('.')[0]


def _fits(kinds: set[type]) -> bool:
    """Check whether a set of law kinds fits one form."""
    return frozenset(kind for kind in kinds if kind not in {Boundary, Event, Flux}) in FORMS


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
    if _fits(set().union(*under.values())):
        return [(name, set(under))] if under else []
    own = {owner: owner_kinds for owner, owner_kinds in under.items() if '/' not in owner[len(scope) :]}
    own_kinds: set[type] = set().union(*own.values())
    if not system.systems or not _fits(own_kinds):
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


def _compile(system: System, backend: str | Backend, owners: frozenset[str] | None) -> Problem:
    """Return the problem of a system, keeping only the laws of the elements in `owners` when it is given."""
    nodes = flatten_system(system)
    declarations = _Declarations(time=symbol('t'))
    blocks = [declarations.declare_component(node) for node in nodes]
    _check_symbol_names(declarations.blocks)
    elements = {block.node.path: block for block in blocks if not _is_interaction(block.node)}
    specs: list[tuple[_KernelSpec, list[Law]]] = []
    for block in blocks:
        if not _is_interaction(block.node):
            spec = _member_spec(block, declarations.time)
        elif isinstance(block.node.component, PairInteraction):
            spec = _pair_spec(block, elements, declarations.time)
        else:
            spec = _interaction_spec(block, elements, declarations.time)
        specs.append((spec, _collect_laws(spec, _component_of(block.node))))
    orders = _Orders.of(blocks)
    specs = _solve_equations(
        [(spec, _with_orders(spec, laws, block, orders)) for (spec, laws), block in zip(specs, blocks, strict=True)],
    )
    noise = getattr(system, 'noise', 'ito')
    if noise not in NOISE_CONVENTIONS:
        error_msg = f'Unknown noise convention {noise!r}. Use one of {", ".join(NOISE_CONVENTIONS)}.'
        raise ValidationError(error_msg)
    hybrid = _Hybrid()
    specs = [(spec, _with_hybrid(spec, laws, declarations, hybrid, NOISE_CONVENTIONS[noise])) for spec, laws in specs]
    fields = _Fields()
    specs = [(spec, _with_fields(spec, laws, declarations, fields)) for spec, laws in specs]
    agents = {block.node.path for block in blocks if block.node.is_agent}
    if owners is not None:
        specs = [(spec, [law for law in laws if _law_owner(spec, law, agents) in owners]) for spec, laws in specs]
    try:
        form = _infer_form([law for _, laws in specs for law in laws])
    except IncompatibleLawsError:
        if owners is not None:
            raise
        kinds: dict[str, set[type]] = defaultdict(set)
        for spec, laws in specs:
            for law in laws:
                kinds[_law_owner(spec, law, agents)].add(type(law))
        plan = _plan_partitions(system, kinds)
        return _compile_partitions(system, backend, plan)
    lowered = _lower_laws(specs, declarations)
    members = sum(block.node.members or 0 for block in blocks if not block.node.is_agent)
    stepwise = form in {Form.AGENT, Form.LATTICE, Form.MONTE_CARLO}
    chosen = resolve_backend('numpy' if stepwise else backend, members)
    fused = resolve_backend(backend, members).name == 'numba'
    functions = _lower(form, lowered, declarations, chosen)
    states, parameters = Layout(declarations.states), Layout(declarations.parameters)
    functions['boundary'] = _wrapping_angles(cast('VectorFunction | None', functions.get('boundary')), states)
    functions['equations'] = _written_equations(cast('dict[str, Expression]', functions['equations']), states, fields)
    volumes, constrained = _with_discretizations(functions, specs, declarations, blocks)
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
    compiled_interactions = tuple(
        CompiledInteraction(
            path=block.node.path,
            interaction=block.node.component,
            roles={
                role: _resolve_role(block.node, role, elements).node.path
                for role in declarations_of(block.node.component, VariableKind.ROLE)
            },
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
        events=_compile_events(hybrid, declarations.time),
        breakpoints=_breakpoints(specs, hybrid, declarations.time),
        history=_history(system, states),
        lattice=_compile_lattice(specs, declarations, fused),
        stable_step=min(fields.limits, default=None),
        linear=tuple(fields.linear),
        volumes=volumes,
        constrained=constrained[0],
        **functions,  # type: ignore[arg-type]
    )


def _lattice_context(spec: _KernelSpec, declarations: _Declarations, fused: bool) -> LatticeContext:
    """Return the lattice states and the arguments of the symbols a component's laws read."""
    states, arguments = {}, {}
    for item, reference in spec.references.items():
        entry = reference.entry
        owner, _, variable = entry.name.rpartition('.')
        block = declarations.blocks.get(owner)
        space = block.node.space if block is not None else None
        is_lattice = isinstance(space, LatticeSpace) and block is not None and block.node.members is not None
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
    compilers = {LocalRule: compile_rule, Drive: compile_drive, Relaxation: compile_relaxation}
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


def _history(system: System, states: Layout) -> dict[str, Any]:
    """Return the declared history of a system after checking that it names states."""
    history = dict(getattr(system, 'history', None) or {})
    unknown = [name for name in history if name not in states]
    if unknown:
        error_msg = (
            f'The history names {", ".join(map(repr, unknown))}, which are not states. Available states are: '
            f'{", ".join(states.names)}.'
        )
        raise ValidationError(error_msg)
    return history


def _wrapping_angles(boundary: VectorFunction | None, states: Layout) -> VectorFunction | None:
    """Return a boundary function that also wraps every angle state into one turn."""
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
        state[indices] = np.mod(state[indices], 2 * np.pi)
        return state

    return wrapped


def _role_paths(system: System, scope: str = '') -> set[str]:
    """Return the paths from the root of everything that roles of a system tree bind to."""
    paths = {
        f'{scope}/{value}' if scope else value
        for interaction in system.interactions
        for name in declarations_of(interaction, VariableKind.ROLE)
        if isinstance(value := getattr(interaction, name), str)
    }
    for child in system.systems:
        paths |= _role_paths(child, f'{scope}/{child.label}' if scope else child.label)
    return paths


def _with_held_observables(system: System, read: set[str], scope: str = '') -> tuple[System, list[str]]:
    """Return a system tree in which every observable that a role binds to is also an aggregate, and their paths."""
    held: list[str] = []
    aggregates = []
    for observable in getattr(system, 'observables', ()):
        path = f'{scope}/{observable.label}' if scope else observable.label
        if path not in read:
            continue
        if observable.group is None or observable.expression is not None:
            error_msg = (
                f'A role binds to the observable {path}, but laws read only observables that reduce a variable or '
                'count the members of a group. Declare it with `group` and `variable`.'
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
    copied.elements = [*system.elements, *aggregates]
    copied.systems = children
    return copied, held


def compile_system(system: System, backend: str | Backend = 'auto') -> Problem:
    """Compile a system into a problem of the form its law kinds determine.

    Elements and groups are compiled once each, whatever the size of a group. Laws written for one member are
    evaluated for every member, and pair interactions are evaluated for every pair within their cutoff.

    Args:
        system:
            The system to compile.

        backend:
            `'auto'`, a registered backend name such as `'numpy'` or `'numba'`, or a backend instance. `'auto'` picks
            Numba for systems whose groups hold at least 1000 members in total and NumPy otherwise.

    Returns:
        problem:
            The compiled problem.

    Raises:
        ValidationError:
            A child has the wrong type, a value is outside its domain or has the wrong shape, or a law is malformed.

        LabelError:
            A label is not valid, duplicated, or produces a symbol collision.

        RoleError:
            A role refers to an element that does not exist, or an interaction binds two groups.

        SpaceError:
            A vector variable has no enclosing space.

        IncompatibleLawsError:
            The system mixes law kinds that no form supports.

        BackendError:
            The backend is not registered.

    Examples:
        >>> from skcomplex.compile import compile_system
        >>> from skcomplex.model import System
        >>> from skcomplex.models.ecology import ExponentialGrowth, Population
        >>> problem = compile_system(
        ...     System(
        ...         elements=[Population('rabbits', n=10)],
        ...         interactions=[ExponentialGrowth('growth', target='rabbits', r=0.5)],
        ...     ),
        ... )
        >>> problem.form.name
        'ODE'
        >>> problem.equations
        {'rabbits.n': n_rabbits*r_growth}
    """
    if isinstance(system, MesaSystem):
        return _compile_mesa(system)
    system, held = _with_held_observables(system, _role_paths(system))
    problem = _compile(system, backend, None)
    if held and problem.form is not Form.COUPLED:
        error_msg = (
            f'Laws read the observables {", ".join(held)} in a system of one form. Laws read observables only from '
            'other partitions of a co-simulation, so move the reading laws into another subsystem of a different form.'
        )
        raise ValidationError(error_msg)
    return problem


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
        >>> from skcomplex.compile import partition_system
        >>> from skcomplex.model import System
        >>> from skcomplex.models.ecology import ExponentialGrowth, Population
        >>> system = System(
        ...     systems=[
        ...         System('a', elements=[Population('x')], interactions=[ExponentialGrowth('g', target='x')]),
        ...         System('b', elements=[Population('y')], interactions=[ExponentialGrowth('g', target='y')]),
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
