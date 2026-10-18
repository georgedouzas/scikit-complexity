"""Compile a system into a problem of one mathematical form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import SR

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
    Element,
    Group,
    Jump,
    Law,
    MesaSystem,
    Noise,
    PairInteraction,
    Rate,
    Rule,
    Space,
    Symbols,
    System,
    Update,
)
from ._backends import Backend, resolve_backend
from ._kernels import Kernel, Output, PairSearch, Reference, Source
from ._layout import Entry, Layout
from ._lowering import symbolic_components
from ._problem import CompiledAgents, CompiledElement, CompiledInteraction, Partition, Problem, Vector, VectorFunction
from ._tree import Node, coerce_value, flatten_system, shape_of

FORMS: dict[frozenset[type], Form] = {
    frozenset(): Form.ODE,
    frozenset({Rate}): Form.ODE,
    frozenset({Noise}): Form.SDE,
    frozenset({Rate, Noise}): Form.SDE,
    frozenset({Jump}): Form.JUMP,
    frozenset({Update}): Form.DISCRETE,
    frozenset({Rule}): Form.AGENT,
}
ASSIGNED_KINDS = (Update, Boundary)


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
    kinds = Rate | Noise | Jump | Update | Boundary | Rule
    if not isinstance(laws, list | tuple) or not all(isinstance(law, kinds) for law in laws):
        error_msg = f'`laws` of {spec.owner} must return a list of Rate, Noise, Jump, Update, Boundary, or Rule laws.'
        raise ValidationError(error_msg)
    if spec.pairs is not None and not all(isinstance(law, Rate) for law in laws):
        error_msg = f'Pair interaction {spec.owner} returns laws other than Rate, which pair interactions do not take.'
        raise ValidationError(error_msg)
    return list(laws)


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
            if isinstance(law, Rule):
                continue
            if isinstance(law, Jump):
                output = _outputs_of(law.rate, [None], f'Jump law of {spec.owner}')[0]
                changes = [
                    (_target_references(target, spec)[0], float(increment)) for target, increment in law.changes.items()
                ]
                lowered.jump_changes[id(output)] = changes
                outputs[Jump].append(output)
                continue
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
    kinds = frozenset(type(law) for law in laws if not isinstance(law, Boundary))
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
    return node.component_type.visual


def _law_owner(spec: _KernelSpec, law: Law, agents: set[str]) -> str:
    """Return the path of the element or group whose state a law changes, or that a rule acts for.

    A rule acts for the first agent group among its roles, or for its first role when none is an agent group.
    """
    if isinstance(law, Rule):
        paths = list(spec.roles.values())
        return next((path for path in paths if path in agents), next(iter(paths), spec.owner))
    target = next(iter(law.changes)) if isinstance(law, Jump) else law.target
    return _target_references(target, spec)[0].entry.name.rpartition('.')[0]


def _fits(kinds: set[type]) -> bool:
    """Check whether a set of law kinds fits one form."""
    return frozenset(kind for kind in kinds if kind is not Boundary) in FORMS


def _mixed_error(name: str, kinds: set[type]) -> IncompatibleLawsError:
    """Return the error of a subsystem whose own laws fit no form."""
    found = ', '.join(sorted(kind.__name__ for kind in kinds if kind is not Boundary))
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
    chosen = resolve_backend('numpy' if form is Form.AGENT else backend, members)
    functions = _lower(form, lowered, declarations, chosen)
    states, parameters = Layout(declarations.states), Layout(declarations.parameters)
    functions['boundary'] = _wrapping_angles(cast('VectorFunction | None', functions.get('boundary')), states)
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
        initial_state=np.concatenate([np.zeros(0), *declarations.initial_states]),
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
        **functions,  # type: ignore[arg-type]
    )


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
