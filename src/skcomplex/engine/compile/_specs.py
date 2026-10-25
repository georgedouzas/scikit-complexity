"""Build the kernel specs that the laws of every component are lowered with: namespaces, references, and rows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from sage.all__sagemath_symbolics import SR

from ...core.base import (
    BaseLaw,
    BaseSpace,
    Expression,
    LabelError,
    NumericDomain,
    Registry,
    RoleError,
    SymbolicVector,
    ValidationError,
    VariableKind,
    components_of,
    declarations_of,
    symbol,
    vector,
)
from ...core.components import GroupComponent
from ...core.laws import JumpLaw, Symbols, has_reductions, neighborhoods_in, reductions_in
from ...core.space import NetworkSpace, SiteSpace, neighbor_pairs
from ._declarations import ACTIVE, ENDS, Block, Declarations, is_interaction
from ._kernels import EdgeSearch, Reference, Search, Source
from ._real_parts import real_components_of
from ._tree import Node

MEMBER_REFERENCE = re.compile(r'^(?P<path>.+)\[(?P<index>-?\d+)\]$')
EDGE_NAMES = ('source', 'target', 'edge')

type Observe = Callable[[Node, object], Observed | None]
type Binding = Callable[[Block, dict[str, Block], Expression], KernelSpec]

BINDINGS = Registry[Binding]('binding')


def _resolve_path(node: Node, role: str, label: object, elements: dict[str, Block]) -> Block:
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


def _resolve_member(node: Node, role: str, reference: str, elements: dict[str, Block]) -> tuple[Block, int] | None:
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


def _resolve_entry(node: Node, role: str, entry: object, elements: dict[str, Block]) -> tuple[Block, int | None]:
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
    elements: dict[str, Block],
) -> list[tuple[Block, int | None]]:
    """Return the blocks and member indices of the entries of a variadic role, in the given order."""
    if not isinstance(entries, list | tuple):
        error_msg = (
            f'Interaction {node.path} binds its variadic role {role!r} to {entries!r}. Bind a list of element paths '
            'and group member references.'
        )
        raise RoleError(error_msg)
    return [_resolve_entry(node, role, entry, elements) for entry in entries]


def _entry_path(block: Block, index: int | None) -> str:
    """Return the path of an entry of a variadic role, with the member index of a member reference."""
    return block.node.path if index is None else f'{block.node.path}[{index}]'


def _entry_namespace(block: Block, index: int | None) -> tuple[Symbols, dict[Expression, Reference]]:
    """Return the namespace of an entry of a variadic role, an element or one member of a group."""
    return block.namespace(Source.SINGLE) if index is None else block.member_namespace(index)


def _resolve_set(node: Node, role: str, labels: Sequence[str], elements: dict[str, Block]) -> list[Block]:
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


def _set_symbols(members: list[Block], index: int) -> dict[Expression, SetSymbol]:
    """Return the symbols of one element of a set with the same symbol of every element."""
    label = ', '.join(member.node.path for member in members)
    symbols = {}
    for variable, value in members[index].symbols.items():
        for column, item in enumerate(real_components_of(value)):
            references = tuple(
                member.references[real_components_of(member.symbols[variable])[column]] for member in members
            )
            symbols[item] = SetSymbol(real_components_of(members[0].symbols[variable])[column], references, label)
    return symbols


def _interaction_specs(
    block: Block,
    elements: dict[str, Block],
    time: Expression,
    observe: Observe,
) -> list[KernelSpec]:
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
        role: resolve_role(node, role, elements)
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
        set_symbols: dict[Expression, SetSymbol] = {}
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
            role_namespace, role_references = item.namespace(
                Source.MEMBER if item.node.members is not None else Source.SINGLE,
            )
            namespaces[role] = with_other(item, role_namespace, role_references)
            references.update(role_references)
        specs.append(
            KernelSpec(
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
    found += [reduction.call for item in expressions for reduction in neighborhoods_in(item)]
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


def _check_aligned(spec: KernelSpec, groups: list[str]) -> None:
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


def _edge_roles(node: Node, elements: dict[str, Block]) -> tuple[Block, Block | None]:
    """Return the group of nodes and the group on edges, if any, that an interaction over neighbours binds."""
    blocks = [resolve_role(node, role, elements) for role in declarations_of(node.component, VariableKind.ROLE)]
    edges = [block for block in blocks if getattr(block.node.component, 'on', None) == 'edges']
    nodes = [block for block in blocks if block not in edges]
    if len(nodes) != 1 or len(edges) > 1 or nodes[0].node.members is None:
        error_msg = (
            f'Interaction {node.path} over neighbours binds {len(nodes)} groups of nodes and {len(edges)} groups on '
            'edges. Bind one role to a group with one member per node or site, and at most one role to a group on '
            'the edges of the same network.'
        )
        raise RoleError(error_msg)
    return nodes[0], next(iter(edges), None)


def _edge_layer(node: Node, nodes: Block, edges: Block | None) -> str | None:
    """Return the layer an interaction over neighbours runs on, checking it against the layer of its edge group."""
    declared = getattr(node.component, 'layer', None)
    if edges is None:
        return declared
    space, edge_space = group_space(nodes), group_space(edges)
    if edge_space is not space:
        error_msg = (
            f'Interaction {node.path} binds the edge group {edges.node.path}, which lives on another space than the '
            f'nodes {nodes.node.path}. Give the edge group the network of the nodes.'
        )
        raise RoleError(error_msg)
    layer = getattr(edges.node.component, 'layer', None)
    if declared is not None and declared != layer:
        error_msg = (
            f'Interaction {node.path} runs on the layer {declared!r}, but its edge group {edges.node.path} lives on '
            f'the layer {layer!r}. Give both one layer.'
        )
        raise RoleError(error_msg)
    return layer


def _changing_search(node: Node, nodes: Block, edges: Block | None, changing: Block) -> EdgeSearch:
    """Return the edge rows of a layer whose edges change, after checking the bound edge group is its group."""
    if edges is not None and edges is not changing:
        error_msg = (
            f'Interaction {node.path} binds the edge group {edges.node.path}, but the layer changes through the '
            f'variable edge group {changing.node.path}. Bind the variable edge group.'
        )
        raise RoleError(error_msg)
    if nodes.node.active is None:
        error_msg = (
            f'Interaction {node.path} runs on a layer whose edges change, but its nodes {nodes.node.path} are not a '
            'variable group. Give the group on the nodes `variable=True`.'
        )
        raise RoleError(error_msg)
    empty = np.zeros(0, dtype=np.intp)
    return EdgeSearch(empty, empty, changing.entries[ENDS[0]], changing.entries[ENDS[1]])


def _edge_search(
    node: Node,
    nodes: Block,
    edges: Block | None,
    elements: dict[str, Block],
) -> EdgeSearch:
    """Return the edges an interaction over neighbours runs on, after checking the sizes of its groups."""
    space = group_space(nodes)
    layer = _edge_layer(node, nodes, edges)
    changing = next(
        (
            block
            for block in elements.values()
            if block.node.active is not None
            and getattr(block.node.component, 'on', None) == 'edges'
            and block.node.space is space
            and getattr(block.node.component, 'layer', None) == layer
        ),
        None,
    )
    if changing is not None:
        return _changing_search(node, nodes, edges, changing)
    if isinstance(space, NetworkSpace):
        sources, targets = space.list_edges(layer)
        size = space.size
    elif isinstance(space, SiteSpace) and edges is None and layer is None:
        pairs = np.array(neighbor_pairs(space, space.size), dtype=np.intp).reshape(-1, 2)
        sources, targets = pairs[:, 0].copy(), pairs[:, 1].copy()
        size = space.size
    else:
        error_msg = (
            f'Interaction {node.path} over neighbours binds {nodes.node.path}, which lives on {space!r}. Bind a group '
            'on the nodes of a network, or on the sites of a lattice without edge groups or layers.'
        )
        raise RoleError(error_msg)
    if nodes.rows != size:
        error_msg = (
            f'Interaction {node.path} over neighbours binds {nodes.node.path} of {nodes.rows} members to a space of '
            f'{size} nodes. Bind a group with one member per node or site.'
        )
        raise RoleError(error_msg)
    if edges is not None and edges.rows != sources.size:
        error_msg = (
            f'Interaction {node.path} binds the edge group {edges.node.path} of {edges.rows} members to a layer of '
            f'{sources.size} edges. Give the edge group one member per edge of its layer.'
        )
        raise RoleError(error_msg)
    return EdgeSearch(sources, targets)


def _edge_spec(block: Block, elements: dict[str, Block], time: Expression) -> KernelSpec:
    """Return the kernel spec of an interaction over neighbours, evaluated once per edge of its layer."""
    node = block.node
    nodes, edges = _edge_roles(node, elements)
    search = _edge_search(node, nodes, edges, elements)
    parameters, references = block.namespace(Source.SINGLE)
    clashes = sorted(set(vars(parameters)) & {*EDGE_NAMES, 't'})
    if clashes:
        error_msg = (
            f'Interaction {node.path} over neighbours declares the parameters {", ".join(clashes)}, whose names its '
            'laws read as the ends and the member of an edge. Rename them.'
        )
        raise LabelError(error_msg)
    source, source_references = nodes.namespace(Source.FIRST, alias='__source')
    target, target_references = nodes.namespace(Source.SECOND, alias='__target')
    references.update(source_references)
    references.update(target_references)
    namespaces: dict[str, Symbols] = {'source': source, 'target': target}
    paths: dict[str, str | tuple[str, ...]] = {'source': nodes.node.path, 'target': nodes.node.path}
    groups: dict[str, int] = {}
    if edges is not None:
        edge, edge_references = edges.namespace(Source.MEMBER)
        references.update(edge_references)
        namespaces['edge'] = edge
        paths['edge'] = edges.node.path
        groups[edges.node.path] = edges.rows
    elif search.source is not None:
        changing = next(block for block in elements.values() if block.entries.get(ENDS[0]) == search.source)
        activity = symbol(f'active_{re.sub(r"[^0-9A-Za-z]", "_", node.path)}_edges', NumericDomain.REAL)
        references[activity] = Reference(changing.entries[ACTIVE], False, 0, Source.MEMBER)
        paths['edge'] = changing.node.path
    namespace = Symbols(t=time, **vars(parameters), **namespaces)
    group = edges.node.path if edges is not None else None
    return KernelSpec(node.path, namespace, references, search.size, search, roles=paths, group=group, groups=groups)


def resolve_role(node: Node, role: str, elements: dict[str, Block]) -> Block:
    """Return the block of the element or group bound to a role of an interaction."""
    return _resolve_path(node, role, getattr(node.component, role), elements)


def with_other(block: Block, namespace: Symbols, references: dict[Expression, Reference]) -> Symbols:
    """Return the namespace of a group's member with `other`, the namespace of a neighbour in reductions over them.

    A group that declares a variable named `other` keeps it, and its laws reduce over neighbours without it.
    """
    if block.node.members is None or 'other' in vars(namespace):
        return namespace
    other, other_references = block.namespace(Source.OTHER, alias='__other')
    references.update(other_references)
    return Symbols(**vars(namespace), other=other)


def member_spec(block: Block, time: Expression) -> KernelSpec:
    """Return the kernel spec of the intrinsic laws of an element or group."""
    namespace, references = block.namespace(Source.MEMBER)
    namespace = with_other(block, namespace, references)
    group = block.node.path if block.node.members is not None else None
    return KernelSpec(block.node.path, Symbols(t=time, **vars(namespace)), references, block.rows, group=group)


def settle_rows(spec: KernelSpec, laws: list[BaseLaw]) -> None:
    """Set the groups an interaction is evaluated over from the member symbols its laws read outside reductions.

    Laws without reductions keep the group their one group role binds to, so a single element bound next to it receives
    the sum over the members. Groups of one size that laws read outside reductions are read member by member, row `i` of
    each together.
    """
    values = [law.rate if isinstance(law, JumpLaw) else getattr(law, 'value', None) for law in laws]
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


def group_space(block: Block) -> BaseSpace | None:
    """Return the space of a group, its own or the enclosing one, shared by the groups of one network."""
    return block.node.space


def state_name(spec: KernelSpec, target: Expression) -> str:
    """Return the dotted name of the state a target symbol of a namespace stands for, or the symbol itself."""
    reference = spec.references.get(target)
    return reference.entry.name if reference is not None else str(target)


def target_references(target: object, spec: KernelSpec) -> list[Reference]:
    """Return the references of the state components a law targets."""
    components = real_components_of(target)
    references = [spec.references.get(component) for component in components]
    if not all(reference is not None and reference.is_state for reference in references):
        error_msg = f'A law of {spec.owner} targets {target!r}, which is not a state symbol of the namespace `s`.'
        raise ValidationError(error_msg)
    return references  # type: ignore[return-value]


def component_of(node: Node) -> object:
    """Return the object whose `laws` method describes a node, a representative member for a group."""
    if isinstance(node.component, GroupComponent):
        return node.component.element(node.component.label, **(node.component.values or {}))
    return node.component


def activity_symbol(spec: KernelSpec, group: str | None) -> Expression | None:
    """Return the symbol of the activity of the rows of a capacity group that a spec reads per member, or `None`.

    Args:
        spec:
            The kernel spec.

        group:
            The path of the group, or `None`.

    Returns:
        activity:
            The symbol, or `None`.
    """
    name = f'{group}.{ACTIVE}'
    return next(
        (
            item
            for item, reference in spec.references.items()
            if reference.entry.name == name and reference.source is Source.MEMBER
        ),
        None,
    )


def mask_of(spec: KernelSpec, declarations: Declarations) -> Expression | None:
    """Return the activity of the rows a spec runs on, the edges of a changing layer or a capacity group's members.

    Args:
        spec:
            The kernel spec.

        declarations:
            The declarations of the components.

    Returns:
        mask:
            The symbol of the activity, or `None` for a spec that runs on no capacity rows.
    """
    if isinstance(spec.pairs, EdgeSearch) and spec.pairs.source is not None:
        name = spec.pairs.source.name.rpartition('.')[0]
        return activity_symbol(spec, name)
    if spec.group in declarations.capacity:
        return activity_symbol(spec, spec.group)
    return None


def masked_value(value: object, mask: Expression) -> object:
    """Return a law value that vanishes on the free rows of a capacity layout.

    Args:
        value:
            The law value.

        mask:
            The activity of the rows.

    Returns:
        value:
            The masked value, or the value itself when it is neither symbolic nor a number.
    """
    if isinstance(value, SymbolicVector):
        return vector(SR(item) * mask for item in value)
    if isinstance(value, Expression | int | float):
        return SR(value) * mask
    return value


def register_binding(over: str, binding: Binding) -> None:
    """Register how the kernel spec of an interaction with a binding is built, such as `'pairs'`.

    Args:
        over:
            The binding, the class variable `over` of the interaction.

        binding:
            The function that returns the kernel spec of an interaction from its block, the blocks of the elements
            and groups by path, and the time symbol.
    """
    BINDINGS.register(over, binding)


def specs_of(block: Block, elements: dict[str, Block], time: Expression, observe: Observe) -> list[KernelSpec]:
    """Return the kernel specs of a component: its intrinsic laws, its edges, its registered binding, or its roles.

    Args:
        block:
            The block of the component.

        elements:
            The blocks of the elements and groups by path.

        time:
            The time symbol.

        observe:
            The observables that roles may bind to.

    Returns:
        specs:
            The kernel specs.
    """
    if not is_interaction(block.node):
        return [member_spec(block, time)]
    over = type(block.node.component).over
    if over == 'neighbors':
        return [_edge_spec(block, elements, time)]
    if over in set(BINDINGS):
        return [BINDINGS.resolve(over)(block, elements, time)]
    return _interaction_specs(block, elements, time, observe)


@dataclass(frozen=True)
class SetSymbol:
    """Symbol of one element of an element set, with the same symbol of every element of the set."""

    canonical: Expression
    references: tuple[Reference, ...]
    label: str


@dataclass
class KernelSpec:
    """Namespace, references, and evaluation indices that the laws of one component are lowered with."""

    owner: str
    namespace: Symbols
    references: dict[Expression, Reference]
    rows: int
    pairs: Search | None = None
    roles: dict[str, str | tuple[str, ...]] = field(default_factory=dict)
    group: str | None = None
    groups: dict[str, int] = field(default_factory=dict)
    sets: dict[Expression, SetSymbol] = field(default_factory=dict)
    set_index: int = 0
    aligned: tuple[str, ...] = ()


@dataclass(frozen=True)
class Observed:
    """Value of an observable that a role reads at the current state, with the references of its symbols."""

    namespace: Symbols
    references: dict[Expression, Reference]
