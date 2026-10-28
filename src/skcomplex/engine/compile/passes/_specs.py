"""Build the kernel spec of every component of a system, the namespace and the rows its laws are lowered with."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import RR, Expression

from ....core.errors import LabelError, RoleError, ValidationError
from ....core.interactions import BaseInteraction, NeighborInteraction, PairInteraction
from ....core.spaces import BaseSpace, CayleySpace, ContinuousSpace, LatticeSpace, NetworkSpace
from ....core.specs import WienerProcessSpec
from ..._utils import declarations_of, symbol
from ...problems import Problem, VariableKind
from ...problems._edge_search import EdgeSearch
from ...problems._pair_search import PairSearch
from ...problems._read_kind import ReadKind
from ...problems._reference import Reference
from .._base import BasePass
from .._block import Block
from .._declarations import _ACTIVE, ENDS, Declarations
from .._kernel_spec import KernelSpec
from .._observables import ObservableEvaluator
from .._observer import Observer
from .._set_symbol import _EDGE_NAMES, _MEMBER_REFERENCE, Observe, SetSymbol
from .._tree import Node


class SpecsPass(BasePass):
    """Pass that builds the kernel specs of every component, one per element set entry of an interaction."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('layout', 'nodes')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('specs',)

    def _run(self, inputs: object = None) -> None:
        """Build the kernel specs of the declared components."""
        compilation = self.compilation_
        declarations = cast('Declarations', compilation.layout)
        blocks = list(declarations.blocks.values())
        elements = {block.node.path: block for block in blocks if not block.node._is_interaction()}
        elements.update(declarations.processes)
        nodes = cast('list[Node]', compilation.nodes)
        observe = Observer(declarations, ObservableEvaluator._lowerable_observables(compilation.system, nodes))
        compilation.specs = [
            (spec, block)
            for block in blocks
            for spec in self._specs_of(block, elements, declarations.time, observe.evaluate)
        ]

    @staticmethod
    def _group_space(block: Block) -> BaseSpace | None:
        """Return the space of a group, its own or the enclosing one, shared by the groups of one network."""
        return block.node.space

    @staticmethod
    def _space_period(node: Node) -> float | None:
        """Return the side of the periodic box a pair interaction measures minimum image distances in, or `None`."""
        space = node.space
        if not isinstance(space, ContinuousSpace) or not space.periodic_:
            return None
        if len(set(space.extent_)) > 1:
            error_msg = (
                f'Pair interaction {node.path} lives in the periodic box {space.extent_}, whose sides differ. Pair '
                'searches measure minimum image distances in boxes with equal sides.'
            )
            raise ValidationError(error_msg)
        return space.extent_[0]

    @staticmethod
    def _set_symbols(members: list[Block], index: int) -> dict[Expression, SetSymbol]:
        """Return the symbols of one element of a set with the same symbol of every element."""
        label = ', '.join(member.node.path for member in members)
        symbols = {}
        for variable, value in members[index].symbols.items():
            for column, item in enumerate(Problem._real_components_of(value)):
                references = tuple(
                    member.references[Problem._real_components_of(member.symbols[variable])[column]]
                    for member in members
                )
                symbols[item] = SetSymbol(
                    Problem._real_components_of(members[0].symbols[variable])[column], references, label,
                )
        return symbols

    @staticmethod
    def _resolve_path(node: Node, role: str, label: object, elements: dict[str, Block]) -> Block:
        """Return the block of an element, a group, or Wiener processes a role of an interaction binds to by label."""
        name = label.label if isinstance(label, WienerProcessSpec) else label
        path = f'{node.scope}{name}' if isinstance(name, str) else None
        if path not in elements:
            where = node.scope.rstrip('/') or 'the root system'
            error_msg = (
                f'Interaction {node.path} binds the role {role!r} to {label!r}, which is not an element or group path '
                f'relative to {where}.'
            )
            raise RoleError(error_msg)
        return elements[path]

    @staticmethod
    def _resolve_role(node: Node, role: str, elements: dict[str, Block]) -> Block:
        """Return the block of the element or group bound to a role of an interaction."""
        return SpecsPass._resolve_path(node, role, getattr(node.component, role), elements)

    @staticmethod
    def _resolve_set(node: Node, role: str, labels: Sequence[str], elements: dict[str, Block]) -> list[Block]:
        """Return the blocks of the single elements of one type that a role binds to as an element set."""
        blocks = [SpecsPass._resolve_path(node, role, label, elements) for label in labels]
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

    @staticmethod
    def _resolve_member(node: Node, role: str, reference: str, elements: dict[str, Block]) -> tuple[Block, int] | None:
        """Return the group block and member index of a member reference such as `sites[3]`, or `None` for a path."""
        match = _MEMBER_REFERENCE.match(reference)
        if match is None or f'{node.scope}{reference}' in elements:
            return None
        block = SpecsPass._resolve_path(node, role, match['path'], elements)
        index, size = int(match['index']), block.node.members
        if size is None or not 0 <= index < size:
            error_msg = (
                f'Interaction {node.path} binds {reference!r} to its role {role!r}, but {block.node.path} holds '
                f'{size or "no"} members. Give a member index from 0 to the group size minus 1.'
            )
            raise RoleError(error_msg)
        return block, index

    @staticmethod
    def _resolve_entry(node: Node, role: str, entry: object, elements: dict[str, Block]) -> tuple[Block, int | None]:
        """Return the block of one entry of a variadic role, with the member index of a member reference."""
        member = SpecsPass._resolve_member(node, role, entry, elements) if isinstance(entry, str) else None
        if member is not None:
            return member
        block = SpecsPass._resolve_path(node, role, entry, elements)
        if block.node.members is not None:
            error_msg = (
                f'Interaction {node.path} binds the group {block.node.path} as an entry of its variadic role {role!r}. '
                f'Bind single elements or members of the group, such as {entry}[0].'
            )
            raise RoleError(error_msg)
        return block, None

    @staticmethod
    def _resolve_entries(
        node: Node,
        role: str,
        entries: object,
        elements: dict[str, Block],
    ) -> list[tuple[Block, int | None]]:
        """Return the blocks and member indices of the entries of a variadic role, in the given order."""
        if not isinstance(entries, list | tuple):
            error_msg = (
                f'Interaction {node.path} binds its variadic role {role!r} to {entries!r}. Bind a list of element '
                'paths and group member references.'
            )
            raise RoleError(error_msg)
        return [SpecsPass._resolve_entry(node, role, entry, elements) for entry in entries]

    @staticmethod
    def _entry_namespace(block: Block, index: int | None) -> tuple[SimpleNamespace, dict[Expression, Reference]]:
        """Return the namespace of an entry of a variadic role, an element or one member of a group."""
        return block.namespace(ReadKind.SINGLE) if index is None else block.member_namespace(index)

    @staticmethod
    def _entry_path(block: Block, index: int | None) -> str:
        """Return the path of an entry of a variadic role, with the member index of a member reference."""
        return block.node.path if index is None else f'{block.node.path}[{index}]'

    @staticmethod
    def _with_other(
        block: Block, namespace: SimpleNamespace, references: dict[Expression, Reference],
    ) -> SimpleNamespace:
        """Return the namespace of a group's member with `other`, the namespace of a neighbour in reductions over them.

        A group that declares a variable named `other` keeps it, and its laws reduce over neighbours without it.
        """
        if block.node.members is None or 'other' in vars(namespace):
            return namespace
        other, other_references = block.namespace(ReadKind.OTHER, alias='__other')
        references.update(other_references)
        return SimpleNamespace(**vars(namespace), other=other)

    @staticmethod
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
                f'Interaction {node.path} runs on a layer whose edges change, but its nodes {nodes.node.path} are not '
                'a variable group. Give the group on the nodes `variable=True`.'
            )
            raise RoleError(error_msg)
        empty = np.zeros(0, dtype=np.intp)
        return EdgeSearch(empty, empty, changing.entries[ENDS[0]], changing.entries[ENDS[1]]).check()

    @staticmethod
    def _edge_layer(node: Node, nodes: Block, edges: Block | None) -> str | None:
        """Return the layer an interaction over neighbours runs on, checking it against the layer of its edge group."""
        declared = getattr(node.component, 'layer', None)
        if edges is None:
            return declared
        space, edge_space = SpecsPass._group_space(nodes), SpecsPass._group_space(edges)
        if edge_space is not space:
            error_msg = (
                f'Interaction {node.path} binds the edge group {edges.node.path}, which lives on another space than '
                f'the nodes {nodes.node.path}. Give the edge group the network of the nodes.'
            )
            raise RoleError(error_msg)
        layer = getattr(edges.node.component, 'layer', None)
        if declared is not None and declared != layer:
            error_msg = (
                f'Interaction {node.path} runs on the layer {declared!r}, but its edge group {edges.node.path} lives '
                f'on the layer {layer!r}. Give both one layer.'
            )
            raise RoleError(error_msg)
        return layer

    @staticmethod
    def _edge_roles(node: Node, elements: dict[str, Block]) -> tuple[Block, Block | None]:
        """Return the group of nodes and the group on edges, if any, that an interaction over neighbours binds."""
        blocks = [
            SpecsPass._resolve_role(node, role, elements) for role in declarations_of(node.component, VariableKind.ROLE)
        ]
        edges = [block for block in blocks if getattr(block.node.component, 'on', None) == 'edges']
        nodes = [block for block in blocks if block not in edges]
        if len(nodes) != 1 or len(edges) > 1 or nodes[0].node.members is None:
            error_msg = (
                f'Interaction {node.path} over neighbours binds {len(nodes)} groups of nodes and {len(edges)} groups '
                'on edges. Bind one role to a group with one member per node or site, and at most one role to a group '
                'on the edges of the same network.'
            )
            raise RoleError(error_msg)
        return nodes[0], next(iter(edges), None)

    @staticmethod
    def _edge_search(
        node: Node,
        nodes: Block,
        edges: Block | None,
        elements: dict[str, Block],
    ) -> EdgeSearch:
        """Return the edges an interaction over neighbours runs on, after checking the sizes of its groups."""
        space = SpecsPass._group_space(nodes)
        layer = SpecsPass._edge_layer(node, nodes, edges)
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
            return SpecsPass._changing_search(node, nodes, edges, changing)
        if isinstance(space, NetworkSpace):
            if layer not in space.edges_:
                known = ', '.join(repr(label) for label in space.edges_)
                error_msg = (
                    f'Interaction {node.path} runs on the layer {layer!r}, but the layers of its network are {known}.'
                )
                raise RoleError(error_msg)
            sources, targets = space.edges_[layer]
            size = space.size_
        elif isinstance(space, LatticeSpace | CayleySpace) and edges is None and layer is None:
            size = int(space.size_ or 0)
            pairs = np.array(space.check().prepare()._neighbor_pairs(), dtype=np.intp).reshape(-1, 2)
            sources, targets = pairs[:, 0].copy(), pairs[:, 1].copy()
        else:
            error_msg = (
                f'Interaction {node.path} over neighbours binds {nodes.node.path}, which lives on {space!r}. Bind a '
                'group on the nodes of a network, or on the sites of a lattice without edge groups or layers.'
            )
            raise RoleError(error_msg)
        if nodes.rows != size:
            error_msg = (
                f'Interaction {node.path} over neighbours binds {nodes.node.path} of {nodes.rows} members to a space '
                f'of {size} nodes. Bind a group with one member per node or site.'
            )
            raise RoleError(error_msg)
        if edges is not None and edges.rows != sources.size:
            error_msg = (
                f'Interaction {node.path} binds the edge group {edges.node.path} of {edges.rows} members to a layer of '
                f'{sources.size} edges. Give the edge group one member per edge of its layer.'
            )
            raise RoleError(error_msg)
        return EdgeSearch(sources, targets).check()

    @staticmethod
    def _edge_spec(block: Block, elements: dict[str, Block], time: Expression) -> KernelSpec:
        """Return the kernel spec of an interaction over neighbours, evaluated once per edge of its layer."""
        node = block.node
        nodes, edges = SpecsPass._edge_roles(node, elements)
        search = SpecsPass._edge_search(node, nodes, edges, elements)
        parameters, references = block.namespace(ReadKind.SINGLE)
        clashes = sorted(set(vars(parameters)) & {*_EDGE_NAMES, 't'})
        if clashes:
            error_msg = (
                f'Interaction {node.path} over neighbours declares the parameters {", ".join(clashes)}, whose names '
                'its laws read as the ends and the member of an edge. Rename them.'
            )
            raise LabelError(error_msg)
        source, source_references = nodes.namespace(ReadKind.FIRST, alias='__source')
        target, target_references = nodes.namespace(ReadKind.SECOND, alias='__target')
        references.update(source_references)
        references.update(target_references)
        namespaces: dict[str, SimpleNamespace] = {'source': source, 'target': target}
        paths: dict[str, str | tuple[str, ...]] = {'source': nodes.node.path, 'target': nodes.node.path}
        groups: dict[str, int] = {}
        if edges is not None:
            edge, edge_references = edges.namespace(ReadKind.MEMBER)
            references.update(edge_references)
            namespaces['edge'] = edge
            paths['edge'] = edges.node.path
            groups[edges.node.path] = edges.rows
        elif search.source is not None:
            changing = next(block for block in elements.values() if block.entries.get(ENDS[0]) == search.source)
            activity = symbol(f'active_{re.sub(r"[^0-9A-Za-z]", "_", node.path)}_edges', RR)
            references[activity] = Reference(changing.entries[_ACTIVE], False, 0, ReadKind.MEMBER)
            paths['edge'] = changing.node.path
        namespace = SimpleNamespace(t=time, **vars(parameters), **namespaces)
        group = edges.node.path if edges is not None else None
        return KernelSpec(
            node.path, namespace, references, search.size, search, roles=paths, group=group, groups=groups,
        )

    @staticmethod
    def _member_spec(block: Block, time: Expression) -> KernelSpec:
        """Return the kernel spec of the intrinsic laws of an element or group."""
        namespace, references = block.namespace(ReadKind.MEMBER)
        namespace = SpecsPass._with_other(block, namespace, references)
        group = block.node.path if block.node.members is not None else None
        return KernelSpec(
            block.node.path, SimpleNamespace(t=time, **vars(namespace)), references, block.rows, group=group,
        )

    @staticmethod
    def _pair_spec(block: Block, elements: dict[str, Block], time: Expression) -> KernelSpec:
        """Return the kernel spec of a pair interaction that is evaluated once per pair within its cutoff."""
        node = block.node
        interaction: BaseInteraction = node.component
        first, second = (
            SpecsPass._resolve_role(node, 'first', elements),
            SpecsPass._resolve_role(node, 'second', elements),
        )
        position = getattr(type(interaction), 'position', 'x')
        for bound in (first, second):
            entry = bound.entries.get(position)
            if entry is None or not entry.member_shape:
                error_msg = f'Pair interaction {node.path} needs the vector state {position!r} in {bound.node.path}.'
                raise ValidationError(error_msg)
        parameters, references = block.namespace(ReadKind.SINGLE)
        first_namespace, first_references = first.namespace(ReadKind.FIRST)
        second_namespace, second_references = second.namespace(ReadKind.SECOND, alias='_pair')
        references.update(first_references)
        references.update(second_references)
        period = getattr(interaction, 'period', None)
        period = period if period is not None else SpecsPass._space_period(node)
        pairs = PairSearch(first.entries[position], second.entries[position], block.entries['cutoff'], period).check()
        namespace = SimpleNamespace(t=time, **vars(parameters), first=first_namespace, second=second_namespace)
        paths: dict[str, str | tuple[str, ...]] = {'first': first.node.path, 'second': second.node.path}
        return KernelSpec(node.path, namespace, references, 0, pairs, roles=paths)

    @staticmethod
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
            role: SpecsPass._resolve_entries(node, role, labels[role], elements)
            for role, declaration in declared.items()
            if declaration.variadic
        }
        sets = {
            role: SpecsPass._resolve_set(node, role, label, elements)
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
            if (member := SpecsPass._resolve_member(node, role, label, elements)) is not None
        }
        bound = {
            role: SpecsPass._resolve_role(node, role, elements)
            for role in labels
            if role not in sets and role not in observed and role not in variadic and role not in single_members
        }
        sizes = {len(members) for members in sets.values()}
        if len(sizes) > 1:
            error_msg = (
                f'Interaction {node.path} binds roles to element sets of different sizes. Give the sets one size.'
            )
            raise RoleError(error_msg)
        groups = {item.node.path: item.rows for item in bound.values() if item.node.members is not None}
        group = next(iter(groups)) if len(groups) == 1 else None
        paths: dict[str, str | tuple[str, ...]] = {role: item.node.path for role, item in bound.items()}
        paths.update({role: tuple(member.node.path for member in members) for role, members in sets.items()})
        paths.update({role: f'{node.scope}{labels[role]}' for role in observed})
        paths.update(
            {role: tuple(SpecsPass._entry_path(*entry) for entry in entries) for role, entries in variadic.items()},
        )
        paths.update({role: SpecsPass._entry_path(*member) for role, member in single_members.items()})
        specs = []
        for index in range(next(iter(sizes), 1)):
            parameters, references = block.namespace(ReadKind.SINGLE)
            namespaces: dict[str, SimpleNamespace | tuple[SimpleNamespace, ...]] = {}
            set_symbols: dict[Expression, SetSymbol] = {}
            for role, members in sets.items():
                namespaces[role], role_references = members[index].namespace(ReadKind.SINGLE)
                references.update(role_references)
                set_symbols.update(SpecsPass._set_symbols(members, index))
            for role, value in observed.items():
                namespaces[role] = value.namespace
                references.update(value.references)
            for role, entries in variadic.items():
                read = [SpecsPass._entry_namespace(*entry) for entry in entries]
                namespaces[role] = tuple(namespace for namespace, _ in read)
                for _, entry_references in read:
                    references.update(entry_references)
            for role, (member_block, position) in single_members.items():
                namespaces[role], role_references = member_block.member_namespace(position)
                references.update(role_references)
            for role, item in bound.items():
                role_namespace, role_references = item.namespace(
                    ReadKind.MEMBER if item.node.members is not None else ReadKind.SINGLE,
                )
                namespaces[role] = SpecsPass._with_other(item, role_namespace, role_references)
                references.update(role_references)
            specs.append(
                KernelSpec(
                    node.path,
                    SimpleNamespace(t=time, **vars(parameters), **namespaces),
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

    @staticmethod
    def _specs_of(block: Block, elements: dict[str, Block], time: Expression, observe: Observe) -> list[KernelSpec]:
        """Return the kernel specs of a component: its intrinsic laws, its edges, its pairs, or its roles.

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
        if not block.node._is_interaction():
            return [SpecsPass._member_spec(block, time)]
        if isinstance(block.node.component, NeighborInteraction):
            return [SpecsPass._edge_spec(block, elements, time)]
        if isinstance(block.node.component, PairInteraction):
            return [SpecsPass._pair_spec(block, elements, time)]
        return SpecsPass._interaction_specs(block, elements, time, observe)
