"""Build the kernel spec of every component of a system.

The spec holds the namespace and the rows its laws are lowered with.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import RR, SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....components.elements import FieldElement
from ....components.interactions import (
    GroupingInteraction,
    HyperedgeInteraction,
    NeighborInteraction,
    PairInteraction,
    TopologyInteraction,
)
from ....components.laws.utils._reductions import MAX_NEIGHBOR_HYPEREDGE_SIZE
from ....components.spaces import (
    BaseSpace,
    CayleySpace,
    CellComplexSpace,
    ContinuousSpace,
    LatticeSpace,
    MeshSpace,
    NetworkSpace,
    PhaseSpace,
)
from ....components.specs import NodeSpec
from ....errors import LabelError, RoleError, ValidationError
from ....specs import EntrySpec, NeighborReadSpec, WienerProcessSpec
from ....types import Indices, VariableKind
from ..._utils import bound_labels, declarations_of, domain_of, symbol, vector
from ...problems import ProblemSpec
from ...problems._block_spec import BlockSpec
from ...problems._declarations_spec import ACTIVE, ENDS, MEMBERS, DeclarationsSpec
from ...problems._edge_search import EdgeSearch
from ...problems._kernel_spec import KernelSpec
from ...problems._observed_spec import ObservedSpec
from ...problems._pair_search import PairSearch
from ...problems._reference_spec import ReferenceSpec
from ...problems._set_symbol_spec import EDGE_NAMES, MEMBER_REFERENCE, SetSymbolSpec
from ...types import ReadKind
from .._base import BasePass
from .._compilation import Compilation
from .._observables import ObservableEvaluator
from .._observer import Observer
from ._hyperedges import MEMBER_LAWS, HyperedgeTables, Tables
from ._layout import LayoutPass

END_SUFFIXES = ('__source', '__target')
HYPEREDGE_NAMES = ('member', 'others', 'members', 'hyperedge', 'edge', 't')
MAX_HYPEREDGE_ROLES = 2
SECOND_PARTS = ('edges', 'hyperedges')


class SpecsPass(BasePass):
    """Pass that builds the kernel specs of every component.

    There is one spec per element set entry of an interaction.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('layout', 'nodes')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('specs',)

    def _run(self, inputs: object = None) -> None:
        """Build the kernel specs of the declared components."""
        compilation = self.compilation_
        declarations = Compilation._get_required(compilation.layout, 'layout')
        blocks = list(declarations.blocks.values())
        elements = {block.node.path: block for block in blocks if not block.node.interaction}
        elements.update(declarations.processes)
        nodes = Compilation._get_required(compilation.nodes, 'nodes')
        live = ObservableEvaluator._get_lowerable_observables(compilation.system, nodes)
        observer = Observer(declarations, live).prepare()
        compilation.specs = []
        for block in blocks:
            specs, declarations = self._get_specs_of(block, elements, declarations, observer)
            compilation.specs += [(spec, block) for spec in specs]
        compilation.layout = declarations

    @staticmethod
    def _get_namespace(
        block: BlockSpec,
        source: ReadKind,
        alias: str = '',
    ) -> tuple[SimpleNamespace, dict[Expression, ReferenceSpec]]:
        """Return the namespace of the symbols of a block and their references.

        They serve one way of reading the symbols.

        Args:
            block:
                The block.

            source:
                How the components of the block are read.

            alias:
                A suffix that gives the symbols of the block distinct names, used for the second member of a pair.

        Returns:
            namespace:
                The namespace and the references of its symbols.
        """
        mapping = {item: symbol(f'{item}{alias}', domain_of(item)) for item in block.references}
        references = {
            mapping[item]: ReferenceSpec(reference.entry, reference.is_state, reference.column, source)
            for item, reference in block.references.items()
        }
        return SpecsPass._get_renamed_namespace(block, mapping), references

    @staticmethod
    def _get_member_namespace(block: BlockSpec, index: int) -> tuple[SimpleNamespace, dict[Expression, ReferenceSpec]]:
        """Return the namespace of one member of a group and its symbol references.

        The member is read like a single element.

        Args:
            block:
                The block of the group.

            index:
                The index of the member.

        Returns:
            namespace:
                The namespace and the references of its symbols, whose columns count from the group's first component.
        """
        mapping = {item: symbol(f'{item}__m{index}', domain_of(item)) for item in block.references}
        references = {
            mapping[item]: ReferenceSpec(
                reference.entry,
                reference.is_state,
                index * reference.entry.columns + reference.column,
                ReadKind.SINGLE,
            )
            for item, reference in block.references.items()
        }
        return SpecsPass._get_renamed_namespace(block, mapping), references

    @staticmethod
    def _get_renamed_namespace(block: BlockSpec, mapping: dict[Expression, Expression]) -> SimpleNamespace:
        """Return the namespace of the variables of a block in renamed symbols.

        Per-entry parameters are tuples.
        """
        symbols: dict[str, object] = {}
        for name, value in block.symbols.items():
            if not isinstance(value, FreeModuleElement):
                symbols[name] = SpecsPass._get_renamed(value, mapping)
            elif name in block.per:
                symbols[name] = tuple(SpecsPass._get_renamed(item, mapping) for item in value)
            elif name in block.entries and (tensor := block.entries[name].tensor) is not None:
                symbols[name] = tensor.get_matrix_of(SpecsPass._get_renamed(item, mapping) for item in value)
            else:
                symbols[name] = vector(SpecsPass._get_renamed(item, mapping) for item in value)
        ports = getattr(block.node.component_type, 'ports', None)
        if ports:
            symbols['ports'] = ports
        return SimpleNamespace(**symbols)

    @staticmethod
    def _get_renamed(value: Expression, mapping: dict[Expression, Expression]) -> Expression:
        """Return the renamed symbol of a variable, or a renamed complex value.

        A complex value has its real and imaginary symbols renamed. A symbol without a variable, such as the handle of a
        Wiener process, keeps its name.
        """
        return mapping.get(value, value) if value.operator() is None else value.subs(mapping)

    @staticmethod
    def _get_group_space(block: BlockSpec) -> BaseSpace | None:
        """Return the space of a group, its own or the enclosing one.

        The groups of one network share it.
        """
        return block.node.space

    @staticmethod
    def _get_space_period(node: NodeSpec) -> float | None:
        """Return the side of the periodic box of a pair interaction, or `None`.

        The pair interaction measures minimum image distances in the box.
        """
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
    def _set_symbols(members: list[BlockSpec], index: int) -> dict[Expression, SetSymbolSpec]:
        """Return the symbols of one element of a set, the same for every element."""
        label = ', '.join(member.node.path for member in members)
        symbols = {}
        for variable, value in members[index].symbols.items():
            for column, item in enumerate(ProblemSpec._get_real_components_of(value)):
                references = tuple(
                    member.references[ProblemSpec._get_real_components_of(member.symbols[variable])[column]]
                    for member in members
                )
                symbols[item] = SetSymbolSpec(
                    ProblemSpec._get_real_components_of(members[0].symbols[variable])[column],
                    references,
                    label,
                )
        return symbols

    @staticmethod
    def _resolve_path(node: NodeSpec, role: str, label: object, elements: dict[str, BlockSpec]) -> BlockSpec:
        """Return the block a role of an interaction binds to by label.

        The block is that of an element, a group, or Wiener processes.
        """
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
    def _resolve_role(node: NodeSpec, role: str, elements: dict[str, BlockSpec]) -> BlockSpec:
        """Return the block of the element or group bound to a role of an interaction."""
        return SpecsPass._resolve_path(node, role, getattr(node.component, role), elements)

    @staticmethod
    def _get_end_names(node: NodeSpec, ends: Sequence[str]) -> tuple[str, ...]:
        """Return the names the laws of an interaction give edge ends or pair members.

        Each end, such as `'source'`, `'target'`, `'edge'`, `'first'`, or `'second'`, takes the name the interaction's
        `_get_role` returns for it, the end itself by default.
        """
        component = node.as_interaction
        return tuple(component._get_role(end) for end in ends)

    @staticmethod
    def _resolve_set(
        node: NodeSpec,
        role: str,
        labels: Sequence[str],
        elements: dict[str, BlockSpec],
    ) -> list[BlockSpec]:
        """Return the blocks of the single elements a role binds to as an element set.

        The elements are of one type.
        """
        blocks = [SpecsPass._resolve_path(node, role, label, elements) for label in labels]
        grouped = [block.node.path for block in blocks if block.node.members is not None]
        if grouped or not blocks:
            error_msg = (
                f'Interaction {node.path} binds the role {role!r} to a set that holds the groups {", ".join(grouped)}. '
                'An element set lists single elements of one type, and a role binds to a group by its path.'
            )
            raise RoleError(error_msg)
        types = list(dict.fromkeys(type(block.node.component).__name__ for block in blocks))
        if len(types) > 1:
            error_msg = (
                f'Interaction {node.path} binds the role {role!r} to elements of the types {" and ".join(types)}. An '
                'element set holds elements of one type.'
            )
            raise RoleError(error_msg)
        return blocks

    @staticmethod
    def _resolve_member(
        node: NodeSpec,
        role: str,
        reference: str,
        elements: dict[str, BlockSpec],
    ) -> tuple[BlockSpec, int] | None:
        """Return the group block and member index of a member reference.

        An example reference is `sites[3]`, and the result is `None` for a path.
        """
        match = MEMBER_REFERENCE.match(reference)
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
    def _resolve_entry(
        node: NodeSpec,
        role: str,
        entry: object,
        elements: dict[str, BlockSpec],
    ) -> tuple[BlockSpec, int | None]:
        """Return the block of one entry of a variadic role.

        It comes with the member index of a member reference. An entry that names a whole group binds its agents, which
        agent rules read by the role, with no member index.
        """
        member = SpecsPass._resolve_member(node, role, entry, elements) if isinstance(entry, str) else None
        if member is not None:
            return member
        return SpecsPass._resolve_path(node, role, entry, elements), None

    @staticmethod
    def _resolve_entries(
        node: NodeSpec,
        role: str,
        entries: object,
        elements: dict[str, BlockSpec],
    ) -> list[tuple[BlockSpec, int | None]]:
        """Return the blocks and member indices of the entries of a variadic role.

        They are in the given order. An entry is an element path or a group member reference, alone or paired with the
        name of a state, and the entries may be the keys of a mapping, where a state name alone names a state of another
        role and binds nothing. A role bound to `None` has no entries.
        """
        if entries is None:
            return []
        labels = bound_labels(entries)
        if isinstance(entries, Mapping):
            entries = [entry for entry in entries if not isinstance(entry, str)]
        if not isinstance(entries, list | tuple) or len(labels) != len(entries):
            error_msg = (
                f'Interaction {node.path} binds its variadic role {role!r} to {entries!r}. Bind a list of element '
                'paths and group member references, or of pairs of one of them and a state name.'
            )
            raise RoleError(error_msg)
        return [SpecsPass._resolve_entry(node, role, label, elements) for label in labels]

    @staticmethod
    def _get_entry_namespace(
        block: BlockSpec,
        index: int | None,
    ) -> tuple[SimpleNamespace, dict[Expression, ReferenceSpec]]:
        """Return the namespace of an entry of a variadic role.

        The entry is an element or one member of a group. A whole group, whose agents rules read by the role, gives an
        empty namespace, since symbolic laws read single elements or members.
        """
        if index is None and block.node.members is not None:
            return SimpleNamespace(), {}
        if index is None:
            return SpecsPass._get_namespace(block, ReadKind.SINGLE)
        return SpecsPass._get_member_namespace(block, index)

    @staticmethod
    def _get_source_namespace(
        block: BlockSpec,
        index: int | None,
        bound: Mapping[str, BlockSpec],
    ) -> tuple[SimpleNamespace, dict[Expression, ReferenceSpec]]:
        """Return the namespace of a source of an interaction.

        It is a whole field read across a phase space, or else an entry. A field on a position lattice is read by the
        target on a phase space with that position, broadcast along its class axes. A field on a phase space is read
        member by member by a target on a position lattice, whose laws read its moments. A group on a cell complex is
        read member by member by the exterior calculus of the target.
        """
        if index is None and block.node.members is not None and isinstance(block.node.space, CellComplexSpace):
            return SpecsPass._get_namespace(block, ReadKind.MEMBER)
        place = block.node.component
        targets = [item.node.component for item in bound.values() if isinstance(item.node.component, FieldElement)]
        if index is not None or not isinstance(place, FieldElement) or not targets:
            return SpecsPass._get_entry_namespace(block, index)
        space, target = place.space, targets[0].space
        if isinstance(space, PhaseSpace) and not isinstance(target, PhaseSpace):
            return SpecsPass._get_namespace(block, ReadKind.MEMBER, alias='__across')
        if isinstance(target, PhaseSpace) and not isinstance(space, PhaseSpace):
            return SpecsPass._broadcast_namespace(block, target)
        return SpecsPass._get_entry_namespace(block, index)

    @staticmethod
    def _broadcast_namespace(
        block: BlockSpec,
        target: PhaseSpace,
    ) -> tuple[SimpleNamespace, dict[Expression, ReferenceSpec]]:
        """Return the namespace of a field over a position lattice on a phase space.

        The field is read at every point of the phase space.

        Args:
            block:
                The block of the field.

            target:
                The phase space that reads it.

        Returns:
            namespace:
                The namespace and the references of its symbols, neighbour reads of the site at the same position.

        Raises:
            ValidationError:
                The field does not have as many sites as the phase space has positions.
        """
        target.prepare()
        if block.rows != target.position_size_:
            error_msg = (
                f'{block.node.path} has {block.rows} sites, but it is read by a field on a phase space of '
                f'{target.position_size_} positions. Give the field the position lattice of the phase space.'
            )
            raise ValidationError(error_msg)
        table = np.repeat(np.arange(block.rows), target.size_ // target.position_size_)
        read = NeighborReadSpec(table, None, None, None, 0)
        namespace, references = SpecsPass._get_namespace(block, ReadKind.NEIGHBOR, alias='__broadcast')
        return namespace, {item: replace(reference, neighbor=read) for item, reference in references.items()}

    @staticmethod
    def _get_entry_path(block: BlockSpec, index: int | None) -> str:
        """Return the path of an entry of a variadic role.

        It comes with the member index of a member reference.
        """
        return block.node.path if index is None else f'{block.node.path}[{index}]'

    @staticmethod
    def _copy_with_other(
        block: BlockSpec,
        namespace: SimpleNamespace,
        references: dict[Expression, ReferenceSpec],
    ) -> SimpleNamespace:
        """Return the namespace of a group's member with `other`.

        `other` is the namespace of a neighbour in reductions over them. A group that declares a variable named `other`
        keeps it, and its laws reduce over neighbours without it.
        """
        if block.node.members is None or 'other' in vars(namespace):
            return SpecsPass._copy_with_hyperedges(block, namespace, references)
        other, other_references = SpecsPass._get_namespace(block, ReadKind.OTHER, alias='__other')
        references.update(other_references)
        return SpecsPass._copy_with_hyperedges(block, SimpleNamespace(**vars(namespace), other=other), references)

    @staticmethod
    def _copy_with_source(
        block: BlockSpec,
        namespace: SimpleNamespace,
        references: dict[Expression, ReferenceSpec],
    ) -> SimpleNamespace:
        """Return a namespace with `source`.

        `source` holds the values of the node a topology request starts from. A group that declares a variable named
        `source` keeps it.
        """
        if 'source' in vars(namespace):
            return namespace
        source, source_references = SpecsPass._get_namespace(block, ReadKind.SOURCE, alias='__source')
        references.update(source_references)
        return SimpleNamespace(**vars(namespace), source=source)

    @staticmethod
    def _copy_with_hyperedges(
        block: BlockSpec,
        namespace: SimpleNamespace,
        references: dict[Expression, ReferenceSpec],
    ) -> SimpleNamespace:
        """Return the namespace of a hyperedge member with the slots of one hyperedge.

        The member is a member of a group on a network with hyperedges or in a continuous space.

        The slots are `others`, the other members of a hyperedge around the member in its order, as many as the largest
        hyperedge has, and `members`, all its members, which only the reductions over the hyperedges around a member
        read. In a continuous space the hyperedges are sets of current neighbours, of at most the largest size a
        reduction over them takes. A group that declares a variable named like a slot keeps it.
        """
        space = SpecsPass._get_group_space(block)
        if block.node.members is None or {'others', 'members'} & set(vars(namespace)):
            return namespace
        if isinstance(space, ContinuousSpace):
            size = MAX_NEIGHBOR_HYPEREDGE_SIZE
        elif isinstance(space, NetworkSpace) and space.hyperedges_:
            size = max(len(item) for item in space.hyperedges_)
        else:
            return namespace
        others = []
        for position in range(size - 1):
            other, other_references = SpecsPass._get_namespace(block, ReadKind.OTHER, alias=f'__others{position}')
            references.update(other_references)
            others.append(other)
        members, member_references = SpecsPass._get_namespace(block, ReadKind.OTHER, alias='__members')
        references.update(member_references)
        return SimpleNamespace(**vars(namespace), others=tuple(others), members=members)

    @staticmethod
    def _get_changing_search(
        node: NodeSpec,
        nodes: BlockSpec,
        edges: BlockSpec | None,
        changing: BlockSpec,
    ) -> EdgeSearch:
        """Return the edge rows of a layer whose edges change.

        It first checks that the bound edge group is its group.
        """
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
        return EdgeSearch(empty, empty, changing.entries[ENDS[0]], changing.entries[ENDS[1]]).prepare()

    @staticmethod
    def _get_edge_layer(node: NodeSpec, nodes: BlockSpec, edges: BlockSpec | None) -> str | None:
        """Return the layer an interaction over neighbours runs on.

        It is checked against the layer of its edge group.
        """
        declared = getattr(node.component, 'layer', None)
        if edges is None:
            return declared
        space, edge_space = SpecsPass._get_group_space(nodes), SpecsPass._get_group_space(edges)
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
    def _get_edge_roles(
        node: NodeSpec,
        elements: dict[str, BlockSpec],
    ) -> tuple[BlockSpec, BlockSpec | None, BlockSpec | None]:
        """Return the node group, edge group, and second node group of an interaction.

        The edge group and the second node group may be absent. The first role bound to a group of nodes binds the
        members the laws run for, and a second one the other members that their reductions run over. A role bound to
        `None` binds nothing.
        """
        blocks = [
            SpecsPass._resolve_role(node, role, elements)
            for role in declarations_of(node.component, VariableKind.ROLE)
            if getattr(node.component, role) is not None
        ]
        edges = [block for block in blocks if getattr(block.node.component, 'on', None) == 'edges']
        nodes = [block for block in blocks if block not in edges]
        if (
            not 1 <= len(nodes) <= 1 + int(not edges)
            or len(edges) > 1
            or any(block.node.members is None for block in nodes)
        ):
            error_msg = (
                f'Interaction {node.path} over neighbours binds {len(nodes)} groups of nodes and {len(edges)} groups '
                'on edges. Bind one role to a group with one member per node or site, and at most one role to a group '
                'on the edges of the same network or to a second group of nodes.'
            )
            raise RoleError(error_msg)
        return nodes[0], next(iter(edges), None), next(iter(nodes[1:]), None)

    @staticmethod
    def _get_edge_search(
        node: NodeSpec,
        nodes: BlockSpec,
        edges: BlockSpec | None,
        elements: dict[str, BlockSpec],
    ) -> EdgeSearch:
        """Return the edges an interaction over neighbours runs on.

        It first checks the sizes of its groups.
        """
        space = SpecsPass._get_group_space(nodes)
        layer = SpecsPass._get_edge_layer(node, nodes, edges)
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
            return SpecsPass._get_changing_search(node, nodes, edges, changing)
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
            pairs = np.array(space.prepare()._get_neighbor_pairs(), dtype=np.intp).reshape(-1, 2)
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
        return EdgeSearch(sources, targets).prepare()

    @staticmethod
    def _get_degrees(nodes: BlockSpec, search: EdgeSearch) -> np.ndarray | None:
        """Return the number of fixed edges of a layer at every node, or `None`.

        On a directed graph, they are counted both ways. A layer whose edges change has no fixed degrees.
        """
        if search.source is not None:
            return None
        size = nodes.rows
        return (np.bincount(search.sources, minlength=size) + np.bincount(search.targets, minlength=size)).astype(float)

    @staticmethod
    def _copy_with_degree(
        namespace: SimpleNamespace,
        references: dict[Expression, ReferenceSpec],
        name: str,
        reference: ReferenceSpec,
    ) -> SimpleNamespace:
        """Return the namespace of an end of an edge with `degree`.

        It is left out when the members declare a variable so named.
        """
        if 'degree' in vars(namespace):
            return namespace
        item = symbol(name, RR)
        references[item] = reference
        return SimpleNamespace(**vars(namespace), degree=item)

    @staticmethod
    def _declare_degrees(
        node: NodeSpec,
        nodes: BlockSpec,
        search: EdgeSearch,
        declarations: DeclarationsSpec,
    ) -> tuple[DeclarationsSpec, dict[str, ReferenceSpec]]:
        """Return the declarations with a hidden degree parameter and its references.

        The parameter holds the degree of every node, and the references are by end.

        Args:
            node:
                The interaction over neighbours.

            nodes:
                The block of the group on the nodes.

            search:
                The edges of the interaction.

            declarations:
                The declarations to extend.

        Returns:
            degrees:
                The declarations, and the reference of the degree of the source under `'source'` and of the target
                under `'target'`, none for a layer whose edges change. The degree of a node is the number of edges of
                the layer at it, both ways on a directed graph.
        """
        degrees = SpecsPass._get_degrees(nodes, search)
        if degrees is None:
            return declarations, {}
        entry = EntrySpec(f'{node.path}._degree', declarations.ends[False], (nodes.rows,), RR, nodes.rows)
        declarations = LayoutPass._copy_with_entry(declarations, entry, degrees, is_state=False)
        references = {
            end: ReferenceSpec(entry, False, 0, source)
            for end, source in zip(EDGE_NAMES[:2], (ReadKind.FIRST, ReadKind.SECOND), strict=True)
        }
        return declarations, references

    @staticmethod
    def _get_end_spec(edge: KernelSpec, nodes: BlockSpec) -> KernelSpec:
        """Return the spec of the laws of a neighbour interaction that read one edge end.

        The interaction is an interaction over neighbours. The spec reads the laws in the namespace of an edge, and
        evaluates them once per member of the group on the nodes. Every symbol of an end of an edge has a symbol of the
        member, and one of the other member, which the reductions of the laws read.

        Args:
            edge:
                The spec of the edges.

            nodes:
                The block of the group on the nodes.

        Returns:
            spec:
                The spec of the laws of the members.
        """
        references = {
            item: reference
            for item, reference in edge.references.items()
            if reference.source in {ReadKind.SINGLE, ReadKind.OTHER}
        }
        for item, reference in edge.references.items():
            if reference.source not in {ReadKind.FIRST, ReadKind.SECOND}:
                continue
            member, other = SpecsPass._get_end_symbols(item)
            references[member] = replace(reference, source=ReadKind.MEMBER)
            references[other] = replace(reference, source=ReadKind.OTHER)
        return KernelSpec(
            edge.owner,
            edge.namespace,
            references,
            nodes.rows,
            roles=edge.roles,
            group=nodes.node.path,
            groups={nodes.node.path: nodes.rows},
            edges=edge,
            place=edge.place,
        )

    @staticmethod
    def _get_end_symbols(item: Expression) -> tuple[Expression, Expression]:
        """Return the member and other member symbols an edge end symbol stands for.

        Args:
            item:
                A symbol of the source or the target of an edge.

        Returns:
            symbols:
                The symbol of the member, and the symbol of the other member.

        Examples:
            >>> from sage.all__sagemath_symbolics import var
            >>> from skcomplex.core.engine.compile.passes._specs import SpecsPass
            >>> SpecsPass._end_symbols(var('x_cells__source'))
            (x_cells, x_cells__other)
        """
        name = str(item).removesuffix(END_SUFFIXES[0]).removesuffix(END_SUFFIXES[1])
        return symbol(name, domain_of(item)), symbol(f'{name}__other', domain_of(item))

    @staticmethod
    def _get_ends_spec(
        block: BlockSpec,
        nodes: BlockSpec,
        parameters: SimpleNamespace,
        references: dict[Expression, ReferenceSpec],
        time: Expression,
        *,
        others: BlockSpec | None = None,
        place: FieldElement | None = None,
    ) -> KernelSpec:
        """Return the spec of the laws of the members of a neighbour interaction.

        The interaction is over neighbours, and its members have no edges. The laws read one end as the member and,
        inside their reductions, the other end as the other members, as on a lattice or a network, and the interaction
        has no laws per edge. With a second group of nodes, the source end reads its members, the other members of the
        reductions, and the target end the members of the first group.

        Args:
            block:
                The block of the interaction.

            nodes:
                The block of the group the interaction binds.

            parameters:
                The namespace of the parameters of the interaction.

            references:
                The references of the parameters, which the references of the ends extend.

            time:
                The time symbol.

            others:
                The block of the second group of nodes, or `None` for the group itself.

            place:
                The field on the nodes, whose ends on a lattice have the degree of the nearest neighbours, or `None`.

        Returns:
            spec:
                The spec of the laws of one end.
        """
        namespaces = {}
        ends = (nodes if others is None else others, nodes)
        names = SpecsPass._get_end_names(block.node, EDGE_NAMES[:2])
        for end, group, source, suffix in zip(
            names,
            ends,
            (ReadKind.FIRST, ReadKind.SECOND),
            END_SUFFIXES,
            strict=True,
        ):
            namespaces[end], end_references = SpecsPass._get_namespace(group, source, alias=suffix)
            references.update(end_references)
            if place is not None and isinstance(place.space, LatticeSpace) and 'degree' not in vars(namespaces[end]):
                degree = SR(2 * place.space.prepare().n_dim_)
                namespaces[end] = SimpleNamespace(**vars(namespaces[end]), degree=degree)
            namespaces[end] = SpecsPass._copy_with_hyperedges(group, namespaces[end], references)
        namespace = SimpleNamespace(t=time, **vars(parameters), **namespaces)
        paths: dict[str, str | tuple[str, ...]] = {names[0]: ends[0].node.path, names[1]: nodes.node.path}
        edge = KernelSpec(block.node.path, namespace, references, 0, roles=paths, place=place)
        return SpecsPass._get_end_spec(edge, nodes)

    @staticmethod
    def _get_edge_specs(
        block: BlockSpec,
        elements: dict[str, BlockSpec],
        declarations: DeclarationsSpec,
    ) -> tuple[list[KernelSpec], DeclarationsSpec]:
        """Return the specs of an interaction over neighbours.

        There is one per edge of its layer and one per node member. Members of a space without edges, such as a
        continuous space, get the spec of the laws of one end alone.

        Args:
            block:
                The block of the interaction.

            elements:
                The blocks of the elements and groups by path.

            declarations:
                The declarations, which the degrees of the ends extend.

        Returns:
            specs:
                The spec of the edges, if any, and the spec of the laws of one end, with the extended declarations.

        Raises:
            LabelError:
                The interaction declares a parameter named like an end or the member of an edge.
        """
        node = block.node
        nodes, edges, others = SpecsPass._get_edge_roles(node, elements)
        parameters, references = SpecsPass._get_namespace(block, ReadKind.SINGLE)
        names = SpecsPass._get_end_names(node, EDGE_NAMES)
        clashes = sorted(set(vars(parameters)) & {*names, 't'})
        if clashes:
            error_msg = (
                f'Interaction {node.path} over neighbours declares the parameters {", ".join(clashes)}, whose names '
                'its laws read as the ends and the member of an edge. Rename them.'
            )
            raise LabelError(error_msg)
        component = nodes.node.component
        place = component if isinstance(component, FieldElement) and edges is None and others is None else None
        spread = (
            place is not None
            and isinstance(place.space, LatticeSpace | MeshSpace)
            and getattr(node.component, 'layer', None) is None
        )
        if (
            others is not None
            or spread
            or (
                edges is None
                and not isinstance(SpecsPass._get_group_space(nodes), LatticeSpace | NetworkSpace | CayleySpace)
            )
        ):
            spec = SpecsPass._get_ends_spec(
                block,
                nodes,
                parameters,
                references,
                declarations.time,
                others=others,
                place=place,
            )
            return [spec], declarations
        search = SpecsPass._get_edge_search(node, nodes, edges, elements)
        declarations, degrees = SpecsPass._declare_degrees(node, nodes, search, declarations)
        namespaces: dict[str, SimpleNamespace] = {}
        base = re.sub(r'[^0-9A-Za-z]', '_', node.path)
        ends = zip(EDGE_NAMES[:2], names[:2], (ReadKind.FIRST, ReadKind.SECOND), END_SUFFIXES, strict=True)
        for end, name, source, suffix in ends:
            namespace, end_references = SpecsPass._get_namespace(nodes, source, alias=suffix)
            references.update(end_references)
            if end in degrees:
                namespace = SpecsPass._copy_with_degree(namespace, references, f'degree_{base}{suffix}', degrees[end])
            namespaces[name] = SpecsPass._copy_with_hyperedges(nodes, namespace, references)
        paths: dict[str, str | tuple[str, ...]] = {names[0]: nodes.node.path, names[1]: nodes.node.path}
        groups: dict[str, int] = {}
        if edges is not None:
            edge, edge_references = SpecsPass._get_namespace(edges, ReadKind.MEMBER)
            references.update(edge_references)
            namespaces[names[2]] = edge
            paths[names[2]] = edges.node.path
            groups[edges.node.path] = edges.rows
        elif search.source is not None:
            changing = next(block for block in elements.values() if block.entries.get(ENDS[0]) == search.source)
            activity = symbol(f'active_{re.sub(r"[^0-9A-Za-z]", "_", node.path)}_edges', RR)
            references[activity] = ReferenceSpec(changing.entries[ACTIVE], False, 0, ReadKind.MEMBER)
            paths[names[2]] = changing.node.path
        namespace = SimpleNamespace(t=declarations.time, **vars(parameters), **namespaces)
        group = edges.node.path if edges is not None else None
        spec = KernelSpec(
            node.path,
            namespace,
            references,
            search.size,
            search,
            roles=paths,
            group=group,
            groups=groups,
            place=place if search.source is None else None,
        )
        return [spec, SpecsPass._get_end_spec(spec, nodes)], declarations

    @staticmethod
    def _get_hyperedge_roles(node: NodeSpec, elements: dict[str, BlockSpec]) -> tuple[BlockSpec, BlockSpec | None]:
        """Return the node group of a hyperedge interaction and the second role's group.

        The second role may bind no group.
        """
        blocks = [
            SpecsPass._resolve_role(node, role, elements)
            for role in declarations_of(node.component, VariableKind.ROLE)
            if getattr(node.component, role) is not None
        ]
        parts = [getattr(block.node.component, 'on', None) for block in blocks]
        valid = (
            len(blocks) <= MAX_HYPEREDGE_ROLES
            and parts[0] not in SECOND_PARTS
            and blocks[0].node.members is not None
            and all(part in SECOND_PARTS for part in parts[1:])
        )
        if not valid:
            error_msg = (
                f'Hyperedge interaction {node.path} binds {", ".join(block.node.path for block in blocks)}. Bind its '
                'first role to a group with one member per node, and at most a second role to a group on the '
                'hyperedges or on the edges of the same network.'
            )
            raise RoleError(error_msg)
        return blocks[0], next(iter(blocks[1:]), None)

    @staticmethod
    def _get_gathered(
        block: BlockSpec,
        alias: str,
        table: Indices | None,
        entry: EntrySpec | None = None,
        order: Indices | None = None,
    ) -> tuple[SimpleNamespace, dict[Expression, ReferenceSpec]]:
        """Return the namespace of one slot of a hyperedge.

        Its symbols read a group through a gather table. The slot of the members of the hyperedge has no table, and its
        symbols, which only reductions over the members read, have no references. A table of positions in the entry of
        the members of a growing hypergraph reads its members from that entry, and a table of positions in an order
        drawn at every step reads the members at those positions of the order.
        """
        namespace, references = SpecsPass._get_namespace(block, ReadKind.NEIGHBOR, alias=alias)
        if table is None:
            return namespace, {}
        read = NeighborReadSpec(table, None, None, None, 0, entry, order)
        return namespace, {item: replace(reference, neighbor=read) for item, reference in references.items()}

    @staticmethod
    def _get_hyperedge_spec(
        block: BlockSpec,
        roles: tuple[BlockSpec, BlockSpec | None],
        tables: tuple[str, Tables],
        time: Expression,
        *,
        growing: BlockSpec | None = None,
        order: Indices | None = None,
    ) -> KernelSpec:
        """Return the spec of the laws of one kind and one size of hyperedge.

        It reads them through the tables of its rows. On a network whose hyperedges are the rows of a variable group,
        the members read their nodes from the entry of the members of the group, and the spec reads the activity of the
        hyperedge of every row. The tables of a grouping drawn at every step hold positions in an order, which the spec
        reads through.
        """
        nodes, second = roles
        mode, (focal, others, hyperedges, edges) = tables
        entry = None if growing is None else growing.entries[MEMBERS]
        parameters, references = SpecsPass._get_namespace(block, ReadKind.SINGLE)
        namespaces: dict[str, object] = {}
        namespaces['member'], gathered = SpecsPass._get_gathered(nodes, '__member', focal, entry, order)
        references.update(gathered)
        read = [
            SpecsPass._get_gathered(nodes, f'__other{position}', table, entry, order)
            for position, table in enumerate(others)
        ]
        namespaces['others'] = tuple(namespace for namespace, _ in read)
        for _, gathered in read:
            references.update(gathered)
        namespaces['members'], _ = SpecsPass._get_gathered(nodes, '__members', None)
        paths: dict[str, str | tuple[str, ...]] = {'member': nodes.node.path}
        if second is not None:
            part = 'edge' if getattr(second.node.component, 'on', None) == 'edges' else 'hyperedge'
            table = edges if part == 'edge' else hyperedges
            namespaces[part], gathered = SpecsPass._get_gathered(second, f'__{part}', table)
            references.update(gathered)
            paths[part] = second.node.path
        if growing is not None:
            activity = symbol(f'active_{re.sub(r"[^0-9A-Za-z]", "_", block.node.path)}_{mode}_hyperedges', RR)
            read_activity = NeighborReadSpec(hyperedges, None, None, None, 0)
            references[activity] = ReferenceSpec(growing.entries[ACTIVE], False, 0, ReadKind.NEIGHBOR, read_activity)
        return KernelSpec(
            block.node.path,
            SimpleNamespace(t=time, **vars(parameters), **namespaces),
            references,
            int(focal.size),
            roles=paths,
            hyperedge=mode,
        )

    @staticmethod
    def _get_hyperedge_specs(block: BlockSpec, elements: dict[str, BlockSpec], time: Expression) -> list[KernelSpec]:
        """Return the specs of a hyperedge interaction.

        They are the specs of the laws of the focal member and of the hyperedge per size.

        Args:
            block:
                The block of the interaction.

            elements:
                The blocks of the elements and groups by path.

            time:
                The time symbol.

        Returns:
            specs:
                Two specs per size of hyperedge, the laws of the focal member first.

        Raises:
            LabelError:
                The interaction declares a parameter named like a slot of a hyperedge.
        """
        node = block.node
        roles = SpecsPass._get_hyperedge_roles(node, elements)
        SpecsPass._check_slot_names(block)
        growing = next(
            (
                item
                for item in elements.values()
                if item.node.active is not None
                and getattr(item.node.component, 'on', None) == 'hyperedges'
                and item.node.space is roles[0].node.space
            ),
            None,
        )
        if growing is None:
            return [
                SpecsPass._get_hyperedge_spec(block, roles, tables, time)
                for tables in HyperedgeTables._get_tables_of(node, *roles)
            ]
        return [
            SpecsPass._get_hyperedge_spec(block, roles, tables, time, growing=growing)
            for tables in HyperedgeTables._get_growing_tables_of(node, *roles, growing)
        ]

    @staticmethod
    def _check_slot_names(block: BlockSpec) -> None:
        """Raise an error for an interaction parameter named like a hyperedge slot.

        Args:
            block:
                The block of the interaction.

        Raises:
            LabelError:
                The interaction declares a parameter named like a slot of a hyperedge.
        """
        parameters, _ = SpecsPass._get_namespace(block, ReadKind.SINGLE)
        clashes = sorted(set(vars(parameters)) & set(HYPEREDGE_NAMES))
        if clashes:
            error_msg = (
                f'Interaction {block.node.path} declares the parameters {", ".join(clashes)}, whose names its laws '
                'read as the slots of a hyperedge or a group. Rename them.'
            )
            raise LabelError(error_msg)

    @staticmethod
    def _get_grouping_specs(block: BlockSpec, elements: dict[str, BlockSpec], time: Expression) -> list[KernelSpec]:
        """Return the specs of a symbolic grouping, one per size of its groups.

        Their rows are the members of a group. Pairs and partitions read the members at positions of an order drawn at
        every step, and neighbourhoods read the members of the groups of the space.

        Args:
            block:
                The block of the grouping.

            elements:
                The blocks of the elements and groups by path.

            time:
                The time symbol.

        Returns:
            specs:
                One spec per size of group, whose update laws reduce over the groups that hold a member.

        Raises:
            RoleError:
                The target is not a group, or its groups do not fit its space.
        """
        node = block.node
        component = node.component
        assert isinstance(component, GroupingInteraction)
        nodes = SpecsPass._resolve_role(node, component._get_role('target'), elements)
        SpecsPass._check_slot_names(block)
        groups = HyperedgeTables._get_groups_of(node, nodes)
        order = None if component.groups == 'neighborhood' else np.arange(nodes.rows, dtype=np.intp)
        counts = np.bincount(np.concatenate([members.ravel() for members in groups]), minlength=nodes.rows)
        specs = []
        for members in groups:
            ids = np.arange(members.shape[0], dtype=np.intp)
            focal, others, hyperedges, _ = HyperedgeTables._get_tables(members, ids, MEMBER_LAWS)
            tables = (MEMBER_LAWS, (focal, others, hyperedges, None))
            spec = SpecsPass._get_hyperedge_spec(block, (nodes, None), tables, time, order=order)
            specs.append(replace(spec, reduce=component.reduce, counts=counts[focal]))
        return specs

    @staticmethod
    def _get_member_spec(block: BlockSpec, time: Expression) -> KernelSpec:
        """Return the kernel spec of the intrinsic laws of an element or group."""
        namespace, references = SpecsPass._get_namespace(block, ReadKind.MEMBER)
        namespace = SpecsPass._copy_with_other(block, namespace, references)
        group = block.node.path if block.node.members is not None else None
        return KernelSpec(
            block.node.path,
            SimpleNamespace(t=time, **vars(namespace)),
            references,
            block.rows,
            group=group,
        )

    @staticmethod
    def _get_pair_spec(block: BlockSpec, elements: dict[str, BlockSpec], time: Expression) -> KernelSpec:
        """Return the kernel spec of a pair interaction.

        It is evaluated once per pair within its cutoff.
        """
        node = block.node
        interaction = node.as_interaction
        names = SpecsPass._get_end_names(node, ('first', 'second'))
        first, second = (SpecsPass._resolve_role(node, name, elements) for name in names)
        position = getattr(interaction, 'position', 'x')
        for bound in (first, second):
            entry = bound.entries.get(position)
            if entry is None or not entry.member_shape:
                error_msg = f'Pair interaction {node.path} needs the vector state {position!r} in {bound.node.path}.'
                raise ValidationError(error_msg)
        parameters, references = SpecsPass._get_namespace(block, ReadKind.SINGLE)
        first_namespace, first_references = SpecsPass._get_namespace(first, ReadKind.FIRST)
        second_namespace, second_references = SpecsPass._get_namespace(second, ReadKind.SECOND, alias='_pair')
        references.update(first_references)
        references.update(second_references)
        period = getattr(interaction, 'period', None)
        period = period if period is not None else SpecsPass._get_space_period(node)
        pairs = PairSearch(first.entries[position], second.entries[position], block.entries['cutoff'], period).prepare()
        namespace = SimpleNamespace(
            t=time,
            **vars(parameters),
            **{names[0]: first_namespace, names[1]: second_namespace},
        )
        paths: dict[str, str | tuple[str, ...]] = {names[0]: first.node.path, names[1]: second.node.path}
        return KernelSpec(node.path, namespace, references, 0, pairs, roles=paths)

    @staticmethod
    def _get_growth_specs(specs: list[KernelSpec], declarations: DeclarationsSpec) -> list[KernelSpec]:
        """Return the specs of an addition of one node per step.

        Their namespace holds the node being added.
        """
        nodes = next(
            (block for block in declarations.capacity.values() if getattr(block.node.component, 'on', None) == 'sites'),
            None,
        )
        if nodes is None:
            return specs
        grown = []
        for spec in specs:
            references = dict(spec.references)
            namespace = SpecsPass._copy_with_source(nodes, spec.namespace, references)
            grown.append(replace(spec, namespace=namespace, references=references))
        return grown

    @staticmethod
    def _get_interaction_specs(
        block: BlockSpec,
        elements: dict[str, BlockSpec],
        time: Expression,
        observer: Observer,
    ) -> list[KernelSpec]:
        """Return the kernel specs of an interaction.

        There is one per element of its element sets, or else one.
        """
        node = block.node
        declared = declarations_of(node.component, VariableKind.ROLE)
        labels = {
            role: getattr(node.component, role)
            for role, declaration in declared.items()
            if declaration.variadic or getattr(node.component, role) is not None
        }
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
            if role not in sets
            and role not in variadic
            and isinstance(found := observer.run((node, label)).observed_, ObservedSpec)
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
            {role: tuple(SpecsPass._get_entry_path(*entry) for entry in entries) for role, entries in variadic.items()},
        )
        paths.update({role: SpecsPass._get_entry_path(*member) for role, member in single_members.items()})
        specs = []
        for index in range(next(iter(sizes), 1)):
            parameters, references = SpecsPass._get_namespace(block, ReadKind.SINGLE)
            namespaces: dict[str, SimpleNamespace | tuple[SimpleNamespace, ...]] = {}
            set_symbols: dict[Expression, SetSymbolSpec] = {}
            for role, members in sets.items():
                namespaces[role], role_references = SpecsPass._get_namespace(members[index], ReadKind.SINGLE)
                references.update(role_references)
                set_symbols.update(SpecsPass._set_symbols(members, index))
            for role, value in observed.items():
                namespaces[role] = value.namespace
                references.update(value.references)
            for role, entries in variadic.items():
                read = [SpecsPass._get_source_namespace(*entry, bound) for entry in entries]
                namespaces[role] = tuple(namespace for namespace, _ in read)
                for _, entry_references in read:
                    references.update(entry_references)
            for role, (member_block, position) in single_members.items():
                namespaces[role], role_references = SpecsPass._get_member_namespace(member_block, position)
                references.update(role_references)
            for role, item in bound.items():
                role_namespace, role_references = SpecsPass._get_namespace(
                    item,
                    ReadKind.MEMBER if item.node.members is not None else ReadKind.SINGLE,
                )
                namespaces[role] = SpecsPass._copy_with_other(item, role_namespace, role_references)
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
    def _binds_separate(block: BlockSpec, bound: list[str], elements: dict[str, BlockSpec]) -> bool:
        """Check whether a neighbour interaction binds a role to separate elements.

        The interaction is over neighbours, and the role binds an element set or a single element.
        """
        node = block.node
        for role in bound:
            label = getattr(node.component, role)
            if isinstance(label, list | tuple):
                return True
            if SpecsPass._resolve_member(node, role, label, elements) is not None:
                continue
            if SpecsPass._resolve_role(node, role, elements).node.members is None:
                return True
        return False

    @staticmethod
    def _separate_specs(
        block: BlockSpec,
        bound: list[str],
        elements: dict[str, BlockSpec],
        time: Expression,
        observer: Observer,
    ) -> list[KernelSpec]:
        """Return the specs of a neighbour interaction that binds separate elements.

        There is one spec per element of a set. The first role binds `namespace.target`. When the second role binds a
        group, `namespace.source` reads its members, and otherwise it is the tuple of the namespaces of the elements the
        second role binds, or of the other elements of the set the first role binds. The ends take the names the
        interaction's `_get_role` returns.
        """
        node = block.node
        source_name, target_name = SpecsPass._get_end_names(node, EDGE_NAMES[:2])
        first, second = bound[0], next(iter(bound[1:]), None)
        sources = getattr(node.component, second or first)
        separate = second is None or SpecsPass._binds_separate(block, [second], elements)
        specs = []
        for spec in SpecsPass._get_interaction_specs(block, elements, time, observer):
            namespace = dict(vars(spec.namespace))
            namespace[target_name] = namespace.pop(first)
            if second is not None:
                namespace[source_name] = namespace.pop(second)
            references = dict(spec.references)
            if separate:
                others = [sources] if isinstance(sources, str) else list(sources)
                if second is None:
                    others = [label for index, label in enumerate(others) if index != spec.set_index]
                namespace[source_name] = tuple(
                    SpecsPass._get_other_namespace(node, second or first, label, elements, references)
                    for label in others
                )
            specs.append(replace(spec, namespace=SimpleNamespace(**namespace), references=references))
        return specs

    @staticmethod
    def _get_other_namespace(
        node: NodeSpec,
        role: str,
        label: str,
        elements: dict[str, BlockSpec],
        references: dict[Expression, ReferenceSpec],
    ) -> SimpleNamespace:
        """Return the namespace of one separate element a neighbour interaction reads.

        The reductions of the interaction over neighbours read it.
        """
        other, other_references = SpecsPass._get_namespace(
            SpecsPass._resolve_path(node, role, label, elements),
            ReadKind.SINGLE,
        )
        references.update(other_references)
        return other

    @staticmethod
    def _get_specs_of(
        block: BlockSpec,
        elements: dict[str, BlockSpec],
        declarations: DeclarationsSpec,
        observer: Observer,
    ) -> tuple[list[KernelSpec], DeclarationsSpec]:
        """Return the kernel specs of a component.

        They come from its intrinsic laws, its edges, its pairs, or its roles.

        Args:
            block:
                The block of the component.

            elements:
                The blocks of the elements and groups by path.

            declarations:
                The declarations, with the time symbol, which the hidden parameters of the specs extend.

            observer:
                The prepared observer of the observables that roles may bind to.

        Returns:
            specs:
                The kernel specs, with the extended declarations.
        """
        time = declarations.time
        if not block.node.interaction:
            return [SpecsPass._get_member_spec(block, time)], declarations
        component = block.node.component
        bound = [role for role in declarations_of(component, VariableKind.ROLE) if getattr(component, role) is not None]
        if isinstance(component, NeighborInteraction) and SpecsPass._binds_separate(block, bound, elements):
            return SpecsPass._separate_specs(block, bound, elements, time, observer), declarations
        if isinstance(component, NeighborInteraction) and bound:
            return SpecsPass._get_edge_specs(block, elements, declarations)
        grouped = isinstance(component, GroupingInteraction) and component.form == 'symbolic'
        if (isinstance(component, HyperedgeInteraction) and bound) or (
            isinstance(component, GroupingInteraction) and grouped and component.groups != 'state'
        ):
            gathered = SpecsPass._get_grouping_specs if grouped else SpecsPass._get_hyperedge_specs
            return gathered(block, elements, time), declarations
        if isinstance(block.node.component, PairInteraction):
            return [SpecsPass._get_pair_spec(block, elements, time)], declarations
        specs = SpecsPass._get_interaction_specs(block, elements, time, observer)
        grows = isinstance(component, TopologyInteraction) and component.action == 'add-node' and not bound
        return (SpecsPass._get_growth_specs(specs, declarations) if grows else specs), declarations
