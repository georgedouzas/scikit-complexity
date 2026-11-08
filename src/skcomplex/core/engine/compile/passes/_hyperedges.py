"""Build the gather tables of the hyperedges of a network and expand reductions.

There is one table per size, and the reductions over their members are expanded.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from numbers import Number
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....components.laws import BaseLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....components.laws.utils import maximum, minimum, reductions_in, where
from ....components.spaces import NetworkSpace
from ....components.specs import NodeSpec
from ....errors import RoleError, ValidationError
from ....specs import ReductionSpec
from ....types import Indices
from ..._utils import vector
from ...problems._block_spec import BlockSpec
from ...problems._declarations_spec import _ACTIVE, MEMBERS
from ...problems._kernel_spec import KernelSpec
from ...types import ReadKind
from .._base import BasePass

MEMBER_LAWS = 'member'
HYPEREDGE_LAWS = 'hyperedge'
HYPEREDGE_REDUCTIONS = ('sum', 'mean', 'count', 'max', 'min')

type Tables = tuple[Indices, list[Indices], Indices, Indices | None]


class HyperedgeTables:
    """Gather tables of the members of the hyperedges of one size.

    The laws of a hyperedge interaction read them. The laws of the focal member run over one row per member of every
    hyperedge, and the laws of the member of a group on hyperedges or edges over one row per hyperedge. Every symbol of
    the laws reads its value through the table of its slot: the focal member, one of the other members, the hyperedge,
    or the edge. On a network whose hyperedges are the rows of a variable group, the tables of the members hold
    positions in the parameter entry of the members of every row, which topology requests rewrite during a run.
    """

    @staticmethod
    def _space(node: NodeSpec, nodes: BlockSpec, second: BlockSpec | None) -> NetworkSpace:
        """Return the network of the nodes.

        It first checks the sizes of the groups the roles bind.
        """
        space = nodes.node.space
        if not isinstance(space, NetworkSpace) or not space.hyperedges_:
            error_msg = (
                f'Hyperedge interaction {node.path} binds {nodes.node.path}, which lives on {space!r}. Bind a group on '
                'the nodes of a network space with hyperedges.'
            )
            raise RoleError(error_msg)
        if nodes.rows != space.size_:
            error_msg = (
                f'Hyperedge interaction {node.path} binds {nodes.node.path} of {nodes.rows} members to a network of '
                f'{space.size_} nodes. Bind a group with one member per node.'
            )
            raise RoleError(error_msg)
        if second is not None and second.node.space is not space:
            error_msg = (
                f'Hyperedge interaction {node.path} binds {second.node.path}, which lives on another space than its '
                f'nodes {nodes.node.path}. Give it the network of the nodes.'
            )
            raise RoleError(error_msg)
        return space

    @staticmethod
    def _edge_indices(node: NodeSpec, space: NetworkSpace, edges: BlockSpec, ends: tuple[Indices, Indices]) -> Indices:
        """Return the edge of a layer that joins the first two members of every hyperedge.

        Args:
            node:
                The node of the interaction.

            space:
                The prepared network space.

            edges:
                The block of the edges of the layer.

            ends:
                The node indices of the first and the second member of every hyperedge.

        Returns:
            indices:
                The index of the edge that joins the first two members of every hyperedge.

        Raises:
            RoleError:
                The first two members of a hyperedge are joined by no edge of the layer.
        """
        layer = getattr(edges.node.component, 'layer', None)
        sources, targets = space.edges_[layer]
        directed = space.graphs_[layer].is_directed()
        index = {
            (int(first), int(second)): number
            for number, (first, second) in enumerate(zip(sources, targets, strict=True))
        }
        found = []
        for first, second in zip(ends[0].tolist(), ends[1].tolist(), strict=True):
            key = (first, second) if directed else (min(first, second), max(first, second))
            if key not in index:
                error_msg = (
                    f'Hyperedge interaction {node.path} binds the edge group {edges.node.path}, but no edge of its '
                    f'layer joins the nodes {first} and {second}, the first two members of a hyperedge. Order every '
                    'hyperedge so an edge joins its first two members.'
                )
                raise RoleError(error_msg)
            found.append(index[key])
        return np.asarray(found, dtype=np.intp)

    @staticmethod
    def _tables(members: np.ndarray, ids: Indices, mode: str) -> tuple[Indices, list[Indices], Indices, np.ndarray]:
        """Return the rows of the focal member, the other members, and the hyperedge.

        It also returns the members of the hyperedges.

        Args:
            members:
                The node index of every member of every hyperedge of one size, one row per hyperedge.

            ids:
                The index of every hyperedge among the hyperedges of the network.

            mode:
                `'member'` for one row per member of every hyperedge, `'hyperedge'` for one row per hyperedge.

        Returns:
            tables:
                The focal member, the other members in hyperedge order, and the hyperedge of every row, and the members
                of the hyperedge of every row.
        """
        size = members.shape[1]
        slots = range(size) if mode == MEMBER_LAWS else range(1)
        focal = np.concatenate([members[:, slot] for slot in slots])
        others = [
            np.concatenate(
                [members[:, [column for column in range(size) if column != slot][position]] for slot in slots],
            )
            for position in range(size - 1)
        ]
        rows = np.concatenate([members for _ in slots])
        return focal.astype(np.intp), [item.astype(np.intp) for item in others], np.tile(ids, len(slots)), rows

    @staticmethod
    def _incidences(space: NetworkSpace, size: int) -> tuple[Indices, list[Indices]]:
        """Return the hyperedges of one size around every node.

        They are rows grouped by node, with their other members.

        Args:
            space:
                The prepared network space.

            size:
                The number of members of the hyperedges.

        Returns:
            rows:
                The start of the rows of every node, with the number of rows last, and per position the other member
                of the hyperedge of every row, in hyperedge order.

        Examples:
            >>> import networkx as nx
            >>> from skcomplex.core.components.spaces import NetworkSpace
            >>> from skcomplex.core.engine.compile.passes._hyperedges import HyperedgeTables
            >>> space = NetworkSpace(nx.complete_graph(4), hyperedges=[(0, 1, 2), (1, 2, 3)]).prepare()
            >>> indptr, others = HyperedgeTables._incidences(space, 3)
            >>> indptr.tolist(), [item.tolist() for item in others]
            ([0, 1, 3, 5, 6], [[1, 2, 0, 1, 0, 1], [2, 3, 2, 3, 1, 2]])
        """
        ids = np.asarray([number for number, item in enumerate(space.hyperedges_) if len(item) == size], dtype=np.intp)
        members = np.asarray([space.hyperedges_[number] for number in ids], dtype=np.intp).reshape(-1, size)
        focal, others, _, _ = HyperedgeTables._tables(members, ids, MEMBER_LAWS)
        order = np.argsort(focal, kind='stable')
        counts = np.bincount(focal, minlength=space.size_)
        indptr = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(counts)]).astype(np.intp)
        return indptr, [item[order] for item in others]

    @staticmethod
    def _sizes(node: NodeSpec, space: NetworkSpace) -> list[int]:
        """Return the sizes of the hyperedges an interaction runs on.

        It is its own size, or every size of the network.

        Args:
            node:
                The node of the interaction.

            space:
                The prepared network space.

        Returns:
            sizes:
                The sizes of the hyperedges in increasing order.

        Raises:
            RoleError:
                The network has no hyperedge of the size of the interaction.
        """
        sizes = sorted({len(hyperedge) for hyperedge in space.hyperedges_})
        size = getattr(node.component, 'size', None)
        if size is None:
            return sizes
        if size not in sizes:
            error_msg = (
                f'Hyperedge interaction {node.path} runs on hyperedges of size {size}, but the hyperedges of its '
                f'network have the sizes {", ".join(map(str, sizes))}. Give one of these sizes.'
            )
            raise RoleError(error_msg)
        return [size]

    @staticmethod
    def _tables_of(node: NodeSpec, nodes: BlockSpec, second: BlockSpec | None) -> list[tuple[str, Tables]]:
        """Return the gather tables of every hyperedge size an interaction runs on.

        Tables are given for both kinds of laws.

        Args:
            node:
                The hyperedge interaction.

            nodes:
                The block of the group on the nodes.

            second:
                The block of the group on hyperedges or edges that the second role binds, or `None`.

        Returns:
            tables:
                Per size, the kind of laws, `'member'` then `'hyperedge'`, with the focal member, the other members,
                the hyperedge, and the edge, or `None`, of every row.

        Raises:
            RoleError:
                The groups do not fit the network or its hyperedges.
        """
        space = HyperedgeTables._space(node, nodes, second)
        part = getattr(second.node.component, 'on', None) if second is not None else None
        if second is not None and part == 'hyperedges' and second.rows != len(space.hyperedges_):
            error_msg = (
                f'Hyperedge interaction {node.path} binds {second.node.path} of {second.rows} members to a network of '
                f'{len(space.hyperedges_)} hyperedges. Give the group one member per hyperedge.'
            )
            raise RoleError(error_msg)
        found: list[tuple[str, Tables]] = []
        for size in HyperedgeTables._sizes(node, space):
            ids = np.asarray(
                [number for number, item in enumerate(space.hyperedges_) if len(item) == size],
                dtype=np.intp,
            )
            members = np.asarray([space.hyperedges_[number] for number in ids], dtype=np.intp).reshape(-1, size)
            for mode in (MEMBER_LAWS, HYPEREDGE_LAWS):
                focal, others, hyperedges, rows = HyperedgeTables._tables(members, ids, mode)
                edges = None
                if second is not None and part == 'edges':
                    edges = HyperedgeTables._edge_indices(node, space, second, (rows[:, 0], rows[:, 1]))
                found.append((mode, (focal, others, hyperedges, edges)))
        return found

    @staticmethod
    def _growing_tables_of(
        node: NodeSpec,
        nodes: BlockSpec,
        second: BlockSpec | None,
        growing: BlockSpec,
    ) -> list[tuple[str, Tables]]:
        """Return the gather tables of the rows of a variable group on hyperedges.

        Tables are given for both kinds of laws. The tables of the focal and the other members hold the positions of the
        members in the parameter entry of the members of every row, and the table of the hyperedge holds its row.

        Args:
            node:
                The hyperedge interaction.

            nodes:
                The block of the variable group on the nodes.

            second:
                The block of the group that the second role binds, which is the variable group on hyperedges, or
                `None`.

            growing:
                The block of the variable group on the hyperedges of the network.

        Returns:
            tables:
                The kind of laws, `'member'` then `'hyperedge'`, with the focal member, the other members, the
                hyperedge, and no edge, of every row.

        Raises:
            RoleError:
                The nodes are not a variable group, the second role binds another group, the group has a rank, or
                the interaction runs on another size than the hyperedges of the group.
        """
        size = growing.entries[MEMBERS].columns
        if getattr(growing.node.component, 'rank', None) is not None:
            error_msg = (
                f'Hyperedge interaction {node.path} runs on {growing.node.path}, whose hyperedges have every size up '
                'to its rank. Laws per hyperedge run on a variable group of one size, without a rank.'
            )
            raise RoleError(error_msg)
        if nodes.node.active is None:
            error_msg = (
                f'Hyperedge interaction {node.path} binds {nodes.node.path} on a network whose hyperedges change. '
                'Bind the variable group on its nodes.'
            )
            raise RoleError(error_msg)
        if second is not None and second is not growing:
            error_msg = (
                f'Hyperedge interaction {node.path} binds {second.node.path} on a network whose hyperedges change. '
                f'Bind the variable group on its hyperedges, {growing.node.path}.'
            )
            raise RoleError(error_msg)
        if getattr(node.component, 'size', None) not in {None, size}:
            error_msg = (
                f'Hyperedge interaction {node.path} runs on hyperedges of size {node.component.size}, but the '
                f'hyperedges of {growing.node.path} have size {size}. Give the size {size}.'
            )
            raise RoleError(error_msg)
        rows = growing.rows
        positions = np.arange(rows * size, dtype=np.intp).reshape(rows, size)
        found: list[tuple[str, Tables]] = []
        for mode in (MEMBER_LAWS, HYPEREDGE_LAWS):
            focal, others, hyperedges, _ = HyperedgeTables._tables(positions, np.arange(rows, dtype=np.intp), mode)
            found.append((mode, (focal, others, hyperedges, None)))
        return found

    @staticmethod
    def _activity(spec: KernelSpec) -> Expression | None:
        """Return the symbol of the hyperedge activity of every row, or `None`.

        The hyperedges are those of a growing hypergraph.
        """
        return next(
            (
                item
                for item, reference in spec.references.items()
                if reference.source is ReadKind.NEIGHBOR and reference.entry.name.endswith(f'.{_ACTIVE}')
            ),
            None,
        )

    @staticmethod
    def _scaled(spec: KernelSpec, law: BaseLaw, value: object, activity: Expression) -> Expression | FreeModuleElement:
        """Return a scalar or a vector value multiplied by an activity.

        A value that is not symbolic is refused.
        """
        if not isinstance(value, Expression | FreeModuleElement | Number):
            error_msg = (
                f'{spec.owner} returns {type(law).__name__} with the value {value!r} for the members of hyperedges '
                'that change during a run, but a free row of the hypergraph switches off only a symbolic value. Write '
                'the value as an expression, or keep the hyperedges fixed.'
            )
            raise ValidationError(error_msg)
        if isinstance(value, FreeModuleElement):
            return vector(SR(item) * activity for item in value)
        return SR(value) * activity

    @staticmethod
    def _masked(spec: KernelSpec, law: BaseLaw) -> BaseLaw:
        """Return a law of a spec on a growing hypergraph that vanishes on free rows.

        The free rows are those of the group on hyperedges. A rate, a noise, or a jump is multiplied by the activity of
        the hyperedge. An update of the hyperedge member writes only its own row, which a new hyperedge starts afresh,
        so it stays as it is.

        Args:
            spec:
                The spec of the hyperedge interaction for one kind of laws.

            law:
                The law, with its reductions over the members written out.

        Returns:
            law:
                The masked law, or the law itself on a hypergraph whose hyperedges do not change.

        Raises:
            ValidationError:
                A law of the focal member other than a rate, a noise, or a jump runs on a growing hypergraph, or its
                value is not symbolic.
        """
        activity = HyperedgeTables._activity(spec)
        if activity is None:
            return law
        if isinstance(law, RateLaw | NoiseLaw):
            return BasePass._replaced(law, value=HyperedgeTables._scaled(spec, law, law.value, activity)).prepare(spec.owner)
        if isinstance(law, JumpLaw):
            return BasePass._replaced(law, rate=HyperedgeTables._scaled(spec, law, law.rate, activity)).prepare(spec.owner)
        if isinstance(law, UpdateLaw) and spec.hyperedge == HYPEREDGE_LAWS:
            return law
        error_msg = (
            f'{spec.owner} returns {type(law).__name__} for the members of hyperedges that change during a run, where '
            'its free rows would write to a node. Write a rate, a noise, or a jump, or update the hyperedge member.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _neighborhoods(node: NodeSpec, nodes: BlockSpec) -> list[list[int]]:
        """Return every member of a group on the sites of a space and its neighbours.

        Every member is followed by the members around it.

        Args:
            node:
                The node of the grouping.

            nodes:
                The block of the group.

        Returns:
            neighborhoods:
                Every member followed by the members around it, one list per member.

        Raises:
            RoleError:
                The space of the group has no neighbourhoods, or the group has not one member per site.
        """
        space = nodes.node.space
        size = getattr(space, 'size_', None)
        if space is None or size is None or nodes.rows != size:
            error_msg = (
                f'Grouping {node.path} forms the neighbourhoods of {nodes.node.path}, which lives on {space!r}. Bind a '
                'group with one member per site of a lattice or node of a network.'
            )
            raise RoleError(error_msg)
        return [[site, *space._get_neighbors(site).tolist()] for site in range(size)]

    @staticmethod
    def _groups_of(node: NodeSpec, nodes: BlockSpec) -> list[np.ndarray]:
        """Return the members of the groups of a grouping.

        There is one array per size, with one row per group. Pairs and partitions hold positions in an order drawn at
        every step: consecutive positions form a group, the positions left over form the last group of a partition, and
        an odd position forms no pair. Neighbourhoods hold the members of the groups of the space.

        Args:
            node:
                The grouping.

            nodes:
                The block of the group it binds.

        Returns:
            groups:
                The members or positions of every group, by size.

        Raises:
            RoleError:
                The target is not a group, or its space has no neighbourhoods for groups of neighbourhoods.
        """
        if nodes.node.members is None:
            error_msg = f'Grouping {node.path} binds {nodes.node.path}, a single element. Bind a group.'
            raise RoleError(error_msg)
        kind = getattr(node.component, 'groups', None)
        if kind == 'neighborhood':
            found = HyperedgeTables._neighborhoods(node, nodes)
            sizes = sorted({len(group) for group in found})
            return [np.asarray([group for group in found if len(group) == size], dtype=np.intp) for size in sizes]
        size = 2 if kind == 'pairs' else int(getattr(node.component, 'size', 2))
        whole = nodes.rows // size * size
        groups = [np.arange(whole, dtype=np.intp).reshape(-1, size)]
        if kind == 'partition' and whole < nodes.rows:
            groups.append(np.arange(whole, nodes.rows, dtype=np.intp)[np.newaxis])
        return [group for group in groups if group.size]

    @staticmethod
    def _components(value: object) -> list[Expression]:
        """Return the symbols of a namespace entry.

        They are the components of a vector, or the one symbol.
        """
        if isinstance(value, FreeModuleElement):
            return [SR(item) for item in value]
        if isinstance(value, tuple):
            return [SR(component) for item in value for component in HyperedgeTables._components(item)]
        return [SR(value)]

    @staticmethod
    def _slots(spec: KernelSpec) -> dict[Expression, list[Expression]]:
        """Return the focal and other member symbols that every member symbol stands for."""
        namespace = spec.namespace
        slots: dict[Expression, list[Expression]] = {}
        for name, value in vars(namespace.members).items():
            readers = [namespace.member, *namespace.others]
            columns = [HyperedgeTables._components(getattr(reader, name)) for reader in readers]
            for position, item in enumerate(HyperedgeTables._components(value)):
                slots[item] = [column[position] for column in columns]
        return slots

    @staticmethod
    def _around(spec: KernelSpec, kind: ReadKind) -> tuple[dict[Expression, list[Expression]], dict[Expression, int]]:
        """Return the slots of a hyperedge around a member, by member symbol.

        The namespace of a spec holds the slots. A namespace of a member of a group on a network with hyperedges, the
        namespace of a spec or one of its entries, holds `members`, every member of a hyperedge around the member, and
        `others`, the other members.

        Args:
            spec:
                The spec.

            kind:
                How the symbols of the slots are read.

        Returns:
            slots:
                The symbols of the other members, in hyperedge order, that every symbol of the members stands for, and
                the position among the other members of every symbol of the other members.
        """
        candidates = [
            spec.namespace,
            *(item for item in vars(spec.namespace).values() if isinstance(item, SimpleNamespace)),
        ]
        columns: dict[Expression, list[Expression]] = {}
        positions: dict[Expression, int] = {}
        for namespace in candidates:
            members, others = getattr(namespace, 'members', None), getattr(namespace, 'others', None)
            if not isinstance(members, SimpleNamespace) or not isinstance(others, tuple):
                continue
            for name, value in vars(members).items():
                found = [HyperedgeTables._components(getattr(other, name)) for other in others]
                for index, item in enumerate(HyperedgeTables._components(value)):
                    reference = spec.references.get(item)
                    if reference is None or reference.source is not kind:
                        continue
                    columns[item] = [column[index] for column in found]
                    positions.update({column[index]: position for position, column in enumerate(found)})
        return columns, positions

    @staticmethod
    def _expanded(reduction: ReductionSpec, slots: dict[Expression, list[Expression]]) -> Expression:
        """Return a reduction over the members of a hyperedge written out over slots.

        It is written as a sum, a mean, or an extremum of slots.

        Args:
            reduction:
                The reduction over the members of a hyperedge.

            slots:
                The symbols of the slots of a hyperedge for every symbol the reduction reads.

        Returns:
            expression:
                The reduction written out over the slots.

        Raises:
            ValidationError:
                The reduction has weights or a kind that a hyperedge does not reduce.
        """
        if reduction.weights is not None or reduction.kind not in HYPEREDGE_REDUCTIONS:
            error_msg = (
                f'The reduction {reduction.call} over the members of a hyperedge has weights or the kind '
                f'{reduction.kind!r}. Reduce the members with total, mean, count, maximum, or minimum, without weights.'
            )
            raise ValidationError(error_msg)
        size = len(next(iter(slots.values())))
        mappings = [{item: columns[position] for item, columns in slots.items()} for position in range(size)]
        arguments = [SR(reduction.argument).subs(mapping) for mapping in mappings]
        conditions = [
            SR(1 if reduction.condition is None else reduction.condition).subs(mapping) for mapping in mappings
        ]
        if reduction.kind in {'max', 'min'}:
            if reduction.condition is not None:
                error_msg = f'The extremum {reduction.call} over the members of a hyperedge takes no condition.'
                raise ValidationError(error_msg)
            return maximum(*arguments) if reduction.kind == 'max' else minimum(*arguments)
        counted = sum((SR(where(condition, 1, 0)) for condition in conditions), SR(0))
        if reduction.kind == 'count':
            return counted
        selected = zip(arguments, conditions, strict=True)
        summed = sum((SR(where(condition, argument, 0)) for argument, condition in selected), SR(0))
        return summed if reduction.kind == 'sum' else summed / counted

    @staticmethod
    def _expansions(spec: KernelSpec, expressions: list[Expression]) -> dict[Expression, Expression]:
        """Return every hyperedge member reduction in expressions with its expansion.

        The expansion is over the slots.

        Args:
            spec:
                The spec of the hyperedge interaction.

            expressions:
                The expressions of a law.

        Returns:
            expansions:
                The expansion of every reduction call over the members.
        """
        return HyperedgeTables._expansions_over(HyperedgeTables._slots(spec), expressions)

    @staticmethod
    def _expansions_over(
        slots: dict[Expression, list[Expression]],
        expressions: list[Expression],
    ) -> dict[Expression, Expression]:
        """Return every hyperedge member reduction with its expansion over given slots.

        The reductions are those in expressions.

        Args:
            slots:
                The symbols of the focal and the other members that every symbol of the members stands for.

            expressions:
                The expressions.

        Returns:
            expansions:
                The expansion of every reduction call over the members.
        """
        expansions: dict[Expression, Expression] = {}
        for expression in expressions:
            for reduction in reductions_in(expression):
                inner = HyperedgeTables._expansions_over(slots, [reduction.argument])
                over = reduction.over
                if (over is not None and over in slots) or (
                    over is None and set(SR(reduction.argument).variables()) & set(slots)
                ):
                    argument = SR(reduction.argument).subs(inner) if inner else reduction.argument
                    expansions[reduction.call] = HyperedgeTables._expanded(replace(reduction, argument=argument), slots)
                else:
                    expansions.update(inner)
        return expansions
