"""Build the gather tables of the hyperedges of a network, one per size, and expand reductions over their members."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.laws.utils import maximum, minimum, reductions_in, where
from ....core.spaces import NetworkSpace
from ....core.specs import NodeSpec
from ....errors import RoleError, ValidationError
from ....specs import ReductionSpec
from ....types import Indices
from ...problems import BlockSpec, KernelSpec

MEMBER_LAWS = 'member'
HYPEREDGE_LAWS = 'hyperedge'
HYPEREDGE_REDUCTIONS = ('sum', 'mean', 'count', 'max', 'min')

type Tables = tuple[Indices, list[Indices], Indices, Indices | None]


class HyperedgeTables:
    """Gather tables of the members of the hyperedges of one size, which the laws of a hyperedge interaction read.

    The laws of the focal member run over one row per member of every hyperedge, and the laws of the member of a group
    on hyperedges or edges over one row per hyperedge. Every symbol of the laws reads its value through the table of
    its slot: the focal member, one of the other members, the hyperedge, or the edge.
    """

    @staticmethod
    def _space(node: NodeSpec, nodes: BlockSpec, second: BlockSpec | None) -> NetworkSpace:
        """Return the network of the nodes, after checking the sizes of the groups the roles bind."""
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
        """Return the rows of the focal member, of every other member, and of the hyperedge, and the hyperedge members.

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
    def _sizes(node: NodeSpec, space: NetworkSpace) -> list[int]:
        """Return the sizes of the hyperedges an interaction runs on, its own size or every size of the network.

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
        """Return the gather tables of every size of hyperedge an interaction runs on, for both kinds of laws.

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
    def _components(value: object) -> list[Expression]:
        """Return the symbols of a namespace entry, the components of a vector or the one symbol."""
        if isinstance(value, FreeModuleElement):
            return [SR(item) for item in value]
        if isinstance(value, tuple):
            return [SR(component) for item in value for component in HyperedgeTables._components(item)]
        return [SR(value)]

    @staticmethod
    def _slots(spec: KernelSpec) -> dict[Expression, list[Expression]]:
        """Return the symbols of the focal and the other members that every symbol of the members stands for."""
        namespace = spec.namespace
        slots: dict[Expression, list[Expression]] = {}
        for name, value in vars(namespace.members).items():
            readers = [namespace.member, *namespace.others]
            columns = [HyperedgeTables._components(getattr(reader, name)) for reader in readers]
            for position, item in enumerate(HyperedgeTables._components(value)):
                slots[item] = [column[position] for column in columns]
        return slots

    @staticmethod
    def _expanded(reduction: ReductionSpec, slots: dict[Expression, list[Expression]]) -> Expression:
        """Return a reduction over the members of a hyperedge written out as a sum, a mean, or an extremum of slots.

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
        """Return every reduction over the members of a hyperedge in expressions with its expansion over the slots.

        Args:
            spec:
                The spec of the hyperedge interaction.

            expressions:
                The expressions of a law.

        Returns:
            expansions:
                The expansion of every reduction call over the members.
        """
        slots = HyperedgeTables._slots(spec)
        expansions: dict[Expression, Expression] = {}
        for expression in expressions:
            for reduction in reductions_in(expression):
                inner = HyperedgeTables._expansions(spec, [reduction.argument])
                over = reduction.over
                if (over is not None and over in slots) or (
                    over is None and set(SR(reduction.argument).variables()) & set(slots)
                ):
                    argument = SR(reduction.argument).subs(inner) if inner else reduction.argument
                    expansions[reduction.call] = HyperedgeTables._expanded(replace(reduction, argument=argument), slots)
                else:
                    expansions.update(inner)
        return expansions
