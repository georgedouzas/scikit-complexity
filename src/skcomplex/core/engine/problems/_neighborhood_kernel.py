"""Neighbourhood reduction that laws read, one value per member.

It is compiled into a function of its symbols.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import itertools
from collections.abc import Callable, Mapping

import networkx as nx
import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import Expression
from scipy.sparse.csgraph import connected_components

from ...components.spaces import (
    AdjacencyNeighborhood,
    BaseNeighborhood,
    BaseSpace,
    CayleySpace,
    CellComplexSpace,
    CofaceNeighborhood,
    ConeNeighborhood,
    ContinuousSpace,
    LatticeNeighborhood,
    LatticeSpace,
    MemoryNeighborhood,
    NearestNeighborhood,
    NetworkSpace,
    RadiusNeighborhood,
)
from ...errors import ValidationError
from ...specs import DirectionSpec, EntrySpec, PairReadSpec, RunStateSpec, VariableSpec
from ...types import Array, Indices
from ..types import Pairs
from ._base import BaseKernel
from ._edge_row_search import EdgeRowSearch
from ._expression_printer import ExpressionPrinter
from ._metric_search import MetricSearch
from ._static_search import StaticSearch

_SPACES_OF: tuple[tuple[type | tuple[type, ...], tuple[type, ...]], ...] = (
    (AdjacencyNeighborhood, (NetworkSpace,)),
    ((LatticeNeighborhood, MemoryNeighborhood), (LatticeSpace, CayleySpace)),
    ((RadiusNeighborhood, NearestNeighborhood, ConeNeighborhood), (ContinuousSpace,)),
    (CofaceNeighborhood, (CellComplexSpace,)),
)
"""Spaces every kind of neighbourhood is read on."""

type NeighborSearch = StaticSearch | EdgeRowSearch | MetricSearch

_PLANE = 2


class NeighborhoodKernel(BaseKernel):
    """Neighbourhood reduction that laws read, one value per member.

    It is compiled into a function of its symbols.

    Args:
        symbol:
            The placeholder symbol that kernels read instead of the reduction.

        kind:
            One of `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, `'normalized-mean'`, `'argmax'`, the components
            reductions `'component-label'`, `'component-size'`, `'component-largest'`, and `'component-giant'`, and
            the shortest paths reductions `'path-betweenness'` and `'path-closeness'`.

        argument:
            The value of one neighbour, or the activity of one link for a components or a shortest paths reduction,
            with nested reductions replaced by their placeholders.

        weight:
            The weight of one neighbour, 1 without one.

        entry:
            The hidden entry that holds one value per member, after the parameters in the extended parameter vector.

        group:
            The path of the group whose members read the reduction and are each other's neighbours.

        search:
            The neighbours of every member.

        reads:
            A mapping from every symbol of the argument and the weight to how it is read.

        time:
            The time symbol.

        coefficients:
            The factor of every pair of the search, such as the oriented entries of an incidence, or `None` for 1.

        hyperedge:
            The size of the hyperedges that every member forms with sets of its current neighbours, which the
            reduction runs over instead of single neighbours, or `None` for single neighbours.

    Attributes:
        paths_:
            The active links of the last shortest paths reduction with its values, which a run reuses while the links
            stay the same, or `None` before the first one.
    """

    values_: None
    paths_: tuple[Indices, Indices, Array] | None = None

    def __init__(
        self,
        symbol: Expression,
        kind: str,
        argument: Expression,
        weight: Expression,
        entry: EntrySpec,
        *,
        group: str,
        search: NeighborSearch,
        reads: dict[Expression, PairReadSpec],
        time: Expression,
        coefficients: Array | None = None,
        hyperedge: int | None = None,
    ) -> None:
        self.symbol = symbol
        self.kind = kind
        self.argument = argument
        self.weight = weight
        self.entry = entry
        self.group = group
        self.search = search
        self.reads = reads
        self.time = time
        self.coefficients = coefficients
        self.hyperedge = hyperedge

    @functools.cached_property
    def _symbols(self) -> list[Expression]:
        """Symbols the argument and the weight read, sorted by name."""
        return sorted(self.reads, key=str)

    @functools.cached_property
    def _function(self) -> Callable[..., object]:
        """NumPy function of the argument and the weight, compiled once."""
        printer = ExpressionPrinter(time=self.time, compiled=True).prepare(self._symbols)
        return ExpressionPrinter._function_of(printer.run([self.argument, self.weight]).code_)

    @property
    def symbols(self) -> list[Expression]:
        """Symbols the argument and the weight read.

        They come in the order the compiled function takes them after time.
        """
        return list(self._symbols)

    @property
    def size(self) -> int:
        """Number of members."""
        return self.entry.rows

    def _gather(self, x: Array, q: Array, readers: Indices, others: Indices, edges: Indices | None) -> list[Array]:
        """Return the values of every symbol for every pair.

        For a reduction over neighbour hyperedges, the pairs are the sets of neighbours, with one column of `others`
        per other member.
        """
        values = []
        differences: dict[int | None, Array] = {}
        for item in self._symbols:
            pair = self.reads[item]
            read = pair.read
            column = (x if read.is_state else q)[read.index]
            neighbours = others if pair.position is None else others[:, pair.position]
            if pair.side == 'shared':
                values.append(column)
            elif pair.side == 'reader':
                values.append(column[readers])
            elif pair.side == 'edge':
                values.append(column[edges])
            elif pair.side == 'pair':
                values.append(column[np.arange(others.size)])
            elif pair.image:
                if not isinstance(self.search, MetricSearch):
                    error_msg = f'The reduction {self.symbol} reads a periodic image, which only a metric search finds.'
                    raise ValidationError(error_msg)
                if pair.position not in differences:
                    differences[pair.position] = self.search.differences(x, readers, neighbours)
                values.append(column[readers] - differences[pair.position][:, read.column])
            elif pair.table is not None:
                values.append(column[pair.table[neighbours]])
            else:
                values.append(column[neighbours])
        return values

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Write the reduction of every member into the extended parameter vector.

        The vector is the `p` of `inputs`.
        """
        self.values_ = None
        state = self._state_of(inputs)
        t, x, q = state.t, state.x, state.p
        indptr, others, edges = self.search.run(state).pairs_
        size = self.size
        if self.hyperedge is not None:
            indptr, others = self._neighbor_sets(indptr, others, self.hyperedge - 1)
        counts = np.diff(indptr)
        readers = np.repeat(np.arange(size, dtype=np.intp), counts)
        argument, weight = ExpressionPrinter._values_of(self._function(t, *self._gather(x, q, readers, others, edges)))
        pairs = readers.size
        argument = np.broadcast_to(np.asarray(argument, dtype=float), (pairs,))
        weight = np.broadcast_to(np.asarray(weight, dtype=float), (pairs,))
        if self.kind.startswith('component-'):
            active = argument != 0
            q[self.entry.indices] = self._components_of(self.kind, readers[active], others[active], size)
            return
        if self.kind.startswith('path-'):
            q[self.entry.indices] = self._cached_paths(readers[argument != 0], others[argument != 0])
            return
        if self.coefficients is not None:
            weight = weight * self.coefficients
        q[self.entry.indices] = self._reduce_pairs(self.kind, readers, weight, argument, counts)

    @staticmethod
    def _neighbor_sets(indptr: Indices, others: Indices, number: int) -> tuple[Indices, Indices]:
        """Return the sets of a number of neighbours of every member in CSR form.

        The sets of a member are the combinations of its neighbours, each in increasing neighbour order. The members of
        equal neighbour counts share one table of combinations.

        Args:
            indptr:
                The row starts of the neighbours of every member.

            others:
                The neighbours of every member, in increasing order within a row.

            number:
                The number of neighbours in a set.

        Returns:
            sets:
                The row starts of the sets of every member, and one row of neighbours per set.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> indptr, others = np.array([0, 3, 4]), np.array([1, 2, 3, 0])
            >>> indptr, sets = NeighborhoodKernel._neighbor_sets(indptr, others, 2)
            >>> indptr.tolist(), sets.tolist()
            ([0, 3, 3], [[1, 2], [1, 3], [2, 3]])
        """
        counts = np.diff(indptr)
        found = np.zeros(counts.size, dtype=np.intp)
        rows = [np.zeros((0, number), dtype=np.intp)]
        readers = [np.zeros(0, dtype=np.intp)]
        for count in np.unique(counts[counts >= number]):
            combinations = np.asarray(list(itertools.combinations(range(int(count)), number)), dtype=np.intp)
            members = np.flatnonzero(counts == count)
            found[members] = len(combinations)
            places = indptr[members][:, None, None] + combinations[None, :, :]
            rows.append(others[places].reshape(-1, number))
            readers.append(np.repeat(members, len(combinations)))
        order = np.argsort(np.concatenate(readers), kind='stable')
        sets = np.concatenate(rows)[order].astype(np.intp)
        return np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(found)]).astype(np.intp), sets

    def _cached_paths(self, readers: Indices, others: Indices) -> Array:
        """Return the shortest paths reduction of the active links.

        It is computed again only when the links changed.

        Args:
            readers:
                The first member of every active link.

            others:
                The second member of every active link.

        Returns:
            values:
                One value per member.
        """
        cached = self.paths_
        if cached is not None and np.array_equal(cached[0], readers) and np.array_equal(cached[1], others):
            return cached[2]
        values = self._paths_of(self.kind, readers, others, self.size)
        self.paths_ = (readers.copy(), others.copy(), values)
        return values

    @staticmethod
    def _paths_of(kind: str, readers: Indices, others: Indices, size: int) -> Array:
        """Return the betweenness or the closeness of every member.

        They are taken on the undirected graph of the active links.

        Args:
            kind:
                `'path-betweenness'` or `'path-closeness'`.

            readers:
                The first member of every active link.

            others:
                The second member of every active link.

            size:
                The number of members.

        Returns:
            values:
                One value per member, by NetworkX.

        Raises:
            ValidationError:
                The kind is unknown.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> readers, others = np.array([0, 1, 2]), np.array([1, 2, 3])
            >>> NeighborhoodKernel._paths_of('path-betweenness', readers, others, 5).tolist()
            [0.0, 2.0, 2.0, 0.0, 0.0]
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(size))
        graph.add_edges_from(zip(readers.tolist(), others.tolist(), strict=True))
        if kind == 'path-betweenness':
            found = nx.betweenness_centrality(graph, normalized=False)
        elif kind == 'path-closeness':
            found = nx.closeness_centrality(graph)
        else:
            error_msg = f'Unknown shortest paths reduction {kind!r}.'
            raise ValidationError(error_msg)
        return np.asarray([found[member] for member in range(size)], dtype=float)

    @staticmethod
    def _components_of(kind: str, readers: Indices, others: Indices, size: int) -> Array:
        """Return a giant component reduction of every member.

        The reduction is the label, the size, the largest size, or the membership of the giant component. The components
        are those of the undirected graph of the active links. The label of a component is its smallest member, and the
        giant component is the largest one, the one of the smallest label among components of equal size.

        Args:
            kind:
                One of `'component-label'`, `'component-size'`, `'component-largest'`, and `'component-giant'`.

            readers:
                The first member of every active link.

            others:
                The second member of every active link.

            size:
                The number of members.

        Returns:
            values:
                One value per member.

        Raises:
            ValidationError:
                The kind is unknown.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> readers, others = np.array([1, 3, 4]), np.array([2, 4, 5])
            >>> NeighborhoodKernel._components_of('component-label', readers, others, 6).tolist()
            [0.0, 1.0, 1.0, 3.0, 3.0, 3.0]
            >>> NeighborhoodKernel._components_of('component-size', readers, others, 6).tolist()
            [1.0, 2.0, 2.0, 3.0, 3.0, 3.0]
            >>> NeighborhoodKernel._components_of('component-giant', readers, others, 6).tolist()
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        """
        links = scipy.sparse.csr_array((np.ones(readers.size), (readers, others)), shape=(size, size))
        _, found = connected_components(links, directed=False)
        smallest = np.full(size, size, dtype=np.intp)
        np.minimum.at(smallest, found, np.arange(size, dtype=np.intp))
        labels = smallest[found]
        sizes = np.bincount(labels, minlength=size)[labels]
        if kind == 'component-label':
            return labels.astype(float)
        if kind == 'component-size':
            return sizes.astype(float)
        largest = sizes.max(initial=0)
        if kind == 'component-largest':
            return np.full(size, float(largest))
        if kind == 'component-giant':
            giant = labels[sizes == largest].min(initial=size)
            return (labels == giant).astype(float)
        error_msg = f'Unknown components reduction {kind!r}.'
        raise ValidationError(error_msg)

    @staticmethod
    def _reduce_pairs(kind: str, readers: Indices, weight: Array, argument: Array, counts: Indices) -> Array:
        """Return the reduction of the weighted values of the pairs of every member.

        A member without neighbours gets 0.

        Args:
            kind:
                One of `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, `'normalized-mean'`, and `'argmax'`.

            readers:
                The member of every pair, in increasing order.

            weight:
                The weight of every pair.

            argument:
                The value of every pair.

            counts:
                The number of neighbours of every member.

        Returns:
            values:
                One value per member.

        Raises:
            ValidationError:
                The kind is unknown.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> readers, counts = np.array([0, 0, 2]), np.array([2, 0, 1])
            >>> values = np.array([1.0, 3.0, 5.0])
            >>> NeighborhoodKernel._reduce_pairs('mean', readers, np.ones(3), values, counts).tolist()
            [2.0, 0.0, 5.0]
            >>> NeighborhoodKernel._reduce_pairs('argmax', readers, np.array([0.5, 2.0, 1.0]), values, counts).tolist()
            [3.0, 0.0, 5.0]
        """
        size = counts.size
        if kind == 'argmax':
            values = np.zeros(size)
            order = np.lexsort((-weight, readers))
            starts = np.cumsum(counts) - counts
            present = counts > 0
            values[present] = argument[order[starts[present]]]
            return values
        if kind in {'max', 'min'}:
            values = np.full(size, -np.inf if kind == 'max' else np.inf)
            (np.maximum if kind == 'max' else np.minimum).at(values, readers, weight * argument)
            return np.where(counts > 0, values, 0.0)
        if kind == 'count':
            return np.bincount(readers, weights=weight * (argument != 0), minlength=size).astype(float)
        total = np.bincount(readers, weights=weight * argument, minlength=size).astype(float)
        if kind == 'sum':
            return total
        if kind == 'mean':
            return np.divide(total, counts, out=np.zeros(size), where=counts > 0)
        if kind == 'normalized-mean':
            weights = np.bincount(readers, weights=weight, minlength=size).astype(float)
            return np.divide(total, weights, out=np.zeros(size), where=weights != 0)
        error_msg = f'Unknown neighbourhood reduction {kind!r}.'
        raise ValidationError(error_msg)

    @staticmethod
    def _network_rows(space: NetworkSpace, neighborhood: AdjacencyNeighborhood) -> tuple[Indices, Indices, Indices]:
        """Return the neighbours of every node of a layer in CSR form, with their edges.

        On a directed graph, the neighbours are the successors.

        Args:
            space:
                The network space.

            neighborhood:
                The graph neighbourhood of a layer, with the node itself when it is centred.

        Returns:
            rows:
                The row starts and the neighbours of every node, in node order, and the edge of the layer that joins
                every node to every neighbour, -1 for the node itself.

        Examples:
            >>> import networkx as nx
            >>> from skcomplex.core.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> from skcomplex.core.components.spaces import AdjacencyNeighborhood, NetworkSpace
            >>> space = NetworkSpace(nx.path_graph(3)).prepare()
            >>> [item.tolist() for item in NeighborhoodKernel._network_rows(space, AdjacencyNeighborhood())]
            [[0, 1, 3, 4], [1, 0, 2, 1], [0, 0, 1, 1]]
        """
        indptr, others, edges = space.adjacency_[neighborhood.layer]
        size = space.size_
        if not neighborhood.center:
            return indptr, others, edges
        readers = np.repeat(np.arange(size, dtype=np.intp), np.diff(indptr))
        keep = readers != others
        readers, others = MetricSearch._with_center(readers[keep], others[keep], size)
        edges = np.concatenate([edges[keep], np.full(size, -1, dtype=np.intp)])
        order = np.lexsort((others, readers))
        counts = np.bincount(readers, minlength=size)
        indptr = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(counts)]).astype(np.intp)
        return indptr, others[order].astype(np.intp), edges[order].astype(np.intp)

    @staticmethod
    def _lattice_rows(space: BaseSpace, neighborhood: LatticeNeighborhood | MemoryNeighborhood) -> Pairs:
        """Return the neighbours of every site of a space in CSR form.

        They come in neighbourhood order, without outside sites.

        Args:
            space:
                The prepared lattice or other space of sites.

            neighborhood:
                The neighbourhood of offsets, or the memory set of a group space.

        Returns:
            rows:
                The row starts and the neighbours of every site.

        Examples:
            >>> from skcomplex.core.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
            >>> space = LatticeSpace(3, boundary='fixed').prepare()
            >>> rows = NeighborhoodKernel._lattice_rows(space, LatticeNeighborhood(((-1,), (1,))))
            >>> [item.tolist() for item in rows]
            [[0, 1, 3, 4], [1, 0, 2, 1]]
        """
        table = space._index_table(neighborhood)
        inside = table < len(table)
        indptr = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(inside.sum(axis=1))]).astype(np.intp)
        return indptr, table[inside].astype(np.intp)

    @staticmethod
    def _default_neighborhood(
        space: object,
    ) -> AdjacencyNeighborhood | LatticeNeighborhood | MemoryNeighborhood | None:
        """Return the neighbourhood that a reduction without one reads.

        It is a network's graph, or a lattice's or group's own.

        Args:
            space:
                The prepared space of the group.

        Returns:
            neighbourhood:
                The neighbourhood, or `None` for a space without one of its own.

        Examples:
            >>> from skcomplex.core.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> from sage.all__sagemath_gap import libgap
            >>> from skcomplex.core.components.spaces import CayleySpace, LatticeSpace
            >>> lattice = LatticeSpace((4, 4)).prepare()
            >>> len(NeighborhoodKernel._default_neighborhood(lattice).prepare().offsets_)
            8
            >>> cayley = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=1).prepare()
            >>> NeighborhoodKernel._default_neighborhood(cayley).elements
            ('a', 'a^-1', 'b', 'b^-1')
        """
        if isinstance(space, NetworkSpace | LatticeSpace | CayleySpace):
            return space._default_neighborhood()
        return None

    @staticmethod
    def _position_state(states: Mapping[str, VariableSpec]) -> str | None:
        """Return the position state of an element type.

        It is the first vector state with the role position or site.

        Args:
            states:
                The state declarations by name.

        Returns:
            name:
                The name of the position state, or `None` without one.
        """
        return next((name for name, item in states.items() if item.is_place), None)

    @staticmethod
    def _cone_direction(cone: ConeNeighborhood, states: Mapping[str, VariableSpec], n_dim: int) -> str:
        """Return the state that holds the direction of a cone.

        It is the named state, else the derivative of the position.

        Args:
            cone:
                The cone neighbourhood.

            states:
                The state declarations of the element type of the members.

            n_dim:
                The dimension of the space.

        Returns:
            name:
                The name of the direction state, a vector state or in two dimensions a scalar angle.

        Raises:
            ValidationError:
                The members have no such state.

        Examples:
            >>> from typing import Annotated
            >>> from skcomplex.core.components.elements import Element
            >>> from skcomplex.core.specs import StateSpec
            >>> from skcomplex.core.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> from skcomplex.core.types import VariableKind
            >>> from skcomplex.core.engine._utils import declarations_of
            >>> from skcomplex.core.components.spaces import ConeNeighborhood
            >>> class BoidElement(Element):
            ...     def __init__(
            ...         self,
            ...         label,
            ...         x: Annotated[object, StateSpec(vector=True, role='position')] = 0.0,
            ...         theta: Annotated[float, StateSpec()] = 0.0,
            ...     ):
            ...         super().__init__(label)
            ...         self.x = x
            ...         self.theta = theta
            >>> states = declarations_of(BoidElement, VariableKind.STATE)
            >>> NeighborhoodKernel._cone_direction(ConeNeighborhood(1.0, 1.0, direction='theta'), states, 2)
            'theta'
        """
        if cone.direction is not None:
            declaration = states.get(cone.direction)
            if declaration is not None and (declaration.vector or n_dim == _PLANE):
                return cone.direction
            error_msg = (
                f'The cone reads the direction {cone.direction!r}, which is not a vector state of the members, or a '
                'scalar angle state in two dimensions.'
            )
            raise ValidationError(error_msg)
        position = NeighborhoodKernel._position_state(states)
        if position is None or states[position].order <= 1:
            error_msg = (
                'The cone has no direction: the position of the members has no first derivative. Declare the position '
                "of order 2, `state(vector=True, order=2, role='position')`, or name a direction state, such as "
                "`ConeNeighborhood(r, angle, direction='theta')`."
            )
            raise ValidationError(error_msg)
        return f'{position}_d1'

    @staticmethod
    def _check_rows(space: BaseSpace, rows: int) -> None:
        """Raise an error unless a group holds one member per node or site of its space.

        Args:
            space:
                The space of the group.

            rows:
                The number of members.

        Raises:
            ValidationError:
                The group does not hold one member per node or site.
        """
        size = getattr(space, 'size_', None)
        if rows != size:
            error_msg = (
                f'A reduction over the neighbours of {space!r} needs one member per node or site, {size}, but the '
                f'group has {rows} members.'
            )
            raise ValidationError(error_msg)

    @staticmethod
    def _neighbor_search(
        over: (
            AdjacencyNeighborhood
            | LatticeNeighborhood
            | MemoryNeighborhood
            | RadiusNeighborhood
            | NearestNeighborhood
            | ConeNeighborhood
        ),
        space: object,
        rows: int,
        entries: Mapping[str, EntrySpec],
        states: Mapping[str, VariableSpec],
        *,
        changing: Mapping[str | None, tuple[EntrySpec, EntrySpec, EntrySpec]] | None = None,
    ) -> NeighborSearch:
        """Return the search of the neighbours of the members of a group in its space.

        Args:
            over:
                The neighbourhood.

            space:
                The space of the group.

            rows:
                The number of members.

            entries:
                The layout entries of the group's variables by name.

            states:
                The state declarations of the element type of the members.

            changing:
                The source, target, and activity entries of the variable edge group of every layer of a network whose
                edges change, or `None` for fixed edges.

        Returns:
            search:
                The search, static on a fixed network or a lattice, by edge rows on a changing network, and by positions
                in a continuous space.

        Raises:
            ValidationError:
                The space does not offer the neighbourhood, the group does not hold one member per node or site, or the
                members lack a position or a direction.
        """
        offered = NeighborhoodKernel._spaces_of(over)
        if not isinstance(space, offered):
            names = ' or '.join(item.__name__ for item in offered)
            error_msg = f'The neighbourhood {over!r} needs a {names}, but the group lives on {space!r}.'
            raise ValidationError(error_msg)
        if isinstance(over, AdjacencyNeighborhood) and isinstance(space, NetworkSpace):
            if changing is not None and over.layer in changing:
                source, target, active = changing[over.layer]
                directed = space.graphs_[over.layer].is_directed()
                return EdgeRowSearch(source, target, active, rows, directed, center=over.center).prepare()
            NeighborhoodKernel._check_rows(space, rows)
            return StaticSearch(*NeighborhoodKernel._network_rows(space, over)).prepare()
        if isinstance(over, LatticeNeighborhood | MemoryNeighborhood) and isinstance(space, BaseSpace):
            NeighborhoodKernel._check_rows(space, rows)
            return StaticSearch(*NeighborhoodKernel._lattice_rows(space, over)).prepare()
        position = NeighborhoodKernel._position_state(states)
        if position is None or not isinstance(space, ContinuousSpace):
            error_msg = (
                f'A reduction over {over!r} measures distances between positions, but the members declare no vector '
                "position state. Declare one with `state(vector=True, role='position')`."
            )
            raise ValidationError(error_msg)
        continuous = space
        periods = continuous.extent_ if continuous.periodic_ else None
        direction = None
        if isinstance(over, ConeNeighborhood):
            name = NeighborhoodKernel._cone_direction(over, states, continuous.n_dim_)
            direction = DirectionSpec(entries[name], not states[name].vector) if name in entries else None
            if direction is None:
                error_msg = f'The cone reads the direction {name!r}, which the members do not hold.'
                raise ValidationError(error_msg)
        return MetricSearch(over, entries[position], periods, direction).prepare()

    @staticmethod
    def _spaces_of(neighborhood: BaseNeighborhood) -> tuple[type, ...]:
        """Return the classes of the spaces a neighbourhood is read on.

        Args:
            neighborhood:
                The neighbourhood.

        Returns:
            spaces:
                The space classes.

        Examples:
            >>> from skcomplex.core.components.spaces import RadiusNeighborhood
            >>> from skcomplex.core.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> NeighborhoodKernel._spaces_of(RadiusNeighborhood(1.0))
            (<class 'skcomplex.core.components.spaces._continuous.ContinuousSpace'>,)
        """
        return next(spaces for neighborhoods, spaces in _SPACES_OF if isinstance(neighborhood, neighborhoods))
