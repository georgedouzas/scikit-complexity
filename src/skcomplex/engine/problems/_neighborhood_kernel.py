"""Neighbourhood reduction that laws read, one value per member, compiled into a function of its symbols."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...core.spaces import (
    AdjacencyNeighborhood,
    BaseNeighborhood,
    BaseSpace,
    CayleySpace,
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
from ...specs import EntrySpec, HeadingSpec, PairReadSpec, RunStateSpec, VariableSpec
from ...types import Array, Indices, Pairs
from ._base import BaseKernel
from ._edge_row_search import EdgeRowSearch
from ._expression_printer import ExpressionPrinter
from ._metric_search import MetricSearch
from ._static_search import StaticSearch

_SPACES_OF: tuple[tuple[type | tuple[type, ...], tuple[type, ...]], ...] = (
    (AdjacencyNeighborhood, (NetworkSpace,)),
    ((LatticeNeighborhood, MemoryNeighborhood), (LatticeSpace, CayleySpace)),
    ((RadiusNeighborhood, NearestNeighborhood, ConeNeighborhood), (ContinuousSpace,)),
)
"""Spaces every kind of neighbourhood is read on."""


type Search = StaticSearch | EdgeRowSearch | MetricSearch


class NeighborhoodKernel(BaseKernel):
    """Neighbourhood reduction that laws read, one value per member, compiled into a function of its symbols.

    Args:
        symbol:
            The placeholder symbol that kernels read instead of the reduction.

        kind:
            One of `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, and `'normalized-mean'`.

        argument:
            The value of one neighbour, with nested reductions replaced by their placeholders.

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
    """

    def __init__(  # noqa: PLR0917
        self,
        symbol: Expression,
        kind: str,
        argument: Expression,
        weight: Expression,
        entry: EntrySpec,
        group: str,
        search: Search,
        reads: dict[Expression, PairReadSpec],
        time: Expression,
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

    @functools.cached_property
    def _symbols(self) -> list[Expression]:
        """Symbols the argument and the weight read, sorted by name."""
        return sorted(self.reads, key=str)

    @functools.cached_property
    def _function(self) -> Callable[..., list[Any]]:
        """NumPy function of the argument and the weight, compiled once."""
        printer = ExpressionPrinter(time=self.time, compiled=True).check().prepare(self._symbols)
        return printer.run([self.argument, self.weight])

    @property
    def symbols(self) -> list[Expression]:
        """Symbols the argument and the weight read, in the order the compiled function takes them after time."""
        return list(self._symbols)

    @property
    def size(self) -> int:
        """Number of members."""
        return self.entry.rows

    def _gather(self, x: Array, q: Array, readers: Indices, others: Indices, edges: Indices | None) -> list[Any]:
        """Return the values of every symbol for every pair."""
        values = []
        differences = None
        for item in self._symbols:
            pair = self.reads[item]
            read = pair.read
            column = (x if read.is_state else q)[read.index]
            if pair.side == 'shared':
                values.append(column)
            elif pair.side == 'reader':
                values.append(column[readers])
            elif pair.side == 'edge':
                values.append(column[edges])
            elif pair.image:
                if differences is None:
                    differences = self.search.differences(x, readers, others)  # type: ignore[union-attr]
                values.append(column[readers] - differences[:, read.column])
            else:
                values.append(column[others])
        return values

    def _run(self, inputs: RunStateSpec) -> None:
        """Write the reduction of every member into the extended parameter vector of the state, the `p` of `inputs`."""
        t, x, q = inputs.t, inputs.x, inputs.p
        indptr, others, edges = self.search.run(inputs)
        size = self.size
        counts = np.diff(indptr)
        readers = np.repeat(np.arange(size, dtype=np.intp), counts)
        argument, weight = self._function(t, *self._gather(x, q, readers, others, edges))
        pairs = others.size
        argument = np.broadcast_to(np.asarray(argument, dtype=float), (pairs,))
        weight = np.broadcast_to(np.asarray(weight, dtype=float), (pairs,))
        q[self.entry.indices] = self._reduce_pairs(self.kind, readers, weight, argument, counts)

    @staticmethod
    def _reduce_pairs(kind: str, readers: Indices, weight: Array, argument: Array, counts: Indices) -> Array:
        """Return the reduction of the weighted values of the pairs of every member, 0 for a member without neighbours.

        Args:
            kind:
                One of `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, and `'normalized-mean'`.

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
            >>> from skcomplex.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> readers, counts = np.array([0, 0, 2]), np.array([2, 0, 1])
            >>> values = np.array([1.0, 3.0, 5.0])
            >>> NeighborhoodKernel._reduce_pairs('mean', readers, np.ones(3), values, counts).tolist()
            [2.0, 0.0, 5.0]
        """
        size = counts.size
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
        """Return the neighbours of every node of a layer in CSR form, successors on a directed graph, with their edges.

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
            >>> from skcomplex.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> from skcomplex.core.spaces import AdjacencyNeighborhood, NetworkSpace
            >>> space = NetworkSpace(nx.path_graph(3)).check().prepare()
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
        """Return the neighbours of every site of a space in CSR form, in neighbourhood order, without outside sites.

        Args:
            space:
                The prepared lattice or other space of sites.

            neighborhood:
                The neighbourhood of offsets, or the memory set of a group space.

        Returns:
            rows:
                The row starts and the neighbours of every site.

        Examples:
            >>> from skcomplex.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> from skcomplex.core.spaces import LatticeNeighborhood, LatticeSpace
            >>> space = LatticeSpace(3, boundary='fixed').check().prepare()
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
        """Return the neighbourhood a reduction without one reads: a network's graph, or a lattice's or group's own.

        Args:
            space:
                The prepared space of the group.

        Returns:
            neighbourhood:
                The neighbourhood, or `None` for a space without one of its own.

        Examples:
            >>> from skcomplex.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> from sage.all__sagemath_gap import libgap
            >>> from skcomplex.core.spaces import CayleySpace, LatticeSpace
            >>> lattice = LatticeSpace((4, 4)).check().prepare()
            >>> len(NeighborhoodKernel._default_neighborhood(lattice).check().prepare().offsets_)
            8
            >>> cayley = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=1).check().prepare()
            >>> NeighborhoodKernel._default_neighborhood(cayley).elements
            ('a', 'a^-1', 'b', 'b^-1')
        """
        if isinstance(space, NetworkSpace | LatticeSpace | CayleySpace):
            return space._default_neighborhood()
        return None

    @staticmethod
    def _position_state(states: Mapping[str, VariableSpec]) -> str | None:
        """Return the position state of an element type: the vector state declared as position, else the vector `x`.

        Args:
            states:
                The state declarations by name.

        Returns:
            name:
                The name of the position state, or `None` without one.
        """
        declared = next((name for name, item in states.items() if item.position and item.vector), None)
        if declared is not None:
            return declared
        found = states.get('x')
        return 'x' if found is not None and found.vector else None

    @staticmethod
    def _cone_heading(cone: ConeNeighborhood, states: Mapping[str, VariableSpec], n_dim: int) -> str:
        """Return the state that holds the heading of a cone: the named state, else the velocity of the position.

        Args:
            cone:
                The cone neighbourhood.

            states:
                The state declarations of the element type of the members.

            n_dim:
                The dimension of the space.

        Returns:
            name:
                The name of the heading state, a vector state or in two dimensions a scalar angle.

        Raises:
            ValidationError:
                The members have no such state.

        Examples:
            >>> from typing import Annotated
            >>> from skcomplex.core.elements import Element
            >>> from skcomplex.specs import StateSpec
            >>> from skcomplex.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> from skcomplex.types import VariableKind
            >>> from skcomplex.engine._utils import declarations_of
            >>> from skcomplex.core.spaces import ConeNeighborhood
            >>> class BoidElement(Element):
            ...     def __init__(
            ...         self,
            ...         label,
            ...         x: Annotated[object, StateSpec(vector=True, position=True)] = 0.0,
            ...         theta: Annotated[float, StateSpec()] = 0.0,
            ...     ):
            ...         super().__init__(label)
            ...         self.x = x
            ...         self.theta = theta
            >>> states = declarations_of(BoidElement, VariableKind.STATE)
            >>> NeighborhoodKernel._cone_heading(ConeNeighborhood(1.0, 1.0, heading='theta'), states, 2)
            'theta'
        """
        if cone.heading is not None:
            declaration = states.get(cone.heading)
            if declaration is not None and (declaration.vector or n_dim == 2):  # noqa: PLR2004
                return cone.heading
            error_msg = (
                f'The cone reads the heading {cone.heading!r}, which is not a vector state of the members, or a scalar '
                'angle state in two dimensions.'
            )
            raise ValidationError(error_msg)
        position = NeighborhoodKernel._position_state(states)
        velocity = next(
            (name for name, item in states.items() if item.vector and position and item.velocity_of == position),
            None,
        )
        if velocity is None and position is not None and states[position].order > 1:
            velocity = f'{position}_d1'
        if velocity is None:
            error_msg = (
                'The cone has no heading: the members declare no vector velocity of their position. Declare one with '
                "`state(vector=True, velocity_of='x')`, or name a heading state, such as "
                "`Cone(r, angle, heading='theta')`."
            )
            raise ValidationError(error_msg)
        return velocity

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
    ) -> Search:
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
                members lack a position or a heading.
        """
        offered = NeighborhoodKernel._spaces_of(over)
        if not isinstance(space, offered):
            names = ' or '.join(item.__name__ for item in offered)
            error_msg = f'The neighbourhood {over!r} needs a {names}, but the group lives on {space!r}.'
            raise ValidationError(error_msg)
        if isinstance(over, AdjacencyNeighborhood) and changing is not None and over.layer in changing:
            source, target, active = changing[over.layer]
            directed = cast('NetworkSpace', space).graphs_[over.layer].is_directed()
            return EdgeRowSearch(source, target, active, rows, directed, over.center).check()
        if isinstance(over, AdjacencyNeighborhood | LatticeNeighborhood | MemoryNeighborhood):
            size = getattr(space, 'size_', None)
            if rows != size:
                error_msg = (
                    f'A reduction over the neighbours of {space!r} needs one member per node or site, {size}, but the '
                    f'group has {rows} members.'
                )
                raise ValidationError(error_msg)
            if isinstance(over, AdjacencyNeighborhood):
                return StaticSearch(*NeighborhoodKernel._network_rows(space, over)).check()  # type: ignore[arg-type]
            return StaticSearch(*NeighborhoodKernel._lattice_rows(space, over)).check()  # type: ignore[arg-type]
        position = NeighborhoodKernel._position_state(states)
        if position is None:
            error_msg = (
                f'A reduction over {over!r} measures distances between positions, but the members declare no vector '
                "position state. Declare one with `state(vector=True, position=True)`."
            )
            raise ValidationError(error_msg)
        continuous: ContinuousSpace = space  # type: ignore[assignment]
        periods = continuous.extent_ if continuous.periodic_ else None
        heading = None
        if isinstance(over, ConeNeighborhood):
            name = NeighborhoodKernel._cone_heading(over, states, continuous.n_dim_)
            heading = HeadingSpec(entries[name], not states[name].vector) if name in entries else None
            if heading is None:
                error_msg = f'The cone reads the heading {name!r}, which the members do not hold.'
                raise ValidationError(error_msg)
        return MetricSearch(over, entries[position], periods, heading).check()

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
            >>> from skcomplex.core.spaces import RadiusNeighborhood
            >>> from skcomplex.engine.problems._neighborhood_kernel import NeighborhoodKernel
            >>> NeighborhoodKernel._spaces_of(RadiusNeighborhood(1.0))
            (<class 'skcomplex.core.spaces._continuous.ContinuousSpace'>,)
        """
        return next(spaces for neighborhoods, spaces in _SPACES_OF if isinstance(neighborhood, neighborhoods))
