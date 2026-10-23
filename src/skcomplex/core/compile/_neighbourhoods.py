"""Find the neighbours of the members of a group and evaluate the reductions of laws over them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ..base import Expression, ValidationError, Variable
from ..space import Adjacency, Cone, ContinuousSpace, LatticeSpace, Nearest, Neighborhood, NetworkSpace, Radius
from ._layout import Entry
from ._printing import numpy_function
from ._reductions import Read

type Vector = npt.NDArray[np.float64]
type Indices = npt.NDArray[np.intp]
type Rows = tuple[Indices, Indices]
type Found = tuple[Indices, Indices, Indices | None]

SIDES = ('shared', 'reader', 'other', 'edge')
ACTIVE_THRESHOLD = 0.5


def _rows(readers: Indices, others: Indices, size: int) -> Rows:
    """Return the row starts and the neighbours, sorted by reader and then by neighbour, of a list of pairs."""
    order = np.lexsort((others, readers))
    counts = np.bincount(readers, minlength=size)
    indptr = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(counts)]).astype(np.intp)
    return indptr, others[order].astype(np.intp)


def _with_center(readers: Indices, others: Indices, size: int) -> tuple[Indices, Indices]:
    """Return pairs with every member paired with itself."""
    own = np.arange(size, dtype=np.intp)
    return np.concatenate([readers, own]), np.concatenate([others, own])


def network_rows(space: NetworkSpace, neighborhood: Adjacency) -> tuple[Indices, Indices, Indices]:
    """Return the neighbours of every node of a layer in CSR form, successors on a directed graph, with their edges.

    Args:
        space:
            The network space.

        neighborhood:
            The graph neighbourhood of a layer, with the node itself when it is centred.

    Returns:
        rows:
            The row starts and the neighbours of every node, in node order, and the edge of the layer that joins every
            node to every neighbour, -1 for the node itself.

    Examples:
        >>>         >>> from skcomplex.core.compile._neighbourhoods import network_rows
        >>> from skcomplex.core.space import Adjacency, NetworkSpace
        >>> [item.tolist() for item in network_rows(NetworkSpace(nx.path_graph(3)), Adjacency())]
        [[0, 1, 3, 4], [1, 0, 2, 1], [0, 0, 1, 1]]
    """
    indptr, others, edges = space.compute_adjacency(neighborhood.layer)
    size = space.size
    if not neighborhood.center:
        return indptr, others, edges
    readers = np.repeat(np.arange(size, dtype=np.intp), np.diff(indptr))
    keep = readers != others
    readers, others = _with_center(readers[keep], others[keep], size)
    edges = np.concatenate([edges[keep], np.full(size, -1, dtype=np.intp)])
    order = np.lexsort((others, readers))
    counts = np.bincount(readers, minlength=size)
    indptr = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(counts)]).astype(np.intp)
    return indptr, others[order].astype(np.intp), edges[order].astype(np.intp)


def lattice_rows(space: LatticeSpace, neighborhood: Neighborhood) -> Rows:
    """Return the neighbours of every site of a lattice in CSR form, in neighbourhood order, without outside sites.

    Args:
        space:
            The lattice space.

        neighborhood:
            The neighbourhood of offsets.

    Returns:
        rows:
            The row starts and the neighbours of every site.

    Examples:
        >>> from skcomplex.core.compile._neighbourhoods import lattice_rows
        >>> from skcomplex.core.space import LatticeSpace, Neighborhood
        >>> rows = lattice_rows(LatticeSpace(3, boundary='fixed'), Neighborhood(((-1,), (1,))))
        >>> [item.tolist() for item in rows]
        [[0, 1, 3, 4], [1, 0, 2, 1]]
    """
    table = space.index_table(neighborhood)
    inside = table < space.size
    indptr = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(inside.sum(axis=1))]).astype(np.intp)
    return indptr, table[inside].astype(np.intp)


def default_neighbourhood(space: object) -> Adjacency | Neighborhood | None:
    """Return the neighbourhood a reduction without one reads: the graph of a network or the lattice's own.

    Args:
        space:
            The space of the group.

    Returns:
        neighbourhood:
            The neighbourhood, or `None` for a space without one of its own.

    Examples:
        >>> from skcomplex.core.compile._neighbourhoods import default_neighbourhood
        >>> from skcomplex.core.space import LatticeSpace
        >>> len(default_neighbourhood(LatticeSpace((4, 4))))
        8
    """
    if isinstance(space, NetworkSpace):
        return Adjacency()
    if not isinstance(space, LatticeSpace):
        return None
    if space.geometry == 'triangular':
        return Neighborhood.triangular()
    if space.neighborhood == 'von_neumann':
        return Neighborhood.von_neumann(n_dim=space.n_dim)
    return Neighborhood.moore(n_dim=space.n_dim)


@dataclass(frozen=True)
class StaticSearch:
    """Neighbours that do not change during a run, the edges of a network or the offsets of a lattice.

    Args:
        indptr:
            The start of the neighbours of every member, and the number of pairs last.

        indices:
            The neighbours of every member in turn.

        edges:
            The edge of a network layer that joins every member to every neighbour, or `None` on a lattice.
    """

    indptr: Indices
    indices: Indices
    edges: Indices | None = None

    def find(self, x: Vector, q: Vector) -> Found:
        """Return the neighbours of every member.

        Args:
            x:
                The state vector, which the neighbours do not depend on.

            q:
                The extended parameter vector, which the neighbours do not depend on.

        Returns:
            rows:
                The row starts, the neighbours, and the edge of every neighbour on a network.
        """
        del x, q
        return self.indptr, self.indices, self.edges


@dataclass(frozen=True)
class EdgeRows:
    """Neighbours of the nodes of a network layer whose edges change, read from the rows of its variable edge group.

    Args:
        source:
            The parameter entry of the source node of every edge row.

        target:
            The parameter entry of the target node of every edge row.

        active:
            The parameter entry of the activity of every edge row.

        size:
            The number of node rows, the capacity of the groups on the nodes.

        directed:
            Whether the edges are directed, so a node reads its successors.

        center:
            Whether a node is its own neighbour.
    """

    source: Entry
    target: Entry
    active: Entry
    size: int
    directed: bool = False
    center: bool = False

    def find(self, x: Vector, q: Vector) -> Found:
        """Return the neighbours of every node along the active edges.

        Args:
            x:
                The state vector, which the neighbours do not depend on.

            q:
                The extended parameter vector, which holds the edge rows.

        Returns:
            rows:
                The row starts, the neighbours, and the edge row of every neighbour, -1 for the node itself.
        """
        del x
        keep = self.active.view(q)[:, 0] > ACTIVE_THRESHOLD
        edges = np.flatnonzero(keep).astype(np.intp)
        sources = self.source.view(q)[keep, 0].astype(np.intp)
        targets = self.target.view(q)[keep, 0].astype(np.intp)
        readers = sources if self.directed else np.concatenate([sources, targets])
        others = targets if self.directed else np.concatenate([targets, sources])
        ids = edges if self.directed else np.concatenate([edges, edges])
        if self.center:
            own = np.arange(self.size, dtype=np.intp)
            readers, others = np.concatenate([readers, own]), np.concatenate([others, own])
            ids = np.concatenate([ids, np.full(self.size, -1, dtype=np.intp)])
        order = np.argsort(readers * self.size + others, kind='stable')
        counts = np.bincount(readers, minlength=self.size)
        indptr = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(counts)]).astype(np.intp)
        return indptr, others[order].astype(np.intp), ids[order].astype(np.intp)


@dataclass(frozen=True)
class Heading:
    """State that holds the heading of the members, a vector or a heading angle in two dimensions.

    Args:
        entry:
            The entry of the state.

        angle:
            Whether the state is a scalar angle, whose heading is `(cos, sin)`.
    """

    entry: Entry
    angle: bool

    def vectors(self, x: Vector) -> Vector:
        """Return the heading of every member as one row of components.

        Args:
            x:
                The state vector.

        Returns:
            headings:
                One row per member.
        """
        values = self.entry.view(x)
        if self.angle:
            return np.column_stack([np.cos(values[:, 0]), np.sin(values[:, 0])])
        return values


@dataclass(frozen=True)
class MetricSearch:
    """Neighbours in a continuous space found from the positions at every evaluation with a k-d tree.

    Args:
        neighbourhood:
            The radius, nearest, or cone neighbourhood.

        positions:
            The entry of the position state of the members.

        periods:
            The sides of a periodic box, or `None` for distances without minimum images.

        heading:
            The heading state of a cone, or `None`.
    """

    neighbourhood: Radius | Nearest | Cone
    positions: Entry
    periods: tuple[float, ...] | None = None
    heading: Heading | None = None

    def wrapped(self, x: Vector) -> Vector:
        """Return the positions of the members, wrapped into a periodic box.

        Args:
            x:
                The state vector.

        Returns:
            positions:
                One row of position components per member.
        """
        positions = self.positions.view(x)
        if self.periods is None:
            return positions
        periods = np.asarray(self.periods)
        wrapped = np.mod(positions, periods)
        return np.where(wrapped >= periods, wrapped - periods, wrapped)

    def differences(self, x: Vector, readers: Indices, others: Indices) -> Vector:
        """Return the displacement from every reader to its neighbour, the minimum image in a periodic box.

        Args:
            x:
                The state vector.

            readers:
                The reader of every pair.

            others:
                The neighbour of every pair.

        Returns:
            differences:
                One row of components per pair, `x_reader - x_other` wrapped.
        """
        positions = self.positions.view(x)
        difference = positions[readers] - positions[others]
        if self.periods is None:
            return difference
        periods = np.asarray(self.periods)
        return difference - periods * np.rint(difference / periods)

    def find(self, x: Vector, q: Vector) -> Found:
        """Return the neighbours of every member at the current positions.

        Args:
            x:
                The state vector.

            q:
                The extended parameter vector, which the neighbours do not depend on.

        Returns:
            rows:
                The row starts and the neighbours, sorted by neighbour within every row, without edges.
        """
        del q
        positions = self.wrapped(x)
        size = positions.shape[0]
        tree = cKDTree(positions, boxsize=np.asarray(self.periods) if self.periods is not None else None)
        neighbourhood = self.neighbourhood
        if isinstance(neighbourhood, Nearest):
            readers, others = self._nearest(tree, positions, neighbourhood.k)
        else:
            pairs = tree.query_pairs(neighbourhood.radius, output_type='ndarray').astype(np.intp)
            readers = np.concatenate([pairs[:, 0], pairs[:, 1]])
            others = np.concatenate([pairs[:, 1], pairs[:, 0]])
            if isinstance(neighbourhood, Cone):
                keep = self._inside(x, readers, others, neighbourhood)
                readers, others = readers[keep], others[keep]
        if neighbourhood.center:
            readers, others = _with_center(readers, others, size)
        return (*_rows(readers, others, size), None)

    @staticmethod
    def _nearest(tree: cKDTree, positions: Vector, k: int) -> tuple[Indices, Indices]:
        """Return the pairs of every member with its `k` nearest other members."""
        size = positions.shape[0]
        k = min(k, size - 1)
        if k < 1:
            return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
        _, found = tree.query(positions, k=k + 1)
        found = np.asarray(found, dtype=np.intp).reshape(size, k + 1)
        own = found == np.arange(size)[:, None]
        own[~own.any(axis=1), -1] = True
        return np.repeat(np.arange(size, dtype=np.intp), k), found[~own].astype(np.intp)

    def _inside(self, x: Vector, readers: Indices, others: Indices, cone: Cone) -> npt.NDArray[np.bool_]:
        """Return whether every pair has its neighbour inside the reader's cone."""
        assert self.heading is not None
        direction = -self.differences(x, readers, others)
        heading = self.heading.vectors(x)[readers]
        dot = np.sum(heading * direction, axis=1)
        norms = np.sqrt(np.sum(heading * heading, axis=1)) * np.sqrt(np.sum(direction * direction, axis=1))
        return dot >= np.cos(cone.angle) * norms


type Search = StaticSearch | EdgeRows | MetricSearch


def position_state(states: Mapping[str, Variable]) -> str | None:
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


def cone_heading(cone: Cone, states: Mapping[str, Variable], n_dim: int) -> str:
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
        >>> from skcomplex.core.base import state
        >>> from skcomplex.core.components import Element
        >>> from skcomplex.core.compile._neighbourhoods import cone_heading
        >>> from skcomplex.core.base import VariableKind, declarations_of
        >>> from skcomplex.core.space import Cone
        >>> Boid = Element.define('Boid', states={'x': state(vector=True, position=True), 'theta': 0.0})
        >>> cone_heading(Cone(1.0, 1.0, heading='theta'), declarations_of(Boid, VariableKind.STATE), 2)
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
    position = position_state(states)
    velocity = next(
        (name for name, item in states.items() if item.vector and position and item.velocity_of == position),
        None,
    )
    if velocity is None and position is not None and states[position].order > 1:
        velocity = f'{position}_d1'
    if velocity is None:
        error_msg = (
            'The cone has no heading: the members declare no vector velocity of their position. Declare one with '
            "`state(vector=True, velocity_of='x')`, or name a heading state, such as `Cone(r, angle, heading='theta')`."
        )
        raise ValidationError(error_msg)
    return velocity


def neighbour_search(
    over: Adjacency | Neighborhood | Radius | Nearest | Cone,
    space: object,
    rows: int,
    entries: Mapping[str, Entry],
    states: Mapping[str, Variable],
    *,
    changing: Mapping[str | None, tuple[Entry, Entry, Entry]] | None = None,
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
    kind = getattr(space, 'space_kind', None)
    if kind is None or not over.requires.accepts('space', kind):
        offered = ', '.join(sorted(str(item) for item in over.requires.space))
        error_msg = f'The neighbourhood {over!r} needs a {offered} space, but the group lives on {space!r}.'
        raise ValidationError(error_msg)
    if isinstance(over, Adjacency) and changing is not None and over.layer in changing:
        source, target, active = changing[over.layer]
        directed = cast('NetworkSpace', space).resolve_graph(over.layer).is_directed()
        return EdgeRows(source, target, active, rows, directed, over.center)
    if isinstance(over, Adjacency | Neighborhood):
        size = space.graph.number_of_nodes() if isinstance(space, NetworkSpace) else space.size  # type: ignore[attr-defined]
        if rows != size:
            error_msg = (
                f'A reduction over the neighbours of {space!r} needs one member per node or site, {size}, but the '
                f'group has {rows} members.'
            )
            raise ValidationError(error_msg)
        if isinstance(over, Adjacency):
            return StaticSearch(*network_rows(space, over))  # type: ignore[arg-type]
        return StaticSearch(*lattice_rows(space, over))  # type: ignore[arg-type]
    position = position_state(states)
    if position is None:
        error_msg = (
            f'A reduction over {over!r} measures distances between positions, but the members declare no vector '
            "position state. Declare one with `state(vector=True, position=True)`."
        )
        raise ValidationError(error_msg)
    continuous: ContinuousSpace = space  # type: ignore[assignment]
    periods = tuple(continuous.extent) if continuous.boundary == 'periodic' else None
    heading = None
    if isinstance(over, Cone):
        name = cone_heading(over, states, continuous.n_dim)
        heading = Heading(entries[name], not states[name].vector) if name in entries else None
        if heading is None:
            error_msg = f'The cone reads the heading {name!r}, which the members do not hold.'
            raise ValidationError(error_msg)
    return MetricSearch(over, entries[position], periods, heading)


@dataclass(frozen=True)
class PairRead:
    """How a neighbourhood reduction reads one symbol for every pair of a member and a neighbour.

    Args:
        read:
            The values of the symbol, one per member or one shared value.

        side:
            `'shared'` for one value, `'reader'` for the value of the member that reads the reduction, `'other'`
            for the value of the neighbour, or `'edge'` for the value of the member of a group on the edges of a
            network layer that joins them.

        image:
            Whether the symbol is a position component of the neighbour, read by its minimum image from the reader.
    """

    read: Read
    side: str
    image: bool = False


@dataclass
class CompiledNeighbourhood:
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

    symbol: Expression
    kind: str
    argument: Expression
    weight: Expression
    entry: Entry
    group: str
    search: Search
    reads: dict[Expression, PairRead]
    time: Expression
    _function: Callable[..., list[Any]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Compile the argument and the weight into one NumPy function."""
        self._symbols = sorted(self.reads, key=str)
        self._function = numpy_function(self.time, self._symbols, [self.argument, self.weight])

    @property
    def symbols(self) -> list[Expression]:
        """Symbols the argument and the weight read, in the order the compiled function takes them after time."""
        return list(self._symbols)

    @property
    def size(self) -> int:
        """Number of members."""
        return self.entry.rows

    def _gather(self, x: Vector, q: Vector, readers: Indices, others: Indices, edges: Indices | None) -> list[Any]:
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

    def evaluate(self, t: float, x: Vector, q: Vector) -> None:
        """Write the reduction of every member into the extended parameter vector.

        Args:
            t:
                The time.

            x:
                The state vector.

            q:
                The extended parameter vector, which holds the parameters and the reductions before this one.
        """
        indptr, others, edges = self.search.find(x, q)
        size = self.size
        counts = np.diff(indptr)
        readers = np.repeat(np.arange(size, dtype=np.intp), counts)
        argument, weight = self._function(t, *self._gather(x, q, readers, others, edges))  # type: ignore[misc]
        pairs = others.size
        argument = np.broadcast_to(np.asarray(argument, dtype=float), (pairs,))
        weight = np.broadcast_to(np.asarray(weight, dtype=float), (pairs,))
        q[self.entry.indices] = reduce_pairs(self.kind, readers, weight, argument, counts)


def reduce_pairs(kind: str, readers: Indices, weight: Vector, argument: Vector, counts: Indices) -> Vector:
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
        >>> from skcomplex.core.compile._neighbourhoods import reduce_pairs
        >>> readers, counts = np.array([0, 0, 2]), np.array([2, 0, 1])
        >>> reduce_pairs('mean', readers, np.ones(3), np.array([1.0, 3.0, 5.0]), counts).tolist()
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
