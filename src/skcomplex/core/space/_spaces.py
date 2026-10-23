"""Define the spaces that vector states live in."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from ..base import Requires, SpaceKind, ValidationError


class Space(BaseEstimator):
    """Base class of spaces, which give vector states and parameters their dimension.

    Subclasses expose the number of dimensions as `n_dim`, and name in `lacks` the options of their kind they do not
    offer, such as `'regular-geometry'` for the checkerboard schedule of a square lattice and `'image'` for image
    visuals.
    """

    space_kind: ClassVar[SpaceKind]
    lacks: ClassVar[frozenset[str]] = frozenset()

    n_dim: int


@dataclass(frozen=True)
class Adjacency:
    """Neighbourhood of a node of a graph: the nodes along its edges, in node order, after the node itself if included.

    Rules receive the values of the neighbours padded to the largest degree of the graph.

    Args:
        center:
            Whether the node itself comes first.

    Examples:
        >>> from skcomplex.core.space import Adjacency
        >>> Adjacency(center=True).center
        True
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.NETWORK})

    center: bool = False


class NetworkSpace(Space):
    """Network whose nodes agents occupy or lattice sites sit on, with the neighbours of a node given by its edges.

    Lattice groups on a network hold one member per node, in the order of `graph.nodes`.

    Args:
        graph:
            The NetworkX graph.

        capacity:
            The largest number of agents on a node, or `None` for no limit.

        fill:
            The value lattice rules read in the padding slots of nodes with fewer neighbours than the largest degree.

        hyperedges:
            The hyperedges of the network, each a sequence of at least two nodes, which interactions with the binding
            `'hyperedges'` and groups on hyperedges read.

    Raises:
        ValidationError:
            A hyperedge has fewer than two nodes or a node outside the graph.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.space import NetworkSpace
        >>> NetworkSpace(nx.cycle_graph(5)).graph.number_of_nodes()
        5
        >>> NetworkSpace(nx.complete_graph(3), hyperedges=[(0, 1, 2)]).hyperedges
        [(0, 1, 2)]
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.NETWORK

    n_dim = 1

    def __init__(
        self,
        graph: nx.Graph,
        *,
        capacity: int | None = None,
        fill: float = 0.0,
        hyperedges: Sequence[Sequence[object]] = (),
    ) -> None:
        self.graph = graph
        self.capacity = capacity
        self.fill = fill
        self.hyperedges = hyperedges
        nodes = set(graph.nodes)
        for hyperedge in hyperedges:
            if len(hyperedge) < 2 or not set(hyperedge) <= nodes:  # noqa: PLR2004
                error_msg = f'The hyperedge {tuple(hyperedge)} needs at least two nodes of the graph.'
                raise ValidationError(error_msg)

    def index_table(self, neighborhood: Adjacency) -> npt.NDArray[np.intp]:
        """Return the node index of every neighbour of every node, padded with the number of nodes.

        Args:
            neighborhood:
                The graph neighbourhood.

        Returns:
            table:
                One row per node and one column per neighbour slot, as wide as the largest degree, plus one column
                for the node itself when the neighbourhood includes it.

        Raises:
            ValidationError:
                The neighbourhood is not a graph neighbourhood.

        Examples:
            >>> import networkx as nx
            >>> from skcomplex.core.space import Adjacency, NetworkSpace
            >>> NetworkSpace(nx.star_graph(2)).index_table(Adjacency()).tolist()
            [[1, 2], [0, 3], [0, 3]]
        """
        if not isinstance(neighborhood, Adjacency):
            error_msg = f'A network reads the neighbourhood Adjacency(), but it was given {neighborhood!r}.'
            raise ValidationError(error_msg)
        size = self.graph.number_of_nodes()
        matrix = nx.to_scipy_sparse_array(self.graph, nodelist=list(self.graph.nodes), format='csr')
        matrix.sort_indices()
        degrees = np.diff(matrix.indptr)
        width = int(degrees.max(initial=0))
        table = np.full((size, width), size, dtype=np.intp)
        slots = np.arange(matrix.indices.size) - np.repeat(matrix.indptr[:-1], degrees)
        table[np.repeat(np.arange(size), degrees), slots] = matrix.indices
        if neighborhood.center:
            return np.column_stack([np.arange(size, dtype=np.intp), table])
        return table


BOUNDARIES = ('open', 'periodic', 'reflecting')


class ContinuousSpace(Space):
    """Continuous space of real positions, a box with a boundary or the whole of `R^d`.

    A box has an extent along every dimension, from 0 to the extent. Its boundary acts on every position state of the
    components on it: `'open'` leaves positions alone, `'periodic'` joins opposite sides and measures pair distances by
    the minimum image, and `'reflecting'` mirrors positions back into the box and reverses the velocity across the wall.
    Without an extent the space is unbounded and has `n_dim` dimensions. Agents on a two-dimensional box move in it and
    find neighbours within a radius.

    Args:
        *sides:
            The side of the box along every dimension, or none for an unbounded space.

        extent:
            The sides as one sequence, instead of `sides`.

        n_dim:
            The number of dimensions of an unbounded space, or `None` for the number of sides of the box.

        boundary:
            `'open'`, `'periodic'`, or `'reflecting'`.

    Raises:
        ValidationError:
            The boundary is unknown, the space has neither an extent nor a dimension, a side is not positive, or an
            unbounded space has a boundary other than open.

    Examples:
        >>> from skcomplex.core.space import ContinuousSpace
        >>> ContinuousSpace(10.0, 5.0, boundary='periodic').height
        5.0
        >>> ContinuousSpace(n_dim=3).n_dim
        3
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.CONTINUOUS

    def __init__(
        self,
        *sides: float,
        extent: Sequence[float] = (),
        n_dim: int | None = None,
        boundary: str = 'open',
    ) -> None:
        given = tuple(float(side) for side in (sides or extent))
        self.extent = extent if not sides and isinstance(extent, tuple) and extent == given else given
        self.n_dim = n_dim if n_dim is not None else len(self.extent)
        self.boundary = boundary
        if boundary not in BOUNDARIES:
            error_msg = f'Unknown boundary {boundary!r}. Use one of {", ".join(BOUNDARIES)}.'
            raise ValidationError(error_msg)
        if not self.n_dim:
            error_msg = 'A continuous space needs the sides of its box or a number of dimensions.'
            raise ValidationError(error_msg)
        if self.extent and (len(self.extent) != self.n_dim or min(self.extent) <= 0):
            error_msg = f'The box {self.extent} needs one positive side for each of its {self.n_dim} dimensions.'
            raise ValidationError(error_msg)
        if not self.extent and boundary != 'open':
            error_msg = f'An unbounded space has no walls for its boundary {boundary!r}. Give the sides of a box.'
            raise ValidationError(error_msg)

    @property
    def width(self) -> float:
        """Return the side of the box along the first dimension."""
        return self.extent[0]

    @property
    def height(self) -> float:
        """Return the side of the box along the second dimension."""
        return self.extent[1]

    @property
    def torus(self) -> bool:
        """Check whether opposite sides of the box are joined."""
        return self.boundary == 'periodic'

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Return the extent, the dimension, and the boundary, so the space clones."""
        return {'extent': self.extent, 'n_dim': self.n_dim, 'boundary': self.boundary}

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the extent or the dimension, and the boundary."""
        sides = ', '.join(map(repr, self.extent)) if self.extent else f'n_dim={self.n_dim}'
        boundary = '' if self.boundary == 'open' else f', boundary={self.boundary!r}'
        return f'ContinuousSpace({sides}{boundary})'
