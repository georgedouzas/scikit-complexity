"""Define the spaces that vector states live in."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from ..base import SpaceKind, ValidationError


class Space(BaseEstimator):
    """Base class of spaces, which give vector states and parameters their dimension.

    Subclasses expose the number of dimensions as `n_dim`.
    """

    space_kind: ClassVar[SpaceKind]

    n_dim: int


class GridSpace(Space):
    """Grid of cells that agents occupy, with Moore or von Neumann neighbourhoods.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        neighborhood:
            `'moore'` for the 8 surrounding cells or `'von_neumann'` for the 4 adjacent ones.

        torus:
            Whether opposite edges are joined.

        capacity:
            The largest number of agents in a cell, or `None` for no limit.

    Examples:
        >>> from skcomplex.core.model import GridSpace
        >>> GridSpace(20, 20, capacity=1).n_dim
        2
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.LATTICE

    n_dim = 2

    def __init__(
        self,
        width: int,
        height: int,
        *,
        neighborhood: str = 'moore',
        torus: bool = False,
        capacity: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.neighborhood = neighborhood
        self.torus = torus
        self.capacity = capacity


@dataclass(frozen=True)
class Adjacency:
    """Neighbourhood of a node of a graph: the nodes along its edges, in node order, after the node itself if included.

    Rules receive the values of the neighbours padded to the largest degree of the graph.

    Args:
        center:
            Whether the node itself comes first.

    Examples:
        >>> from skcomplex.core.model import Adjacency
        >>> Adjacency(center=True).center
        True
    """

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

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.model import NetworkSpace
        >>> NetworkSpace(nx.cycle_graph(5)).graph.number_of_nodes()
        5
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.NETWORK

    n_dim = 1

    def __init__(self, graph: nx.Graph, *, capacity: int | None = None, fill: float = 0.0) -> None:
        self.graph = graph
        self.capacity = capacity
        self.fill = fill

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
            >>> from skcomplex.core.model import Adjacency, NetworkSpace
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


class ContinuousSpace(Space):
    """Rectangle in which agents have real positions and find neighbours within a radius.

    Args:
        width:
            The width.

        height:
            The height.

        torus:
            Whether opposite edges are joined.

    Examples:
        >>> from skcomplex.core.model import ContinuousSpace
        >>> ContinuousSpace(10.0, 5.0).height
        5.0
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.CONTINUOUS

    n_dim = 2

    def __init__(self, width: float, height: float, *, torus: bool = False) -> None:
        self.width = width
        self.height = height
        self.torus = torus


class EuclideanSpace(Space):
    """Euclidean space with Cartesian coordinates.

    Args:
        n_dim:
            The number of dimensions.

    Examples:
        >>> from skcomplex.core.model import EuclideanSpace
        >>> EuclideanSpace(n_dim=2).n_dim
        2
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.CONTINUOUS

    def __init__(self, n_dim: int = 3) -> None:
        self.n_dim = n_dim
