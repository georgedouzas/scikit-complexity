"""Define the spaces that vector states live in."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx
from sklearn.base import BaseEstimator


class Space(BaseEstimator):
    """Base class of spaces, which give vector states and parameters their dimension.

    Subclasses expose the number of dimensions as `n_dim`.
    """

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
        >>> from skcomplex.model import GridSpace
        >>> GridSpace(20, 20, capacity=1).n_dim
        2
    """

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


class NetworkSpace(Space):
    """Network whose nodes agents occupy, with the neighbours of a node given by its edges.

    Args:
        graph:
            The NetworkX graph.

        capacity:
            The largest number of agents on a node, or `None` for no limit.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.model import NetworkSpace
        >>> NetworkSpace(nx.cycle_graph(5)).graph.number_of_nodes()
        5
    """

    n_dim = 1

    def __init__(self, graph: nx.Graph, *, capacity: int | None = None) -> None:
        self.graph = graph
        self.capacity = capacity


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
        >>> from skcomplex.model import ContinuousSpace
        >>> ContinuousSpace(10.0, 5.0).height
        5.0
    """

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
        >>> from skcomplex.model import EuclideanSpace
        >>> EuclideanSpace(n_dim=2).n_dim
        2
    """

    def __init__(self, n_dim: int = 3) -> None:
        self.n_dim = n_dim
