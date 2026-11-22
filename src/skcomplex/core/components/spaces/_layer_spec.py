"""Define the spec of a named layer of edges over the nodes of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import networkx as nx

from ...base import BaseSpec
from ...errors import ValidationError


@dataclass(frozen=True)
class LayerSpec(BaseSpec):
    """A spec of a named relation over the nodes of a network space.

    It holds a second set of edges over the same people or places as the network, such as who works with whom in
    `LayerSpec('work', nx.path_graph(3))`. Interactions and groups on edges name the layer they use, and a law reads the
    neighbours along it with `AdjacencyNeighborhood(layer='work')`. People meet at home and at work in an epidemic,
    banks lend and trade on separate networks, and cities are joined by roads and by flights.

    Read more in the user guide, under Spaces.

    Args:
        label:
            The name of the layer, unique in its network space.

        graph:
            The NetworkX graph of the relation, over the nodes of the network space.

    See Also:
        NetworkSpace: Holds the layers over its nodes.
        AdjacencyNeighborhood: Reads the neighbours along one layer.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.components.spaces import LayerSpec, NetworkSpace
        >>> space = NetworkSpace(nx.path_graph(3), layers=[LayerSpec('work', nx.complete_graph(3))]).prepare()
        >>> space.graphs_['work'].number_of_edges()
        3
    """

    label: str
    graph: nx.Graph

    def _check(self) -> None:
        """Raise an error for a label that is not a string.

        Raises:
            ValidationError:
                The label is not a string.
        """
        if not isinstance(self.label, str):
            error_msg = f'The label of a layer must be a string, but it is {self.label!r}.'
            raise ValidationError(error_msg)
