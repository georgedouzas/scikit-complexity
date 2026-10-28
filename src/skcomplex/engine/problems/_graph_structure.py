"""Host graph of a graph rewriting system at one generation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx

from ._structure import Structure


class GraphStructure(Structure):
    """Host graph of a graph rewriting system at one generation.

    Args:
        graph:
            The graph, which the structure copies and freezes.

        generation:
            The generation.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.engine.problems._graph_structure import GraphStructure
        >>> GraphStructure(nx.cycle_graph(4), 0).parts
        4
    """

    def __init__(self, graph: nx.Graph, generation: int) -> None:
        super().__init__(generation)
        self.graph = nx.freeze(graph.copy())

    @property
    def parts(self) -> int:
        """Number of nodes."""
        return int(self.graph.number_of_nodes())

    def count(self, name: str) -> int:
        """Return the number of nodes with a label.

        Args:
            name:
                The label.

        Returns:
            count:
                The number of nodes whose `label` attribute is the label.
        """
        return sum(1 for _, label in self.graph.nodes(data='label') if label == name)
