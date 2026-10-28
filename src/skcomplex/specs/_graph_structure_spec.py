"""Define the graph structure spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import networkx as nx

from ._base_structure_spec import BaseStructureSpec


@dataclass(frozen=True)
class GraphStructureSpec(BaseStructureSpec):
    """Host graph of a graph rewriting system at one generation.

    Args:
        graph:
            The graph, a frozen copy that later generations do not change.

        generation:
            The generation.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.specs import GraphStructureSpec
        >>> GraphStructureSpec(nx.freeze(nx.cycle_graph(4)), 0).parts
        4
    """

    graph: nx.Graph
    generation: int

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
