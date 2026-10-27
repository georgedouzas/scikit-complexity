"""Define graph production laws, the double-pushout productions that rewrite host graphs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

import networkx as nx

from ..errors import RewritingError
from ..spaces import DiscreteTime
from ..specs import RequirementSpec
from ._base import BaseLaw


class GraphProductionLaw(BaseLaw):
    """Double-pushout graph production: a left graph, an interface, and a right graph that share nodes by name.

    A match of the left graph in a host graph deletes the nodes and edges of the left graph that are not in the
    interface and adds the nodes and edges of the right graph that are not in it. Nodes and edges match when their
    `label` attributes agree, where the left graph gives one. A host graph element returns its graph productions from
    `_get_laws`, and every generation applies all of them at once.

    Args:
        left:
            The left graph.

        interface:
            The interface, whose nodes are nodes of both other graphs.

        right:
            The right graph.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.laws import GraphProductionLaw
        >>> interface = nx.Graph()
        >>> interface.add_nodes_from('ab')
        >>> production = GraphProductionLaw(nx.Graph([('a', 'b')]), interface, nx.Graph([('a', 'c'), ('c', 'b')]))
        >>> sorted(production.check().run({}))
        ['c']
    """

    def __init__(self, left: nx.Graph, interface: nx.Graph, right: nx.Graph, *, label: str = '') -> None:
        super().__init__(label)
        self.left = left
        self.interface = interface
        self.right = right

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states, since a graph production rewrites the host graph of its element as a whole."""
        return ()

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _check(self) -> Self:
        """Raise an error for an interface node that is not a node of both the left and the right graph.

        Returns:
            law:
                The law itself.
        """
        missing = set(self.interface.nodes) - (set(self.left.nodes) & set(self.right.nodes))
        if missing:
            error_msg = (
                f'The interface nodes {sorted(map(str, missing))} are not nodes of both the left and right graph.'
            )
            raise RewritingError(error_msg)
        return self

    def _run(self, inputs: object) -> object:
        """Return the nodes the production adds, the nodes of the right graph outside the interface.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value, unused.

        Returns:
            nodes:
                The added nodes.
        """
        return set(self.right.nodes) - set(self.interface.nodes)
