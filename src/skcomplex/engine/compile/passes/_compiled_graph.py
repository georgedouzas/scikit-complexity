"""Graph rewriting system of one host graph, with double-pushout productions matched by NetworkX."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx.algorithms import isomorphism

from ....core.elements import HostGraphElement
from ....core.laws import GraphProductionLaw
from ....errors import RewritingError
from .._compiled_word import IDENTIFIER
from .._graph_structure import GraphStructure


class CompiledGraph:
    """Graph rewriting system of one host graph, with double-pushout productions matched by NetworkX.

    Args:
        path:
            The path of the host graph.

        host:
            The host graph element.

        productions:
            The graph productions that target it, applied one after the other within a generation.
    """

    def __init__(self, path: str, host: HostGraphElement, productions: Sequence[GraphProductionLaw]) -> None:
        self.path = path
        self.graph = host.graph
        self.max_size = int(host.max_size)
        self.max_matches = int(host.max_matches)
        self.productions = tuple(productions)
        for production in self.productions:
            outside = set(production.interface.nodes) - (set(production.left.nodes) & set(production.right.nodes))
            if outside:
                error_msg = (
                    f'The interface of a production of {path!r} has the nodes {sorted(map(str, outside))}, which are '
                    'not nodes of both its left and its right graph.'
                )
                raise RewritingError(error_msg)
        labels = [label for _, label in self.graph.nodes(data='label')]
        labels += [label for item in self.productions for _, label in item.right.nodes(data='label')]
        self.counted = tuple(
            sorted({label for label in labels if isinstance(label, str) and IDENTIFIER.match(label)}),
        )

    def initial(self, values: npt.NDArray[np.float64]) -> GraphStructure:
        """Return the host graph at generation 0.

        Args:
            values:
                The parameter values, unused.

        Returns:
            graph:
                The initial graph.
        """
        del values
        return GraphStructure(self.graph, 0)

    def counts(self, graph: GraphStructure) -> list[float]:
        """Return the count of nodes of every label and the number of nodes.

        Args:
            graph:
                The graph.

        Returns:
            counts:
                The counts, in the order of the states.
        """
        return [float(graph.count(name)) for name in self.counted] + [float(graph.parts)]

    def _key(self, graph: nx.Graph, first: Any, second: Any) -> Any:  # noqa: ANN401
        """Return the key of an edge, ordered for directed graphs."""
        return (first, second) if graph.is_directed() else frozenset((first, second))

    def _select(
        self,
        graph: nx.Graph,
        production: GraphProductionLaw,
        rng: np.random.Generator,
    ) -> list[dict[Any, Any]]:
        """Return a maximal set of parallel independent matches that satisfy the dangling condition."""
        matcher_type = isomorphism.DiGraphMatcher if graph.is_directed() else isomorphism.GraphMatcher
        matcher = matcher_type(graph, production.left, node_match=self._matches, edge_match=self._matches)
        found = [
            {pattern: host for host, pattern in item.items()}
            for item in itertools.islice(matcher.subgraph_monomorphisms_iter(), self.max_matches)
        ]
        left, interface = production.left, production.interface
        used: set[Any] = set()
        deleted: set[Any] = set()
        chosen = []
        for index in rng.permutation(len(found)):
            match = found[index]
            edges = {(self._key(graph, match[u], match[v]), interface.has_edge(u, v)) for u, v in left.edges}
            nodes = {match[node] for node in left.nodes}
            removed_nodes = {match[node] for node in left.nodes if node not in interface}
            removed = removed_nodes | {key for key, kept in edges if not kept}
            items = nodes | {key for key, _ in edges}
            incident = {self._key(graph, u, v) for node in removed_nodes for u, v in nx.edges(graph, node)}
            if graph.is_directed():
                incident |= {self._key(graph, u, v) for node in removed_nodes for u, v in graph.in_edges(node)}
            if not incident <= removed or items & deleted or removed & used:
                continue
            used |= items
            deleted |= removed
            chosen.append(match)
        return chosen

    def _apply(self, graph: nx.Graph, production: GraphProductionLaw, matches: list[dict[Any, Any]]) -> None:
        """Delete and glue the parts of every chosen match."""
        left, interface, right = production.left, production.interface, production.right
        for match in matches:
            for u, v in left.edges:
                if not interface.has_edge(u, v) and graph.has_edge(match[u], match[v]):
                    graph.remove_edge(match[u], match[v])
            graph.remove_nodes_from([match[node] for node in left.nodes if node not in interface])
        fresh = itertools.count(max((node for node in graph.nodes if isinstance(node, int)), default=-1) + 1)
        for match in matches:
            names = {node: match[node] for node in interface.nodes}
            for node in right.nodes:
                if node not in names:
                    names[node] = next(item for item in fresh if not graph.has_node(item))
                graph.add_node(names[node], **right.nodes[node])
            for u, v, attributes in right.edges(data=True):
                if not interface.has_edge(u, v):
                    graph.add_edge(names[u], names[v], **attributes)

    def rewrite(
        self,
        structure: GraphStructure,
        values: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> GraphStructure:
        """Return the host graph of the next generation.

        Args:
            structure:
                The host graph.

            values:
                The parameter values, unused.

            rng:
                The random stream that orders the matches.

        Returns:
            graph:
                The next host graph.

        Raises:
            RewritingError:
                The next graph would exceed the maximum size.
        """
        del values
        graph = nx.DiGraph(structure.graph) if structure.graph.is_directed() else nx.Graph(structure.graph)
        for production in self.productions:
            self._apply(graph, production, self._select(graph, production, rng))
            if graph.number_of_nodes() > self.max_size:
                error_msg = (
                    f'Host graph {self.path!r} would reach {graph.number_of_nodes()} nodes at generation '
                    f'{structure.generation + 1}, above its maximum size {self.max_size}.'
                )
                raise RewritingError(error_msg)
        return GraphStructure(graph, structure.generation + 1)

    @staticmethod
    def _matches(attributes: dict[str, Any], pattern: dict[str, Any]) -> bool:
        """Check that a host node or edge has the label of a pattern node or edge, where the pattern gives one."""
        return 'label' not in pattern or attributes.get('label') == pattern['label']
