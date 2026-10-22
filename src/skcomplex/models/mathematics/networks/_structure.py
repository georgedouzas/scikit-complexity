"""Network structure model: a recorded network with its centralities, mixing, communities, and motifs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

import networkx as nx
import numpy as np

from ....core.analysis import assortativity, centrality, communities, motif_zscores, summarize_network
from ....core.base import Domain, VisualSpec, state
from ....core.components import Element
from ....core.laws import Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView
from ....core.space import NetworkSpace
from ....core.system import Observable, System

BLOCKS = 4
BLOCK_SIZE = 100
IN_DEGREE = 16
OUT_DEGREE = 2

NetworkMeasures = Element.define(
    'NetworkMeasures',
    parameters={
        'path_length': 0.0,
        'assortativity': 0.0,
        'modularity': 0.0,
        'agreement': 0.0,
        'triangle_zscore': 0.0,
    },
)


class NetworkNode(Element):
    """Node of a network that carries its degree, clustering, betweenness, and community.

    Args:
        label:
            The label of the agent or group.

        degree:
            The degree of the node.

        clustering:
            The clustering coefficient of the node.

        betweenness:
            The normalized betweenness centrality of the node.

        community:
            The index of the community of the node.

    Examples:
        >>> from skcomplex.models.mathematics.networks import NetworkNode
        >>> NetworkNode.group('nodes', 3)
        Group('nodes', element=NetworkNode, size=3)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'community'})
    degree: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    clustering: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    betweenness: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    community: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule that keeps the node in place.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.stay)]

    def stay(self, agent: AgentView, model: ModelView) -> None:
        """Keep the node and its measures unchanged.

        Args:
            agent:
                The node.

            model:
                The model.
        """


def _path_length(s: Any) -> Any:  # noqa: ANN401
    """Return the mean shortest path length of the largest component."""
    return s.measures.path_length


def _assortativity(s: Any) -> Any:  # noqa: ANN401
    """Return the degree assortativity."""
    return s.measures.assortativity


def _modularity(s: Any) -> Any:  # noqa: ANN401
    """Return the modularity of the detected communities."""
    return s.measures.modularity


def _agreement(s: Any) -> Any:  # noqa: ANN401
    """Return the agreement of the detected communities with the reference partition."""
    return s.measures.agreement


def _triangle_zscore(s: Any) -> Any:  # noqa: ANN401
    """Return the z-score of triangles against degree-preserving edge switches."""
    return s.measures.triangle_zscore


def _planted_partition(random_state: int | None) -> nx.Graph:
    """Return a stochastic block model of 4 blocks of 100 nodes with mean in-block degree 16 and out-block degree 2."""
    inside = IN_DEGREE / (BLOCK_SIZE - 1)
    outside = OUT_DEGREE / ((BLOCKS - 1) * BLOCK_SIZE)
    probabilities = [[inside if row == column else outside for column in range(BLOCKS)] for row in range(BLOCKS)]
    return nx.stochastic_block_model([BLOCK_SIZE] * BLOCKS, probabilities, seed=random_state)


def _reference(graph: nx.Graph, attribute: str | None) -> list[object] | None:
    """Return the reference community of every node from an attribute, or `None` when a node lacks it."""
    if attribute is None:
        return None
    values = nx.get_node_attributes(graph, attribute)
    return [values[node] for node in graph.nodes()] if len(values) == graph.number_of_nodes() else None


def _structure_observables(system: System) -> list[Observable]:
    """Return the clustering, path length, assortativity, modularity, agreement, and triangle z-score.

    The agreement is 0 without a reference partition, and the triangle z-score is 0 without null samples.
    """
    return [
        Observable('clustering', group='nodes', variable='clustering', reduce='mean'),
        Observable('path_length', _path_length),
        Observable('assortativity', _assortativity),
        Observable('modularity', _modularity),
        Observable('agreement', _agreement),
        Observable('triangle_zscore', _triangle_zscore),
    ]


@register_model(
    field='mathematics/networks',
    tags=('network', 'centrality', 'communities', 'motifs', 'assortativity'),
    references=('Newman 2010', 'Milo et al. 2002', 'Blondel et al. 2008', 'Bertin 2021'),
    defaults={'until': 1, 'dt': 1},
    observables=_structure_observables,
)
def network_structure_analysis(
    graph: nx.Graph | None = None,
    *,
    reference: str | None = 'block',
    null_samples: int = 10,
    random_state: int | None = 0,
) -> System:
    """Return a network recorded with its centralities, degree mixing, communities, and triangle z-score.

    Every node is an agent that carries its degree, clustering, betweenness, and the community found by Louvain
    modularity maximization. The observables report the clustering, the mean path length of the largest component, the
    degree assortativity, the modularity, the agreement of the communities with a reference partition, and the z-score
    of triangles against degree-preserving edge switches.

    Args:
        graph:
            The undirected NetworkX graph, or `None` for a stochastic block model of 4 blocks of 100 nodes with mean
            in-block degree 16 and out-block degree 2.

        reference:
            The node attribute that holds the reference partition, such as `'block'` of a stochastic block model or
            `'club'` of Zachary's karate club, or `None`. The agreement is reported only when every node has it.

        null_samples:
            The number of edge-switched graphs of the triangle z-score, which is reported only when positive.

        random_state:
            The seed of the block model, the community detection, and the null graphs.

    Returns:
        system:
            The model, with the node group `nodes` and the network measures `measures`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.mathematics.networks import network_structure_analysis
        >>> system = network_structure_analysis(nx.karate_club_graph(), reference='club', null_samples=0)
        >>> system.elements[0].size
        34
    """
    network = _planted_partition(random_state) if graph is None else graph
    found = communities(network, reference=_reference(network, reference), random_state=random_state)
    summary = summarize_network(network)
    clustering = nx.clustering(network)
    betweenness = centrality(network, 'betweenness')
    nodes = list(network.nodes())
    group = NetworkNode.group(
        'nodes',
        len(nodes),
        location=nodes,
        degree=np.asarray([network.degree(node) for node in nodes], dtype=float),
        clustering=np.asarray([clustering[node] for node in nodes], dtype=float),
        betweenness=np.asarray([betweenness[node] for node in nodes], dtype=float),
        community=found.labels.astype(float),
    )
    zscore = (
        motif_zscores(network, samples=null_samples, random_state=random_state).zscores['triangle']
        if null_samples
        else 0.0
    )
    measures = NetworkMeasures(
        'measures',
        path_length=summary.path_length,
        assortativity=float(assortativity(network)),
        modularity=found.modularity,
        agreement=0.0 if found.agreement is None else found.agreement,
        triangle_zscore=zscore,
    )
    return System(components=[group, measures], space=NetworkSpace(network))
