"""Network structure system: a recorded network with its centralities, mixing, communities, and motifs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np

from ....core.analysis import motifs, network_measures
from ....core.components.elements import Element, GroupElement
from ....core.components.spaces import NetworkSpace
from ....core.components.system import System
from ....core.errors import AnalysisError
from ....core.specs import ObservableSpec, ParameterSpec, VisualSpec
from ..._catalog import register_system
from ..common._network_node import NetworkNodeElement


class NetworkMeasuresElement(Element):
    """Measures of a whole network, held as parameters that the observables of the system report.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        path_length:
            The mean shortest path length of the largest component.

        assortativity:
            The degree assortativity.

        modularity:
            The modularity of the communities found.

        agreement:
            The agreement of the communities found with a reference partition.

        triangle_zscore:
            The z-score of the number of triangles against degree-preserving edge switches.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import NetworkMeasuresElement
        >>> NetworkMeasuresElement('measures', modularity=0.4).modularity
        0.4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        path_length: Annotated[float, ParameterSpec()] = 0.0,
        assortativity: Annotated[float, ParameterSpec()] = 0.0,
        modularity: Annotated[float, ParameterSpec()] = 0.0,
        agreement: Annotated[float, ParameterSpec()] = 0.0,
        triangle_zscore: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.path_length = path_length
        self.assortativity = assortativity
        self.modularity = modularity
        self.agreement = agreement
        self.triangle_zscore = triangle_zscore


BLOCKS = 4


BLOCK_SIZE = 100


IN_DEGREE = 16


OUT_DEGREE = 2


def _path_length(s: SimpleNamespace) -> object:
    """Return the mean shortest path length of the largest component."""
    return s.measures.path_length


def _assortativity(s: SimpleNamespace) -> object:
    """Return the degree assortativity."""
    return s.measures.assortativity


def _modularity(s: SimpleNamespace) -> object:
    """Return the modularity of the detected communities."""
    return s.measures.modularity


def _agreement(s: SimpleNamespace) -> object:
    """Return the agreement of the detected communities with the reference partition."""
    return s.measures.agreement


def _triangle_zscore(s: SimpleNamespace) -> object:
    """Return the z-score of triangles against degree-preserving edge switches."""
    return s.measures.triangle_zscore


def _planted_partition(random_state: int | None) -> nx.Graph:
    """Return a stochastic block system of 4 blocks of 100 nodes with mean in-block degree 16 and out-block degree 2."""
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


def _structure_observables(system: System) -> list[ObservableSpec]:
    """Return the clustering, path length, assortativity, modularity, agreement, and triangle z-score.

    The agreement is 0 without a reference partition, and the triangle z-score is 0 without null samples.
    """
    return [
        ObservableSpec('clustering', group='nodes', variable='clustering', reduce='mean'),
        ObservableSpec('path_length', _path_length),
        ObservableSpec('assortativity', _assortativity),
        ObservableSpec('modularity', _modularity),
        ObservableSpec('agreement', _agreement),
        ObservableSpec('triangle_zscore', _triangle_zscore),
    ]


@register_system(
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
            The undirected NetworkX graph, or `None` for a stochastic block system of 4 blocks of 100 nodes with mean
            in-block degree 16 and out-block degree 2.

        reference:
            The node attribute that holds the reference partition, such as `'block'` of a stochastic block system or
            `'club'` of Zachary's karate club, or `None`. The agreement is reported only when every node has it.

        null_samples:
            The number of edge-switched graphs of the triangle z-score, which is reported only when positive.

        random_state:
            The seed of the block system, the community detection, and the null graphs.

    Returns:
        system:
            The system, with the node group `nodes` and the network measures `measures`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.mathematics.networks import network_structure_analysis
        >>> system = network_structure_analysis(nx.karate_club_graph(), reference='club', null_samples=0)
        >>> system.elements[0].prepare().size_
        34
    """
    network = _planted_partition(random_state) if graph is None else graph
    found = network_measures(
        network,
        kind='communities',
        reference=_reference(network, reference),
        random_state=random_state,
    )
    summary = network_measures(network)
    clustering = nx.clustering(network)
    centrality = network_measures(network, kind='centrality', measure='betweenness').centrality
    assortativity = network_measures(network, kind='assortativity').assortativity
    labels, modularity, path_length = found.labels, found.modularity, summary.path_length
    if labels is None or modularity is None or path_length is None or centrality is None or assortativity is None:
        error_msg = 'The network measures miss the communities, the summary, the centrality, or the assortativity.'
        raise AnalysisError(error_msg)
    betweenness = centrality[0]
    nodes = list(network.nodes())
    group = GroupElement(
        'nodes',
        element=NetworkNodeElement,
        size=len(nodes),
        values={
            'degree': np.asarray([network.degree(node) for node in nodes], dtype=float),
            'clustering': np.asarray([clustering[node] for node in nodes], dtype=float),
            'betweenness': np.asarray([betweenness[node] for node in nodes], dtype=float),
            'community': labels[0].astype(float),
        },
        location=nodes,
    )
    zscore = 0.0
    if null_samples:
        scores = motifs(network, kind='z-scores', samples=null_samples, random_state=random_state).z_scores
        zscore = 0.0 if scores is None else scores[0]['triangle']
    measures = NetworkMeasuresElement(
        'measures',
        path_length=float(path_length[0]),
        assortativity=float(assortativity[0]),
        modularity=float(modularity[0]),
        agreement=0.0 if found.agreement is None else float(found.agreement[0]),
        triangle_zscore=zscore,
    )
    return System(elements=[group, measures], space=NetworkSpace(network))
