"""Networks: growth, turnover, ensembles, robustness, cascades, walks, diffusion, PageRank, structure, percolation."""

from ._copying_vertex import CopyingVertexElement
from ._davidsen_ebel_bornholdt import ClosingNodeElement, davidsen_ebel_bornholdt
from ._dependent_node import DependentNodeElement
from ._duplication_deletion import DuplicatingNodeElement, duplication_deletion
from ._ensemble_node import EnsembleNodeElement
from ._ensembles import (
    configuration_model,
    interdependent_network_cascade,
    kronecker_graph,
    network_attack,
    random_graph_ensembles,
)
from ._explosive import ClusterNodeElement, explosive_percolation
from ._failing import FailingElement
from ._fitness_vertex import FitnessVertexElement
from ._giant_component import GiantComponentInteraction
from ._growth import barabasi_albert, graphlet_arrival, holme_kim, popularity_similarity_network, vertex_copying
from ._hypergraphs import growing_simplicial_complex, hypergraph_random_walk
from ._link import LinkElement
from ._loaded_node import LoadedNodeElement
from ._network_measures import NetworkMeasuresElement
from ._network_node import NetworkNodeElement
from ._network_walker import NetworkWalkerElement
from ._node_content import NodeContentElement
from ._node_merging import MergingNodeElement, node_merging
from ._node_rank import NodeRankElement
from ._overload import motter_lai_overload
from ._processes import network_diffusion, network_random_walk, pagerank
from ._regulated_link import RegulatedLinkElement
from ._regulated_node import RegulatedNodeElement
from ._similarity_vertex import SimilarityVertexElement
from ._simplex import SimplexElement
from ._simplex_ageing import SimplexAgeingInteraction
from ._size_biased_move import SizeBiasedMoveInteraction
from ._spatial_triadic import (
    PercolatingNodeElement,
    PercolationInteraction,
    SignedLinkElement,
    SignedRegulationInteraction,
    spatial_triadic_percolation,
)
from ._structure import network_structure_analysis
from ._superposition import SuperposedNodeElement, network_superposition
from ._triad_vertex import TriadVertexElement
from ._triadic import triadic_percolation
from ._triadic_regulation import TriadicRegulationInteraction
from ._vertex import VertexElement
from ._walker_density import WalkerDensityElement

__all__: list[str] = [
    'ClosingNodeElement',
    'ClusterNodeElement',
    'CopyingVertexElement',
    'DependentNodeElement',
    'DuplicatingNodeElement',
    'EnsembleNodeElement',
    'FailingElement',
    'FitnessVertexElement',
    'GiantComponentInteraction',
    'LinkElement',
    'LoadedNodeElement',
    'MergingNodeElement',
    'NetworkMeasuresElement',
    'NetworkNodeElement',
    'NetworkWalkerElement',
    'NodeContentElement',
    'NodeRankElement',
    'PercolatingNodeElement',
    'PercolationInteraction',
    'RegulatedLinkElement',
    'RegulatedNodeElement',
    'SignedLinkElement',
    'SignedRegulationInteraction',
    'SimilarityVertexElement',
    'SimplexAgeingInteraction',
    'SimplexElement',
    'SizeBiasedMoveInteraction',
    'SuperposedNodeElement',
    'TriadVertexElement',
    'TriadicRegulationInteraction',
    'VertexElement',
    'WalkerDensityElement',
    'barabasi_albert',
    'configuration_model',
    'davidsen_ebel_bornholdt',
    'duplication_deletion',
    'explosive_percolation',
    'graphlet_arrival',
    'growing_simplicial_complex',
    'holme_kim',
    'hypergraph_random_walk',
    'interdependent_network_cascade',
    'kronecker_graph',
    'motter_lai_overload',
    'network_attack',
    'network_diffusion',
    'network_random_walk',
    'network_structure_analysis',
    'network_superposition',
    'node_merging',
    'pagerank',
    'popularity_similarity_network',
    'random_graph_ensembles',
    'spatial_triadic_percolation',
    'triadic_percolation',
    'vertex_copying',
]
