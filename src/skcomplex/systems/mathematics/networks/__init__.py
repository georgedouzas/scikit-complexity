"""Networks: growth, turnover, ensembles, robustness, cascades, walks, diffusion, PageRank, structure, percolation."""

from ..common._link import LinkElement
from ..common._network_node import NetworkNodeElement
from ..common._regulated_node import RegulatedNodeElement
from ..common._vertex import VertexElement
from ._davidsen_ebel_bornholdt import ClosingNodeElement, davidsen_ebel_bornholdt
from ._duplication_deletion import DuplicatingNodeElement, duplication_deletion
from ._ensembles import (
    DependentNodeElement,
    EnsembleNodeElement,
    FailingElement,
    configuration_model,
    interdependent_network_cascade,
    kronecker_graph,
    network_attack,
    random_graph_ensembles,
)
from ._explosive import ClusterNodeElement, explosive_percolation
from ._growth import (
    CopyingVertexElement,
    FitnessVertexElement,
    SimilarityVertexElement,
    TriadVertexElement,
    barabasi_albert,
    graphlet_arrival,
    holme_kim,
    popularity_similarity_network,
    vertex_copying,
)
from ._hypergraphs import (
    SimplexAgeingInteraction,
    SimplexElement,
    SizeBiasedMoveInteraction,
    WalkerDensityElement,
    growing_simplicial_complex,
    hypergraph_random_walk,
)
from ._node_merging import MergingNodeElement, node_merging
from ._overload import LoadedNodeElement, motter_lai_overload
from ._processes import (
    NetworkWalkerElement,
    NodeContentElement,
    NodeRankElement,
    network_diffusion,
    network_random_walk,
    pagerank,
)
from ._spatial_triadic import (
    PercolatingNodeElement,
    PercolationInteraction,
    SignedLinkElement,
    SignedRegulationInteraction,
    spatial_triadic_percolation,
)
from ._structure import NetworkMeasuresElement, network_structure_analysis
from ._superposition import SuperposedNodeElement, network_superposition
from ._triadic import GiantComponentInteraction, RegulatedLinkElement, TriadicRegulationInteraction, triadic_percolation

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
