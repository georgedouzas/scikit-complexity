"""Networks: growth, turnover, ensembles, robustness, cascades, walks, diffusion, PageRank, structure, percolation."""

from ..common._network_node import NetworkNodeElement
from ..common._regulated_node import RegulatedNodeElement
from ..common._vertex import VertexElement
from ._davidsen_ebel_bornholdt import ClosingNodeElement, DavidsenEbelBornholdtSystem
from ._duplication_deletion import DuplicatingNodeElement, DuplicationDeletionSystem
from ._ensembles import (
    ConfigurationModelSystem,
    DependentNodeElement,
    EnsembleNodeElement,
    FailingElement,
    InterdependentNetworkCascadeSystem,
    KroneckerGraphSystem,
    NetworkAttackSystem,
    RandomGraphEnsemblesSystem,
)
from ._explosive import ClusterNodeElement, ExplosivePercolationSystem
from ._growth import (
    BarabasiAlbertSystem,
    CopyingVertexElement,
    FitnessVertexElement,
    GraphletArrivalSystem,
    HolmeKimSystem,
    PopularitySimilarityNetworkSystem,
    SimilarityVertexElement,
    TriadVertexElement,
    VertexCopyingSystem,
)
from ._hypergraphs import (
    GrowingSimplicialComplexSystem,
    HypergraphRandomWalkSystem,
    SimplexAgeingInteraction,
    SimplexElement,
    SizeBiasedMoveInteraction,
    WalkerDensityElement,
)
from ._node_merging import MergingNodeElement, NodeMergingSystem
from ._overload import LoadedNodeElement, MotterLaiOverloadSystem
from ._processes import (
    NetworkDiffusionSystem,
    NetworkRandomWalkSystem,
    NetworkWalkerElement,
    NodeContentElement,
    NodeRankElement,
    PagerankSystem,
)
from ._spatial_triadic import (
    PercolatingNodeElement,
    PercolationInteraction,
    SignedLinkElement,
    SignedRegulationInteraction,
    SpatialTriadicPercolationSystem,
)
from ._structure import NetworkMeasuresElement, NetworkStructureAnalysisSystem
from ._superposition import NetworkSuperpositionSystem, SuperposedNodeElement
from ._triadic import (
    GiantComponentInteraction,
    RegulatedLinkElement,
    TriadicPercolationSystem,
    TriadicRegulationInteraction,
)

__all__: list[str] = [
    'BarabasiAlbertSystem',
    'ClosingNodeElement',
    'ClusterNodeElement',
    'ConfigurationModelSystem',
    'CopyingVertexElement',
    'DavidsenEbelBornholdtSystem',
    'DependentNodeElement',
    'DuplicatingNodeElement',
    'DuplicationDeletionSystem',
    'EnsembleNodeElement',
    'ExplosivePercolationSystem',
    'FailingElement',
    'FitnessVertexElement',
    'GiantComponentInteraction',
    'GraphletArrivalSystem',
    'GrowingSimplicialComplexSystem',
    'HolmeKimSystem',
    'HypergraphRandomWalkSystem',
    'InterdependentNetworkCascadeSystem',
    'KroneckerGraphSystem',
    'LoadedNodeElement',
    'MergingNodeElement',
    'MotterLaiOverloadSystem',
    'NetworkAttackSystem',
    'NetworkDiffusionSystem',
    'NetworkMeasuresElement',
    'NetworkNodeElement',
    'NetworkRandomWalkSystem',
    'NetworkStructureAnalysisSystem',
    'NetworkSuperpositionSystem',
    'NetworkWalkerElement',
    'NodeContentElement',
    'NodeMergingSystem',
    'NodeRankElement',
    'PagerankSystem',
    'PercolatingNodeElement',
    'PercolationInteraction',
    'PopularitySimilarityNetworkSystem',
    'RandomGraphEnsemblesSystem',
    'RegulatedLinkElement',
    'RegulatedNodeElement',
    'SignedLinkElement',
    'SignedRegulationInteraction',
    'SimilarityVertexElement',
    'SimplexAgeingInteraction',
    'SimplexElement',
    'SizeBiasedMoveInteraction',
    'SpatialTriadicPercolationSystem',
    'SuperposedNodeElement',
    'TriadVertexElement',
    'TriadicPercolationSystem',
    'TriadicRegulationInteraction',
    'VertexCopyingSystem',
    'VertexElement',
    'WalkerDensityElement',
]
