"""Evolution systems.

They cover evolutionary games, quasispecies, hypercycles, finite populations, landscapes, and coevolution.
"""

from ..common.populations import ReplicatorInteraction
from ._assembly import (
    AssemblyGrowthSystem,
    AssemblyLevelElement,
    PolymerAssemblyPoolSystem,
    PolymerElement,
    PolymerJoiningInteraction,
)
from ._coevolution import BakSneppenSystem, SpeciesElement
from ._drift import AlleleFrequencyElement, KimuraDiffusionSystem
from ._finite_populations import (
    AdaptiveClimbingInteraction,
    AdaptiveClimbingSystem,
    ClimberElement,
    IndividualElement,
    MoranProcessSystem,
    MoranStepInteraction,
    NKLandscapeSystem,
    TangledNatureLifeCycleInteraction,
    TangledNatureSystem,
)
from ._games import CyclicSpeciesElement, ReplicatorDynamicsSystem, RockPaperScissorsSystem, StrategyElement
from ._networks import (
    ChemicalElement,
    CompetingSpeciesElement,
    CompetitiveJainKrishnaSystem,
    CompetitiveSelectionInteraction,
    DependencyFieldInteraction,
    JainKrishnaSystem,
    SoleManrubiaSystem,
    TaxonElement,
)
from ._neutral_network_analysis import NeutralNetworkResult, neutral_network
from ._nkcs import CoevolutionaryClimbingInteraction, CoevolvingSpeciesElement, NKCSSystem
from ._prebiotic import (
    CatalysisInteraction,
    EigenQuasispeciesSystem,
    GenotypeClassElement,
    HypercyclesSystem,
    MasterSequenceElement,
    MoleculeElement,
)
from ._rna_walks import (
    NucleotideElement,
    RNANeutralNetworksSystem,
    fold_sequence,
    mutate_point,
    read_base_pair_distance,
    read_plane_layout,
    read_sequence,
)
from ._substitutions import LocusElement, OriginFixationWalkSystem, SubstitutionInteraction

__all__: list[str] = [
    'AdaptiveClimbingInteraction',
    'AdaptiveClimbingSystem',
    'AlleleFrequencyElement',
    'AssemblyGrowthSystem',
    'AssemblyLevelElement',
    'BakSneppenSystem',
    'CatalysisInteraction',
    'ChemicalElement',
    'ClimberElement',
    'CoevolutionaryClimbingInteraction',
    'CoevolvingSpeciesElement',
    'CompetingSpeciesElement',
    'CompetitiveJainKrishnaSystem',
    'CompetitiveSelectionInteraction',
    'CyclicSpeciesElement',
    'DependencyFieldInteraction',
    'EigenQuasispeciesSystem',
    'GenotypeClassElement',
    'HypercyclesSystem',
    'IndividualElement',
    'JainKrishnaSystem',
    'KimuraDiffusionSystem',
    'LocusElement',
    'MasterSequenceElement',
    'MoleculeElement',
    'MoranProcessSystem',
    'MoranStepInteraction',
    'NKCSSystem',
    'NKLandscapeSystem',
    'NeutralNetworkResult',
    'NucleotideElement',
    'OriginFixationWalkSystem',
    'PolymerAssemblyPoolSystem',
    'PolymerElement',
    'PolymerJoiningInteraction',
    'RNANeutralNetworksSystem',
    'ReplicatorDynamicsSystem',
    'ReplicatorInteraction',
    'RockPaperScissorsSystem',
    'SoleManrubiaSystem',
    'SpeciesElement',
    'StrategyElement',
    'SubstitutionInteraction',
    'TangledNatureLifeCycleInteraction',
    'TangledNatureSystem',
    'TaxonElement',
    'fold_sequence',
    'mutate_point',
    'neutral_network',
    'read_base_pair_distance',
    'read_plane_layout',
    'read_sequence',
]
