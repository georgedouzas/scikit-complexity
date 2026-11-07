"""Evolution: evolutionary games, quasispecies, hypercycles, finite populations, landscapes, and coevolution."""

from ..common.populations import ReplicatorInteraction
from ._assembly import (
    AssemblyLevelElement,
    PolymerElement,
    PolymerJoiningInteraction,
    assembly_growth,
    polymer_assembly_pool,
)
from ._coevolution import SpeciesElement, bak_sneppen
from ._drift import AlleleFrequencyElement, kimura_diffusion
from ._finite_populations import (
    AdaptiveClimbingInteraction,
    ClimberElement,
    IndividualElement,
    MoranStepInteraction,
    TangledNatureLifeCycleInteraction,
    adaptive_climbing,
    moran_process,
    nk_landscape,
    tangled_nature,
)
from ._games import CyclicSpeciesElement, StrategyElement, replicator_dynamics, rock_paper_scissors
from ._networks import (
    ChemicalElement,
    CompetingSpeciesElement,
    CompetitiveSelectionInteraction,
    DependencyFieldInteraction,
    TaxonElement,
    competitive_jain_krishna,
    jain_krishna,
    sole_manrubia,
)
from ._neutral_network_analysis import NeutralNetworkResult, neutral_network
from ._nkcs import CoevolutionaryClimbingInteraction, CoevolvingSpeciesElement, nkcs
from ._prebiotic import (
    CatalysisInteraction,
    GenotypeClassElement,
    MasterSequenceElement,
    MoleculeElement,
    eigen_quasispecies,
    hypercycles,
)
from ._rna_walks import (
    NucleotideElement,
    fold_sequence,
    mutate_point,
    read_base_pair_distance,
    read_plane_layout,
    read_sequence,
    rna_neutral_networks,
)
from ._substitutions import LocusElement, SubstitutionInteraction, origin_fixation_walk

__all__: list[str] = [
    'AdaptiveClimbingInteraction',
    'AlleleFrequencyElement',
    'AssemblyLevelElement',
    'CatalysisInteraction',
    'ChemicalElement',
    'ClimberElement',
    'CoevolutionaryClimbingInteraction',
    'CoevolvingSpeciesElement',
    'CompetingSpeciesElement',
    'CompetitiveSelectionInteraction',
    'CyclicSpeciesElement',
    'DependencyFieldInteraction',
    'GenotypeClassElement',
    'IndividualElement',
    'LocusElement',
    'MasterSequenceElement',
    'MoleculeElement',
    'MoranStepInteraction',
    'NeutralNetworkResult',
    'NucleotideElement',
    'PolymerElement',
    'PolymerJoiningInteraction',
    'ReplicatorInteraction',
    'SpeciesElement',
    'StrategyElement',
    'SubstitutionInteraction',
    'TangledNatureLifeCycleInteraction',
    'TaxonElement',
    'adaptive_climbing',
    'assembly_growth',
    'bak_sneppen',
    'competitive_jain_krishna',
    'eigen_quasispecies',
    'fold_sequence',
    'hypercycles',
    'jain_krishna',
    'kimura_diffusion',
    'moran_process',
    'mutate_point',
    'neutral_network',
    'nk_landscape',
    'nkcs',
    'origin_fixation_walk',
    'polymer_assembly_pool',
    'read_base_pair_distance',
    'read_plane_layout',
    'read_sequence',
    'replicator_dynamics',
    'rna_neutral_networks',
    'rock_paper_scissors',
    'sole_manrubia',
    'tangled_nature',
]
