"""Evolution: evolutionary games, quasispecies, hypercycles, finite populations, landscapes, and coevolution."""

from ._coevolution import SpeciesElement, bak_sneppen
from ._finite_populations import (
    AdaptiveClimbingInteraction,
    ClimberElement,
    IndividualElement,
    MoranStepInteraction,
    TangledNatureStepInteraction,
    adaptive_climbing,
    moran_process,
    nk_landscape,
    tangled_nature,
)
from ._games import (
    CyclicSpeciesElement,
    ReplicationInteraction,
    StrategyElement,
    replicator_dynamics,
    rock_paper_scissors,
)
from ._networks import (
    CatalyticLinkElement,
    ChemicalElement,
    CouplingDriftInteraction,
    CouplingElement,
    DependencyFieldInteraction,
    TaxonElement,
    jain_krishna,
    sole_manrubia,
)
from ._prebiotic import (
    CatalysisInteraction,
    GenotypeClassElement,
    MasterSequenceElement,
    MoleculeElement,
    MutationInteraction,
    eigen_quasispecies,
    hypercycles,
)
from ._substitutions import LocusElement, SubstitutionInteraction, origin_fixation_walk

__all__: list[str] = [
    'AdaptiveClimbingInteraction',
    'CatalysisInteraction',
    'CatalyticLinkElement',
    'ChemicalElement',
    'ClimberElement',
    'CouplingDriftInteraction',
    'CouplingElement',
    'CyclicSpeciesElement',
    'DependencyFieldInteraction',
    'GenotypeClassElement',
    'IndividualElement',
    'LocusElement',
    'MasterSequenceElement',
    'MoleculeElement',
    'MoranStepInteraction',
    'MutationInteraction',
    'ReplicationInteraction',
    'SpeciesElement',
    'StrategyElement',
    'SubstitutionInteraction',
    'TangledNatureStepInteraction',
    'TaxonElement',
    'adaptive_climbing',
    'bak_sneppen',
    'eigen_quasispecies',
    'hypercycles',
    'jain_krishna',
    'moran_process',
    'nk_landscape',
    'origin_fixation_walk',
    'replicator_dynamics',
    'rock_paper_scissors',
    'sole_manrubia',
    'tangled_nature',
]
