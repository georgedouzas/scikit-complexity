"""Evolution: evolutionary games, quasispecies, hypercycles, finite populations, landscapes, and coevolution."""

from ._coevolution import SpeciesElement, bak_sneppen
from ._finite_populations import (
    AdaptiveClimbingInteraction,
    ClimberElement,
    IndividualElement,
    MoranStepInteraction,
    adaptive_climbing,
    moran_process,
    nk_landscape,
    tangled_nature,
)
from ._games import CyclicSpeciesElement, StrategyElement, replicator_dynamics, rock_paper_scissors
from ._networks import (
    CatalyticLinkElement,
    ChemicalElement,
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
    eigen_quasispecies,
    hypercycles,
)
from ._replicator import ReplicatorInteraction
from ._substitutions import LocusElement, SubstitutionInteraction, origin_fixation_walk
from ._tangled_nature_life_cycle import TangledNatureLifeCycleInteraction

__all__: list[str] = [
    'AdaptiveClimbingInteraction',
    'CatalysisInteraction',
    'CatalyticLinkElement',
    'ChemicalElement',
    'ClimberElement',
    'CyclicSpeciesElement',
    'DependencyFieldInteraction',
    'GenotypeClassElement',
    'IndividualElement',
    'LocusElement',
    'MasterSequenceElement',
    'MoleculeElement',
    'MoranStepInteraction',
    'ReplicatorInteraction',
    'SpeciesElement',
    'StrategyElement',
    'SubstitutionInteraction',
    'TangledNatureLifeCycleInteraction',
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
